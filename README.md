# vla-test

An end-to-end example of training a vision-language-action policy on an [SO-101](https://github.com/TheRobotStudio/SO-ARM100): oracle, demonstration collection, fine-tuning, evaluation, reinforcement learning, hardware.

## Sim

A [ManiSkill3](https://github.com/haosulab/ManiSkill) cell, with the arm under absolute joint-position control, three 30 mm colour blocks on a table, and two cameras on the intrinsic calibrated from the physical arm's InnoMaker U20CAM.

```bash
uv sync
uv run pytest
uv run python sim/scripts/render.py    # writes wrist/top/inspect PNGs
```

Rendering needs a GPU with Vulkan.

## Demonstrations

A scripted oracle plans the pick-and-stack in closed form and drives every parallel environment at once. Layouts are screened on state alone and only the ones it stacks get rendered, so every episode is a success.

An episode starts where a cycle would leave the arm, jaws open or shut, and its first move is out to a stand-off above the block the prompt names. A pick that closed on nothing leaves the arm among those poses, so recovering from one is in the demonstrations.

Every env carries its own cursor through the cycle, so each one arrives at a keypose in the steps its own travel takes and an episode is as long as its own arm was moving.

```bash
uv run python sim/scripts/collect.py OUT
```

360 episodes in [LeRobot](https://github.com/huggingface/lerobot) v3.0 format at 10 Hz, 60 for each of the six colour orderings: the five arm joints and the gripper as `observation.state` and `action`, and 480x480 H.264 from the wrist and top cameras. `meta/train_layouts.jsonl` records the spawn and the opening pose each episode started from, and `meta/eval_layouts.jsonl` 150 more, held out, to measure a policy on. Both are what `reset` takes back, so an eval sets off from where a demonstration did.

`meta/stats.json` gives `observation.state` and `action` the arm's joint travel as their normalization range. pi0.5 spells the state out in its prompt as one of 256 bins across that range and does not clip, so a range drawn from the demonstrations alone would read every pose outside them as the same bin.

## Fine-tuning

`train/pi05_so101_lora_backbone.yaml` is the recipe: LoRA on pi0.5's gemma_2b language backbone, the action expert and the projection heads dense, SigLIP frozen.

```bash
set -a; . ./.env; set +a    # WANDB_API_KEY, see .env.example
uv run lerobot-train --config_path=train/pi05_so101_lora_backbone.yaml \
                     --output_dir=/data/checkpoints/pi05_so101_block_stack_sim \
                     --job_name=pi05_so101_block_stack_sim \
                     --wandb.run_id=pi05_so101_block_stack_sim
```

`uv sync --package vla-test-train` installs this without the sim.

`train/merge.py` folds the adapter into the base weights, for a checkpoint that loads without one:

```bash
uv run python train/merge.py \
    /data/checkpoints/pi05_so101_block_stack_sim/checkpoints/last/pretrained_model \
    /data/checkpoints/pi05_so101_block_stack_sim/merged
```

## Evaluation

The checkpoint is rolled out on the 150 held-out layouts and scored by the gates that screened the oracle. Chunks are stitched with [Real-Time Chunking](https://www.physicalintelligence.company/research/real_time_chunking).

```bash
uv run python sim/scripts/eval.py \
    /data/checkpoints/pi05_so101_block_stack_sim/checkpoints/last/pretrained_model
```

`--video DIR` records each batch, and `--no-rtc` denoises each chunk on its own. `--layouts FILE` scores the checkpoint on another dataset's held-out layouts; the normalization travels with the checkpoint, so only the spawns change. The run reports what the checkpoint costs to run beside what it scores: the weights it loads, the peak it reaches rolling out, and the median time a chunk takes to denoise.

`--int8` holds the weights that read the observation at eight bits, five sevenths of what a policy holds: the language backbone and the vision tower. The action expert and the projections that emit the commands keep the precision they were trained at. The 20k sim checkpoint holds 4.26 GiB of weights under the flag and 7.24 without, and scores 101 of the 150 held-out layouts against 105, six won and ten lost. An eval this size resolves about three points, so that is a difference it cannot separate from none. The weights are unpacked for each matmul, which costs about a tenth of the time a chunk takes.

## Reinforcement learning

The fine-tuned checkpoint is the warm start for PPO in [RLinf](https://github.com/RLinf/RLinf), which drives the same env in parallel and scores the task and nothing else: a step is worth 1 while the commanded block is seated on its target and out of the jaws, and 0 otherwise. RLinf differences that level, so an episode's rewards sum to 1 if the stack is still standing at the end and 0 if it is not, which makes the return the same number the policy is graded on.

`sim/scripts/rl_layouts.py` screens a pool of spawns the oracle stacks, cycling the colour pairs so all six are covered evenly:

```bash
uv run python sim/scripts/rl_layouts.py rl_layouts.jsonl --count 5000
```

The env takes `layouts=PATH` and draws each reset from the file, which keeps the spawns the oracle cannot stack out of the reward. Training draws from the pool at random; the periodic eval takes `sample=False` and the dataset's held-out layouts, so every check scores the same episodes as `sim/scripts/eval.py`.

`rl/to_openpi.py` rewrites a merged checkpoint into the layout RLinf's openpi actor loads:

```bash
uv run python rl/to_openpi.py \
    /data/checkpoints/pi05_so101_block_stack_sim/merged \
    /data/checkpoints/pi05_so101_block_stack_sim/openpi
```

`rl/Dockerfile` builds the environment a run happens in on top of RLinf's `rlinf/rlinf:agentic-rlinf0.4-maniskill_libero` image, which has a venv per embodiment: the openpi one includes ManiSkill, openpi, a CUDA torch and the packages RLinf imports. The build adds sshd, a clone of [RLinf](https://github.com/RLinf/RLinf) at `RLINF_COMMIT`, and a profile script that selects the venv and sources `rl/profile.sh` from the copied repository, so a login shell on the machine has the paths a run reads.

```bash
docker build -f rl/Dockerfile -t danwahl/vla-test-rl .
docker push danwahl/vla-test-rl
```

The image contains the base RLinf code, and the repository is copied onto a machine running it, so a config change needs only an rsync. `rl/patch.py` adds the arm to that clone:

```bash
cp /data/datasets/so101_block_stack_sim/meta/eval_layouts.jsonl .
rsync -a --exclude .venv --exclude .git . root@HOST:$VLA_TEST_DIR/
python $VLA_TEST_DIR/rl/patch.py $RLINF_DIR
```

Then, with `WANDB_API_KEY` and `HF_TOKEN` in the environment:

```bash
hf download drwahl/pi05_so101_block_stack_sim --include 'openpi/*' \
    --local-dir /data/checkpoints/pi05_so101_block_stack_sim

# score the warm start, then PPO from it
python $RLINF_DIR/evaluations/eval_embodied_agent.py \
    --config-path $VLA_TEST_DIR/rl --config-name pi05_so101_ppo
python $EMBODIED_PATH/train_embodied_agent.py \
    --config-path $VLA_TEST_DIR/rl --config-name pi05_so101_ppo
```

The run config stays here; its `searchpath` picks up RLinf's own config tree for the pieces it builds on.

`rl/pi05_so101_grpo.yaml` runs the same warm start without a critic. A group of rollouts shares a spawn and is scored against its own mean, so layout difficulty cancels rather than having to be learned, and the reward is the level `evaluate()` returns: a point for seating the stack, and a step's worth for every step it stands. That level only rises, so no step scores below zero.

```bash
python $EMBODIED_PATH/train_embodied_agent.py \
    --config-path $VLA_TEST_DIR/rl --config-name pi05_so101_grpo
```

`rl/to_lerobot.py` takes a checkpoint back the other way, into a directory `sim/scripts/eval.py` and the rest of lerobot read:

```bash
uv run python rl/to_lerobot.py \
    RESULTS/checkpoints/global_step_60/actor/model_state_dict/full_weights.pt \
    /data/checkpoints/pi05_so101_block_stack_sim/merged \
    /data/checkpoints/pi05_so101_block_stack_rl/global_step_60
```

## Hardware

`hw/` runs the same task on the physical arm through [LeRobot](https://github.com/huggingface/lerobot)'s `SO101Follower`. `hw/robot.py` holds the processor steps that make the arm look like the simulator: degrees to radians, `<joint>.pos` to the sim's bare joint names, and each lens undistorted and reprojected onto `K_SIM` as a 480x480 frame. Episodes are written with `sim/src/sim/dataset.py`, the schema `sim/scripts/collect.py` writes, so hardware and sim episodes are interchangeable.

`uv sync --package vla-test-hw` installs this without the trainer. `hw/robot.py` holds the rig's serial port and the USB port each camera sits in, and `hw/calibration/` each lens' charuco intrinsic.

The table is read from the sim rather than perceived. A layout is planned there, `hw/overlay.py` blends that layout's camera view with the live one, and the blocks are moved onto their sim positions:

```bash
uv run python -m hw.overlay --index 0
```

`hw/oracle_replay.py` does that and then sends the oracle's joint commands open loop, recording the arm's own states and frames against them:

```bash
uv run python -m hw.oracle_replay OUT --indices 0 1 2
```

Both take `--layouts`, which defaults to the spawns the sim demonstrations were collected on, so a policy trained on either set is still scored against layouts it has not seen.

`hw/teleop.py` drives the arm with a 3DConnexion SpaceMouse instead. The cap translates the tool, tilts it and turns the jaw; its two buttons work the gripper. The target is integrated from the home pose and solved by `sim/src/sim/kinematics.py`, the same closed form the oracle plans with, so teleoperated and scripted episodes occupy one joint-space distribution.

```bash
uv run python -m hw.teleop --held red --target green --out OUT
```

Without `--out` it teleoperates and records nothing.

`train/pi05_so101_hw.yaml` fine-tunes on what these record, as a fresh adapter over the merged sim weights:

```bash
uv run lerobot-train --config_path=train/pi05_so101_hw.yaml \
                     --output_dir=/data/checkpoints/pi05_so101_block_stack_hw \
                     --job_name=pi05_so101_block_stack_hw \
                     --wandb.run_id=pi05_so101_block_stack_hw
```

lerobot trains on one dataset, so learning from sim and hardware together means writing both into one. `sim/scripts/combine.py` copies the sim dataset and appends the hardware episodes to the copy, repeated `--repeat` times, which is what sets how much of a batch comes off the arm. `train/pi05_so101_mix.yaml` trains on the result from `lerobot/pi05_base`, at the sim recipe and budget:

```bash
uv run python sim/scripts/combine.py /data/datasets/so101_block_stack_mix

uv run lerobot-train --config_path=train/pi05_so101_mix.yaml \
                     --output_dir=/data/checkpoints/pi05_so101_block_stack_mix \
                     --job_name=pi05_so101_block_stack_mix \
                     --wandb.run_id=pi05_so101_block_stack_mix
```

`train/pi05_so101_hw_base.yaml` is the same recipe on the hardware episodes alone, the control for what the sim start supplies.

`hw/rollout.py` is the evaluation, with the arm where the simulator was. The blocks go anywhere on the table, the console's button starts the episode, and whether it stacked is the operator's call:

```bash
uv run python -m hw.rollout CHECKPOINT --held red --target blue
```

Chunks are stitched the way `sim/scripts/eval.py` stitches them. Denoising the next one takes long enough to see, so the arm runs on the chunk it already has while that happens, and RTC is told how many steps that will take so the two join.
