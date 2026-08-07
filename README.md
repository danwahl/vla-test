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

```bash
uv run python sim/scripts/collect.py OUT
```

360 episodes in [LeRobot](https://github.com/huggingface/lerobot) v3.0 format at 10 Hz, 60 for each of the six colour orderings: the five arm joints and the gripper as `observation.state` and `action`, and 480x480 H.264 from the wrist and top cameras. `meta/train_layouts.jsonl` records the spawn each episode started from, and `meta/eval_layouts.jsonl` 150 more, held out, to measure a policy on.

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

`--video DIR` records each batch.

## Reinforcement learning

The fine-tuned checkpoint is the warm start for PPO in [RLinf](https://github.com/RLinf/RLinf), which drives the same env in parallel and scores the task and nothing else: a step is worth 1 while the commanded block is seated on its target and out of the jaws, and 0 otherwise. RLinf differences that level, so an episode's rewards sum to 1 if the stack is still standing at the end and 0 if it is not, which makes the return the same number the policy is graded on.

`sim/scripts/rl_layouts.py` screens a pool of spawns the oracle stacks, cycling the colour pairs so all six are covered evenly:

```bash
uv run python sim/scripts/rl_layouts.py rl_layouts.jsonl --count 5000
```

The env takes `layouts=PATH` and draws each reset from the file, which keeps the spawns the oracle cannot stack out of the reward and fixes the population an eval is read against.

`rl/convert.py` rewrites a merged checkpoint into the layout RLinf's openpi actor loads:

```bash
uv run python rl/convert.py \
    /data/checkpoints/pi05_so101_block_stack_sim/merged \
    /data/checkpoints/pi05_so101_block_stack_sim/openpi
```

RLinf is `rl/rlinf`, a submodule pinned to the commit this was written against. The run happens in its `rlinf/rlinf:agentic-rlinf0.4-maniskill_libero` image, which carries a venv per embodiment; the openpi one has ManiSkill, openpi, a CUDA torch and the packages RLinf imports, so RLinf itself and this repository's `sim` go on `PYTHONPATH` rather than being installed. `rl/patch.py` copies two modules into the submodule and edits four call sites.

```bash
source /opt/venv/openpi/bin/activate
git submodule update --init
python rl/patch.py

export VLA_TEST_DIR=$PWD
export EMBODIED_PATH=$PWD/rl/rlinf/examples/embodiment
export PYTHONPATH=$PWD/rl/rlinf:$PWD/sim/src
export SFT_CKPT=/data/checkpoints/pi05_so101_block_stack_sim/openpi
export RL_LAYOUTS=$PWD/rl_layouts.jsonl

# score the warm start, then PPO from it
python rl/rlinf/evaluations/eval_embodied_agent.py \
    --config-path $VLA_TEST_DIR/rl --config-name pi05_so101_ppo
python $EMBODIED_PATH/train_embodied_agent.py \
    --config-path $VLA_TEST_DIR/rl --config-name pi05_so101_ppo
```

The run config stays here; its `searchpath` picks up RLinf's own config tree for the pieces it builds on.
