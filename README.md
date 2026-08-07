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

The fine-tuned checkpoint is the warm start for PPO in [RLinf](https://github.com/RLinf/RLinf), which drives the same env in parallel and scores the milestones the arm passes on the way to a stack: a grasp, a lift, the block carried over its target, and the stack itself. Each is banked once reached, so the reward only rises and a step-to-step difference is never negative. A small bonus accrues for every step the stack stands, which is what pays for letting go of it.

`rl/convert.py` rewrites a merged checkpoint into the layout RLinf's openpi actor loads:

```bash
uv run python rl/convert.py \
    /data/checkpoints/pi05_so101_block_stack_sim/merged \
    /data/checkpoints/pi05_so101_block_stack_sim/openpi
```

RLinf is not vendored. In a checkout of it (this was written against `d3aff54`), four files go in:

| from | to |
|---|---|
| `rl/pi05_so101_ppo.yaml` | `examples/embodiment/config/` |
| `rl/env/so101_block_stack.yaml` | `examples/embodiment/config/env/` |
| `rl/task.py` | `rlinf/envs/maniskill/tasks/so101_block_stack.py` |
| `rl/dataconfig.py` | `rlinf/models/embodiment/openpi/dataconfig/so101_block_stack.py` |

and four call sites take a case for this arm. `So101BlockStackDataConfig` joins the registry in `.../dataconfig/__init__.py` under the name `pi05_so101_block_stack`, whose prefix is where the actor reads pi0.5's language-token budget; `rlinf/envs/action_utils.py` returns joint-space actions unchanged; `rlinf/config.py` resolves this robot to `pd_joint_pos`; and `rlinf/envs/maniskill/maniskill_env.py` gains the observation and reward branches the env config names:

```python
if wrap_obs_mode == "so101":
    from rlinf.envs.maniskill.tasks import so101_block_stack

    return so101_block_stack.wrap_obs(raw_obs, self.env.unwrapped)
```
```python
elif getattr(self.cfg, "reward_mode", "default") == "milestone":
    reward = info["reward"]
```

Both entrypoints run from that checkout, and take the paths to this one and to the converted weights:

```bash
export VLA_TEST_DIR=/path/to/vla-test
export SFT_CKPT=/data/checkpoints/pi05_so101_block_stack_sim/openpi
export EMBODIED_PATH=$PWD/examples/embodiment

# score the warm start
python evaluations/eval_embodied_agent.py \
    --config-path $EMBODIED_PATH/config --config-name pi05_so101_ppo
# PPO from it
python $EMBODIED_PATH/train_embodied_agent.py \
    --config-path $EMBODIED_PATH/config --config-name pi05_so101_ppo
```

The actor and the rollout are held on one 80 GB card, each offloaded while the other runs. Scoring builds the env and the rollout and no actor, and reads the model from `rollout.model`, which mirrors the actor's. It draws the spawns the env samples, so it is the number the climb is read against, and a different population from the screened layouts `sim/scripts/eval.py` replays.
