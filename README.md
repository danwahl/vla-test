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
