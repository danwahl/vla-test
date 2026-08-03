# vla-test

An end-to-end example of training a vision-language-action policy on an [SO-101](https://github.com/TheRobotStudio/SO-ARM100): oracle, demonstration collection, fine-tuning, evaluation, reinforcement learning, hardware.

## Sim

A [ManiSkill3](https://github.com/haosulab/ManiSkill) cell, with the arm under absolute joint-position control, three 30 mm colour cubes on a table, and two cameras on the intrinsic calibrated from the physical arm's InnoMaker U20CAM.

```bash
cd sim
uv sync --extra dev
uv run pytest
uv run python scripts/render.py    # writes wrist/top/inspect PNGs
```

Rendering needs a GPU with Vulkan.
