"""Register the block-stacking env into RLinf's ManiSkill task set, and shape what the
rollout reads off it.

Copied to ``rlinf/envs/maniskill/tasks/so101_block_stack.py``, where RLinf imports every
module at startup, so the arm reaches it from ``PYTHONPATH``.
"""

import torch

import sim  # noqa: F401  registers SO101BlockStack-v1 and the arm


def wrap_obs(raw_obs, env):
    """The cameras, joint positions and prompts in the slots the rollout reads."""
    sensors = raw_obs["sensor_data"]
    top = sensors["top"]["rgb"].to(torch.uint8)
    return {
        "main_images": top,
        "wrist_images": sensors["wrist"]["rgb"].to(torch.uint8),
        "extra_view_images": None,
        "states": env.agent.robot.get_qpos()[..., :6].to(top.device, torch.float32),
        "task_descriptions": env.prompts(),
    }
