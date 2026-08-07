"""Register the block-stacking env into RLinf's ManiSkill task set, and shape what the
rollout reads off it.

Copied to ``rlinf/envs/maniskill/tasks/so101_block_stack.py``, where RLinf imports every
module at startup. Point ``VLA_TEST_DIR`` at this repository.
"""

import os
import sys
from pathlib import Path

import torch

# RLinf sweeps this package on import, so a checkout that only runs other embodiments
# carries on without the arm.
package = Path(os.environ.get("VLA_TEST_DIR", Path.home() / "vla-test")) / "sim" / "src"
if package.is_dir():
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
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
