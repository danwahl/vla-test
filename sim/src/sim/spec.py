"""What the arm, its cameras and the task are, apart from any simulation of them.

The simulator and the physical arm agree on these, and the physical arm reaches them
without a physics engine installed: nothing here imports one.
"""

from __future__ import annotations

import numpy as np

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINT_NAMES = [*ARM_JOINTS, "gripper"]

# One width for open and one for closed, so a policy reads a single unambiguous pair. The
# jaw faces stand 57 mm apart open and 22 mm apart closed, either side of the 30 mm block.
GRIPPER_OPEN = 0.6
GRIPPER_CLOSED = 0.09

# The arm folded back over its own base, the physical SO-101's rest pose.
HOME_QPOS = np.array([-0.1414, 0.3624, -1.1317, 1.5485, -0.0915, GRIPPER_OPEN], np.float32)

CAMERAS = ("top", "wrist")

# Ideal pinhole fitted to the physical InnoMaker U20CAM, as a 480x480 centre crop of the
# 640x480 calibration: same rays, so f and cy are unchanged and cx drops by (640-480)/2.
IMAGE_SIZE = 480
K_SIM = np.array([[546.0, 0.0, 240.0], [0.0, 546.0, 240.0], [0.0, 0.0, 1.0]], np.float32)

BLOCK_NAMES = ("red", "green", "blue")

TASK_PROMPT = "stack the {held} block on the {target} block"


def prompt(held, target):
    return TASK_PROMPT.format(held=BLOCK_NAMES[held], target=BLOCK_NAMES[target])
