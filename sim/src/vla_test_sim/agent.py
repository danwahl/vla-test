"""The SO-101 arm, under absolute joint-position control."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import sapien
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINT_NAMES = [*ARM_JOINTS, "gripper"]

# One width for open and one for closed, so a policy reads a single unambiguous pair. The
# jaw faces stand 57 mm apart open and 22 mm apart closed, either side of the 30 mm block.
GRIPPER_OPEN = 0.6
GRIPPER_CLOSED = 0.09

BASE_POSE = sapien.Pose(p=[0.0, 0.0, 0.74])

# The arm folded back over its own base, the physical SO-101's rest pose.
HOME_QPOS = np.array([-0.1414, 0.3624, -1.1317, 1.5485, -0.0915, GRIPPER_OPEN], np.float32)

_GRIP = {"material": "grip", "patch_radius": 0.1, "min_patch_radius": 0.1}


@register_agent()
class SO101(BaseAgent):
    uid = "so101"
    urdf_path = str(Path(__file__).parent / "description" / "so101.urdf")
    urdf_config: ClassVar[dict] = {
        "_materials": {
            "grip": {"static_friction": 1.5, "dynamic_friction": 1.5, "restitution": 0.0}
        },
        "link": {"gripper_link": _GRIP, "moving_jaw_so101_v1_link": _GRIP},
    }
    keyframes: ClassVar[dict] = {"home": Keyframe(qpos=HOME_QPOS, pose=BASE_POSE)}

    arm_joint_names = ARM_JOINTS
    gripper_joint_names: ClassVar[list[str]] = ["gripper"]

    @property
    def _controller_configs(self):
        return {
            "pd_joint_pos": PDJointPosControllerConfig(
                JOINT_NAMES,
                lower=None,
                upper=None,
                # Generic stiffness (1000) lets the shoulder and elbow sag ~0.06 rad
                # reaching down under gravity. The gripper stays soft, so closing inside
                # the block presses it rather than ejecting it.
                stiffness=[6000.0, 6000.0, 4000.0, 6000.0, 8000.0, 300.0],
                damping=[77.5, 77.5, 63.0, 77.5, 89.0, 30.0],
                force_limit=[500.0, 500.0, 500.0, 500.0, 500.0, 50.0],
                normalize_action=False,
                use_delta=False,
                # ManiSkill's setpoint interpolation is CPU-only. On physx_cuda it
                # silently no-ops the drive-target write and the arm snaps to qpos 0.
                interpolate=False,
            )
        }

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.tcp_link = self.robot.links_map["gripper_frame_link"]
        self.wrist_cam_link = self.robot.links_map["wrist_camera_frame_link"]
        self.top_cam_link = self.robot.links_map["top_camera_frame_link"]

    @property
    def tcp_pose(self):
        return self.tcp_link.pose
