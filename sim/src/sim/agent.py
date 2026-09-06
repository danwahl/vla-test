"""The SO-101 arm, under absolute joint-position control."""

from __future__ import annotations

from typing import ClassVar

import sapien
import torch
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent

from .spec import ARM_JOINTS, HOME_QPOS, JOINT_NAMES, URDF_PATH

BASE_POSE = sapien.Pose(p=[0.0, 0.0, 0.74])

_GRIP = {"material": "grip", "patch_radius": 0.1, "min_patch_radius": 0.1}


@register_agent()
class SO101(BaseAgent):
    uid = "so101"
    urdf_path = str(URDF_PATH)
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
        self.jaw_links = [self.robot.links_map[name]
                          for name in ("gripper_link", "moving_jaw_so101_v1_link")]

    @property
    def tcp_pose(self):
        return self.tcp_link.pose

    def is_grasping(self, actor, min_force=0.5):
        """Whether both jaws are pressing on ``actor``."""
        forces = torch.stack([
            torch.linalg.norm(self.scene.get_pairwise_contact_forces(link, actor), dim=1)
            for link in self.jaw_links
        ])
        return (forces >= min_force).all(dim=0)
