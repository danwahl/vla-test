"""Three colour cubes on a table, seen by a wrist camera and an overhead camera."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SimConfig

from .agent import BASE_POSE, HOME_QPOS, SO101  # noqa: F401  (import registers the agent)

# Ideal pinhole fitted to the physical InnoMaker U20CAM, as a 480x480 centre crop of the
# 640x480 calibration: same rays, so f and cy are unchanged and cx drops by (640-480)/2.
IMAGE_SIZE = 480
K_SIM = np.array([[546.0, 0.0, 240.0], [0.0, 546.0, 240.0], [0.0, 0.0, 1.0]], np.float32)

# The URDF's optical-frame links are OpenCV-style (+Z out of the lens); SAPIEN cameras
# are ROS-style (+X out of the lens). Columns are the camera axes in link coordinates.
_MOUNT = np.eye(4, dtype=np.float32)
_MOUNT[:3, :3] = [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
CAM_MOUNT_POSE = sapien.Pose(_MOUNT)

TABLE_POS = (0.30, 0.0, 0.72)
TABLE_SIZE = (0.80, 0.60, 0.04)
TABLE_TOP_Z = 0.74
TABLE_RGBA = (0.55, 0.38, 0.22, 1.0)

CUBE_SIDE = 0.030
CUBE_HALF = CUBE_SIDE / 2
CUBE_MASS = 0.05
CUBE_REST_Z = TABLE_TOP_Z + CUBE_HALF + 0.001
CUBE_COLOURS = {
    "red": (0.85, 0.10, 0.10, 1.0),
    "green": (0.10, 0.65, 0.15, 1.0),
    "blue": (0.10, 0.20, 0.85, 1.0),
}
CUBE_LAYOUT = {"red": (0.29, 0.00), "green": (0.30, 0.09), "blue": (0.29, 0.18)}


@register_env("SO101Blocks-v1", max_episode_steps=200)
class SO101Blocks(BaseEnv):
    SUPPORTED_ROBOTS: ClassVar[list[str]] = ["so101"]
    # ManiSkill defaults to normalized_dense, which raises on the first step() until
    # compute_dense_reward is implemented.
    SUPPORTED_REWARD_MODES: ClassVar[tuple[str, ...]] = ("none",)

    def __init__(self, *args, robot_uids="so101", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        # 400 Hz is required for a 30 mm grasp: below it the contact impulse from
        # closing the jaws rides the arm up and the fingers meet above the cube.
        return SimConfig(sim_freq=400, control_freq=10)

    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig(name, pose=CAM_MOUNT_POSE, width=IMAGE_SIZE, height=IMAGE_SIZE,
                         intrinsic=K_SIM, near=0.001, far=10.0, mount=mount)
            for name, mount in [("wrist", self.agent.wrist_cam_link),
                                ("top", self.agent.top_cam_link)]
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[1.15, -0.95, 1.30], target=[0.12, 0.0, 0.98])
        return CameraConfig("render_camera", pose, 512, 512, fov=np.deg2rad(50), near=0.01, far=100)

    def _load_agent(self, options):
        super()._load_agent(options, BASE_POSE)

    def _load_scene(self, options):
        build_ground(self.scene, altitude=0.0)

        builder = self.scene.create_actor_builder()
        half = [s / 2 for s in TABLE_SIZE]
        builder.add_box_collision(half_size=half)
        builder.add_box_visual(half_size=half,
                               material=sapien.render.RenderMaterial(base_color=TABLE_RGBA))
        builder.initial_pose = sapien.Pose(p=TABLE_POS)
        self.table = builder.build_static(name="table")

        friction = sapien.physx.PhysxMaterial(
            static_friction=1.5, dynamic_friction=1.5, restitution=0.0
        )
        self.cubes = {}
        for name, (x, y) in CUBE_LAYOUT.items():
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[CUBE_HALF] * 3, material=friction,
                                      density=CUBE_MASS / CUBE_SIDE**3)
            builder.add_box_visual(
                half_size=[CUBE_HALF] * 3,
                material=sapien.render.RenderMaterial(base_color=CUBE_COLOURS[name]),
            )
            builder.initial_pose = sapien.Pose(p=[x, y, CUBE_REST_Z])
            self.cubes[name] = builder.build(name=f"cube_{name}")

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            self.agent.robot.set_qpos(
                torch.tensor(HOME_QPOS, device=self.device).repeat(len(env_idx), 1)
            )
            self.agent.robot.set_pose(BASE_POSE)

    def evaluate(self):
        return {}
