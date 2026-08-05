"""Three colour blocks on a table, seen by a wrist camera and an overhead camera."""

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
from mani_skill.utils.structs.pose import Pose
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

BLOCK_SIDE = 0.030
BLOCK_HALF = BLOCK_SIDE / 2
BLOCK_MASS = 0.05
BLOCK_REST_Z = TABLE_TOP_Z + BLOCK_HALF + 0.001
BLOCK_COLOURS = {
    "red": (0.85, 0.10, 0.10, 1.0),
    "green": (0.10, 0.65, 0.15, 1.0),
    "blue": (0.10, 0.20, 0.85, 1.0),
}
BLOCK_NAMES = tuple(BLOCK_COLOURS)
# Spawn box, inside the arm's top-down reach at both grasp and stack height. The
# separation clears a block rotated onto its 42 mm diagonal.
SPAWN_X = (0.17, 0.25)
SPAWN_Y = (-0.12, 0.12)
SPAWN_SEPARATION = 0.07

# The block is 4-fold symmetric top-down, so a quarter turn spans every distinct pose.
SPAWN_YAW = np.pi / 2

# Success gates. A lift clears the table; a stack is seated within a block half-width,
# one block-height up, with neither of the other two blocks shoved out of place.
LIFT_DZ = 0.05
STACK_XY_TOL = 0.018
STACK_Z_TOL = 0.012
DISTURB_TOL = 0.015

TASK_PROMPT = "stack the {held} block on the {target} block"


# Half again the oracle's cycle, and well past where a fine-tuned policy settles.
@register_env("SO101BlockStack-v1", max_episode_steps=200)
class SO101BlockStack(BaseEnv):
    SUPPORTED_ROBOTS: ClassVar[list[str]] = ["so101"]
    # ManiSkill defaults to normalized_dense, which raises on the first step() until
    # compute_dense_reward is implemented.
    SUPPORTED_REWARD_MODES: ClassVar[tuple[str, ...]] = ("none",)

    def __init__(self, *args, robot_uids="so101", **kwargs):
        # Shadows are the depth cue in the wrist view.
        kwargs.setdefault("enable_shadow", True)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        # 400 Hz is required for a 30 mm grasp: below it the contact impulse from
        # closing the jaws rides the arm up and the fingers meet above the block.
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
        pose = sapien_utils.look_at(eye=[0.66, -0.52, 1.12], target=[0.20, 0.02, 0.86])
        return CameraConfig("render_camera", pose, 512, 512, fov=np.deg2rad(52), near=0.01, far=100)

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
        self.blocks = {}
        for i, name in enumerate(BLOCK_NAMES):
            x, y = SPAWN_X[0] + 0.03 * i, SPAWN_Y[0]
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[BLOCK_HALF] * 3, material=friction,
                                      density=BLOCK_MASS / BLOCK_SIDE**3)
            builder.add_box_visual(
                half_size=[BLOCK_HALF] * 3,
                material=sapien.render.RenderMaterial(base_color=BLOCK_COLOURS[name]),
            )
            builder.initial_pose = sapien.Pose(p=[x, y, BLOCK_REST_Z])
            self.blocks[name] = builder.build(name=f"block_{name}")
        self._block_friction = friction

    def _after_reconfigure(self, options):
        # Across parallel scenes a material handed to the builder is reported back
        # correctly but is not what the solver uses, and blocks slide on one another as if
        # frictionless. Assigning it once the scene is up is what takes effect.
        for block in self.blocks.values():
            for obj in block._objs:
                body = obj.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
                for shape in body.collision_shapes:
                    shape.physical_material = self._block_friction

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            b = len(env_idx)
            self.agent.robot.set_qpos(torch.tensor(HOME_QPOS).repeat(b, 1))
            self.agent.robot.set_pose(BASE_POSE)

            # Spawns and colour pair can each be given instead of sampled, so a run can be
            # replayed and a batch can be made to cover the pairs evenly. One value is
            # shared by the whole batch; a batch of them is taken one per env.
            layout = (options or {}).get("layout") or {}

            def given(key, dtype, *shape):
                return torch.as_tensor(layout[key], dtype=dtype,
                                       device=self.device).expand(b, *shape)

            if "xy" in layout:
                xy, yaw = given("xy", torch.float32, 3, 2), given("yaw", torch.float32, 3)
            else:
                xy, yaw = self._sample_layout(b)
            if "held" in layout:
                held, target = given("held", torch.long), given("target", torch.long)
            else:
                held = torch.randint(3, (b,))
                target = (held + 1 + torch.randint(2, (b,))) % 3

            for i, name in enumerate(BLOCK_NAMES):
                pose = torch.zeros(b, 7)
                pose[:, :2] = xy[:, i]
                pose[:, 2] = BLOCK_REST_Z
                pose[:, 3] = torch.cos(yaw[:, i] / 2)
                pose[:, 6] = torch.sin(yaw[:, i] / 2)
                self.blocks[name].set_pose(Pose.create(pose))
                self.blocks[name].set_linear_velocity(torch.zeros(b, 3))
                self.blocks[name].set_angular_velocity(torch.zeros(b, 3))

            self.held, self.target = held, target
            self.spawns = torch.stack(
                [self.blocks[n].pose.p[env_idx] for n in BLOCK_NAMES], dim=1
            )
            self.lifted = torch.zeros(b, dtype=torch.bool)

    def _sample_layout(self, b):
        """Three block spawns, resampled until none of them overlap."""
        with torch.device(self.device):
            lo = torch.tensor([SPAWN_X[0], SPAWN_Y[0]])
            span = torch.tensor([SPAWN_X[1] - SPAWN_X[0], SPAWN_Y[1] - SPAWN_Y[0]])
            xy = lo + span * torch.rand(b, 3, 2)
            for _ in range(64):
                gap = torch.cdist(xy, xy) + torch.eye(3) * SPAWN_SEPARATION
                crowded = (gap < SPAWN_SEPARATION).any(-1).any(-1)
                if not crowded.any():
                    break
                xy[crowded] = lo + span * torch.rand(int(crowded.sum()), 3, 2)
            return xy, SPAWN_YAW * torch.rand(b, 3)

    def _block_positions(self):
        return torch.stack([self.blocks[n].pose.p for n in BLOCK_NAMES], dim=1)

    def layout(self):
        """What was just spawned, per env, in the form ``reset`` takes it back in.

        The blocks stand upright until something pushes them, so a spawn quaternion is a
        yaw and nothing else. Read it before stepping.
        """
        quat = torch.stack([self.blocks[n].pose.q for n in BLOCK_NAMES], dim=1)
        return {"xy": self._block_positions()[..., :2],
                "yaw": 2 * torch.atan2(quat[..., 3], quat[..., 0]),
                "held": self.held, "target": self.target}

    def evaluate(self):
        """The two gates: was the commanded block lifted, and did it come to rest stacked."""
        pos = self._block_positions()
        rows = torch.arange(self.num_envs, device=self.device)
        held, target = pos[rows, self.held], pos[rows, self.target]
        spawn = self.spawns[rows, self.held]

        self.lifted |= (held[:, 2] - spawn[:, 2]) > LIFT_DZ

        moved = torch.linalg.norm((pos - self.spawns)[..., :2], dim=-1)
        moved[rows, self.held] = 0.0
        stack_xy = torch.linalg.norm((held - target)[:, :2], dim=-1)
        stack_z = held[:, 2] - target[:, 2]
        stacked = (
            (stack_xy < STACK_XY_TOL)
            & ((stack_z - BLOCK_SIDE).abs() < STACK_Z_TOL)
            & (moved < DISTURB_TOL).all(-1)
        )
        return {"success": stacked, "lifted": self.lifted.clone(),
                "stack_xy": stack_xy, "stack_z": stack_z}
