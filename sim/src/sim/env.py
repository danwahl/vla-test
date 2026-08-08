"""Three colour blocks on a table, seen by a wrist camera and an overhead camera."""

from __future__ import annotations

import json
from pathlib import Path
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
# The ordered pairs of distinct blocks the task can name.
PAIRS = [(held, target) for held in range(3) for target in range(3) if held != target]
# Spawn box, inside the arm's top-down reach at both grasp and stack height. The
# separation clears a block rotated onto its 42 mm diagonal.
SPAWN_X = (0.17, 0.25)
SPAWN_Y = (-0.12, 0.12)
SPAWN_SEPARATION = 0.07

# The block is 4-fold symmetric top-down, so a quarter turn spans every distinct pose.
SPAWN_YAW = np.pi / 2

# Success gates. A lift clears the table; a stack is seated within a block half-width,
# one block-height up, out of the jaws, with neither of the other two blocks shoved out of
# place.
LIFT_DZ = 0.05
STACK_XY_TOL = 0.018
STACK_Z_TOL = 0.012
DISTURB_TOL = 0.015

# What a step of holding the stack is worth, against a point for first seating it. Set so
# that holding it for the rest of the episode is worth about as much as seating it.
HOLD_REWARD = 0.005

TASK_PROMPT = "stack the {held} block on the {target} block"


def prompt(held, target):
    return TASK_PROMPT.format(held=BLOCK_NAMES[held], target=BLOCK_NAMES[target])


def write_layouts(path, layouts, key):
    """One line per layout, giving the spawn a run can be replayed from."""
    with Path(path).open("w") as out:
        for i, item in enumerate(layouts):
            positions = [[float(x), float(y), BLOCK_REST_Z] for x, y in item["xy"]]
            out.write(json.dumps({
                key: i,
                "held": int(item["held"]),
                "target": int(item["target"]),
                "prompt": prompt(int(item["held"]), int(item["target"])),
                "positions": positions,
                "yaws": [float(yaw) for yaw in item["yaw"]],
                "max_yaw": SPAWN_YAW,
            }) + "\n")


def read_layouts(path):
    """Those lines back, in the form ``reset`` takes a layout."""
    with Path(path).open() as file:
        rows = [json.loads(line) for line in file]
    return {
        "xy": np.array([[p[:2] for p in row["positions"]] for row in rows], np.float32),
        "yaw": np.array([row["yaws"] for row in rows], np.float32),
        "held": np.array([row["held"] for row in rows]),
        "target": np.array([row["target"] for row in rows]),
    }


# Half again the oracle's cycle, and well past where a fine-tuned policy settles.
@register_env("SO101BlockStack-v1", max_episode_steps=200)
class SO101BlockStack(BaseEnv):
    SUPPORTED_ROBOTS: ClassVar[list[str]] = ["so101"]
    # ManiSkill defaults to normalized_dense, which raises on the first step() until
    # compute_dense_reward is implemented.
    SUPPORTED_REWARD_MODES: ClassVar[tuple[str, ...]] = ("none",)

    def __init__(self, *args, robot_uids="so101", layouts=None, sample=True, **kwargs):
        # Shadows are the depth cue in the wrist view.
        kwargs.setdefault("enable_shadow", True)
        # Screened spawns to draw resets from instead of sampling fresh ones. Read before
        # the base class reconfigures, since that reaches _initialize_episode.
        self._pool = read_layouts(layouts) if layouts else None
        # Unset, env i takes layout i from the pool at every reset, so two runs are scored
        # on the same episodes.
        self._sample = sample
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

        # Per-env state spans the whole batch and is written by index, so resetting some
        # envs while the rest keep running leaves those others untouched.
        with torch.device(self.device):
            self.held = torch.zeros(self.num_envs, dtype=torch.long)
            self.target = torch.zeros(self.num_envs, dtype=torch.long)
            self.spawns = torch.zeros(self.num_envs, 3, 3)
            self.milestones = {name: torch.zeros(self.num_envs, dtype=torch.bool)
                               for name in ("grasped", "lifted", "stacked")}
            self.held_steps = torch.zeros(self.num_envs)
            self.counted = torch.full((self.num_envs,), -1)

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            b = len(env_idx)
            self.agent.robot.set_qpos(torch.tensor(HOME_QPOS).repeat(b, 1))
            self.agent.robot.set_pose(BASE_POSE)

            # Spawns and colour pair can each be given instead of sampled, so a run can be
            # replayed and a batch can be made to cover the pairs evenly. One value is
            # shared by the whole batch; a batch of them is taken one per env.
            options = options or {}
            layout = options.get("layout") or {}
            if self._pool is not None:
                layout = self._draw(env_idx, options) | layout

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

            self.held[env_idx], self.target[env_idx] = held, target
            self.spawns[env_idx] = torch.stack(
                [self.blocks[n].pose.p[env_idx] for n in BLOCK_NAMES], dim=1
            )
            for reached in self.milestones.values():
                reached[env_idx] = False
            self.held_steps[env_idx] = 0.0
            self.counted[env_idx] = -1

    @property
    def total_num_trials(self):
        """Layouts a reset can name by index. Without a pool the spawns are sampled fresh
        and an index goes unread."""
        return len(self._pool["held"]) if self._pool is not None else 1

    def _draw(self, env_idx, options):
        """A layout from the pool per env resetting. ``episode_id`` names them by index,
        which is how a group of envs is given the same one. A given layout overrides them.
        """
        rows = options.get("episode_id")
        if rows is None:
            rows = (torch.randint(self.total_num_trials, (len(env_idx),))
                    if self._sample else env_idx)
        rows = rows.cpu().numpy() % self.total_num_trials
        return {key: value[rows] for key, value in self._pool.items()}

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

    def prompts(self):
        """What each env in the batch was asked to do."""
        return [prompt(held, target) for held, target
                in zip(self.held.tolist(), self.target.tolist(), strict=True)]

    def evaluate(self):
        """Whether the commanded block was lifted, and whether it is stacked on its target.

        A stack is seated and out of the jaws, so a block held at stacking height does not
        count. ``grasped`` and ``lifted`` latch; ``seated`` and ``success`` are read fresh
        each step.
        """
        pos = self._block_positions()
        rows = torch.arange(self.num_envs, device=self.device)
        held, target = pos[rows, self.held], pos[rows, self.target]
        spawn = self.spawns[rows, self.held]

        moved = torch.linalg.norm((pos - self.spawns)[..., :2], dim=-1)
        moved[rows, self.held] = 0.0
        stack_xy = torch.linalg.norm((held - target)[:, :2], dim=-1)
        stack_z = held[:, 2] - target[:, 2]
        seated = (
            (stack_xy < STACK_XY_TOL)
            & ((stack_z - BLOCK_SIDE).abs() < STACK_Z_TOL)
            & (moved < DISTURB_TOL).all(-1)
        )
        lifted = (held[:, 2] - spawn[:, 2]) > LIFT_DZ
        grasping = torch.stack(
            [self.agent.is_grasping(self.blocks[name]) for name in BLOCK_NAMES], dim=1
        )[rows, self.held]
        stacked = seated & ~grasping

        for name, reached in [("grasped", grasping), ("lifted", lifted),
                              ("stacked", stacked)]:
            self.milestones[name] |= reached
        # This is read more than once in a step, so the count moves on the clock.
        self.held_steps += stacked & (self.elapsed_steps != self.counted)
        self.counted = self.elapsed_steps.clone()

        return {"success": stacked, "seated": seated,
                # A level that only rises, so differencing it never scores below zero: a
                # point for seating the stack, and a step's worth for every step it stands.
                "reward": self.milestones["stacked"] + HOLD_REWARD * self.held_steps,
                "lifted": self.milestones["lifted"].clone(),
                "grasped": self.milestones["grasped"].clone(),
                "stack_xy": stack_xy, "stack_z": stack_z}
