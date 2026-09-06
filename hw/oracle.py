"""One layout, run in sim, in the form the hardware needs it.

The sim layout is made true of the table: the views here show where its blocks sit, the
operator moves the physical blocks onto them, and then the joint commands here are
replayed open loop.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import gymnasium as gym
import numpy as np
import torch

import sim.env  # noqa: F401  (registers the env)
from hw.robot import PARK_QPOS
from sim.dataset import SIM, root
from sim.oracle import Oracle
from sim.spec import CAMERAS

# The spawns the sim demonstrations were collected on, which is where hardware layouts are
# drawn from too: the held-out set beside it is what a policy trained on either is scored
# against.
LAYOUTS = root(SIM) / "meta" / "train_layouts.jsonl"


@dataclass
class Plan:
    prompt: str
    # Each camera at the spawn, arm parked: what the table should look like while the
    # blocks go down. The arm stands where the real one is standing at that moment rather
    # than at the pose the episode starts from, so its outline is one more thing the
    # operator can see the overlay agreeing on.
    views: dict[str, np.ndarray]
    # The pose the layout opens in: mid-cycle over the table, not the rest pose. The
    # hardware episode starts here, where its sim counterpart starts.
    start: np.ndarray
    # The commanded joint positions, one row per control step, in radians.
    commands: np.ndarray
    # Whether the oracle stacked it in sim. A layout that fails there is not worth
    # carrying to the table.
    stacked: bool


def parked(env):
    """Render each camera with the arm at its park pose, and put it back where it was.

    Only the configuration moves. Velocities and drive targets are untouched, so the
    oracle afterwards runs from the same state the reset left.
    """
    was = env.agent.robot.get_qpos().clone()
    env.agent.robot.set_qpos(torch.as_tensor(PARK_QPOS, dtype=torch.float32)[None])
    env.scene.update_render()
    obs = env.get_obs()
    views = {camera: obs["sensor_data"][camera]["rgb"][0].cpu().numpy()
             for camera in CAMERAS}
    env.agent.robot.set_qpos(was)
    env.scene.update_render()
    return views


def plan(env, index):
    """Run the oracle on layout ``index`` and keep what it commanded."""
    env.reset(options={"episode_id": torch.tensor([index])})
    start = env.agent.robot.get_qpos()[0].cpu().numpy()
    views = parked(env)

    # The layout carries the pose it opens in, so planning it twice plans the same
    # trajectory.
    oracle = Oracle(env)
    commands = []
    oracle.run(env.held, env.target,
               on_step=lambda _: commands.append(oracle.command[0].cpu().numpy()))

    return Plan(prompt=env.prompts()[0], views=views, start=start,
                commands=np.array(commands, np.float32),
                stacked=bool(env.evaluate()["success"][0]))


@contextmanager
def planner(layouts):
    """One sim, planning layouts one after another.

    The env is built once rather than per layout because a dozen make-and-close cycles in
    a process exhaust the render device.
    """
    # `num_envs=1` selects the CPU backend, which is what makes `parked`'s teleport render.
    env = gym.make("SO101BlockStack-v1", num_envs=1, obs_mode="rgb",
                   layouts=str(layouts), sample=False).unwrapped
    try:
        yield partial(plan, env)
    finally:
        env.close()
