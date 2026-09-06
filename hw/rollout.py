"""Roll a policy out on the arm.

    uv run python -m hw.rollout CHECKPOINT --indices 0 1 2
    uv run python -m hw.rollout CHECKPOINT --freehand --steps 1000 --held red --target blue

The policy reads the arm's own cameras and joint positions and its commands go to the bus.
Chunks are stitched and smoothed the way `sim/scripts/eval.py` executes them, and a chunk
runs open loop until the next one replaces it.

Denoising the next chunk takes long enough to see, so the arm runs on the chunk it
already has while that happens. RTC is told how many steps that will take and returns a
chunk beginning with the ones executed meanwhile, which is what makes the two join.

Each layout is planned in sim first, so the console shows where its blocks belong and the
arm can be put in the pose that layout opens in. Those two are the state the sim eval hands
the same policy, so a score here reads against the score there.

``--freehand`` takes the blocks wherever they are put and opens from the rest pose. It
reaches nothing in the sim package, so it runs on a machine with no simulator installed.

Whether it stacked is a call for whoever is standing there. What the arm reports is how far
the jaw closed, which says whether a block was in it.
"""

from __future__ import annotations

import argparse
import threading
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from hw.overlay import serve
from hw.robot import (
    FPS,
    PARK_QPOS,
    actions,
    follower,
    home,
    observations,
    picks,
    release,
    walk_to,
)
from sim.dataset import SIM, root
from sim.policy import load_policy, smooth
from sim.spec import BLOCK_NAMES, CAMERAS, HOME_QPOS, JOINT_NAMES, prompt


def observation(frame, task):
    """The arm's cameras and joint positions in the form the processors take."""
    return {
        "observation.state": torch.tensor([[frame[joint] for joint in JOINT_NAMES]],
                                          dtype=torch.float32),
        **{f"observation.images.{camera}":
           torch.from_numpy(frame[camera]).permute(2, 0, 1)[None].float() / 255
           for camera in CAMERAS},
        "task": [task],
    }


class Replan(threading.Thread):
    """The next chunk, denoised while the arm runs on the one it has."""

    def __init__(self, policy, batch, leftover, delay, horizon):
        super().__init__()
        self.policy, self.batch = policy, batch
        self.leftover, self.delay, self.horizon = leftover, delay, horizon
        self.chunk, self.took, self.error = None, 0.0, None

    def run(self):
        began = time.perf_counter()
        try:
            self.chunk = self.policy.predict_action_chunk(
                self.batch,
                prev_chunk_left_over=self.leftover,
                inference_delay=self.delay,
                execution_horizon=self.horizon,
            )
        except Exception as error:
            self.error = error
        self.took = time.perf_counter() - began

    def join(self, timeout=None):
        """Wait for the chunk, raising here whatever the thread raised there."""
        super().join(timeout)
        if self.error is not None:
            raise self.error


def rollout(robot, policy, preprocessor, postprocessor, task, steps, horizon):
    """Drive the arm for ``steps``, replanning every ``horizon``.

    How many steps a replan is given comes from how long the last one took, since that is
    the best estimate of what the next will cost. It is also held to what the previous
    chunk has left to run, so the arm is never told to keep going past the end of it. The
    first replan has no chunk to run on and no prefix to agree with, so it stands still for
    one denoising pass.

    Returns how far the jaw closed on the best of the episode's grasps, in the follower's
    own gripper units: the narrowest reading inside each close, and the widest of those
    across closes, since a long episode lets the policy try again after a miss. Both widths
    are sampled once per chunk, which is enough: a close spans several chunks.
    """
    to_sim, to_robot = observations(), actions()
    held, delay = None, 0
    commanded, reached = [], []
    leftover, commands = None, ()

    def drive(command, frame):
        nonlocal held
        walk_to(robot, to_robot, held, command, frame)
        held = command

    for _ in range(0, steps, horizon):
        frame = to_sim(robot.get_observation())
        if held is None:
            held = np.array([frame[joint] for joint in JOINT_NAMES])
        # Each reading is taken before that step's command goes out, so a command shows
        # in the reading after it.
        commanded.append(held[JOINT_NAMES.index("gripper")])
        reached.append(frame["gripper"])

        replan = Replan(policy, preprocessor(observation(frame, task)), leftover, delay,
                        horizon)
        replan.start()
        for command in commands[:delay]:
            drive(command, frame)
        replan.join()

        leftover = replan.chunk[:, horizon:]
        commands = smooth(postprocessor(replan.chunk))[0].cpu().numpy()
        for command in commands[delay:horizon]:
            drive(command, frame)
        # What the next chunk will be guided onto, and how much of it will have run by
        # the time that chunk arrives.
        commands = commands[horizon:]
        delay = min(int(np.ceil(replan.took * FPS)), horizon - 1, len(commands))
    return float(np.rad2deg(max((min(reached[window]) for window in picks(commanded)),
                                default=np.nan)))


def episode(robot, console, policy, preprocessor, postprocessor, where, task, views,
            start, steps, horizon):
    """Hold for the blocks, drive the policy, and park. ``None`` if the console skips it."""
    if not console.place(views, f"{where}: {task}"):
        return None
    console.say(f"{where}: running")
    with console.watch():
        # Via the rest pose, so the travel to the opening pose runs through a pose with
        # known clearance over the blocks.
        home(robot)
        home(robot, start)
        jaw = rollout(robot, policy, preprocessor, postprocessor, task, steps, horizon)
        home(robot, PARK_QPOS)
    console.say(f"{where}: jaw closed to {jaw:.1f}")
    return jaw


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--layouts", type=Path,
                        help="default: the held-out layouts beside the demonstrations")
    parser.add_argument("--indices", type=int, nargs="+", default=[0])
    parser.add_argument("--freehand", action="store_true",
                        help="blocks anywhere on the table, opening from rest")
    parser.add_argument("--held", choices=BLOCK_NAMES, default=BLOCK_NAMES[0],
                        help="freehand only")
    parser.add_argument("--target", choices=BLOCK_NAMES, default=BLOCK_NAMES[1],
                        help="freehand only")
    parser.add_argument("--episodes", type=int, default=6, help="freehand only")
    parser.add_argument("--dataset", type=Path, default=root(SIM))
    parser.add_argument("--repo-id", default=SIM)
    parser.add_argument("--horizon", type=int, default=20, help="steps executed per chunk")
    # A cycle can be run from wherever the last one left the arm, so an episode longer than
    # the sim limit is one that lets the policy have another go at a pick it missed.
    parser.add_argument("--steps", type=int, help="default: the sim episode limit")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    steps = args.steps
    if steps is None:
        if args.freehand:
            parser.error("--freehand needs --steps: the default limit comes from the sim env")
        import gymnasium as gym

        import sim.env  # noqa: F401  (registers the env, for the episode length)
        steps = gym.spec("SO101BlockStack-v1").max_episode_steps

    policy, preprocessor, postprocessor = load_policy(
        args.checkpoint, LeRobotDatasetMetadata(args.repo_id, root=args.dataset),
        args.horizon, "cuda")
    print(f"{args.checkpoint}, {steps} steps an episode", flush=True)

    robot = follower()
    try:
        robot.connect()
        home(robot, PARK_QPOS)
        with serve(robot, args.port) as console:
            run = partial(episode, robot, console, policy, preprocessor, postprocessor,
                          steps=steps, horizon=args.horizon)
            if args.freehand:
                task = prompt(BLOCK_NAMES.index(args.held), BLOCK_NAMES.index(args.target))
                for number in range(1, args.episodes + 1):
                    where = f"{number}/{args.episodes}"
                    jaw = run(where=where, task=task, views=None, start=HOME_QPOS)
                    if jaw is not None:
                        print(f"episode {number}: jaw closed to {jaw:.1f}", flush=True)
                return

            # Imported here rather than at the top because it reaches the simulator,
            # which the freehand path above runs without.
            from hw.oracle import LAYOUTS, planner
            with planner(args.layouts or LAYOUTS.with_name("eval_layouts.jsonl")) as plan:
                for number, index in enumerate(args.indices, 1):
                    where = f"{number}/{len(args.indices)}  layout {index}"
                    console.say(f"{where}: planning")
                    laid_out = plan(index)
                    print(f"layout {index}: {laid_out.prompt}", flush=True)

                    jaw = run(where=where, task=laid_out.prompt, views=laid_out.views,
                              start=laid_out.start)
                    if jaw is not None:
                        print(f"layout {index}: jaw closed to {jaw:.1f}", flush=True)
    finally:
        release(robot)


if __name__ == "__main__":
    main()
