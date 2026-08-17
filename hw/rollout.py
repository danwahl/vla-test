"""Roll a policy out on the arm.

    uv run python -m hw.rollout CHECKPOINT --held red --target blue

The policy reads the arm's own cameras and joint positions and its commands go to the bus.
Chunks are stitched with Real-Time Chunking the way `sim/scripts/eval.py` executes them,
and a chunk runs open loop until the next one replaces it.

Denoising the next chunk takes long enough to see, so the arm runs on the chunk it
already has while that happens. RTC is told how many steps that will take and returns a
chunk beginning with the ones executed meanwhile, which is what makes the two join.

The console gates each episode: put the blocks anywhere on the table and press the button.
Whether it stacked is a call for whoever is standing there. What the arm reports is how
far the jaw closed, which says whether a block was in it.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

import sim  # noqa: F401  (registers the env, for the episode length)
from hw.overlay import serve
from hw.robot import FPS, PARK_QPOS, actions, follower, home, observations, walk_to
from sim.agent import JOINT_NAMES
from sim.dataset import CAMERAS
from sim.env import BLOCK_NAMES, prompt
from sim.policy import load_policy


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
        self.chunk, self.took = None, 0.0

    def run(self):
        began = time.perf_counter()
        self.chunk = self.policy.predict_action_chunk(
            self.batch,
            prev_chunk_left_over=self.leftover,
            inference_delay=self.delay,
            execution_horizon=self.horizon,
        )
        self.took = time.perf_counter() - began


def rollout(robot, policy, preprocessor, postprocessor, task, steps, horizon):
    """Drive the arm for ``steps``, replanning every ``horizon``.

    How many steps a replan is given comes from how long the last one took, since that is
    the best estimate of what the next will cost. It is also held to what the previous
    chunk has left to run, so the arm is never told to keep going past the end of it. The
    first replan has no chunk to run on and no prefix to agree with, so it stands still for
    one denoising pass.

    Returns the narrowest the jaw reached, in the follower's own gripper units. It is read
    once per chunk rather than per step, which is enough: once the jaw has closed on a
    block it stays there.
    """
    to_sim, to_robot = observations(), actions()
    held, jaw, delay = None, np.inf, 0
    leftover, commands = None, ()

    def drive(command, frame):
        nonlocal held
        walk_to(robot, to_robot, held, command, frame)
        held = command

    for _ in range(0, steps, horizon):
        frame = to_sim(robot.get_observation())
        jaw = min(jaw, frame["gripper"])
        if held is None:
            held = np.array([frame[joint] for joint in JOINT_NAMES])

        replan = Replan(policy, preprocessor(observation(frame, task)), leftover, delay,
                        horizon)
        replan.start()
        for command in commands[:delay]:
            drive(command, frame)
        replan.join()

        leftover = replan.chunk[:, horizon:]
        commands = postprocessor(replan.chunk)[0].cpu().numpy()
        for command in commands[delay:horizon]:
            drive(command, frame)
        # What the next chunk will be guided onto, and how much of it will have run by
        # the time that chunk arrives.
        commands = commands[horizon:]
        delay = min(int(np.ceil(replan.took * FPS)), horizon - 1, len(commands))
    return float(np.rad2deg(jaw))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--held", choices=BLOCK_NAMES, default=BLOCK_NAMES[0])
    parser.add_argument("--target", choices=BLOCK_NAMES, default=BLOCK_NAMES[1])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--dataset", type=Path,
                        default=Path("/data/datasets/so101_block_stack_sim"))
    parser.add_argument("--repo-id", default="vla-test/so101_block_stack_sim")
    parser.add_argument("--horizon", type=int, default=20, help="steps executed per chunk")
    parser.add_argument("--no-rtc", action="store_true", help="denoise each chunk on its own")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()

    task = prompt(BLOCK_NAMES.index(args.held), BLOCK_NAMES.index(args.target))
    steps = gym.spec("SO101BlockStack-v1").max_episode_steps
    policy, preprocessor, postprocessor = load_policy(
        args.checkpoint, LeRobotDatasetMetadata(args.repo_id, root=args.dataset),
        args.horizon, "cuda", rtc=not args.no_rtc)
    print(f"{task}, {steps} steps an episode", flush=True)

    robot = follower()
    robot.connect()
    try:
        home(robot, PARK_QPOS)
        with serve(robot, args.out, args.port) as console:
            for episode in range(1, args.episodes + 1):
                where = f"{episode}/{args.episodes}"
                if not console.place(None, f"{where}: {task}"):
                    continue
                console.say(f"{where}: running")
                home(robot)
                jaw = rollout(robot, policy, preprocessor, postprocessor, task, steps,
                              args.horizon)
                home(robot, PARK_QPOS)
                console.say(f"{where}: jaw closed to {jaw:.1f}")
                print(f"episode {episode}: jaw closed to {jaw:.1f}", flush=True)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
