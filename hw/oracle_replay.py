"""Replay sim oracle demonstrations on the arm, and record what the arm did.

    uv run python -m hw.oracle_replay OUT --indices 0 1 2

Each layout is planned in sim first, then it goes up on the overlay console so the blocks
can be put where the sim put them, then the oracle's joint commands go out open loop at
the sim's control rate. The commands are the sim's; the states and frames are the arm's,
so the episode reads as a demonstration on hardware and the difference between the two is
what the trajectory lost on the way to the arm.

The arm waits at its park pose while the blocks go down and returns there afterwards, so
the console shows an unobstructed table between episodes and the layout can be judged
before the next one is planned.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

from hw.oracle import LAYOUTS, planner
from hw.overlay import serve
from hw.robot import (
    FPS,
    PARK_QPOS,
    actions,
    follower,
    home,
    observations,
    pick,
    release,
    walk_to,
)
from sim.dataset import HW, create, finalize
from sim.spec import JOINT_NAMES


def replay(robot, commands, dataset, task):
    """Send each command in turn, keeping the observation that preceded it.

    Returns the narrowest the jaw reached over the close ``pick`` finds, in the follower's
    own gripper units. A block between the jaws stops them short of the commanded close, so
    that number says whether one was there.
    """
    to_sim, to_robot = observations(), actions()
    held, jaw = None, []
    for command in commands:
        step = dict(zip(JOINT_NAMES, command, strict=True))
        observation = to_sim(robot.get_observation())
        dataset.add_frame({
            **build_dataset_frame(dataset.features, observation, prefix=OBS_STR),
            **build_dataset_frame(dataset.features, step, prefix=ACTION),
            "task": task,
        })
        jaw.append(observation["gripper"])
        if held is None:
            held = np.array([observation[joint] for joint in JOINT_NAMES])
        walk_to(robot, to_robot, held, command, observation)
        held = command
    dataset.save_episode()
    # Each reading is taken before that step's command goes out, so a command shows in
    # the reading after it.
    return float(np.rad2deg(min(jaw[1:][pick(commands[:, JOINT_NAMES.index("gripper")])],
                                default=np.nan)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--repo-id", default=HW)
    parser.add_argument("--layouts", type=Path, default=LAYOUTS)
    parser.add_argument("--indices", type=int, nargs="+", default=[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resume", action="store_true",
                        help="append to the dataset already at OUT")
    args = parser.parse_args()

    dataset = create(args.out, args.repo_id, FPS, args.resume)
    robot = follower()
    try:
        robot.connect()
        home(robot, PARK_QPOS)
        with serve(robot, args.port) as console, planner(args.layouts) as plan:
            for number, index in enumerate(args.indices, 1):
                where = f"{number}/{len(args.indices)}  layout {index}"
                console.say(f"{where}: planning")
                laid_out = plan(index)
                if not laid_out.stacked:
                    print(f"layout {index}: the oracle misses it in sim, skipping")
                    continue
                print(f"layout {index}: {laid_out.prompt}, {len(laid_out.commands)} steps")

                if not console.place(laid_out.views, f"{where}: {laid_out.prompt}"):
                    continue
                console.say(f"{where}: running")
                # Out to the rest pose first and only then to where the episode begins, so
                # the arm reaches over the blocks from a pose whose clearance is known
                # rather than from wherever it was parked.
                home(robot)
                home(robot, laid_out.start)
                jaw = replay(robot, laid_out.commands, dataset, laid_out.prompt)
                home(robot, PARK_QPOS)
                console.say(f"{where}: jaw closed to {jaw:.1f}")
                print(f"layout {index}: jaw closed to {jaw:.1f}")
    finally:
        release(robot)
        # Where the parquet footers are written, so the episodes a run did record stay
        # readable when it is stopped part way through a list of layouts.
        finalize(dataset)


if __name__ == "__main__":
    main()
