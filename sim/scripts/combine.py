"""Write the sim and the hardware demonstrations into one dataset.

    uv run python sim/scripts/combine.py OUT

lerobot trains on one dataset, so a policy that learns from both is trained on a dataset
holding both. The hardware set is the smaller by an order of magnitude, and how many times
it is written here is what sets how much of a batch comes off the arm; the default puts
about a quarter of the frames on the arm.

The sim tree is copied and the hardware episodes are appended to the copy, so only the
repeated frames are encoded again. Copying also carries the layout files across, which is
what lets an eval read its held-out spawns out of the combined dataset.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from sim.dataset import CAMERAS, create


def episodes(dataset):
    """Each episode of ``dataset``, as the frames a writer takes.

    Read once and kept, since the caller writes them repeatedly and decoding the videos is
    the expensive half. The hardware set is small enough to sit in memory whole.
    """
    def frame(row):
        return {"task": row["task"],
                "observation.state": row["observation.state"].numpy(),
                "action": row["action"].numpy(),
                **{f"observation.images.{camera}":
                   row[f"observation.images.{camera}"].permute(1, 2, 0).numpy()
                   for camera in CAMERAS}}

    table = dataset.meta.episodes
    return [[frame(dataset[index]) for index in range(start, end)]
            for start, end in zip(table["dataset_from_index"], table["dataset_to_index"],
                                  strict=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--sim", type=Path,
                        default=Path("/data/datasets/so101_block_stack_sim"))
    parser.add_argument("--hw", type=Path,
                        default=Path("/data/datasets/so101_block_stack_hw"))
    parser.add_argument("--repeat", type=int, default=8,
                        help="times the hardware set is written")
    parser.add_argument("--repo-id", default="vla-test/so101_block_stack_mix")
    args = parser.parse_args()

    # `return_uint8` hands back the frames as they were stored, so a repeat is a decode and
    # an encode rather than a trip through float.
    hardware = LeRobotDataset("vla-test/so101_block_stack_hw", root=args.hw,
                              return_uint8=True)
    print(f"{hardware.meta.total_episodes} hardware episodes, "
          f"{hardware.meta.total_frames} frames, written {args.repeat} times", flush=True)
    demonstrations = episodes(hardware)

    shutil.copytree(args.sim, args.out)
    dataset = create(args.out, args.repo_id, hardware.meta.fps, resume=True)
    try:
        for repeat in range(1, args.repeat + 1):
            for frames in demonstrations:
                for frame in frames:
                    # `add_frame` takes the keys it wants out of what it is handed, so each
                    # write needs its own dictionary over the frames being kept.
                    dataset.add_frame(dict(frame))
                dataset.save_episode()
            print(f"written {repeat}/{args.repeat}, "
                  f"{dataset.meta.total_episodes} episodes so far", flush=True)
    finally:
        dataset.finalize()
    print(f"{dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames",
          flush=True)


if __name__ == "__main__":
    main()
