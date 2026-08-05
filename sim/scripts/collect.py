"""Collect oracle demonstrations into a LeRobot v3.0 dataset.

    uv run --extra data python scripts/collect.py OUT

Layouts are screened before anything is rendered: a batch is spawned, the oracle runs on
state alone, and only the layouts it stacks are kept. Those are then replayed with the
cameras on and written out as episodes. The eval ones are screened the same way and only
written down, as the held-out set a policy is measured on.

Screening and recording alternate until every colour pair has its quota.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import vla_test_sim  # noqa: F401  (registers the env)
from vla_test_sim.agent import JOINT_NAMES
from vla_test_sim.env import BLOCK_NAMES, BLOCK_REST_Z, IMAGE_SIZE, SPAWN_YAW, TASK_PROMPT
from vla_test_sim.oracle import Oracle

CAMERAS = ("wrist", "top")
PAIRS = [(held, target) for held in range(3) for target in range(3) if held != target]

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (len(JOINT_NAMES),),
                          "names": JOINT_NAMES},
    "action": {"dtype": "float32", "shape": (len(JOINT_NAMES),), "names": JOINT_NAMES},
    **{f"observation.images.{camera}": {"dtype": "video",
                                        "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                                        "names": ["height", "width", "channel"]}
       for camera in CAMERAS},
}


def prompt(held, target):
    return TASK_PROMPT.format(held=BLOCK_NAMES[held], target=BLOCK_NAMES[target])


def screen(env, need, seed):
    """Spawn batches until every colour pair has ``need[pair]`` layouts the oracle stacks."""
    kept = {pair: [] for pair in PAIRS}
    batch = 0
    while short := [pair for pair in PAIRS for _ in range(need[pair] - len(kept[pair]))]:
        pairs = [short[i % len(short)] for i in range(env.num_envs)]
        env.reset(seed=seed + batch, options={"layout": {
            "held": [pair[0] for pair in pairs], "target": [pair[1] for pair in pairs]}})
        layout = {key: value.cpu().numpy() for key, value in env.layout().items()}

        Oracle(env).run(env.held, env.target)
        won = env.evaluate()["success"].cpu().numpy()
        for i in np.flatnonzero(won):
            if len(kept[pairs[i]]) < need[pairs[i]]:
                kept[pairs[i]].append({key: value[i] for key, value in layout.items()})
        print(f"screen batch {batch}: {won.sum()}/{env.num_envs} stacked, "
              f"{sum(len(v) for v in kept.values())}/{sum(need.values())} kept", flush=True)
        batch += 1
    return kept


def rollout(env, layouts):
    """Replay a batch of ``layouts`` with the cameras on, keeping every step.

    Returns the commands, the observations that earned them, and which envs stacked. One
    snapshot is taken before the first command and one after each, so ``snapshots[t]`` is
    what the oracle saw when it chose ``commands[t]``.
    """
    obs, _ = env.reset(options={"layout": {
        key: np.stack([item[key] for item in layouts]) for key in layouts[0]}})
    oracle = Oracle(env)
    snapshots, commands = [], []

    def snapshot(obs):
        snapshots.append((obs["agent"]["qpos"].cpu().numpy(),
                          *(obs["sensor_data"][c]["rgb"].cpu().numpy() for c in CAMERAS)))

    def step(obs):
        commands.append(oracle.command.cpu().numpy())
        snapshot(obs)

    snapshot(obs)
    oracle.run(env.held, env.target, on_step=step)
    return commands, snapshots, env.evaluate()["success"].cpu().numpy()


def record(env, layouts, dataset, quota):
    """Replay ``layouts`` and save each one as an episode, returning the ones written.

    Every env in a batch runs the same phases, so an episode is as long as the batch's
    slowest env; the tail of a short last batch repeats a layout and is dropped.
    """
    written = []
    for start in range(0, len(layouts), env.num_envs):
        batch = layouts[start:start + env.num_envs]
        padded = batch + [batch[-1]] * (env.num_envs - len(batch))
        commands, snapshots, won = rollout(env, padded)

        for i, item in enumerate(batch):
            pair = (int(item["held"]), int(item["target"]))
            if not won[i] or not quota[pair]:
                continue
            quota[pair] -= 1
            task = prompt(*pair)
            for command, (qpos, *images) in zip(commands, snapshots[:-1], strict=True):
                dataset.add_frame({
                    "observation.state": qpos[i].astype(np.float32),
                    "action": command[i].astype(np.float32),
                    **{f"observation.images.{c}": image[i]
                       for c, image in zip(CAMERAS, images, strict=True)},
                    "task": task,
                })
            dataset.save_episode()
            written.append(item)
        print(f"record {len(written)} episodes, {len(commands)} frames each, "
              f"{sum(quota.values())} still owed", flush=True)
    return written


def write_layouts(path, layouts, key):
    """One line per layout, giving the spawn a run can be replayed from."""
    with path.open("w") as out:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--repo-id", default="vla-test/so101_block_stack_sim")
    parser.add_argument("--per-pair", type=int, default=60, help="train episodes per pair")
    parser.add_argument("--eval-per-pair", type=int, default=25, help="held-out layouts per pair")
    parser.add_argument("--envs", type=int, default=16, help="envs stepped in lockstep")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    state_env = gym.make("SO101BlockStack-v1", num_envs=args.envs).unwrapped
    fps = round(1 / state_env.control_timestep)
    dataset = LeRobotDataset.create(repo_id=args.repo_id, fps=fps, features=FEATURES,
                                    root=args.out, robot_type="so101",
                                    # H.264 rather than the AV1 default: every loader
                                    # downstream of here decodes it.
                                    rgb_encoder=RGBEncoderConfig(vcodec="h264"))
    render_env = gym.make("SO101BlockStack-v1", num_envs=args.envs, obs_mode="rgb").unwrapped

    quota = dict.fromkeys(PAIRS, args.per_pair)
    written, seed = [], args.seed
    while any(quota.values()):
        pool = screen(state_env, quota, seed)
        written += record(render_env, [item for pair in PAIRS for item in pool[pair]],
                          dataset, quota)
        seed += 1_000
    render_env.close()
    dataset.finalize()

    # Carrying on from the seed the loop reached keeps the held-out spawns off the train
    # ones. They cycle through the pairs, so a truncated eval still covers all six.
    held_out = screen(state_env, dict.fromkeys(PAIRS, args.eval_per_pair), seed)
    state_env.close()
    interleaved = [held_out[pair][i] for i in range(args.eval_per_pair) for pair in PAIRS]

    meta = args.out / "meta"
    write_layouts(meta / "train_layouts.jsonl", written, "episode_index")
    write_layouts(meta / "eval_layouts.jsonl", interleaved, "index")
    print(f"{len(written)} episodes and {len(interleaved)} held-out layouts in {args.out}")


if __name__ == "__main__":
    main()
