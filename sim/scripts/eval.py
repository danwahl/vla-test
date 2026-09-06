"""Roll a policy out on the held-out layouts.

    uv run python sim/scripts/eval.py CHECKPOINT/pretrained_model

Chunks are stitched with Real-Time Chunking: each one is guided onto the tail of the
chunk it replaces. ``--no-rtc`` denoises each chunk on its own instead, which is how the
RL rollout executes them.

``--no-smooth`` executes each chunk as it arrives, rather than off the cubic
`sim.policy.smooth` reads it back from.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from mani_skill.utils.visualization.misc import images_to_video, tile_images

import sim.env  # noqa: F401  (registers the env)
from sim.dataset import SIM, root
from sim.policy import load_policy, smooth
from sim.spec import BLOCK_NAMES, CAMERAS


def observation(obs, tasks):
    """The env's cameras and joint positions in the form the processors take."""
    return {
        "observation.state": obs["agent"]["qpos"].float(),
        **{f"observation.images.{camera}":
           obs["sensor_data"][camera]["rgb"].permute(0, 3, 1, 2).float() / 255
           for camera in CAMERAS},
        "task": tasks,
    }


def rollout(env, policy, preprocessor, postprocessor, layouts, horizon, steps,
            frames=None, smoothing=True):
    """Drive a batch of layouts for ``steps``, replanning every ``horizon``.

    A stack is scored the first step it holds, since the arm carries on moving afterwards.
    Passing a ``frames`` list also collects the inspection view, tiled over the batch, from
    the pose the layout opens in.

    Also gives back what each chunk took to denoise, and the commands themselves, which say
    how smoothly the arm was driven.
    """
    obs, _ = env.reset(options={"layout": {
        "xy": np.array([[p[:2] for p in item["positions"]] for item in layouts], np.float32),
        "yaw": np.array([item["yaws"] for item in layouts], np.float32),
        "held": np.array([item["held"] for item in layouts]),
        "target": np.array([item["target"] for item in layouts]),
        "qpos": np.array([item["start"] for item in layouts], np.float32),
    }})
    tasks = [item["prompt"] for item in layouts]
    stacked = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def snapshot():
        frames.append(tile_images(env.render_rgb_array().cpu().numpy(),
                                  nrows=round(env.num_envs ** 0.5)))

    if frames is not None:
        snapshot()

    leftover, latencies, sent = None, [], []
    for _ in range(0, steps, horizon):
        began = time.perf_counter()
        chunk = policy.predict_action_chunk(
            preprocessor(observation(obs, tasks)),
            prev_chunk_left_over=leftover,
            inference_delay=0,
            execution_horizon=horizon,
        )
        # The call queues work and returns, so the clock has to wait for the device.
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - began)
        # The next chunk is guided onto this one as the policy wrote it, so smoothing
        # changes what the arm is told and not what the model is asked to agree with.
        leftover = chunk[:, horizon:]
        commands = postprocessor(chunk)
        if smoothing:
            commands = smooth(commands)
        sent.append(commands[:, :horizon].cpu().numpy())
        for step in range(horizon):
            obs, _, success, _, _ = env.step(commands[:, step])
            stacked |= success
            if frames is not None:
                snapshot()
    return (stacked.cpu().numpy(), env.evaluate()["lifted"].cpu().numpy(), latencies,
            np.concatenate(sent, axis=1))


def report(results):
    """The two gates, overall and per colour pair."""
    def line(name, rows):
        print(f"{name}: {sum(r['stacked'] for r in rows)} stacked, "
              f"{sum(r['lifted'] for r in rows)} lifted, of {len(rows)}")

    line("all", results)
    for held in range(len(BLOCK_NAMES)):
        for target in range(len(BLOCK_NAMES)):
            rows = [r for r in results if (r["held"], r["target"]) == (held, target)]
            if rows:
                line(f"  {BLOCK_NAMES[held]} on {BLOCK_NAMES[target]}", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, default=root(SIM))
    parser.add_argument("--repo-id", default=SIM)
    # The normalization comes from the processors saved beside the checkpoint, so a
    # checkpoint can be scored on another dataset's held-out layouts and only the spawns
    # change.
    parser.add_argument("--layouts", type=Path,
                        help="default: the dataset's own held-out layouts")
    parser.add_argument("--horizon", type=int, default=20, help="steps executed per chunk")
    parser.add_argument("--no-rtc", action="store_true", help="denoise each chunk on its own")
    parser.add_argument("--no-smooth", action="store_true",
                        help="execute each chunk as the policy wrote it")
    parser.add_argument("--int8", action="store_true",
                        help="hold the backbone and vision tower as int8")
    parser.add_argument("--envs", type=int, default=16, help="envs stepped in lockstep")
    parser.add_argument("--episodes", type=int, help="default: every held-out layout")
    parser.add_argument("--video", type=Path, help="write an mp4 per batch of layouts")
    # Two policies see the same layouts in the same order, so their outcomes pair up by
    # index and a comparison between them can be read per layout.
    parser.add_argument("--out", type=Path, help="write each layout's outcome")
    # The layouts are fixed, but the policy draws fresh noise for every chunk it denoises.
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    with (args.layouts or args.dataset / "meta" / "eval_layouts.jsonl").open() as file:
        layouts = [json.loads(line) for line in file][:args.episodes]

    env = gym.make("SO101BlockStack-v1", num_envs=args.envs, obs_mode="rgb",
                   render_mode="rgb_array").unwrapped
    steps = gym.spec("SO101BlockStack-v1").max_episode_steps
    # The env is already on the device, so the checkpoint's own footprint is what loading
    # it adds.
    on_device = torch.cuda.memory_allocated()
    policy, preprocessor, postprocessor = load_policy(
        args.checkpoint, LeRobotDatasetMetadata(args.repo_id, root=args.dataset),
        args.horizon, env.device.type, rtc=not args.no_rtc, int8=args.int8)
    weights = torch.cuda.memory_allocated() - on_device
    torch.cuda.reset_peak_memory_stats()

    results, latencies, jerks = [], [], []
    for start in range(0, len(layouts), env.num_envs):
        batch = layouts[start:start + env.num_envs]
        padded = batch + [batch[-1]] * (env.num_envs - len(batch))
        frames = [] if args.video else None
        stacked, lifted, timings, commands = rollout(
            env, policy, preprocessor, postprocessor, padded, args.horizon, steps, frames,
            not args.no_smooth)
        results += [{**item, "stacked": bool(stacked[i]), "lifted": bool(lifted[i])}
                    for i, item in enumerate(batch)]
        latencies += timings
        # The five arm joints of the layouts this batch actually holds, since a short last
        # batch is padded by repeating one of them. A jerky rollout shows as a large second
        # difference here.
        jerks.append(np.abs(np.diff(commands[:len(batch), :, :5], n=2, axis=1)).mean())
        if frames:
            images_to_video(frames, str(args.video), f"{start:03d}",
                            fps=round(1 / env.control_timestep))
        print(f"{len(results)}/{len(layouts)} rolled out, "
              f"{sum(r['stacked'] for r in results)} stacked", flush=True)
    env.close()

    if args.out:
        args.out.write_text("".join(json.dumps(row) + "\n" for row in results))
    report(results)
    # Median over every chunk, so the first one's kernel selection does not read as latency.
    print(f"{weights / 2**30:.2f} GiB of weights, "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak, "
          f"{1000 * float(np.median(latencies)):.0f} ms per chunk at {args.envs} envs")
    print(f"second difference {float(np.mean(jerks)):.5f} rad a step in the commands")


if __name__ == "__main__":
    main()
