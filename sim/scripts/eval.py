"""Roll a policy out on the held-out layouts.

    uv run python sim/scripts/eval.py CHECKPOINT/pretrained_model

Chunks are stitched with Real-Time Chunking: each one is guided onto the tail of the
chunk it replaces. ``--no-rtc`` denoises each chunk on its own instead, which is how the
RL rollout executes them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.rtc import RTCConfig
from lerobot.processor import RenameObservationsProcessorStep
from mani_skill.utils.visualization.misc import images_to_video, tile_images

import sim  # noqa: F401  (registers the env)
from sim.env import BLOCK_NAMES

CAMERAS = ("wrist", "top")


def load_policy(checkpoint, metadata, horizon, device, rtc=True):
    """The checkpoint's policy and the processors saved beside it, which carry the
    camera renaming and the normalization stats from training."""
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = checkpoint
    config.device = device
    config.rtc_config = RTCConfig(execution_horizon=horizon, enabled=rtc)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    # The cameras reach the policy under the DROID slot names it was trained on. Handing
    # make_policy the same map is what tells it the two sets are meant to differ.
    renaming = next(step for step in preprocessor.steps
                    if isinstance(step, RenameObservationsProcessorStep))
    policy = make_policy(cfg=config, ds_meta=metadata, rename_map=renaming.rename_map)
    policy.eval()
    return policy, preprocessor, postprocessor


def observation(obs, tasks):
    """The env's cameras and joint positions in the form the processors take."""
    return {
        "observation.state": obs["agent"]["qpos"].float(),
        **{f"observation.images.{camera}":
           obs["sensor_data"][camera]["rgb"].permute(0, 3, 1, 2).float() / 255
           for camera in CAMERAS},
        "task": tasks,
    }


def rollout(env, policy, preprocessor, postprocessor, layouts, horizon, steps, frames=None):
    """Drive a batch of layouts for ``steps``, replanning every ``horizon``.

    A stack is scored the first step it holds, since the arm carries on moving afterwards.
    Passing a ``frames`` list also collects the inspection view, tiled over the batch.
    """
    obs, _ = env.reset(options={"layout": {
        "xy": np.array([[p[:2] for p in item["positions"]] for item in layouts], np.float32),
        "yaw": np.array([item["yaws"] for item in layouts], np.float32),
        "held": np.array([item["held"] for item in layouts]),
        "target": np.array([item["target"] for item in layouts]),
    }})
    tasks = [item["prompt"] for item in layouts]
    stacked = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    leftover = None
    for _ in range(0, steps, horizon):
        chunk = policy.predict_action_chunk(
            preprocessor(observation(obs, tasks)),
            prev_chunk_left_over=leftover,
            inference_delay=0,
            execution_horizon=horizon,
        )
        leftover = chunk[:, horizon:]
        commands = postprocessor(chunk)
        for step in range(horizon):
            obs, _, success, _, _ = env.step(commands[:, step])
            stacked |= success
            if frames is not None:
                frames.append(tile_images(env.render_rgb_array().cpu().numpy(),
                                          nrows=round(env.num_envs ** 0.5)))
    return stacked.cpu().numpy(), env.evaluate()["lifted"].cpu().numpy()


def report(results):
    """The two gates, overall and per colour pair."""
    def line(name, rows):
        print(f"{name}: {sum(r['stacked'] for r in rows)} stacked, "
              f"{sum(r['lifted'] for r in rows)} lifted, of {len(rows)}")

    line("all", results)
    for held in range(3):
        for target in range(3):
            rows = [r for r in results if (r["held"], r["target"]) == (held, target)]
            if rows:
                line(f"  {BLOCK_NAMES[held]} on {BLOCK_NAMES[target]}", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path,
                        default=Path("/data/datasets/so101_block_stack_sim"))
    parser.add_argument("--repo-id", default="vla-test/so101_block_stack_sim")
    parser.add_argument("--horizon", type=int, default=20, help="steps executed per chunk")
    parser.add_argument("--no-rtc", action="store_true", help="denoise each chunk on its own")
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
    with (args.dataset / "meta" / "eval_layouts.jsonl").open() as file:
        layouts = [json.loads(line) for line in file][:args.episodes]

    env = gym.make("SO101BlockStack-v1", num_envs=args.envs, obs_mode="rgb",
                   render_mode="rgb_array").unwrapped
    steps = gym.spec("SO101BlockStack-v1").max_episode_steps
    policy, preprocessor, postprocessor = load_policy(
        args.checkpoint, LeRobotDatasetMetadata(args.repo_id, root=args.dataset),
        args.horizon, env.device.type, rtc=not args.no_rtc)

    results = []
    for start in range(0, len(layouts), env.num_envs):
        batch = layouts[start:start + env.num_envs]
        padded = batch + [batch[-1]] * (env.num_envs - len(batch))
        frames = [] if args.video else None
        stacked, lifted = rollout(env, policy, preprocessor, postprocessor, padded,
                                  args.horizon, steps, frames)
        results += [{**item, "stacked": bool(stacked[i]), "lifted": bool(lifted[i])}
                    for i, item in enumerate(batch)]
        if frames:
            images_to_video(frames, str(args.video), f"{start:03d}",
                            fps=round(1 / env.control_timestep))
        print(f"{len(results)}/{len(layouts)} rolled out, "
              f"{sum(r['stacked'] for r in results)} stacked", flush=True)
    env.close()

    if args.out:
        args.out.write_text("".join(json.dumps(row) + "\n" for row in results))
    report(results)


if __name__ == "__main__":
    main()
