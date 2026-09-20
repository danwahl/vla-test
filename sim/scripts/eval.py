"""Roll a policy out on the held-out layouts.

    uv run python sim/scripts/eval.py CHECKPOINT/pretrained_model

Chunks are stitched with Real-Time Chunking: each one is guided onto the tail of the
chunk it replaces. ``--no-rtc`` denoises each chunk on its own instead, which is how the
RL rollout executes them. A policy with no stitching of its own executes whole chunks and
takes neither flag.

``--no-smooth`` executes each chunk as it arrives, rather than off the cubic
`sim.policy.smooth` reads it back from. ``--blend N`` eases each chunk onto the previous
one over its first N steps, which is the seam a policy without stitching leaves behind;
``--anchor`` chooses whether position or velocity is what carries across it.
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
from sim.policy import blend, load_policy, smooth
from sim.spec import BLOCK_NAMES, CAMERAS


def observation(obs, tasks, time_axis=False):
    """The env's cameras and joint positions in the form the processors take.

    A policy that asks for image history takes its cameras under a time axis. The rollout
    keeps no buffer, so it is given the current frame alone: a policy that reads history
    predicts what follows rather than observing it.
    """
    images = {f"observation.images.{camera}":
              obs["sensor_data"][camera]["rgb"].permute(0, 3, 1, 2).float() / 255
              for camera in CAMERAS}
    if time_axis:
        images = {key: value.unsqueeze(1) for key, value in images.items()}
    return {"observation.state": obs["agent"]["qpos"].float(), **images, "task": tasks}


def rollout(env, policy, preprocessor, postprocessor, layouts, horizon, steps,
            frames=None, smoothing=True, time_axis=False, blending=0, anchor="velocity"):
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

    # A policy that does not stitch is asked for a chunk and nothing else; it has no tail
    # to guide the next one onto.
    stitching = policy.supports_rtc()
    leftover, latencies, sent = None, [], []
    # The two steps the arm was last driven through, which say where it was heading.
    tail = None
    for done in range(0, steps, horizon):
        # The last chunk is cut to what is left of the episode, so a horizon that does not
        # divide the budget does not buy the policy extra steps to stack in.
        take = min(horizon, steps - done)
        began = time.perf_counter()
        batch = preprocessor(observation(obs, tasks, time_axis))
        chunk = (policy.predict_action_chunk(batch, prev_chunk_left_over=leftover,
                                             inference_delay=0, execution_horizon=horizon)
                 if stitching else policy.predict_action_chunk(batch))
        # The call queues work and returns, so the clock has to wait for the device.
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - began)
        # The next chunk is guided onto this one as the policy wrote it, so smoothing
        # changes what the arm is told and not what the model is asked to agree with.
        leftover = chunk[:, horizon:] if stitching else None
        commands = postprocessor(chunk)
        if smoothing:
            commands = smooth(commands)
        # After smoothing, since this is a fix to what reaches the arm rather than to the
        # wobble inside a chunk.
        commands = blend(commands, tail, blending, anchor)
        sent.append(commands[:, :take].cpu().numpy())
        tail = commands[:, max(take - 2, 0):take]
        for step in range(take):
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
    parser.add_argument("--horizon", type=int,
                        help="steps executed per chunk; default: 20 stitched, the policy's own "
                             "otherwise")
    parser.add_argument("--no-rtc", action="store_true", help="denoise each chunk on its own")
    parser.add_argument("--no-smooth", action="store_true",
                        help="execute each chunk as the policy wrote it")
    parser.add_argument("--blend", type=int, default=0,
                        help="steps a chunk is eased onto the previous one; a stitched "
                             "policy needs none")
    parser.add_argument("--anchor", choices=("velocity", "position"), default="velocity",
                        help="what the seam is made continuous in")
    parser.add_argument("--clip", action="store_true",
                        help="hold each chunk inside the arm's travel before unnormalizing")
    parser.add_argument("--int8", action="store_true",
                        help="hold the backbone and vision tower as int8")
    parser.add_argument("--envs", type=int, default=16, help="envs stepped in lockstep")
    parser.add_argument("--episodes", type=int, help="default: every held-out layout")
    parser.add_argument("--video", type=Path, help="write an mp4 per batch of layouts")
    # Two policies see the same layouts in the same order, so their outcomes pair up by
    # index and a comparison between them can be read per layout.
    parser.add_argument("--out", type=Path, help="write each layout's outcome")
    # What the arm was actually told, which is where a seam shows.
    parser.add_argument("--commands", type=Path,
                        help="write the commands sent, (layouts, steps, joints)")
    # The layouts are fixed, but the policy draws fresh noise for every chunk it denoises.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--denoise-steps", type=int,
                        help="Euler steps a chunk is denoised in; default: the policy's own")
    parser.add_argument("--top-tilt", type=float, default=0.0,
                        help="degrees of top-camera pitch, as a bracket aimed by hand")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    with (args.layouts or args.dataset / "meta" / "eval_layouts.jsonl").open() as file:
        layouts = [json.loads(line) for line in file][:args.episodes]

    env = gym.make("SO101BlockStack-v1", num_envs=args.envs, obs_mode="rgb",
                   render_mode="rgb_array", top_tilt_deg=args.top_tilt).unwrapped
    steps = gym.spec("SO101BlockStack-v1").max_episode_steps
    # The env is already on the device, so the checkpoint's own footprint is what loading
    # it adds.
    on_device = torch.cuda.memory_allocated()
    policy, preprocessor, postprocessor = load_policy(
        args.checkpoint, LeRobotDatasetMetadata(args.repo_id, root=args.dataset),
        args.horizon or 20, env.device.type, rtc=not args.no_rtc, int8=args.int8,
        denoise_steps=args.denoise_steps, clip=args.clip)
    # Without stitching there is no tail to guide the next chunk onto, so a shorter horizon
    # simply discards the rest of the chunk and replans. That is what a policy's own
    # `n_action_steps` does to its `chunk_size`, and `--horizon` sets it for a run.
    horizon = args.horizon or (20 if policy.supports_rtc() else policy.config.n_action_steps)
    # A policy without stitching hands back `n_action_steps` and no more, so a longer
    # horizon would execute steps the chunk does not hold.
    if not policy.supports_rtc() and horizon > policy.config.n_action_steps:
        raise SystemExit(f"--horizon {horizon} is longer than the {policy.config.n_action_steps} "
                         f"steps this policy writes a chunk of")
    # A policy that reads image history says so here; a policy that reads one frame
    # leaves it unset, and the two already scored are both of the second kind.
    time_axis = getattr(policy.config, "image_observation_delta_indices", None) is not None
    weights = torch.cuda.memory_allocated() - on_device
    torch.cuda.reset_peak_memory_stats()

    results, latencies, jerks, sent = [], [], [], []
    for start in range(0, len(layouts), env.num_envs):
        batch = layouts[start:start + env.num_envs]
        padded = batch + [batch[-1]] * (env.num_envs - len(batch))
        frames = [] if args.video else None
        stacked, lifted, timings, commands = rollout(
            env, policy, preprocessor, postprocessor, padded, horizon, steps, frames,
            not args.no_smooth, time_axis, args.blend, args.anchor)
        results += [{**item, "stacked": bool(stacked[i]), "lifted": bool(lifted[i])}
                    for i, item in enumerate(batch)]
        latencies += timings
        # The five arm joints of the layouts this batch actually holds, since a short last
        # batch is padded by repeating one of them. A jerky rollout shows as a large second
        # difference here.
        jerks.append(np.abs(np.diff(commands[:len(batch), :, :5], n=2, axis=1)).mean())
        sent.append(commands[:len(batch)])
        if frames:
            images_to_video(frames, str(args.video), f"{start:03d}",
                            fps=round(1 / env.control_timestep))
        print(f"{len(results)}/{len(layouts)} rolled out, "
              f"{sum(r['stacked'] for r in results)} stacked", flush=True)
    env.close()

    if args.out:
        args.out.write_text("".join(json.dumps(row) + "\n" for row in results))
    if args.commands:
        np.savez_compressed(args.commands, commands=np.concatenate(sent), horizon=horizon)
    report(results)
    # Median over every chunk, so the first one's kernel selection does not read as latency.
    print(f"{weights / 2**30:.2f} GiB of weights, "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak, "
          f"{1000 * float(np.median(latencies)):.0f} ms per chunk at {args.envs} envs, "
          f"{horizon} steps executed per chunk")
    print(f"second difference {float(np.mean(jerks)):.5f} rad a step in the commands")


if __name__ == "__main__":
    main()
