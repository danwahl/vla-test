"""Screen a pool of spawns the oracle can stack, for RL to roll out on.

    uv run python sim/scripts/rl_layouts.py OUT.jsonl --count 5000

The env samples a fresh spawn at every reset and the oracle stacks about 94% of them.
Drawing resets from a screened pool instead takes that floor out of the reward and out of
the eval, and fixes the population the two are read against. Colour pairs are cycled, so
the pool covers all six evenly.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import sim  # noqa: F401  (registers the env)
from sim.env import BLOCK_NAMES, PAIRS, write_layouts
from sim.oracle import Oracle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--count", type=int, default=5000, help="layouts to keep")
    parser.add_argument("--envs", type=int, default=64, help="envs stepped in lockstep")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    env = gym.make("SO101BlockStack-v1", num_envs=args.envs).unwrapped

    kept, batch = [], 0
    while len(kept) < args.count:
        pairs = [PAIRS[(batch * env.num_envs + i) % len(PAIRS)] for i in range(env.num_envs)]
        env.reset(seed=args.seed + batch, options={"layout": {
            "held": [pair[0] for pair in pairs], "target": [pair[1] for pair in pairs]}})
        layout = {key: value.cpu().numpy() for key, value in env.layout().items()}

        Oracle(env).run(env.held, env.target)
        won = env.evaluate()["success"].cpu().numpy()
        kept += [{key: value[i] for key, value in layout.items()}
                 for i in np.flatnonzero(won)]
        print(f"batch {batch}: {won.sum()}/{env.num_envs} stacked, "
              f"{len(kept)}/{args.count} kept", flush=True)
        batch += 1
    env.close()

    kept = kept[:args.count]
    write_layouts(args.out, kept, "index")
    counts = Counter((int(item["held"]), int(item["target"])) for item in kept)
    print(f"\n{len(kept)} layouts in {args.out}, from {batch * args.envs} spawns")
    for held, target in PAIRS:
        print(f"  {BLOCK_NAMES[held]} on {BLOCK_NAMES[target]}: {counts[held, target]}")


if __name__ == "__main__":
    main()
