"""Rewrite a trained actor into the layout lerobot loads.

    uv run python rl/to_lerobot.py WEIGHTS MERGED OUT

``WEIGHTS`` is the ``full_weights.pt`` of a checkpoint RLinf saved, ``MERGED`` the lerobot
one the run warm started from. Training moves the weights and nothing else, so the config
and the processors come across from ``MERGED`` as they were.
"""

import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file

# The critic and the noise head belong to the actor. What remains is the policy, under
# the name lerobot gives it.
RL_ONLY = ("value_head.", "noise_head.")

trained, merged, out = (Path(arg) for arg in sys.argv[1:4])

state = torch.load(trained, map_location="cpu", weights_only=True)
weights = {f"model.{key}": value.contiguous() for key, value in state.items()
           if not key.startswith(RL_ONLY)}

with safe_open(merged / "model.safetensors", framework="pt") as file:
    expected = set(file.keys())
if set(weights) != expected:
    raise SystemExit(f"{len(set(weights) ^ expected)} keys differ from {merged}")

out.mkdir(parents=True, exist_ok=True)
save_file(weights, out / "model.safetensors")
for pattern in ("*.json", "policy_*.safetensors"):
    for path in merged.glob(pattern):
        shutil.copy(path, out)
print(out)
