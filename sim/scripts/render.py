"""Write the two camera views and an inspection view to PNGs."""

import sys
from pathlib import Path

import gymnasium as gym
from PIL import Image

import sim.env  # noqa: F401  (registers the env)

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/so101")
out.mkdir(parents=True, exist_ok=True)

env = gym.make("SO101BlockStack-v1", obs_mode="rgb", render_mode="rgb_array", num_envs=1)
obs, _ = env.reset(seed=0)

views = {name: cam["rgb"][0] for name, cam in obs["sensor_data"].items()}
views["inspect"] = env.unwrapped.render_rgb_array()[0]
for name, image in views.items():
    Image.fromarray(image.cpu().numpy()).save(out / f"{name}.png")
    print(out / f"{name}.png")

env.close()
