import gymnasium as gym

import vla_test_sim  # noqa: F401  (registers the env)
from vla_test_sim.env import IMAGE_SIZE


def test_env_resets_steps_and_renders_both_cameras():
    env = gym.make("SO101Blocks-v1", obs_mode="rgb", num_envs=2)
    obs, _ = env.reset(seed=0)
    assert set(obs["sensor_data"]) == {"wrist", "top"}
    for camera in obs["sensor_data"].values():
        assert camera["rgb"].shape == (2, IMAGE_SIZE, IMAGE_SIZE, 3)
    env.step(env.action_space.sample())
    env.close()
