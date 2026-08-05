import gymnasium as gym
import torch

import vla_test_sim  # noqa: F401  (registers the env)
from vla_test_sim.env import IMAGE_SIZE
from vla_test_sim.oracle import Oracle


def test_env_resets_steps_and_renders_both_cameras():
    env = gym.make("SO101BlockStack-v1", obs_mode="rgb", num_envs=2)
    obs, _ = env.reset(seed=0)
    assert set(obs["sensor_data"]) == {"wrist", "top"}
    for camera in obs["sensor_data"].values():
        assert camera["rgb"].shape == (2, IMAGE_SIZE, IMAGE_SIZE, 3)
    env.step(env.action_space.sample())
    env.close()


def test_a_batch_of_layouts_goes_back_in_the_way_it_came_out():
    """A recorded layout is what a demonstration and its eval run share."""
    env = gym.make("SO101BlockStack-v1", num_envs=4).unwrapped
    env.reset(seed=0)
    out = {key: value.clone() for key, value in env.layout().items()}

    env.reset(seed=1, options={"layout": out})
    back = env.layout()
    assert torch.allclose(back["xy"], out["xy"]) and torch.allclose(back["yaw"], out["yaw"])
    assert (back["held"] == out["held"]).all() and (back["target"] == out["target"]).all()
    env.close()


def test_oracle_stacks_a_block_inside_the_episode_limit():
    """A cycle has to fit the registered limit, or a demonstration cannot be collected."""
    env = gym.make("SO101BlockStack-v1", num_envs=4).unwrapped
    env.reset(seed=0)
    steps = []
    Oracle(env).run(env.held, env.target, on_step=steps.append)
    assert len(steps) <= gym.spec("SO101BlockStack-v1").max_episode_steps
    assert env.evaluate()["success"].any()
    env.close()
