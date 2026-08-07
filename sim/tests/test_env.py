import gymnasium as gym
import numpy as np
import torch

import sim  # noqa: F401  (registers the env)
from sim.env import IMAGE_SIZE, read_layouts, write_layouts
from sim.oracle import Oracle


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


def test_resets_are_drawn_from_a_screened_pool(tmp_path):
    """RL draws its spawns from a screened file rather than sampling them fresh."""
    env = gym.make("SO101BlockStack-v1", num_envs=8).unwrapped
    env.reset(seed=0)
    written = tmp_path / "pool.jsonl"
    write_layouts(written, [{key: value[i].cpu().numpy() for key, value in env.layout().items()}
                            for i in range(env.num_envs)], "index")
    env.close()
    pool = read_layouts(written)

    env = gym.make("SO101BlockStack-v1", num_envs=4, layouts=written).unwrapped
    env.reset(seed=1)
    drawn = env.layout()
    for i in range(env.num_envs):
        matches = np.flatnonzero(
            np.isclose(pool["xy"], drawn["xy"][i].cpu().numpy(), atol=1e-6).all((1, 2)))
        assert len(matches) == 1, "every reset comes from one pool row"
        row = matches[0]
        assert pool["held"][row] == int(drawn["held"][i])
        assert pool["target"][row] == int(drawn["target"][i])
        assert np.allclose(pool["yaw"][row], drawn["yaw"][i].cpu().numpy(), atol=1e-6)
    env.close()


def test_oracle_stacks_a_block_inside_the_episode_limit():
    """A cycle has to fit the registered limit, or a demonstration cannot be collected."""
    env = gym.make("SO101BlockStack-v1", num_envs=4).unwrapped
    env.reset(seed=0)
    steps = []
    Oracle(env).run(env.held, env.target, on_step=steps.append)
    assert len(steps) <= gym.spec("SO101BlockStack-v1").max_episode_steps
    assert env.evaluate()["success"].any()

    # ManiSkill scores a step once, and the reward banks as it goes, so a second reader
    # must not bank it again.
    first = env.evaluate()["reward"]
    second = env.evaluate()["reward"]
    assert first.max() > 0, "a stacked episode has banked something to double"
    assert torch.equal(first, second), "reading twice banked it twice"
    env.close()
