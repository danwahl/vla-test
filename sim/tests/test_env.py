import gymnasium as gym

import vla_test_sim  # noqa: F401  (registers the env)
from vla_test_sim.env import IMAGE_SIZE
from vla_test_sim.oracle import Oracle


def test_env_resets_steps_and_renders_both_cameras():
    env = gym.make("SO101Blocks-v1", obs_mode="rgb", num_envs=2)
    obs, _ = env.reset(seed=0)
    assert set(obs["sensor_data"]) == {"wrist", "top"}
    for camera in obs["sensor_data"].values():
        assert camera["rgb"].shape == (2, IMAGE_SIZE, IMAGE_SIZE, 3)
    env.step(env.action_space.sample())
    env.close()


def test_oracle_stacks_a_cube_inside_the_episode_limit():
    """A cycle has to fit the registered limit, or a demonstration cannot be collected."""
    env = gym.make("SO101Blocks-v1", num_envs=4).unwrapped
    env.reset(seed=0)
    steps = []
    Oracle(env).run(env.held, env.target, on_step=lambda: steps.append(None))
    assert len(steps) <= gym.spec("SO101Blocks-v1").max_episode_steps
    assert env.evaluate()["success"].any()
    env.close()
