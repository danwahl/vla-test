import gymnasium as gym
import numpy as np
import torch

import sim.env  # noqa: F401  (registers the env)
from sim.env import HOLD_REWARD, read_layouts, write_layouts
from sim.oracle import Oracle
from sim.spec import HOME_QPOS, IMAGE_SIZE


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


def test_a_group_of_envs_can_be_given_one_layout(tmp_path):
    """What GRPO rests on: RLinf hands a group of envs one pool row by index."""
    env = gym.make("SO101BlockStack-v1", num_envs=8).unwrapped
    env.reset(seed=0)
    written = tmp_path / "pool.jsonl"
    write_layouts(written, [{key: value[i].cpu().numpy() for key, value in env.layout().items()}
                            for i in range(env.num_envs)], "index")
    env.close()

    env = gym.make("SO101BlockStack-v1", num_envs=8, layouts=written).unwrapped
    assert env.total_num_trials == 8
    env.reset(options={"episode_id": torch.tensor([2] * 4 + [5] * 4)})
    # Each env places its own copy, so a group agrees to float precision, not exactly.
    xy = env.layout()["xy"]
    assert (xy[:4] - xy[0]).abs().max() < 1e-6
    assert (xy[4:] - xy[4]).abs().max() < 1e-6
    assert (xy[0] - xy[4]).abs().max() > 1e-3
    env.close()

    env = gym.make("SO101BlockStack-v1", num_envs=8, layouts=written, sample=False).unwrapped
    env.reset()
    first = env.layout()["xy"].clone()
    env.reset()
    assert (first - env.layout()["xy"]).abs().max() < 1e-6
    env.close()


def test_the_reward_level_only_rises():
    """RL differences this level, so a fall must cost nothing rather than a point."""
    env = gym.make("SO101BlockStack-v1", num_envs=4).unwrapped
    env.reset(seed=0, options={"layout": {"held": [0] * 4, "target": [1] * 4}})

    levels, wins = [], []

    def watch(_obs):
        info = env.evaluate()
        # A second reader in the same step must not advance the count.
        assert torch.equal(info["reward"], env.evaluate()["reward"])
        levels.append(info["reward"].clone())
        wins.append(info["success"].clone())

    Oracle(env).run(env.held, env.target, on_step=watch)
    env.close()

    level, won = torch.stack(levels), torch.stack(wins).float()
    assert won.any(0).any(), "the oracle stacked none of them"
    assert (level.diff(dim=0) >= 0).all()
    # A point once seated, kept where the arm knocked it over, and a step's worth per step
    # it stood.
    assert torch.allclose(level[-1], won.amax(0) + HOLD_REWARD * won.sum(0))


def test_oracle_stacks_a_block_inside_the_episode_limit():
    """A cycle has to fit the registered limit, or a demonstration cannot be collected."""
    env = gym.make("SO101BlockStack-v1", num_envs=4).unwrapped
    env.reset(seed=0)
    steps = []
    Oracle(env).run(env.held, env.target, on_step=steps.append)
    assert len(steps) <= gym.spec("SO101BlockStack-v1").max_episode_steps
    assert env.evaluate()["success"].any()
    env.close()


def test_oracle_recovers_from_the_pose_a_cycle_leaves():
    """The longest cycle is the one that starts shut and away from the rest pose.

    A demonstration collected from there has to fit the same limit, and the pose it starts
    in has to be one the arm actually holds: a start off in the joint limits would be
    recorded as a lurch on the first step.
    """
    env = gym.make("SO101BlockStack-v1", num_envs=4).unwrapped
    limits = env.agent.robot.get_qlimits()[0]

    for seed in range(6):
        env.reset(seed=seed)
        start = Oracle(env).draw_start(torch.Generator(device=env.device).manual_seed(seed))
        assert ((start >= limits[:, 0]) & (start <= limits[:, 1])).all()
        # Every episode opens part way through a cycle, so the rest pose is a fallback for
        # a pose the arm cannot hold.
        at_rest = torch.isclose(start[:, :5].float(),
                                torch.tensor(HOME_QPOS[:5], device=env.device)).all(-1)
        assert not at_rest.any()

        env.reset(options={"layout": {**env.layout(), "qpos": start}})
        # The arm is standing in the pose before a step is taken, so the first frame of a
        # demonstration is not recorded against a pose it is still travelling to.
        oracle = Oracle(env)
        assert torch.allclose(env.agent.robot.get_qpos(), start.float(), atol=1e-5)
        assert torch.allclose(oracle.command, start, atol=1e-5)

        steps = []
        oracle.run(env.held, env.target, on_step=steps.append)
        assert len(steps) <= gym.spec("SO101BlockStack-v1").max_episode_steps
    env.close()


def test_a_layout_carries_the_pose_it_opens_in():
    """A layout has to open in the same pose every time, or a re-take is a different one."""
    env = gym.make("SO101BlockStack-v1", num_envs=4).unwrapped
    env.reset(seed=0)
    start = Oracle(env).draw_start(torch.Generator(device=env.device).manual_seed(7))

    poses = []
    for _ in range(2):
        env.reset(seed=0, options={"layout": {**env.layout(), "qpos": start}})
        poses.append(env.agent.robot.get_qpos().clone())
    env.close()
    assert torch.equal(*poses)
    assert torch.allclose(poses[0], start.float(), atol=1e-5)
