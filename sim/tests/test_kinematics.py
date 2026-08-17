import gymnasium as gym
import numpy as np
import torch

import sim  # noqa: F401  (registers the env)
from sim.agent import HOME_QPOS
from sim.kinematics import HOME_JAW_YAW, HOME_TCP, HOME_TILT, TILT_DOWN, ik


def _tcp(env, q):
    env.agent.robot.set_qpos(torch.as_tensor(q, dtype=torch.float32, device=env.device))
    env.scene._gpu_apply_all()
    env.scene.px.gpu_update_articulation_kinematics()
    env.scene._gpu_fetch_all()
    pose = env.agent.tcp_pose
    return pose.p.cpu().numpy(), pose.q.cpu().numpy()


def test_ik_round_trips_through_sapien_fk():
    env = gym.make("SO101BlockStack-v1", num_envs=64).unwrapped
    env.reset(seed=0)

    rng = np.random.default_rng(0)
    target = np.stack([rng.uniform(0.17, 0.27, 64), rng.uniform(-0.10, 0.14, 64),
                       rng.uniform(0.755, 0.80, 64)], axis=-1)
    yaw = rng.uniform(-np.pi / 4, np.pi / 4, 64)

    q, reachable = ik(target, yaw)
    assert reachable.all()

    p, quat = _tcp(env, torch.cat([q.float(), torch.full((64, 1), 0.5)], dim=-1))
    assert np.abs(p - target).max() < 1e-4

    # Jaw axis on the commanded yaw, tool straight down.
    w, x, y, z = quat.T
    jaw = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)
    down = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)
    assert np.abs(np.arctan2(jaw[:, 1], jaw[:, 0]) - yaw).max() < 1e-4
    assert down[:, 2].max() < -0.9999

    # Tilting the tool in from above still lands the roll axis on target.
    raised = target + np.array([0.0, 0.0, 0.06])
    q_tilt, ok = ik(raised, yaw, tilt=TILT_DOWN - 0.4)
    assert ok.all()
    p_tilt, _ = _tcp(env, torch.cat([q_tilt.float(), torch.full((64, 1), 0.5)], dim=-1))
    assert np.abs(p_tilt[:, 2] - raised[:, 2]).max() < 0.01
    env.close()


def test_home_tool_pose():
    """The constants teleoperation integrates its target from."""
    env = gym.make("SO101BlockStack-v1", num_envs=64).unwrapped
    env.reset(seed=0)

    p, quat = _tcp(env, torch.as_tensor(HOME_QPOS).repeat(64, 1))
    w, x, y, z = quat.T
    jaw = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)], -1)
    down = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)

    assert np.abs(p[0] - HOME_TCP).max() < 1e-5
    assert abs(np.arctan2(jaw[0, 1], jaw[0, 0]) - HOME_JAW_YAW) < 1e-5
    assert abs(np.arctan2(-down[0, 2], np.hypot(down[0, 0], down[0, 1])) - HOME_TILT) < 1e-5
    env.close()
