import numpy as np

from hw.robot import MAX_RELATIVE_TARGET
from hw.teleop import BOUNDS, LIMITS, TILT_RANGE, SpaceMouseToJoints, solve, start_qpos
from sim.agent import ARM_JOINTS, JOINT_NAMES

CHANNELS = ("dx", "dy", "dz", "d_tilt", "d_yaw", "gripper")


def held(step, act):
    """One control frame, as the joint vector the follower is sent."""
    out = step.action(dict(act))
    return np.array([out[joint] for joint in JOINT_NAMES])


def test_starts_where_the_arm_is_homed():
    """The first commanded frame matches `start_qpos`, so nothing moves on the first tick."""
    first = held(SpaceMouseToJoints(), dict.fromkeys(CHANNELS, 0.0))

    assert np.abs(first - start_qpos()).max() == 0.0


def test_no_command_outruns_the_follower():
    """Full deflection on every channel, from anywhere reachable, moves a joint no further
    between frames than the follower's cap, and never leaves a joint limit."""
    rng = np.random.default_rng(0)
    cap = np.deg2rad(MAX_RELATIVE_TARGET)

    for _ in range(40):
        tcp = rng.uniform(BOUNDS[0], BOUNDS[1])
        tilt = float(rng.uniform(*TILT_RANGE))
        angles, solved = solve(tcp, 1.6, tilt)
        if not solved:
            continue
        step = SpaceMouseToJoints()
        step.tcp, step.tilt, step.jaw_yaw, step.angles = tcp, tilt, 1.6, angles

        push = rng.choice([-1.0, 1.0], 5)
        act = dict(zip(CHANNELS[:3], 0.0045 * push[:3], strict=True))
        act |= dict(zip(CHANNELS[3:5], 0.05 * push[3:], strict=True)) | {"gripper": -1.0}

        previous = held(step, act)
        for _ in range(30):
            current = held(step, act)
            assert np.abs(current - previous).max() <= cap
            assert (step.angles >= LIMITS[0]).all() and (step.angles <= LIMITS[1]).all()
            previous = current


def test_joint_limits_come_from_the_description():
    assert LIMITS.shape == (2, len(ARM_JOINTS))
    assert (LIMITS[0] < LIMITS[1]).all()
