import numpy as np

from hw.robot import actions, observations
from sim.agent import HOME_QPOS, JOINT_NAMES
from sim.env import IMAGE_SIZE


def follower_observation(qpos):
    return {**{f"{joint}.pos": float(np.rad2deg(value))
               for joint, value in zip(JOINT_NAMES, qpos, strict=True)},
            **{camera: np.full((480, 640, 3), 40, np.uint8) for camera in ("top", "wrist")}}


def test_units_round_trip():
    """A pose read off the bus and commanded back reaches the same degrees."""
    observation = observations()(follower_observation(HOME_QPOS))
    command = {joint: observation[joint] for joint in JOINT_NAMES}

    for joint, value in actions()((command, observation)).items():
        assert value == follower_observation(HOME_QPOS)[joint]


def test_cameras_land_on_the_sim_pinhole():
    observation = observations()(follower_observation(HOME_QPOS))

    for camera in ("top", "wrist"):
        assert observation[camera].shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        # A flat frame comes through flat, including the rows the wrist lens reprojects
        # from off the sensor, which `BORDER_REPLICATE` fills from the edge.
        assert observation[camera].min() == observation[camera].max() == 40
