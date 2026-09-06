import json

import cv2
import numpy as np

from hw.robot import CALIBRATION, actions, observations, pick, picks
from sim.spec import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_QPOS,
    IMAGE_SIZE,
    JOINT_NAMES,
)

OPEN, SHUT = GRIPPER_OPEN, GRIPPER_CLOSED


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
    """A mark on each lens' own axis comes out on `K_SIM`'s.

    The remap sends an output pixel through `K_SIM` and back out through `K_real`, so the
    two principal points are the one pair of pixels that has to correspond whatever the
    distortion is.
    """
    frame = follower_observation(HOME_QPOS)
    for camera in ("top", "wrist"):
        lens = json.loads((CALIBRATION / f"{camera}.json").read_text())
        centre = np.array(lens["K_real"])[:2, 2]
        cv2.circle(frame[camera], centre.round().astype(int), 4, (255, 255, 255), -1)

    observation = observations()(frame)
    for camera in ("top", "wrist"):
        assert observation[camera].shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        brightest = np.argwhere(observation[camera][..., 0] == 255).mean(axis=0)
        assert np.abs(brightest - IMAGE_SIZE / 2).max() < 1.0


def test_pick_is_the_first_close():
    assert pick([OPEN] * 3 + [SHUT] * 4 + [OPEN] * 2) == slice(3, 7)


def test_pick_skips_a_jaw_that_starts_shut():
    """An episode opening mid-cycle holds a closed jaw before it ever reaches a block."""
    assert pick([SHUT] * 3 + [OPEN] * 2 + [SHUT] * 4 + [OPEN]) == slice(5, 9)


def test_pick_starts_where_a_ramp_crosses_the_midpoint():
    # The middle sample lands exactly on `HALF_SHUT`, which counts as closing.
    assert pick([OPEN, *np.linspace(OPEN, SHUT, 5), SHUT]) == slice(3, 7)


def test_pick_runs_to_the_end_when_the_jaw_never_reopens():
    assert pick([OPEN, SHUT, SHUT]) == slice(1, 3)


def test_pick_is_empty_without_a_close():
    assert pick([OPEN] * 3) == slice(0, 0)


def test_pick_is_empty_without_an_open():
    assert pick([SHUT] * 3) == slice(0, 0)


def test_picks_finds_every_close():
    """A missed grasp is followed by another go, and the second one is the grasp."""
    jaw = [OPEN] * 2 + [SHUT] * 2 + [OPEN] * 3 + [SHUT] * 2 + [OPEN]
    assert list(picks(jaw)) == [slice(2, 4), slice(7, 9)]


def test_picks_gives_the_one_close_of_a_single_cycle():
    jaw = [OPEN] * 3 + [SHUT] * 4 + [OPEN] * 2
    assert list(picks(jaw)) == [pick(jaw)]


def test_picks_is_empty_without_a_close():
    assert list(picks([OPEN] * 3)) == []


def test_picks_ends_on_a_close_that_never_reopens():
    assert list(picks([OPEN, SHUT, OPEN, SHUT, SHUT])) == [slice(1, 2), slice(3, 5)]
