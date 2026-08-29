"""The physical SO-101, presented the way the simulator presents it.

A policy trained in sim reads radians, bare joint names and 480x480 frames on the pinhole
`sim.spec.K_SIM`. The follower reports degrees, `<joint>.pos` names and whatever its lens
does. The processor steps here sit either side of `lerobot`'s record loop and close that
gap, so a hardware episode lands in the same schema `sim/scripts/collect.py` writes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path

import cv2
import numpy as np
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ObservationProcessorStep,
    RobotActionProcessorStep,
    RobotProcessorPipeline,
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.robot_utils import precise_sleep

from sim.spec import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_QPOS,
    IMAGE_SIZE,
    JOINT_NAMES,
    K_SIM,
)

CALIBRATION = Path(__file__).parent / "calibration"

PORT = "/dev/ttyACM0"
FOLLOWER_ID = "so101_follower"
# Which lens is which. The two cameras report the same model and the same serial, so the
# USB port each sits in is the only thing that tells them apart, and the `/dev/video`
# number behind a port changes whenever a camera re-enumerates. OpenCV's V4L2 backend
# opens a device by number rather than by name, so the port is resolved to one on the way
# in. A camera moved to a different port needs its line here changed to match.
BY_PATH = Path("/dev/v4l/by-path")
CAMERA_PORT = {"top": "pci-0000:0e:00.0-usb-0:4.3:1.0-video-index0",
               "wrist": "pci-0000:0e:00.0-usb-0:4.2:1.0-video-index0"}

# The rate the sim env runs its controller at, so a replayed trajectory keeps its timing.
FPS = 10
# The rate `walk_to` writes to the bus at.
COMMAND_HZ = 50
# How long `home` holds its last setpoint, comfortably past the servos' own lag.
SETTLE_SECONDS = 0.5

# How far a joint may be commanded from where it is, in degrees. The oracle's largest step
# at the sim's control rate is 2.9 degrees on the arm, so this is clear of the motion it
# asks for and still catches a lunge.
MAX_RELATIVE_TARGET = 10.0


def camera_index(port):
    """The `/dev/video` number currently behind a USB port."""
    return int((BY_PATH / port).resolve().name.removeprefix("video"))


def follower(port=PORT, cameras=CAMERA_PORT):
    """The arm and its two lenses, at the resolution the calibration was taken at."""
    return SO101Follower(SO101FollowerConfig(
        port=port,
        id=FOLLOWER_ID,
        use_degrees=True,
        max_relative_target=MAX_RELATIVE_TARGET,
        cameras={name: OpenCVCameraConfig(index_or_path=camera_index(usb),
                                          width=640, height=480, fps=30)
                 for name, usb in cameras.items()},
    ))


@dataclass
class ToSimUnits(ObservationProcessorStep):
    """`<joint>.pos` in degrees to the bare radian names the sim dataset carries."""

    def observation(self, observation: RobotObservation) -> RobotObservation:
        for joint in JOINT_NAMES:
            observation[joint] = float(np.deg2rad(observation[f"{joint}.pos"]))
        return observation

    def transform_features(self, features):
        return features


@dataclass
class ToRobotUnits(RobotActionProcessorStep):
    """The same conversion back, for the action on its way to the bus."""

    def action(self, action: RobotAction) -> RobotAction:
        return {f"{joint}.pos": float(np.rad2deg(action[joint])) for joint in JOINT_NAMES}

    def transform_features(self, features):
        return features


@dataclass
class ToSimCameras(ObservationProcessorStep):
    """Undistort each lens and reproject it onto the sim pinhole.

    `cv2.initUndistortRectifyMap` takes the intrinsic the output is wanted in, so asking
    for `K_SIM` at 480x480 does the undistortion and the centre crop in the one remap.

    The top lens covers the whole sim frame. The wrist lens sits 28 px low against
    `K_SIM`'s principal point, so the bottom 10 rows of its reprojection fall off the
    sensor; the edge is replicated into them rather than left black, since a hard black
    band is a landmark the sim frames never carry.
    """

    calibration: Path = CALIBRATION
    _maps: dict = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._maps = {}
        for path in sorted(self.calibration.glob("*.json")):
            lens = json.loads(path.read_text())
            self._maps[path.stem] = cv2.initUndistortRectifyMap(
                np.array(lens["K_real"]), np.array(lens["dist"]), None, K_SIM,
                (IMAGE_SIZE, IMAGE_SIZE), cv2.CV_16SC2,
            )

    def observation(self, observation: RobotObservation) -> RobotObservation:
        for name, (map1, map2) in self._maps.items():
            frame = observation.get(name)
            if frame is not None:
                observation[name] = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REPLICATE)
        return observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def observations():
    """What the follower reports, as the sim dataset holds it."""
    return RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[ToSimCameras(), ToSimUnits()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )


def actions():
    """A joint command in sim units, as the bus takes it."""
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[ToRobotUnits()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


# Where the arm waits between episodes, folded back over its own base and short of where
# the blocks go, so the next layout is placed against a frame it does not obstruct. Three
# of the angles are what the arm allows rather than what is tidy: `shoulder_pan` stays at
# zero because a quarter turn either way rubs the camera post, `shoulder_lift` is at the
# description's lower limit, and `wrist_flex` waits well below the right angle its
# neighbours are at, far enough that the jaw clears the arm it folds back over and short
# of the angle the servo stalls at. The arm has to stand where the sim draws it for the
# overlay to be worth anything.
PARK_QPOS = np.array([*np.deg2rad([0.0, -100.0, 90.0, 70.0, 0.0]), GRIPPER_OPEN])


def _ramp(start, goal, steps):
    """The setpoints from ``start`` to ``goal``, paced on the `COMMAND_HZ` clock.

    Sleeping the whole period on top of the caller's work stretches the motion, and a
    trajectory is paced for the sim's clock.
    """
    for step in range(1, steps + 1):
        began = time.perf_counter()
        yield start + (goal - start) * step / steps
        precise_sleep(max(1 / COMMAND_HZ - (time.perf_counter() - began), 0.0))


def walk_to(robot, to_robot, held, command, observation):
    """Send one control step as the bus writes along the way to it.

    The sim's controller settles inside one of its own steps, so the joint path between
    setpoints is continuous there; a Feetech servo at P_Coefficient 16 reaches the setpoint
    and dwells, which the arm shows as stepping. Dividing the same path into
    `COMMAND_HZ / FPS` writes gives it a reference that moves the way the sim's does, and
    leaves the commanded trajectory unchanged.
    """
    for towards in _ramp(held, command, COMMAND_HZ // FPS):
        robot.send_action(to_robot((dict(zip(JOINT_NAMES, towards, strict=True)),
                                    observation)))


def home(robot, qpos=HOME_QPOS, seconds=3.0):
    """Ramp to ``qpos`` and hold there, at a pace the servos track under their own control.

    A Feetech servo trails its setpoint by up to 90 ms, and the ramp runs to its last
    setpoint at full speed, so the arm is still short of the pose when the ramp ends. The
    hold is what lets it arrive, and whatever reads the arm next reads it standing still.
    """
    start = np.array([robot.get_observation()[f"{joint}.pos"] for joint in JOINT_NAMES])
    goal = np.rad2deg(qpos)
    for blend in chain(_ramp(start, goal, int(seconds * COMMAND_HZ)),
                       _ramp(goal, goal, int(SETTLE_SECONDS * COMMAND_HZ))):
        robot.send_action({f"{joint}.pos": float(value)
                           for joint, value in zip(JOINT_NAMES, blend, strict=True)})


# Midway between the two widths the jaws are ever asked for. A command eases between them
# over several steps, so the width passes through this value rather than resting on it.
HALF_SHUT = (GRIPPER_OPEN + GRIPPER_CLOSED) / 2


def pick(jaw):
    """The steps spanning the first close in a commanded ``jaw`` width, where a block is grasped.

    An episode opens wherever the previous cycle left the arm, so half of them open with the
    jaw already shut on nothing, a pose narrower than any grasp. What the arm reached over
    this window is what says whether a block was between the fingers.
    """
    jaw = np.asarray(jaw)
    opened = np.flatnonzero(jaw > HALF_SHUT)
    shut = np.flatnonzero(jaw <= HALF_SHUT)
    if not len(opened):
        return slice(0, 0)
    closes = shut[shut > opened[0]]
    if not len(closes):
        return slice(0, 0)
    reopens = opened[opened > closes[0]]
    return slice(closes[0], reopens[0] if len(reopens) else len(jaw))


def picks(jaw):
    """Every close in a commanded ``jaw`` width, in order.

    An episode long enough for the policy to try again after a missed grasp holds more
    than one, and what the arm reached on its best attempt is what says whether it ever
    had a block.
    """
    jaw, at = np.asarray(jaw), 0
    while at < len(jaw):
        window = pick(jaw[at:])
        if window.start == window.stop:
            return
        yield slice(at + window.start, at + window.stop)
        at += window.stop
