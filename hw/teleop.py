"""Drive the arm with a SpaceMouse, and record what it does.

    uv run python -m hw.teleop --held red --target green
    uv run python -m hw.teleop --held red --target green --out OUT

The cap moves the tool, tilts it and turns the jaw; the two buttons work the gripper. The
target is integrated from the home pose and solved by the same closed-form IK the oracle
plans with, so a teleoperated episode lands in the joint-space distribution the sim
demonstrations occupy. Episode control is `lerobot-record`'s: the keys it prints on start.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lerobot.processor import (
    RobotActionProcessorStep,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.keyboard_input import init_keyboard_listener

from hw.robot import FPS, MAX_RELATIVE_TARGET, actions, follower, home, observations
from hw.spacemouse import SpaceMouse, SpaceMouseConfig
from sim.agent import ARM_JOINTS, GRIPPER_CLOSED, GRIPPER_OPEN, SO101
from sim.dataset import create
from sim.env import BLOCK_NAMES, TABLE_TOP_Z, prompt
from sim.kinematics import HOME_JAW_YAW, HOME_TCP, HOME_TILT, TILT_DOWN, ik

# Where the tool may go: over the table, clear of its top, and inside the arm's reach.
# The home pose sits at the far end of the x range.
BOUNDS = np.array([[0.11, -0.22, TABLE_TOP_Z + 0.012],
                   [0.36, 0.22, TABLE_TOP_Z + 0.230]])
# Straight down, to nearly horizontal, which is as flat as the reach allows.
TILT_RANGE = (0.40, TILT_DOWN)
# Jaw travel per frame, so a button held for a second covers the whole range.
JAW_RATE = (GRIPPER_OPEN - GRIPPER_CLOSED) / FPS
# As far as a joint may move between frames, set to the follower's own cap. The follower
# caps against where the arm is rather than against where it was last told to go, so
# tracking lag can still clip a full step; this is what bounds the motion asked for. A
# clipped command is still the one recorded.
MAX_JOINT_STEP = np.deg2rad(MAX_RELATIVE_TARGET)


def _joint_limits():
    joints = {joint.get("name"): joint.find("limit")
              for joint in ElementTree.parse(SO101.urdf_path).getroot().findall("joint")}
    return np.array([[float(joints[name].get(edge)) for name in ARM_JOINTS]
                     for edge in ("lower", "upper")])


LIMITS = _joint_limits()


def solve(tcp, jaw_yaw, tilt):
    """The arm joints for a tool pose, and whether the arm can hold them."""
    angles, reachable = ik(tcp, jaw_yaw, tilt)
    angles = angles.numpy()
    return angles, bool(reachable) and bool(((LIMITS[0] <= angles) & (angles <= LIMITS[1])).all())


@dataclass
class SpaceMouseToJoints(RobotActionProcessorStep):
    """Cap increments to joint angles, through the oracle's IK.

    The target is carried here rather than read back from the arm, so letting go of the
    cap holds the pose instead of drifting down against gravity.

    A step the arm cannot follow is refused whole and the tool stays where it was. Three
    ways it cannot: the two-link law has no solution there, a joint would leave its limit,
    or a joint would move further in one frame than the follower will carry. The last one
    also catches the solver changing branch, which reads as most of a turn between frames
    on what is nearly the same pose.
    """

    tcp: np.ndarray = field(default_factory=lambda: np.array(HOME_TCP))
    tilt: float = HOME_TILT
    jaw_yaw: float = HOME_JAW_YAW
    jaw: float = GRIPPER_OPEN
    angles: np.ndarray = field(
        default_factory=lambda: solve(HOME_TCP, HOME_JAW_YAW, HOME_TILT)[0])

    def action(self, action: RobotAction) -> RobotAction:
        step = np.array([action["dx"], action["dy"], action["dz"]])
        tcp = np.clip(self.tcp + step, *BOUNDS)
        tilt = float(np.clip(self.tilt + action["d_tilt"], *TILT_RANGE))
        jaw_yaw = self.jaw_yaw + action["d_yaw"]

        angles, solved = solve(tcp, jaw_yaw, tilt)
        if solved and np.abs(angles - self.angles).max() <= MAX_JOINT_STEP:
            self.tcp, self.tilt, self.jaw_yaw, self.angles = tcp, tilt, jaw_yaw, angles

        self.jaw = float(np.clip(self.jaw + action["gripper"] * JAW_RATE,
                                 GRIPPER_CLOSED, GRIPPER_OPEN))
        return {**dict(zip(ARM_JOINTS, self.angles.tolist(), strict=True)),
                "gripper": self.jaw}

    def reset(self):
        self.tcp = np.array(HOME_TCP)
        self.tilt, self.jaw_yaw, self.jaw = HOME_TILT, HOME_JAW_YAW, GRIPPER_OPEN
        self.angles = solve(HOME_TCP, HOME_JAW_YAW, HOME_TILT)[0]

    def transform_features(self, features):
        return features


def start_qpos():
    """Where the arm is homed for teleoperation.

    The IK's own answer for the home tool pose, which is a few degrees off `HOME_QPOS`:
    the jaw-offset back-out is exact fingers-down and the home pose is half that tilt.
    Homing here rather than there keeps the first commanded frame continuous.
    """
    return np.append(solve(HOME_TCP, HOME_JAW_YAW, HOME_TILT)[0], GRIPPER_OPEN)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--held", choices=BLOCK_NAMES, default="red")
    parser.add_argument("--target", choices=BLOCK_NAMES, default="green")
    parser.add_argument("--out", type=Path, help="write episodes here")
    parser.add_argument("--repo-id", default="vla-test/so101_block_stack_hw")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seconds", type=int, default=60, help="cap on one episode")
    args = parser.parse_args()

    task = prompt(BLOCK_NAMES.index(args.held), BLOCK_NAMES.index(args.target))
    dataset = create(args.out, args.repo_id, FPS) if args.out else None

    robot, mouse = follower(), SpaceMouse(SpaceMouseConfig(id="spacemouse"))
    to_joints = SpaceMouseToJoints()
    listener = None
    # Connecting the follower leaves torque on, so everything from here is unwound in the
    # `finally` whether or not the rest of the setup succeeded.
    try:
        robot.connect()
        mouse.connect()
        listener, events = init_keyboard_listener()
        for episode in range(args.episodes):
            print(f"episode {episode}: {task}")
            home(robot, start_qpos())
            to_joints.reset()
            record_loop(
                robot=robot, events=events, fps=FPS,
                teleop_action_processor=RobotProcessorPipeline[
                    tuple[RobotAction, RobotObservation], RobotAction](
                        steps=[to_joints],
                        to_transition=robot_action_observation_to_transition,
                        to_output=transition_to_robot_action),
                robot_action_processor=actions(),
                robot_observation_processor=observations(),
                dataset=dataset, teleop=mouse, control_time_s=args.seconds,
                single_task=task,
            )
            if dataset is not None:
                if events["rerecord_episode"]:
                    events["rerecord_episode"] = False
                    dataset.clear_episode_buffer()
                else:
                    dataset.save_episode()
            if events["stop_recording"]:
                break
    finally:
        # `init_keyboard_listener` returns None where no backend is usable, which is what
        # a run with its output piped gets.
        if listener is not None:
            listener.stop()
        for device in (mouse, robot):
            if device.is_connected:
                device.disconnect()
        if dataset is not None:
            dataset.finalize()


if __name__ == "__main__":
    main()
