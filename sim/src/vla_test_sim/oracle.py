"""Scripted pick-and-stack expert, planned in closed form and run across all envs at once.

A cycle is a handful of keyposes solved by :mod:`kinematics` and joined in joint space by
splines that cross each one without stopping on it, so the arm comes to rest only where
the jaws work against a still arm. Planning in joint space fixes the solution branch for a
whole move. Every env runs the same phases with its own joint targets, which keeps the
batch in lockstep for rendering.
"""

from __future__ import annotations

import math

import torch

from .agent import HOME_QPOS
from .env import CUBE_SIDE
from .kinematics import ik, ik_clearance, ik_straight_up

GRASP_DZ = -0.003        # clamp just below the cube centre so the jaw tips clear the table
GRASP_LATERAL = 0.020    # along the jaw axis, seating the cube against the fixed jaw
HOVER_DZ = 0.06          # clearance above the grasp and above the target stack
RETRACT_DZ = 0.05        # leave the seated cube by this much, before the tool has to tilt
REST_GAP = 0.001         # release this close to a seated stack, so it settles rather than drops

# One width for open and one for closed, so a policy reads a single unambiguous pair. The
# jaw is asymmetric and brushes a seated cube on its way open, at a cost of about a stack
# in a hundred.
GRIP_OPEN = 0.7
GRIP_CLOSED = 0.09       # a few mm inside the 30 mm cube

# Opening slowly lets the placed cube settle while the jaw is still moving. Closing spends
# its settling time afterwards instead: the grip loads only once the jaws have stopped, and
# the in-grip offset is read the moment that hold ends.
CLOSE_STEPS = 6
RELEASE_STEPS = 18
GRIP_STEPS = 2

JOINT_VEL_MAX = 0.5      # rad/s commanded; faster starts slipping the cube in the grip
PROBE_SAMPLES = 256      # resolution at which a move is measured to size its step count


def fold_yaw(yaw):
    """A cube is 4-fold symmetric, so fold its yaw into the nearest quarter turn."""
    return torch.remainder(yaw + math.pi / 4, math.pi / 2) - math.pi / 4


def _yaw_of(quat):
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _smoothstep(t):
    return t * t * (3 - 2 * t)


def _spline(u, knots, at):
    """Sample a shape-preserving cubic through ``knots``, at knot parameters ``u``, at ``at``.

    The slope is zero at the first and last knot and continuous everywhere between, so the
    curve leaves and arrives at rest yet crosses every interior knot exactly and at speed.
    Interior slopes are the Fritsch-Carlson harmonic mean, which holds each piece inside
    the two values it joins, so the arm never swings past a keypose it is aiming for.

    ``knots`` is ``(..., len(u), channels)``; the leading dimensions ride along, which is
    what lets one call cover every env.
    """
    h = (u[1:] - u[:-1])[:, None]
    d = (knots[..., 1:, :] - knots[..., :-1, :]) / h
    before, after = d[..., :-1, :], d[..., 1:, :]

    # A knot the path doubles back at, or arrives at flat, is a turning point and gets a
    # flat slope. Everywhere else the slope is the harmonic mean of the two segments,
    # which the gentler of them dominates; that is what holds the curve inside them.
    onward = before * after > 0
    w1, w2 = 2 * h[1:] + h[:-1], h[1:] + 2 * h[:-1]
    ones = torch.ones_like(before)
    mean = (w1 + w2) / (w1 / torch.where(onward, before, ones)
                        + w2 / torch.where(onward, after, ones))
    m = torch.zeros_like(knots)
    m[..., 1:-1, :] = torch.where(onward, mean, torch.zeros_like(mean))

    i = torch.clamp(torch.searchsorted(u, at, right=True) - 1, 0, len(u) - 2)
    hi = h[i]
    s = ((at - u[i]) / hi[:, 0])[:, None]
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1) * knots[..., i, :]
            + (s3 - 2 * s2 + s) * hi * m[..., i, :]
            + (3 * s2 - 2 * s3) * knots[..., i + 1, :]
            + (s3 - s2) * hi * m[..., i + 1, :])


class Oracle:
    """Drives every env through one pick-and-stack cycle."""

    def __init__(self, env):
        self.env = env
        self.n = env.num_envs
        self.device = env.device
        self.limits = env.agent.robot.get_qlimits()[0].to(torch.float64)
        self.command = torch.as_tensor(HOME_QPOS, dtype=torch.float64,
                                       device=self.device).repeat(self.n, 1)
        self.on_step = None

    def _up(self, dz):
        return torch.tensor([0.0, 0.0, dz], dtype=torch.float64, device=self.device)

    def _cube(self, idx):
        """(position, folded yaw) of the per-env cube named by index ``idx``."""
        poses = torch.stack([c.pose.raw_pose for c in self.env.cubes.values()], dim=1)
        pose = poses[torch.arange(self.n, device=self.device), idx].to(torch.float64)
        return pose[:, :3], fold_yaw(_yaw_of(pose[:, 3:]))

    def _in_limits(self, q, reachable):
        return reachable & (q >= self.limits[:5, 0]).all(-1) & (q <= self.limits[:5, 1]).all(-1)

    def _grasp_tcp(self, pos, jaw):
        jaw_axis = torch.stack([torch.cos(jaw), torch.sin(jaw), torch.zeros_like(jaw)], -1)
        return pos + GRASP_LATERAL * jaw_axis + self._up(GRASP_DZ)

    def _plan_grasp(self, pos, yaw):
        """Solve the grasp on the quarter turn of the jaw that twists the wrist least.

        All four straddle the cube identically, so among the ones the arm can hold, take
        the one nearest the wrist's current roll rather than the first that works.
        """
        roll_now = self.command[:, 4]
        q_best, jaw_best = None, yaw.clone()
        cost = torch.full((self.n,), math.inf, device=self.device, dtype=torch.float64)
        for quarter in range(4):
            jaw = yaw + quarter * math.pi / 2
            q, reachable = ik(self._grasp_tcp(pos, jaw), jaw)
            twist = (q[:, 4] - roll_now).abs()
            take = self._in_limits(q, reachable) & (twist < cost)
            q_best = q if q_best is None else torch.where(take[:, None], q, q_best)
            jaw_best = torch.where(take, jaw, jaw_best)
            cost = torch.where(take, twist, cost)
        return q_best, jaw_best, cost.isfinite()

    def _plan_place(self, pos, yaw, offset, grasp_jaw):
        """Keyposes that seat a cube held at ``offset`` from the TCP on top of ``pos``.

        Both cubes are 4-fold symmetric, so any quarter turn seats flush; take the one
        nearest ``grasp_jaw``, or the wrist spins most of a turn mid-carry and slings the
        cube out of the jaws.
        """
        turn = fold_yaw(yaw - grasp_jaw)
        jaw = grasp_jaw + turn
        cos, sin = torch.cos(turn), torch.sin(turn)
        turned = torch.stack([cos * offset[:, 0] - sin * offset[:, 1],
                              sin * offset[:, 0] + cos * offset[:, 1], offset[:, 2]], -1)
        tcp = pos + self._up(CUBE_SIDE + REST_GAP) - turned
        q_rest, reachable = ik(tcp, jaw)
        q_above, above_ok = ik_clearance(tcp + self._up(HOVER_DZ), jaw, self.limits)
        q_clear, _ = ik_straight_up(tcp, jaw, self.limits, RETRACT_DZ)
        return q_rest, q_above, q_clear, self._in_limits(q_rest, reachable) & above_ok

    def feasible(self, held, target):
        """Whether the arm can reach the grasp, carry the cube clear, and seat the stack."""
        held_pos, held_yaw = self._cube(held)
        target_pos, target_yaw = self._cube(target)
        _, jaw, grasp_ok = self._plan_grasp(held_pos, held_yaw)
        _, lift_ok = ik_clearance(self._grasp_tcp(held_pos, jaw) + self._up(HOVER_DZ),
                                  jaw, self.limits)
        *_, place_ok = self._plan_place(target_pos, target_yaw,
                                        torch.zeros_like(target_pos), jaw)
        return grasp_ok & lift_ok & place_ok

    # ---- motion -----------------------------------------------------------
    def _act(self, command):
        self.command = command
        self.env.step(command.to(torch.float32))
        if self.on_step is not None:
            self.on_step()

    def _hold(self, steps):
        for _ in range(steps):
            self._act(self.command)

    def _path(self, keyposes, gripper, at):
        """The commanded path from the last command through ``keyposes``, sampled at ``at``.

        The path starts from the previous *command*, which keeps the command stream
        continuous across moves. ``at`` runs over 0..1 of the move and is eased on its way
        into the spline, zeroing the acceleration at the two ends as well as the velocity
        the spline already pins there.
        """
        gripper = torch.as_tensor(gripper, dtype=torch.float64, device=self.device)
        # The jaws take their width at the first keypose and hold it, so they are open
        # before the descent onto a cube begins.
        grip = gripper.expand(self.n, len(keyposes) + 1).clone()
        grip[:, 0] = self.command[:, 5]
        knots = torch.cat([torch.stack([self.command[:, :5], *keyposes], 1),
                           grip[..., None]], -1)

        # Give each segment a share of the move in proportion to how far it travels, so
        # every segment is crossed at much the same speed.
        span = (knots[:, 1:, :5] - knots[:, :-1, :5]).abs().amax((0, 2)).clamp(min=1e-6)
        u = torch.cat([torch.zeros(1, dtype=span.dtype, device=span.device),
                       span.cumsum(0)])
        return _spline(u / u[-1], knots, _smoothstep(at))

    def _move(self, *keyposes, gripper):
        """Flow from the last command through ``keyposes`` as a single continuous move.

        The spline's peak speed depends on how the keyposes fall, so sample the path
        densely and take enough steps to keep every joint under ``JOINT_VEL_MAX``.
        """
        probe = torch.linspace(0, 1, PROBE_SAMPLES, dtype=torch.float64, device=self.device)
        path = self._path(keyposes, gripper, probe)
        peak = float((path[:, 1:, :5] - path[:, :-1, :5]).abs().max()) * (PROBE_SAMPLES - 1)
        steps = max(2, math.ceil(peak / (JOINT_VEL_MAX * self.env.control_timestep)))

        at = torch.linspace(0, 1, steps + 1, dtype=torch.float64, device=self.device)
        for command in self._path(keyposes, gripper, at[1:]).unbind(1):
            self._act(command)

    def _grip(self, width, steps, settle=0):
        """Ease the jaws to ``width`` with the arm held where it is, then wait ``settle``.

        The jaws work against a still arm, so this is the one place the arm stops.
        """
        start = self.command
        goal = start.clone()
        goal[:, 5] = width
        for i in range(1, steps + 1):
            self._act(start + _smoothstep(i / steps) * (goal - start))
        self._hold(settle)

    # ---- one cycle --------------------------------------------------------
    def run(self, held, target, on_step=None):
        """Pick the ``held`` cube and stack it on the ``target`` cube, in every env."""
        self.on_step = on_step
        held_pos, held_yaw = self._cube(held)

        q_grasp, jaw, _ = self._plan_grasp(held_pos, held_yaw)
        q_clear, _ = ik_clearance(self._grasp_tcp(held_pos, jaw) + self._up(HOVER_DZ),
                                  jaw, self.limits)

        # Three moves, parted only where the jaws have to work against a stationary arm.
        self._move(q_clear, q_grasp, gripper=GRIP_OPEN)
        self._grip(GRIP_CLOSED, CLOSE_STEPS, GRIP_STEPS)

        # The cube is held rigidly, so aim its centre rather than the TCP. Read the
        # offset here, still fingers-down: the place is fingers-down too, so carrying it
        # round is a turn of the jaw and nothing else.
        offset = self._cube(held)[0] - self.env.agent.tcp_pose.p.to(torch.float64)

        target_pos, target_yaw = self._cube(target)
        q_rest, q_above, q_retract, _ = self._plan_place(target_pos, target_yaw, offset, jaw)
        self._move(q_clear, q_above, q_rest, gripper=GRIP_CLOSED)
        self._grip(GRIP_OPEN, RELEASE_STEPS)
        # Leave straight up: the jaws still straddle the cube, so the retract trades
        # reach for keeping the tool vertical.
        self._move(q_retract, gripper=GRIP_OPEN)
        self.on_step = None
