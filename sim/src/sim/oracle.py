"""Scripted pick-and-stack expert, planned in closed form and run across all envs at once.

A cycle is a handful of keyposes solved by :mod:`kinematics` and joined in joint space by
splines that cross each one without stopping on it, so the arm comes to rest only at the
stand-off it looks from and where the jaws work against a still arm. Planning in joint
space fixes the solution branch for a whole move. Every env runs the same phases with its
own joint targets, which keeps the batch in lockstep for rendering.

A cycle starts wherever the arm is, which `retract` is there to vary.
"""

from __future__ import annotations

import math

import torch

from .agent import GRIPPER_CLOSED, GRIPPER_OPEN, HOME_QPOS
from .env import BLOCK_REST_Z, BLOCK_SIDE, SPAWN_X, SPAWN_Y
from .kinematics import ik, ik_clearance, ik_straight_up

GRASP_DZ = -0.003        # clamp just below the block centre so the jaw tips clear the table
GRASP_LATERAL = 0.020    # along the jaw axis, seating the block against the fixed jaw
HOVER_DZ = 0.06          # clearance above the grasp and above the target stack
LOOK_DZ = 0.12           # stand off this far over the block before descending on it
RETRACT_DZ = 0.05        # leave the seated block by this much, before the tool has to tilt
REST_GAP = 0.001         # release this close to a seated stack, so it settles rather than drops

# Opening takes longer than closing: it lets the placed block settle while the jaw is moving.
CLOSE_STEPS = 6
RELEASE_STEPS = 8

JOINT_VEL_MAX = 0.5      # rad/s commanded; well under what the arm holds, to keep it smooth
RAMP_STEPS = 4           # steps spent reaching that speed at each end of a move
PROBE_SAMPLES = 256      # resolution at which a move is measured to pace its steps


def fold_yaw(yaw):
    """A block is 4-fold symmetric, so fold its yaw into the nearest quarter turn."""
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
    # flat slope.
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

    def _block(self, idx):
        """(position, folded yaw) of the per-env block named by index ``idx``."""
        poses = torch.stack([c.pose.raw_pose for c in self.env.blocks.values()], dim=1)
        pose = poses[torch.arange(self.n, device=self.device), idx].to(torch.float64)
        return pose[:, :3], fold_yaw(_yaw_of(pose[:, 3:]))

    def _in_limits(self, q, reachable):
        return reachable & (q >= self.limits[:5, 0]).all(-1) & (q <= self.limits[:5, 1]).all(-1)

    def _grasp_tcp(self, pos, jaw):
        jaw_axis = torch.stack([torch.cos(jaw), torch.sin(jaw), torch.zeros_like(jaw)], -1)
        return pos + GRASP_LATERAL * jaw_axis + self._up(GRASP_DZ)

    def _plan_grasp(self, pos, yaw):
        """Solve the grasp on the quarter turn of the jaw that leaves the wrist near home.

        All four straddle the block identically, so among the ones the arm can hold, take
        the one whose roll is nearest the rest pose's.
        """
        q_best, jaw_best = None, yaw.clone()
        cost = torch.full((self.n,), math.inf, device=self.device, dtype=torch.float64)
        for quarter in range(4):
            jaw = yaw + quarter * math.pi / 2
            q, reachable = ik(self._grasp_tcp(pos, jaw), jaw)
            twist = (q[:, 4] - HOME_QPOS[4]).abs()
            take = self._in_limits(q, reachable) & (twist < cost)
            q_best = q if q_best is None else torch.where(take[:, None], q, q_best)
            jaw_best = torch.where(take, jaw, jaw_best)
            cost = torch.where(take, twist, cost)
        return q_best, jaw_best

    def _place_tcp(self, pos, offset, turn):
        """Where the TCP goes to rest a block held at ``offset`` on top of ``pos``.

        Turning the wrist by ``turn`` swings the held block around the tool axis, so the
        offset to its centre turns with it.
        """
        cos, sin = torch.cos(turn), torch.sin(turn)
        turned = torch.stack([cos * offset[:, 0] - sin * offset[:, 1],
                              sin * offset[:, 0] + cos * offset[:, 1], offset[:, 2]], -1)
        return pos + self._up(BLOCK_SIDE + REST_GAP) - turned

    def _plan_place(self, pos, yaw, offset, grasp_jaw, roll_now):
        """Keyposes that seat a block held at ``offset`` from the TCP on top of ``pos``.

        Both blocks are 4-fold symmetric, so all four quarter turns seat flush. The roll
        joint spans less than a full turn, and the pan differs between the two blocks, so
        the quarter that turns the jaw least is not the one that turns the wrist least.
        Take the one that leaves the wrist nearest ``roll_now``, or it unwinds most of a
        turn mid-carry and slings the block out of the jaws.
        """
        square = grasp_jaw + fold_yaw(yaw - grasp_jaw)
        jaw_best = square.clone()
        cost = torch.full((self.n,), math.inf, device=self.device, dtype=torch.float64)
        for quarter in range(4):
            jaw = square + quarter * math.pi / 2
            q, reachable = ik(self._place_tcp(pos, offset, jaw - grasp_jaw), jaw)
            twist = (q[:, 4] - roll_now).abs()
            take = self._in_limits(q, reachable) & (twist < cost)
            jaw_best = torch.where(take, jaw, jaw_best)
            cost = torch.where(take, twist, cost)

        tcp = self._place_tcp(pos, offset, jaw_best - grasp_jaw)
        q_rest, _ = ik(tcp, jaw_best)
        q_above, _ = ik_clearance(tcp + self._up(HOVER_DZ), jaw_best, self.limits)
        q_clear, _ = ik_straight_up(tcp, jaw_best, self.limits, RETRACT_DZ)
        return q_rest, q_above, q_clear

    # ---- motion -----------------------------------------------------------
    def _act(self, command):
        self.command = command
        obs, *_ = self.env.step(command.to(torch.float32))
        if self.on_step is not None:
            self.on_step(obs)

    def _path(self, keyposes, gripper, at):
        """The commanded path from the last command through ``keyposes``, sampled at ``at``.

        The path starts from the previous *command*, which keeps the command stream
        continuous across moves. ``at`` runs over 0..1 of the spline.
        """
        gripper = torch.as_tensor(gripper, dtype=torch.float64, device=self.device)
        # The jaws take their width at the first keypose and hold it. Every caller sets off
        # at the width it asks for, so the channel is flat and the arm alone is what moves.
        grip = gripper.expand(self.n, len(keyposes) + 1).clone()
        grip[:, 0] = self.command[:, 5]
        knots = torch.cat([torch.stack([self.command[:, :5], *keyposes], 1),
                           grip[..., None]], -1)

        # Give each segment a share of the move in proportion to how far it travels, so
        # every segment is crossed at much the same speed.
        span = (knots[:, 1:, :5] - knots[:, :-1, :5]).abs().amax((0, 2)).clamp(min=1e-6)
        u = torch.cat([torch.zeros(1, dtype=span.dtype, device=span.device),
                       span.cumsum(0)])
        return _spline(u / u[-1], knots, at)

    def _move(self, *keyposes, gripper):
        """Flow from the last command through ``keyposes`` as a single continuous move.

        Steps are placed by distance travelled, so the move holds ``JOINT_VEL_MAX``
        throughout, ramped over ``RAMP_STEPS`` at each end.
        """
        probe = torch.linspace(0, 1, PROBE_SAMPLES, dtype=torch.float64, device=self.device)
        path = self._path(keyposes, gripper, probe)
        # How far the joint with furthest to go gets, in the env with furthest to go.
        travel = (path[:, 1:, :5] - path[:, :-1, :5]).abs().amax((0, 2))
        covered = torch.cat([torch.zeros_like(travel[:1]), travel.cumsum(0)])

        stride = JOINT_VEL_MAX * self.env.control_timestep
        steps = max(2, math.ceil(float(covered[-1]) / stride) + RAMP_STEPS)
        i = torch.arange(1, steps + 1, dtype=torch.float64, device=self.device)
        speed = torch.minimum(i, steps + 1 - i).clamp(max=RAMP_STEPS + 1)
        reach = covered[-1] * speed.cumsum(0) / speed.sum()

        # Read back the spline parameter that has covered each of those distances.
        j = torch.searchsorted(covered, reach).clamp(1, PROBE_SAMPLES - 1)
        span = (covered[j] - covered[j - 1]).clamp(min=1e-12)
        at = (j - 1 + (reach - covered[j - 1]) / span) / (PROBE_SAMPLES - 1)
        for command in self._path(keyposes, gripper, at).unbind(1):
            self._act(command)

    def _grip(self, width, steps):
        """Ease the jaws to ``width`` with the arm held where it is.

        The jaws work against a still arm, so this is the one place the arm stops.
        """
        start = self.command
        goal = start.clone()
        goal[:, 5] = width
        for i in range(1, steps + 1):
            self._act(start + _smoothstep(i / steps) * (goal - start))

    # ---- where a cycle leaves the arm -------------------------------------
    def retract(self):
        """Stand the arm where a cycle would have left it, jaws open or shut.

        A cycle ends withdrawn over the table, and a cycle whose grasp closed on nothing
        ends there holding nothing. Starting an episode in that pose is what puts the
        state after a missed pick in front of a policy while what to do next is still
        being demonstrated.

        The stack is imagined anywhere a block spawns, and the jaws take a quarter turn,
        which spans every orientation a square block leaves them in. Where the arm cannot
        hold the pose it starts folded up at the rest pose instead.
        """
        span = torch.tensor([[SPAWN_X[1] - SPAWN_X[0], SPAWN_Y[1] - SPAWN_Y[0], 0.0]],
                            dtype=torch.float64, device=self.device)
        low = torch.tensor([[SPAWN_X[0], SPAWN_Y[0], BLOCK_REST_Z + BLOCK_SIDE + REST_GAP]],
                           dtype=torch.float64, device=self.device)
        tcp = low + span * torch.rand(self.n, 3, dtype=torch.float64, device=self.device)
        jaw = math.pi / 2 * torch.rand(self.n, dtype=torch.float64, device=self.device)

        # Back off the way `run` leaves a block it has seated, anywhere from touching the
        # stack to as far as the arm can hold, so the pose varies in height as well as
        # over the table. Re-solving there reads whether that pose holds.
        _, rise = ik_straight_up(tcp, jaw, self.limits, RETRACT_DZ)
        tcp[:, 2] += rise * torch.rand(self.n, dtype=torch.float64, device=self.device)
        q, reachable = ik(tcp, jaw)
        home = torch.as_tensor(HOME_QPOS, dtype=torch.float64, device=self.device)
        q = torch.where(self._in_limits(q, reachable)[:, None], q, home[:5])

        shut = torch.rand(self.n, device=self.device) < 0.5
        width = torch.where(shut, GRIPPER_CLOSED, GRIPPER_OPEN)
        self.command = torch.cat([q, width[:, None].to(torch.float64)], -1)
        self.env.agent.robot.set_qpos(self.command.to(torch.float32))
        # On the GPU backend a write outside of a reset reaches the solver a step later and
        # the render state not at all, so the cameras would go on showing the rest pose.
        # This is the flush `reset` itself ends with.
        if self.env.gpu_sim_enabled:
            self.env.scene._gpu_apply_all()
            self.env.scene.px.gpu_update_articulation_kinematics()
            self.env.scene._gpu_fetch_all()

    # ---- one cycle --------------------------------------------------------
    def run(self, held, target, on_step=None):
        """Pick the ``held`` block and stack it on the ``target`` block, in every env.

        The cycle runs from wherever the arm was left, so the same commands recover from a
        missed pick as begin a fresh episode.

        ``on_step`` is handed each step's observation, alongside the ``self.command`` that
        produced it, so a demonstration can be recorded without rendering the scene twice.
        """
        self.on_step = on_step
        held_pos, held_yaw = self._block(held)

        q_grasp, jaw = self._plan_grasp(held_pos, held_yaw)
        tcp = self._grasp_tcp(held_pos, jaw)
        q_look, _ = ik_clearance(tcp + self._up(LOOK_DZ), jaw, self.limits)
        q_clear, _ = ik_clearance(tcp + self._up(HOVER_DZ), jaw, self.limits)

        # Whatever the jaws were left holding, they are holding nothing now, so they open
        # against a still arm the way every other width change is made. The arm then sets
        # off already able to take the block.
        self._grip(GRIPPER_OPEN, RELEASE_STEPS)
        # Come to the block from above and from a stand-off, whatever the arm was doing
        # before, so the wrist camera arrives looking down at what the prompt names.
        self._move(q_look, gripper=GRIPPER_OPEN)
        self._move(q_clear, q_grasp, gripper=GRIPPER_OPEN)
        self._grip(GRIPPER_CLOSED, CLOSE_STEPS)

        # The block is held rigidly, so aim its centre rather than the TCP. Read the
        # offset here, still fingers-down: the place is fingers-down too, so carrying it
        # round is a turn of the jaw and nothing else.
        offset = self._block(held)[0] - self.env.agent.tcp_pose.p.to(torch.float64)

        target_pos, target_yaw = self._block(target)
        q_rest, q_above, q_retract = self._plan_place(target_pos, target_yaw, offset, jaw,
                                                      q_grasp[:, 4])
        self._move(q_clear, q_above, q_rest, gripper=GRIPPER_CLOSED)
        self._grip(GRIPPER_OPEN, RELEASE_STEPS)
        # Leave straight up: the jaws still straddle the block, so the retract trades
        # reach for keeping the tool vertical.
        self._move(q_retract, gripper=GRIPPER_OPEN)
        self.on_step = None
