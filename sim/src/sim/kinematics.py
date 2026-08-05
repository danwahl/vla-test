"""Closed-form inverse kinematics for the SO-101, batched over environments.

shoulder_pan is vertical, shoulder_lift / elbow_flex / wrist_flex are parallel and
horizontal, and wrist_roll is the tool approach axis. So the arm is a pan joint plus a
planar 3R plus a roll: fixing the tool tilt leaves a pan angle and a two-link cosine law,
and the roll is free to set the jaw yaw.
"""

from __future__ import annotations

import math

import torch

from .agent import BASE_POSE

BASE_P = torch.tensor(BASE_POSE.p, dtype=torch.float64)

# Link geometry in the pan frame (URDF joint origins).
PAN_ORIGIN = torch.tensor([0.045200, 0.000100, 0.069600], dtype=torch.float64)
LIFT_X, LIFT_Z = 0.030400, 0.054200
L1, A1 = 0.116030, math.atan2(-0.112600, 0.028000)   # shoulder_lift -> elbow_flex
L2, A2 = 0.135000, math.atan2(-0.005200, 0.134900)   # elbow_flex -> wrist_flex
L3 = 0.159258                                        # wrist_flex -> TCP along the roll axis

# The TCP sits 7.9 mm off the roll axis (the fixed jaw), at a fixed phase from the jaw
# axis, and the roll joint's zero is a quarter turn from it. Measured from FK; the
# round-trip test in tests/test_kinematics.py pins them.
JAW_OFFSET = 0.007903
JAW_PHASE = 0.027605
ROLL_ZERO = -1.522120
AXIS_Y = -0.000177          # the roll axis' lateral offset from the pan axis

TILT_DOWN = math.pi / 2     # tool straight down


def ik(tcp, jaw_yaw, tilt=TILT_DOWN):
    """Arm joint angles placing the TCP at world ``tcp`` with the tool at ``tilt``.

    ``tilt`` is the flex sum: ``TILT_DOWN`` points the tool straight down, less than that
    tilts it in from above, and ``jaw_yaw`` orients the jaw opening axis. The TCP's 7.9 mm
    offset from the roll axis is taken out, so ``tcp`` is where the tool actually lands;
    the back-out treats that offset as horizontal, which is exact pointing straight down
    and a fraction of a millimetre out at the tilts a clearance waypoint needs.

    Returns ``(q, reachable)`` with ``q`` shaped ``(..., 5)``. Angles where ``reachable``
    is false are meaningless (the two-link law has no solution there).
    """
    tcp = torch.as_tensor(tcp, dtype=torch.float64)
    kw = {"dtype": torch.float64, "device": tcp.device}
    jaw_yaw = torch.as_tensor(jaw_yaw, **kw).expand(tcp.shape[:-1])
    tilt = torch.as_tensor(tilt, **kw).expand(tcp.shape[:-1])

    ang = jaw_yaw + JAW_PHASE
    off = JAW_OFFSET * torch.stack([torch.cos(ang), torch.sin(ang),
                                    torch.zeros_like(ang)], dim=-1)
    v = tcp - BASE_P.to(tcp.device) - off - PAN_ORIGIN.to(tcp.device)

    # Pan places the roll axis, which sits a fraction of a millimetre off that axis.
    rho = torch.hypot(v[..., 0], v[..., 1])
    radial = torch.sqrt(torch.clamp(rho * rho - AXIS_Y * AXIS_Y, min=0.0))
    pan = (torch.atan2(torch.full_like(radial, AXIS_Y), radial)
           - torch.atan2(v[..., 1], v[..., 0]))

    # Drop the wrist, whose direction the tilt already fixes, and solve the two links.
    u = radial - L3 * torch.cos(tilt) - LIFT_X
    w = v[..., 2] + L3 * torch.sin(tilt) - LIFT_Z
    cos_elbow = (u * u + w * w - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    reachable = (cos_elbow.abs() <= 1.0) & (rho > abs(AXIS_Y))
    elbow = torch.acos(torch.clamp(cos_elbow, -1.0, 1.0))

    lift = (torch.atan2(-w, u)
            - torch.atan2(L2 * torch.sin(elbow), L1 + L2 * torch.cos(elbow)) - A1)
    flex = tilt - lift - elbow + A2 - A1
    roll = torch.remainder(jaw_yaw + pan + ROLL_ZERO + math.pi, 2 * math.pi) - math.pi
    return torch.stack([pan, lift, elbow - A2 + A1, flex, roll], dim=-1), reachable


def ik_clearance(tcp, jaw_yaw, limits):
    """As ``ik``, but for an approach or retract waypoint above the reach ceiling.

    A top-down hover more than ~70 mm over the table is out of reach, so walk the tilt
    back from straight down and keep the flattest one every joint can hold.
    """
    tcp = torch.as_tensor(tcp, dtype=torch.float64)
    tilts = torch.linspace(TILT_DOWN, 0.5, 24, dtype=torch.float64, device=tcp.device)
    lo, hi = limits[:5, 0].to(torch.float64), limits[:5, 1].to(torch.float64)

    best = None
    found = torch.zeros(tcp.shape[:-1], dtype=torch.bool, device=tcp.device)
    for tilt in tilts:
        q, ok = ik(tcp, jaw_yaw, tilt)
        ok = ok & (q >= lo).all(-1) & (q <= hi).all(-1)
        best = q if best is None else torch.where((ok & ~found)[..., None], q, best)
        found = found | ok
    return best, found


def ik_straight_up(tcp, jaw_yaw, limits, height, steps=12):
    """Back off vertically from ``tcp``, as far as the arm can while staying fingers-down.

    Withdrawing from a seated block keeps the tool vertical, since the jaws still bracket
    it, so this takes the highest reachable rise at that fixed orientation.
    """
    tcp = torch.as_tensor(tcp, dtype=torch.float64)
    lo, hi = limits[:5, 0].to(torch.float64), limits[:5, 1].to(torch.float64)
    rises = torch.linspace(height, 0.0, steps, dtype=torch.float64, device=tcp.device)

    best = None
    achieved = torch.zeros(tcp.shape[:-1], dtype=torch.float64, device=tcp.device)
    found = torch.zeros(tcp.shape[:-1], dtype=torch.bool, device=tcp.device)
    for rise in rises:
        raised = tcp.clone()
        raised[..., 2] = raised[..., 2] + rise
        q, ok = ik(raised, jaw_yaw)
        ok = ok & (q >= lo).all(-1) & (q <= hi).all(-1)
        take = ok & ~found
        best = q if best is None else torch.where(take[..., None], q, best)
        achieved = torch.where(take, rise, achieved)
        found = found | ok
    return best, achieved
