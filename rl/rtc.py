"""Real-Time Chunking guidance, in the parametrization RLinf's openpi sampler uses.

Each chunk is denoised toward the unexecuted tail of the chunk it replaces, so the arm does
not jump where one chunk hands over to the next. ``rl/patch.py`` copies this next to the
sampler and calls it from ``sample_mean_var_val``, which the rollout and the update both go
through, so the executed chunk and the log-density of that chunk stay the same function.

LeRobot corrects the velocity, ``v_t - gw * weights * (prev - x0_pred)``. This sampler works
in an x0/x1 interpolant whose mean is ``x_t - delta * v_t`` under the ``flow_ode`` and
``flow_noise`` weightings, so the same correction is a shift of the mean by
``gw * delta * weights * (prev - x0_pred)``. ``rl/tests/test_rtc.py`` checks the two against
each other.
"""

from __future__ import annotations

import torch

MAX_GUIDANCE_WEIGHT = 10.0


def prefix_weights(execution_horizon, horizon, device, dtype):
    """How hard each step of the new chunk is held to the old one, at inference delay zero.

    A ramp down across the window that will execute before the next replan, and zero past it,
    where the arm is free. The ramp is the interior of a linspace, so it starts just under 1
    and ends just over 0: neither end of the window is fully pinned or fully free.
    """
    steps = max(min(int(execution_horizon), int(horizon)), 0)
    weights = torch.zeros(horizon, device=device, dtype=dtype)
    if steps:
        weights[:steps] = torch.linspace(1, 0, steps + 2, device=device, dtype=dtype)[1:-1]
    return weights[None, :, None]


def guidance_weight(time, max_weight=MAX_GUIDANCE_WEIGHT):
    """The weight the paper puts on the correction, at flow time ``time``.

    Time runs from 1 at the first denoising step toward 0. The weight diverges as 1 is
    approached, which is what ``max_weight`` caps.
    """
    remaining = 1 - time
    c = torch.nan_to_num(time / remaining, posinf=max_weight)
    inv_r2 = torch.nan_to_num((time**2 + remaining**2) / time**2, posinf=max_weight)
    return torch.clamp(c * inv_r2, max=max_weight)


def guide(x_t_mean, x0_pred, prev_remaining, time, delta, execution_horizon,
          valid=None, max_weight=MAX_GUIDANCE_WEIGHT):
    """``x_t_mean`` shifted toward ``prev_remaining``.

    ``prev_remaining`` is the previous chunk with its executed steps dropped, so its step 0
    lines up with step 0 of the chunk being denoised. It is a constant with respect to the
    network parameters, which is what keeps the guided sampler an ordinary policy whose
    log-density is exact.

    ``valid`` marks which rows of the batch have a previous chunk at all: the first chunk of
    an episode has none, and neither does an env that reset part way through a rollout. Those
    rows denoise unguided, since a tail of zeros reads as a command to travel to the origin.
    """
    horizon = x_t_mean.shape[1]
    overlap = min(prev_remaining.shape[1], horizon)
    width = min(prev_remaining.shape[2], x_t_mean.shape[2])

    target = torch.zeros_like(x_t_mean)
    target[:, :overlap, :width] = prev_remaining[:, :overlap, :width]

    # The window shortens to the tail, as LeRobot does before building the same ramp, so the
    # ramp spans only steps the tail covers.
    window = min(int(execution_horizon), overlap)
    weights = prefix_weights(window, horizon, x_t_mean.device, x_t_mean.dtype)
    if valid is not None:
        weights = weights * valid.reshape(-1, 1, 1).to(weights.dtype)

    return x_t_mean + guidance_weight(time, max_weight) * delta * weights * (target - x0_pred)


class Tail:
    """The unexecuted end of the chunk each env was last given, held across a rollout.

    ``inputs`` hands the sampler something to denoise toward and a mask of which rows have
    one; ``advance`` records the chunk just handed out. Both return real tensors from the
    first chunk onward, including the one where no row has a previous chunk: the buffer
    stacks its keys off the first step, so a key missing there is missing from the whole
    trajectory.

    An env whose episode ended during the chunk just executed starts the next one unguided.
    Its tail describes a layout that is gone.
    """

    def __init__(self, horizon, executed, dim):
        self.executed = int(executed)
        self.leftover = int(horizon) - int(executed)
        self.dim = int(dim)
        self.dtype = None
        self.prev = None
        self.valid = None

    def reset(self):
        self.prev = None
        self.valid = None

    def inputs(self, batch_size, device, dtype, dones=None):
        # The buffer stacks a key across chunk steps, so every step's pair matches the one
        # the first step set.
        self.dtype = dtype
        if self.prev is None or self.prev.shape[0] != batch_size:
            self.prev = torch.zeros(
                (batch_size, self.leftover, self.dim), device=device, dtype=dtype)
            self.valid = torch.zeros(batch_size, device=device, dtype=dtype)
        if dones is not None:
            ended = dones.reshape(dones.shape[0], -1).any(dim=-1)
            self.valid = self.valid * (~ended.to(device)).to(self.valid.dtype)
        return {"rtc_prev": self.prev, "rtc_valid": self.valid}

    def advance(self, model_actions):
        # A copy, since the buffer holds this until the update reads it a rollout later and
        # the sampler's caller owns the chunk it is a slice of.
        chunk = model_actions.detach().to(self.dtype)
        self.prev = chunk[:, self.executed:].clone()
        self.valid = torch.ones(
            chunk.shape[0], device=chunk.device, dtype=chunk.dtype)
