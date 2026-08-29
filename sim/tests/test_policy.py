"""The chunk smoother holds the motion a chunk carries and drops the wobble on top of it."""

import numpy as np
import pytest
import torch

from sim.policy import CONTROL_POINTS, smooth


def wobbly(seed=0, envs=4, steps=50, joints=6, wobble=0.01):
    """A smooth path with per-step noise added, as a chunk arrives.

    Half a period over the chunk, so the motion's own curvature sits well under the wobble
    on top of it, which is the ratio a rollout shows.
    """
    generator = torch.Generator().manual_seed(seed)
    at = torch.linspace(0, 1, steps)[None, :, None]
    phase = torch.rand(envs, 1, joints, generator=generator) * 2 * np.pi
    path = torch.sin(np.pi * at + phase)
    return path + wobble * torch.randn(envs, steps, joints, generator=generator), path


def test_keeps_the_control_points():
    chunk, _ = wobbly()
    at = torch.linspace(0, chunk.shape[1] - 1, CONTROL_POINTS).round().long().unique()
    assert torch.allclose(smooth(chunk)[:, at], chunk[:, at], atol=1e-6)


def test_drops_the_wobble():
    chunk, path = wobbly()
    def jerk(x):
        return x.diff(n=2, dim=1).abs().mean()
    assert jerk(smooth(chunk)) < jerk(chunk) / 4
    # Closer to the path the wobble was added to than the chunk it arrived as.
    assert (smooth(chunk) - path).abs().mean() < (chunk - path).abs().mean()


def test_never_leaves_the_span_of_its_control_points():
    """Fritsch-Carlson slopes, so no piece swings past the points it joins."""
    chunk, _ = wobbly(seed=1, wobble=0.05)
    at = torch.linspace(0, chunk.shape[1] - 1, CONTROL_POINTS).round().long().unique()
    knots = chunk[:, at]
    smoothed = smooth(chunk)
    for piece, (lo, hi) in enumerate(zip(at[:-1].tolist(), at[1:].tolist(), strict=True)):
        span = knots[:, piece:piece + 2]
        inside = smoothed[:, lo:hi + 1]
        assert (inside >= span.amin(1, keepdim=True) - 1e-6).all()
        assert (inside <= span.amax(1, keepdim=True) + 1e-6).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_runs_where_the_chunk_is():
    """A chunk arrives on whatever device the policy ran on, and the curve is built there."""
    chunk, _ = wobbly()
    on_gpu = smooth(chunk.cuda()).cpu()
    assert torch.allclose(on_gpu, smooth(chunk), atol=1e-6)


def test_passes_a_straight_line_through():
    """A chunk with no wobble in it is already what the curve would draw."""
    ramp = torch.linspace(0, 1, 50)[None, :, None] * torch.arange(1, 7)
    assert torch.allclose(smooth(ramp), ramp, atol=1e-5)
