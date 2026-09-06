"""A trained checkpoint, loaded the way both the sim eval and the arm need it."""

from __future__ import annotations

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.rtc import RTCConfig
from lerobot.processor import RenameObservationsProcessorStep
from torchao.quantization import Int8WeightOnlyConfig, quantize_

# Both run once per chunk and hand the action expert a cache, so they sit the far side of
# the model from the commands. The weights are unpacked for each matmul.
OBSERVATION_MODULES = ("paligemma.model.language_model", "paligemma.model.vision_tower")

# How many of a chunk's steps `smooth` keeps.
CONTROL_POINTS = 13


def _end_slope(near, beyond, h_near, h_beyond):
    """The slope at an end knot, from the two secants next to it.

    A one-sided quadratic through the first three knots, held to the direction it starts
    out in: a slope disagreeing in sign with the secant beside it flattens, and one running
    away from a turning point is held to three times that secant. Both hold the end piece
    inside the interval it spans.
    """
    slope = ((2 * h_near + h_beyond) * near - h_near * beyond) / (h_near + h_beyond)
    slope = torch.where(torch.sign(slope) != torch.sign(near),
                        torch.zeros_like(slope), slope)
    runaway = (torch.sign(near) != torch.sign(beyond)) & (slope.abs() > 3 * near.abs())
    return torch.where(runaway, 3 * near, slope)


def smooth(chunk):
    """``chunk`` (envs, steps, joints) read back off a cubic through some of its steps.

    A chunk carries the motion the policy intends and a step-to-step wobble on top of it.
    Keeping `CONTROL_POINTS` of the steps and curving between them holds the motion and
    drops the wobble.

    The slopes are Fritsch-Carlson, the same shape-preserving rule the oracle's keyposes
    use, so no piece leaves the interval its two control points span. A chunk is smoothed
    whole and executed in part, so the steps that reach the arm sit inside the curve.
    """
    steps = chunk.shape[1]
    at = torch.linspace(0, steps - 1, CONTROL_POINTS,
                        device=chunk.device).round().long().unique()
    knots = chunk[:, at]
    widths = (at[1:] - at[:-1]).to(chunk.dtype)[None, :, None]
    secants = (knots[:, 1:] - knots[:, :-1]) / widths

    before, after = secants[:, :-1], secants[:, 1:]
    # A knot the path turns at gets a flat slope, which is what stops the curve rounding a
    # corner into an overshoot.
    onward = before * after > 0
    w1, w2 = 2 * widths[:, 1:] + widths[:, :-1], widths[:, 1:] + 2 * widths[:, :-1]
    ones = torch.ones_like(before)
    harmonic = (w1 + w2) / (w1 / torch.where(onward, before, ones)
                            + w2 / torch.where(onward, after, ones))
    slopes = torch.zeros_like(knots)
    slopes[:, 1:-1] = torch.where(onward, harmonic, torch.zeros_like(harmonic))
    slopes[:, 0] = _end_slope(secants[:, 0], secants[:, 1], widths[:, 0], widths[:, 1])
    slopes[:, -1] = _end_slope(secants[:, -1], secants[:, -2], widths[:, -1], widths[:, -2])

    grid = torch.arange(steps, device=chunk.device, dtype=chunk.dtype)
    piece = (torch.searchsorted(at.to(grid.dtype).contiguous(), grid.contiguous(),
                                right=True) - 1).clamp(0, len(at) - 2)
    width = widths[:, piece]
    s = ((grid - at[piece].to(grid.dtype)) / width[0, :, 0])[None, :, None]
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1) * knots[:, piece]
            + (s3 - 2 * s2 + s) * width * slopes[:, piece]
            + (-2 * s3 + 3 * s2) * knots[:, piece + 1]
            + (s3 - s2) * width * slopes[:, piece + 1])


def load_policy(checkpoint, metadata, horizon, device, rtc=True, int8=False):
    """The checkpoint's policy and the processors saved beside it, which carry the
    camera renaming and the normalization stats from training."""
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = checkpoint
    # Built in host memory and moved once it is the size it will run at.
    config.device = "cpu"
    config.rtc_config = RTCConfig(execution_horizon=horizon, enabled=rtc)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    # The cameras reach the policy under the DROID slot names it was trained on. Handing
    # make_policy the same map is what tells it the two sets are meant to differ.
    renaming = next(step for step in preprocessor.steps
                    if isinstance(step, RenameObservationsProcessorStep))
    policy = make_policy(cfg=config, ds_meta=metadata, rename_map=renaming.rename_map)
    # A checkpoint saved with its adapter comes back wrapped, which puts the module holding
    # the backbone a level deeper than one saved without.
    inner = policy.model
    model = inner if hasattr(inner, "paligemma_with_expert") else inner.model
    # Actions leave through `action_out_proj`, so neither vocabulary head is on the path a
    # chunk takes.
    model.paligemma_with_expert.paligemma.lm_head = None
    model.paligemma_with_expert.gemma_expert.lm_head = None
    if int8:
        # `version=2` scales each output channel on its own, so a channel with a narrow
        # range keeps its resolution. Setting inductor's config is left alone because it
        # turns on TF32 for every float32 matmul in the process, the vision tower included.
        quantize_(model.paligemma_with_expert,
                  Int8WeightOnlyConfig(version=2, set_inductor_config=False),
                  filter_fn=lambda module, name: (isinstance(module, torch.nn.Linear)
                                                  and name.startswith(OBSERVATION_MODULES)))
    policy.to(device)
    policy.eval()
    return policy, preprocessor, postprocessor
