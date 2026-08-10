"""The guidance in ``rl/rtc.py`` against the LeRobot implementation the eval scores with.

LeRobot corrects the velocity; ``rl/rtc.py`` shifts the mean of the same denoising step. The
two are the same operation written in different variables, so a nonlinear stand-in denoiser
is enough to catch a mistake in either: anything reading the model's Jacobian would differ.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rtc

CHUNK, DIM, STEPS = 50, 32, 4
EXECUTED = 20


@pytest.fixture
def processor():
    lerobot_rtc = pytest.importorskip("lerobot.policies.rtc.modeling_rtc")
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    return lerobot_rtc.RTCProcessor(RTCConfig(execution_horizon=EXECUTED, enabled=True))


def timesteps(steps):
    """The sampler's schedule: flow time falling from 1, with the endpoint appended."""
    return torch.cat([torch.linspace(1, 1 / steps, steps), torch.zeros(1)])


@pytest.mark.parametrize("steps,step,executed,leftover", [
    *[(STEPS, step, EXECUTED, CHUNK - EXECUTED) for step in range(STEPS)],
    # More steps executed per chunk than the chunk leaves over, where the window shortens.
    (STEPS, 1, 40, 10),
    # Ten denoising steps, where the weight runs past its cap and the cap is what binds.
    *[(10, step, EXECUTED, CHUNK - EXECUTED) for step in (0, 1, 5)],
])
def test_matches_lerobot(processor, steps, step, executed, leftover):
    torch.manual_seed(step)
    denoiser = torch.nn.Sequential(
        torch.nn.Linear(DIM, 16), torch.nn.Tanh(), torch.nn.Linear(16, DIM))
    x_t = torch.randn(3, CHUNK, DIM)
    prev = torch.randn(3, leftover, DIM)

    schedule = timesteps(steps)
    time, delta = schedule[step], schedule[step] - schedule[step + 1]

    def mean_of(velocity):
        """The sampler's flow_noise mean, which is the flow_ode mean as well."""
        x0_pred = x_t - velocity * time
        x1_pred = x_t + velocity * (1 - time)
        return x0_pred * (1 - (time - delta)) + x1_pred * (time - delta)

    guided = processor.denoise_step(
        x_t=x_t, prev_chunk_left_over=prev, inference_delay=0, time=time.item(),
        original_denoise_step_partial=denoiser, execution_horizon=executed)

    velocity = denoiser(x_t)
    ours = rtc.guide(mean_of(velocity), x_t - velocity * time, prev, time, delta, executed)

    torch.testing.assert_close(ours, mean_of(guided), rtol=1e-4, atol=1e-5)


def test_no_guidance_past_the_executed_window(processor):
    """Steps the arm will replan before reaching are left alone."""
    weights = rtc.prefix_weights(EXECUTED, CHUNK, torch.device("cpu"), torch.float32)
    assert weights[0, :EXECUTED, 0].min() > 0
    assert weights[0, EXECUTED:, 0].abs().max() == 0


def test_window_shortens_to_the_tail():
    """A window longer than the tail ramps across the tail, not across the window."""
    leftover = 10
    weights = rtc.guide(
        torch.zeros(1, CHUNK, DIM), torch.zeros(1, CHUNK, DIM),
        torch.ones(1, leftover, DIM), torch.tensor(0.5), torch.tensor(1.0),
        execution_horizon=EXECUTED)[0, :, 0] / rtc.guidance_weight(torch.tensor(0.5))

    assert weights[leftover:].abs().max() == 0
    torch.testing.assert_close(weights[0], torch.tensor(leftover / (leftover + 1.0)))


def test_action_dims_the_tail_does_not_cover_are_guided_to_zero():
    """A tail narrower than the denoising space guides the dims it does not cover toward
    zero, because the weights are per step and the target is zero-filled. pi0.5 pads 6 real
    dimensions to 32 and this is what the measured eval does, so the port keeps it."""
    torch.manual_seed(0)
    mean, x0_pred = torch.randn(2, CHUNK, DIM), torch.randn(2, CHUNK, DIM)
    guided = rtc.guide(mean, x0_pred, torch.randn(2, CHUNK - EXECUTED, 6),
                       torch.tensor(0.5), torch.tensor(0.25), EXECUTED)

    assert not torch.allclose(guided[:, :EXECUTED, 6:], mean[:, :EXECUTED, 6:])
    torch.testing.assert_close(guided[:, EXECUTED:], mean[:, EXECUTED:])


def test_rows_without_a_previous_chunk_are_left_alone():
    """A zeroed tail is an absence of a previous chunk, not a command to go to zero."""
    torch.manual_seed(0)
    mean, x0_pred = torch.randn(4, CHUNK, DIM), torch.randn(4, CHUNK, DIM)
    prev = torch.randn(4, CHUNK - EXECUTED, DIM)
    valid = torch.tensor([1.0, 0.0, 1.0, 0.0])

    guided = rtc.guide(mean, x0_pred, prev, torch.tensor(0.5), torch.tensor(0.25),
                       EXECUTED, valid=valid)

    torch.testing.assert_close(guided[1], mean[1])
    torch.testing.assert_close(guided[3], mean[3])
    assert not torch.allclose(guided[0], mean[0])


def tail_inputs(tail, batch=4, dones=None):
    return tail.inputs(batch, torch.device("cpu"), torch.float32, dones=dones)


def test_tail_hands_out_real_tensors_before_any_chunk_exists():
    """The buffer stacks its keys off the first chunk step, so both keys are present there."""
    first = tail_inputs(rtc.Tail(CHUNK, EXECUTED, DIM))

    assert first["rtc_prev"].shape == (4, CHUNK - EXECUTED, DIM)
    assert first["rtc_valid"].shape == (4,)
    assert first["rtc_valid"].abs().max() == 0


def test_tail_carries_the_unexecuted_end_of_the_last_chunk():
    tail = rtc.Tail(CHUNK, EXECUTED, DIM)
    tail_inputs(tail)
    chunk = torch.randn(4, CHUNK, DIM)
    tail.advance(chunk)

    carried = tail_inputs(tail)
    torch.testing.assert_close(carried["rtc_prev"], chunk[:, EXECUTED:])
    assert carried["rtc_valid"].min() == 1


def test_a_finished_episode_starts_the_next_chunk_unguided():
    """The env reports a flag per step of the chunk, not one per env."""
    tail = rtc.Tail(CHUNK, EXECUTED, DIM)
    tail_inputs(tail)
    tail.advance(torch.randn(4, CHUNK, DIM))

    dones = torch.zeros(4, EXECUTED, dtype=torch.bool)
    dones[1, -1] = True
    dones[3, 7] = True

    carried = tail_inputs(tail, dones=dones)
    assert carried["rtc_valid"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert carried["rtc_valid"].shape == (4,)


def test_advancing_does_not_rewrite_what_was_already_handed_out():
    """forward_inputs keeps the tensor by reference, so a later chunk must not reach it."""
    tail = rtc.Tail(CHUNK, EXECUTED, DIM)
    handed = tail_inputs(tail)["rtc_prev"]
    tail.advance(torch.randn(4, CHUNK, DIM))

    assert handed.abs().max() == 0


def test_a_new_rollout_epoch_starts_without_the_last_one_s_tail():
    tail = rtc.Tail(CHUNK, EXECUTED, DIM)
    tail_inputs(tail)
    tail.advance(torch.randn(4, CHUNK, DIM))
    tail.reset()

    assert tail_inputs(tail)["rtc_valid"].abs().max() == 0


def test_weight_is_bounded_near_the_first_step():
    """The weight diverges as the flow time approaches 1. Four and ten denoising steps put
    their first step at 1 exactly, where the cap stands in for the divergence; twenty puts a
    step at 0.95, where the cap is what binds."""
    assert rtc.guidance_weight(torch.tensor(1.0)) == rtc.MAX_GUIDANCE_WEIGHT
    assert rtc.guidance_weight(torch.tensor(0.95)) == rtc.MAX_GUIDANCE_WEIGHT
    assert rtc.guidance_weight(torch.tensor(0.95), max_weight=100) == pytest.approx(19.05,
                                                                                   abs=1e-2)
    assert rtc.guidance_weight(torch.tensor(0.5)) == pytest.approx(2.0)
    assert rtc.guidance_weight(torch.tensor(0.9)) == pytest.approx(9.111, abs=1e-3)
