"""A trained checkpoint, loaded the way both the sim eval and the arm need it."""

from __future__ import annotations

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.rtc import RTCConfig
from lerobot.processor import RenameObservationsProcessorStep


def load_policy(checkpoint, metadata, horizon, device, rtc=True):
    """The checkpoint's policy and the processors saved beside it, which carry the
    camera renaming and the normalization stats from training."""
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = checkpoint
    config.device = device
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
    policy.eval()
    return policy, preprocessor, postprocessor
