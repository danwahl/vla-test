"""A trained checkpoint, loaded the way both the sim eval and the arm need it."""

from __future__ import annotations

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.rtc import RTCConfig
from lerobot.processor import RenameObservationsProcessorStep
from torchao.quantization import Int8WeightOnlyConfig, quantize_

# Both run once per chunk and hand the action expert a cache, so they sit the far side of
# the model from the commands. The weights are unpacked for each matmul, which buys memory
# and not speed.
OBSERVATION_MODULES = ("paligemma.model.language_model", "paligemma.model.vision_tower")


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
