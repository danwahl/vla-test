"""Fold the LoRA adapter into the base weights, for a checkpoint that stands on its own.

    uv run python train/merge.py CHECKPOINT/pretrained_model OUT

The processors are copied across as they are, so OUT loads the same way the adapter did.
"""

import sys
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from peft import PeftConfig, PeftModel

checkpoint, out = Path(sys.argv[1]), Path(sys.argv[2])

config = PreTrainedConfig.from_pretrained(checkpoint)
# Folding weights together is arithmetic; it asks nothing of a GPU.
config.device = "cpu"
base = PI05Policy.from_pretrained(PeftConfig.from_pretrained(checkpoint).base_model_name_or_path,
                                 config=config)
policy = PeftModel.from_pretrained(base, checkpoint).merge_and_unload()

# These two are what send a loader to the adapter; cleared, OUT is read as plain weights.
policy.config.use_peft = False
policy.config.pretrained_path = None
policy.save_pretrained(out)

preprocessor, postprocessor = make_pre_post_processors(policy_cfg=config,
                                                       pretrained_path=checkpoint)
preprocessor.save_pretrained(out)
postprocessor.save_pretrained(out)
print(out)
