"""The openpi transforms the RL rollout feeds this policy.

Copied to ``rlinf/models/embodiment/openpi/dataconfig/so101_block_stack.py`` and registered
as ``pi05_so101_block_stack``, which is what ``actor.model.openpi.config_name`` names. The
prefix carries meaning there: the actor reads pi0.5's language-token budget out of it.

The camera slots and the absolute 6-D joint actions are the ones the checkpoint was
fine-tuned on, so the policy sees at rollout what it saw at training.
"""

import dataclasses
import pathlib

import einops
import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override


def _image(image):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    return einops.rearrange(image, "c h w -> h w c") if image.shape[0] == 3 else image


@dataclasses.dataclass(frozen=True)
class So101Inputs(_transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        top, wrist = _image(data["observation/image"]), _image(data["observation/wrist_image"])
        inputs = {
            "state": data["observation/state"],
            "image": {"base_0_rgb": top, "left_wrist_0_rgb": wrist,
                      "right_wrist_0_rgb": np.zeros_like(top)},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_,
                           "right_wrist_0_rgb": np.False_},
        }
        return inputs | {key: data[key] for key in ("actions", "prompt") if key in data}


@dataclasses.dataclass(frozen=True)
class So101Outputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :6])}


@dataclasses.dataclass(frozen=True)
class So101BlockStackDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path,
               model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=_transforms.Group(
                inputs=[So101Inputs(model_type=model_config.model_type)],
                outputs=[So101Outputs()],
            ),
            model_transforms=ModelTransformFactory()(model_config),
        )
