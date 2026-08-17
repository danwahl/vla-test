"""The one schema demonstrations are written in, wherever they were collected.

Sim and hardware episodes have to be interchangeable to a policy, so they share this
rather than each declaring their own.
"""

from __future__ import annotations

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .agent import JOINT_NAMES
from .env import IMAGE_SIZE

CAMERAS = ("top", "wrist")

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (len(JOINT_NAMES),),
                          "names": JOINT_NAMES},
    "action": {"dtype": "float32", "shape": (len(JOINT_NAMES),), "names": JOINT_NAMES},
    **{f"observation.images.{camera}": {"dtype": "video",
                                        "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                                        "names": ["height", "width", "channel"]}
       for camera in CAMERAS},
}


def create(root, repo_id, fps, resume=False):
    """Somewhere to write episodes, either a new dataset or the one already at ``root``."""
    # H.264 rather than the AV1 default: every loader downstream of here decodes it.
    encoder = RGBEncoderConfig(vcodec="h264")
    if resume:
        return LeRobotDataset.resume(repo_id=repo_id, root=root, rgb_encoder=encoder)
    return LeRobotDataset.create(repo_id=repo_id, fps=fps, features=FEATURES, root=root,
                                 robot_type="so101", rgb_encoder=encoder)
