"""The one schema demonstrations are written in, wherever they were collected.

Sim and hardware episodes have to be interchangeable to a policy, so they share this
rather than each declaring their own.
"""

from __future__ import annotations

import json

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .agent import JOINT_NAMES, LIMITS
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


def finalize(dataset):
    """Close the episodes out, with both joint features normalized on the arm's travel.

    pi0.5 reads ``observation.state`` as one of 256 bins spanning q01 to q99, spelled out in
    the prompt as text, and nothing clips it: a joint outside that span reads as one of the
    span's edges whatever angle it is really at. ``action`` is a joint position too, so it
    takes the same span. The arm's travel holds every pose it can reach and means the same
    thing in every dataset the arm appears in.

    Callers close a dataset out from a ``finally``, where a run stopped before it saved an
    episode has statistics to write nothing about.
    """
    dataset.finalize()
    path = dataset.root / "meta" / "stats.json"
    if not path.exists():
        return
    stats = json.loads(path.read_text())
    for key in ("observation.state", "action"):
        stats[key]["q01"] = LIMITS[:, 0].tolist()
        stats[key]["q99"] = LIMITS[:, 1].tolist()
    path.write_text(json.dumps(stats))
