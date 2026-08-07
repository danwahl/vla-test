"""Rewrite a merged checkpoint into the layout RLinf's openpi actor loads.

    uv run python rl/convert.py MERGED OUT

Both sides are the same PyTorch pi0.5, so the weights carry over tensor for tensor and
only the names change. The normalization stats travel as the JSON openpi reads.
"""

import json
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

# openpi looks the stats up by asset id, which is the dataset they were measured on.
REPO_ID = "vla-test/so101_block_stack_sim"
NORMALIZED = {"observation.state": "state", "action": "actions"}
STATISTICS = ("mean", "std", "q01", "q99")

merged, out = Path(sys.argv[1]), Path(sys.argv[2])

weights = load_file(merged / "model.safetensors")
out.mkdir(parents=True, exist_ok=True)
# The actor loads with strict=False, so a key it has no use for costs nothing and a key
# it wanted costs a tensor left at its initialization.
save_file({key.removeprefix("model."): value for key, value in weights.items()},
          out / "model.safetensors")

stats = load_file(next(merged.glob("policy_preprocessor_step_*_normalizer_processor.safetensors")))
(out / REPO_ID).mkdir(parents=True, exist_ok=True)
(out / REPO_ID / "norm_stats.json").write_text(json.dumps({"norm_stats": {
    name: {statistic: stats[f"{key}.{statistic}"].tolist() for statistic in STATISTICS}
    for key, name in NORMALIZED.items()
}}))
print(out)
