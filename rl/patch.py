"""Teach an RLinf checkout about this arm.

    uv run python rl/patch.py RLINF_DIR --clone

RLinf dispatches environments, observations, actions and control modes through if-else
chains rather than a registry, and its own guide for adding an environment says to edit
them in place. So this copies two modules in and edits four call sites. It is idempotent,
and each edit asserts a single match of its anchor, so a checkout that has moved on fails
here rather than halfway through a run.

``COMMIT`` is the pin. It is checked against the checkout, so this is the one place the
version is written down and no run can quietly use another. RLinf's dependencies are not
declared here: its ``embodied`` extra carries neither ManiSkill nor openpi nor a CUDA
torch, which come from the image the run happens in, so an installable that resolved would
not be an environment that runs.

The run config is not copied: hydra reads it from this repo, and its ``searchpath`` picks
up RLinf's own config tree. See the README for the invocation.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).parent

REPOSITORY = "https://github.com/RLinf/RLinf"
COMMIT = "d3aff547c06d66e2e792a62a71122abed86e8aee"

# Modules with no call site to share, which are simply imported by name once the edits
# below reference them.
FILES = {
    "task.py": "rlinf/envs/maniskill/tasks/so101_block_stack.py",
    "dataconfig.py": "rlinf/models/embodiment/openpi/dataconfig/so101_block_stack.py",
}

EDITS = [
    # The arm takes absolute joint targets, and get_robot_control_mode raises on a robot
    # it does not know.
    (
        "rlinf/config.py",
        '                elif "panda" in robot:\n',
        '                elif "so101" in robot:\n'
        '                    return "pd_joint_pos"\n'
        '                elif "panda" in robot:\n',
    ),
    # Joint-space actions reach the env as the policy emits them. The panda arms already
    # take this path.
    (
        "rlinf/envs/action_utils.py",
        '    if "panda" in policy:\n',
        '    if "panda" in policy or "so101" in policy:\n',
    ),
    # Two cameras and a prompt that differs per env, which none of the stock modes carry.
    (
        "rlinf/envs/maniskill/maniskill_env.py",
        '            return infos["extracted_obs"]\n',
        '            return infos["extracted_obs"]\n'
        "\n"
        '        if wrap_obs_mode == "so101":\n'
        "            from rlinf.envs.maniskill.tasks import so101_block_stack\n"
        "\n"
        "            return so101_block_stack.wrap_obs(raw_obs, self.env.unwrapped)\n",
    ),
    # The openpi transforms, under the name actor.model.openpi.config_name gives.
    (
        "rlinf/models/embodiment/openpi/dataconfig/__init__.py",
        "from rlinf.models.embodiment.openpi.dataconfig.isaaclab_dataconfig import (\n",
        "from rlinf.models.embodiment.openpi.dataconfig.so101_block_stack import (\n"
        "    So101BlockStackDataConfig,\n"
        ")\n"
        "from rlinf.models.embodiment.openpi.dataconfig.isaaclab_dataconfig import (\n",
    ),
    (
        "rlinf/models/embodiment/openpi/dataconfig/__init__.py",
        "_CONFIGS = [\n",
        "_CONFIGS = [\n"
        "    TrainConfig(\n"
        '        name="pi05_so101_block_stack",\n'
        "        model=pi0_config.Pi0Config(pi05=True, action_horizon=50,\n"
        '                                   paligemma_variant="gemma_2b",\n'
        '                                   action_expert_variant="gemma_300m"),\n'
        "        data=So101BlockStackDataConfig(\n"
        '            repo_id="vla-test/so101_block_stack_sim",\n'
        "            base_config=DataConfig(prompt_from_task=True),\n"
        "        ),\n"
        "    ),\n",
    ),
]


def clone(target):
    """Fetch the pinned commit alone, which is a fraction of the history."""
    subprocess.run(["git", "init", "-q", str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "fetch", "-q", "--depth", "1",
                    REPOSITORY, COMMIT], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "-q", "FETCH_HEAD"], check=True)


def head(target):
    return subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rlinf", type=Path, help="an RLinf checkout")
    parser.add_argument("--clone", action="store_true",
                        help="fetch the pinned commit into RLINF_DIR first")
    parser.add_argument("--any-commit", action="store_true",
                        help="patch a checkout that is not on the pin")
    args = parser.parse_args()

    if args.clone:
        clone(args.rlinf)
    at = head(args.rlinf)
    if at != COMMIT and not args.any_commit:
        raise SystemExit(f"{args.rlinf} is at {at[:12]}, not the pinned {COMMIT[:12]}. "
                         "Pass --any-commit to patch it anyway.")
    print(f"rlinf    {at[:12]}")

    for source, destination in FILES.items():
        target = args.rlinf / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(HERE / source, target)
        print(f"copied   {destination}")

    for path, anchor, replacement in EDITS:
        text = (args.rlinf / path).read_text()
        if replacement in text:
            print(f"in place {path}")
            continue
        count = text.count(anchor)
        if count != 1:
            raise SystemExit(f"{path}: anchor matched {count} times, expected 1")
        (args.rlinf / path).write_text(text.replace(anchor, replacement))
        print(f"patched  {path}")


if __name__ == "__main__":
    main()
