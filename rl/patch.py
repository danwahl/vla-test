"""Teach an RLinf checkout about this arm.

    python rl/patch.py $RLINF_DIR

RLinf dispatches environments, observations, actions and control modes through if-else
chains, and its guide for adding an environment says to edit them in place, so this copies
modules in and edits those chains. Each edit asserts a single match of its anchor, and
re-running is a no-op, so a version this was not written against fails here, before a run
starts. That version is ``RLINF_COMMIT`` in ``rl/Dockerfile``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).parent

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
    # A level the task computes, which RLinf differences into the step reward.
    (
        "rlinf/envs/maniskill/maniskill_env.py",
        '        elif getattr(self.cfg, "reward_mode", "default") == "only_success":\n',
        '        elif getattr(self.cfg, "reward_mode", "default") == "task":\n'
        '            reward = info["reward"]\n'
        '        elif getattr(self.cfg, "reward_mode", "default") == "only_success":\n',
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rlinf", type=Path, help="an RLinf checkout")
    args = parser.parse_args()

    if not (args.rlinf / "rlinf").is_dir():
        raise SystemExit(f"{args.rlinf} holds no RLinf checkout")

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
