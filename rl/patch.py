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

ROLLOUT = "rlinf/workers/rollout/hf/huggingface_worker.py"
ENV = "rlinf/workers/env/env_worker.py"
ACTOR = "rlinf/workers/actor/fsdp_actor_worker.py"

# Modules with no call site to share, imported by name once the edits below reference
# them.
FILES = {
    "task.py": "rlinf/envs/maniskill/tasks/so101_block_stack.py",
    "dataconfig.py": "rlinf/models/embodiment/openpi/dataconfig/so101_block_stack.py",
    "rtc.py": "rlinf/models/embodiment/openpi/vla_test_rtc.py",
}

MODEL = "rlinf/models/embodiment/openpi/openpi_action_model.py"

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
    # Real-Time Chunking, model side. `rtc_prev` reaches the one denoising step the rollout
    # and the update both go through, so whatever guidance the rollout applies is the
    # guidance the ratio is computed under. Passing None leaves the sampler unguided, which
    # is what the bootstrap value pass and every unpatched caller do.
    (
        MODEL,
        "from rlinf.utils.logging import get_logger\n",
        "from rlinf.models.embodiment.openpi import vla_test_rtc\n"
        "from rlinf.utils.logging import get_logger\n",
    ),
    (
        MODEL,
        "        sample_method,\n"
        "        denoise_steps,\n"
        "        compute_values=True,\n"
        "    ):\n"
        '        """\n'
        "        Sample the mean, variance and value of the action at a given timestep.\n",
        "        sample_method,\n"
        "        denoise_steps,\n"
        "        compute_values=True,\n"
        "        rtc_prev=None,\n"
        "        rtc_valid=None,\n"
        "    ):\n"
        '        """\n'
        "        Sample the mean, variance and value of the action at a given timestep.\n",
    ),
    (
        MODEL,
        "        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight\n"
        "        return x_t_mean, x_t_std, value_t, v_t\n",
        "        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight\n"
        "        if rtc_prev is not None:\n"
        "            if rtc_valid is None:\n"
        "                raise ValueError(\n"
        '                    "RTC guidance needs rtc_valid: without it a row with no "\n'
        '                    "previous chunk is held to a tail of zeros"\n'
        "                )\n"
        '            if sample_method not in ("flow_ode", "flow_noise"):\n'
        "                raise ValueError(\n"
        '                    "RTC guidance is derived for the flow_ode and flow_noise "\n'
        '                    f"weightings, not {sample_method}"\n'
        "                )\n"
        "            x_t_mean = vla_test_rtc.guide(\n"
        "                x_t_mean,\n"
        "                x0_pred,\n"
        "                rtc_prev,\n"
        "                t_input,\n"
        "                delta,\n"
        "                self.config.action_chunk,\n"
        "                valid=rtc_valid,\n"
        "            )\n"
        "        return x_t_mean, x_t_std, value_t, v_t\n",
    ),
    # The rollout path: sample_actions down to the denoising loop.
    (
        MODEL,
        "        observation: _model.Observation,\n"
        "        noise=None,\n"
        '        mode="train",\n'
        "        compute_values=True,\n"
        "    ) -> torch.Tensor:\n"
        '        """Do a full inference forward and compute the action',
        "        observation: _model.Observation,\n"
        "        noise=None,\n"
        '        mode="train",\n'
        "        compute_values=True,\n"
        "        rtc_prev=None,\n"
        "        rtc_valid=None,\n"
        "    ) -> torch.Tensor:\n"
        '        """Do a full inference forward and compute the action',
    ),
    (
        MODEL,
        "            past_key_values,\n"
        "            noise=noise,\n"
        "            mode=mode,\n"
        "            compute_values=compute_values,\n"
        "        )\n",
        "            past_key_values,\n"
        "            noise=noise,\n"
        "            mode=mode,\n"
        "            compute_values=compute_values,\n"
        "            rtc_prev=rtc_prev,\n"
        "            rtc_valid=rtc_valid,\n"
        "        )\n",
    ),
    (
        MODEL,
        "        noise=None,\n"
        '        mode="train",\n'
        "        compute_values=True,\n"
        "    ) -> torch.Tensor:\n"
        "        bsize = state.shape[0]\n",
        "        noise=None,\n"
        '        mode="train",\n'
        "        compute_values=True,\n"
        "        rtc_prev=None,\n"
        "        rtc_valid=None,\n"
        "    ) -> torch.Tensor:\n"
        "        bsize = state.shape[0]\n",
    ),
    (
        MODEL,
        "                sample_method,\n"
        "                num_steps,\n"
        "                compute_values,\n"
        "            )\n"
        "            # Euler step",
        "                sample_method,\n"
        "                num_steps,\n"
        "                compute_values,\n"
        "                rtc_prev,\n"
        "                rtc_valid,\n"
        "            )\n"
        "            # Euler step",
    ),
    # The update path: default_forward down to get_log_prob_value. `rtc_prev` rides in
    # forward_inputs, which the rollout writes and the actor reads back.
    (
        MODEL,
        "        chains,\n"
        "        denoise_inds,\n"
        "        compute_values=False,\n"
        "    ):\n"
        "        bsize = state.shape[0]\n"
        "        batch_indices = torch.arange(bsize)\n",
        "        chains,\n"
        "        denoise_inds,\n"
        "        compute_values=False,\n"
        "        rtc_prev=None,\n"
        "        rtc_valid=None,\n"
        "    ):\n"
        "        bsize = state.shape[0]\n"
        "        batch_indices = torch.arange(bsize)\n",
    ),
    (
        MODEL,
        "                self.config.noise_method,\n"
        "                self.config.num_steps,\n"
        "                compute_values,\n"
        "            )\n"
        "            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)\n",
        "                self.config.noise_method,\n"
        "                self.config.num_steps,\n"
        "                compute_values,\n"
        "                rtc_prev,\n"
        "                rtc_valid,\n"
        "            )\n"
        "            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)\n",
    ),
    (
        MODEL,
        "            chains,\n"
        "            denoise_inds,\n"
        "            compute_values,\n"
        "        )\n"
        "        log_probs = log_probs[\n",
        "            chains,\n"
        "            denoise_inds,\n"
        "            compute_values,\n"
        '            forward_inputs.get("rtc_prev"),\n'
        '            forward_inputs.get("rtc_valid"),\n'
        "        )\n"
        "        log_probs = log_probs[\n",
    ),
    # Real-Time Chunking, rollout side. `rtc_prev` and `rtc_valid` ride from the rollout
    # loop, which owns the tail, down to the sampler, and into forward_inputs so the update
    # recomputes its log-density under the same guidance the rollout executed. The bootstrap
    # value pass goes through predict without them, so it neither advances the tail nor
    # guides.
    (
        MODEL,
        '        rtc_context: RTCGuidanceContext | None = None,\n'
        "        **kwargs,\n",
        '        rtc_context: RTCGuidanceContext | None = None,\n'
        "        rtc_prev=None,\n"
        "        rtc_valid=None,\n"
        "        **kwargs,\n",
    ),
    (
        MODEL,
        "                outputs = self.sample_actions(\n"
        "                    observation, mode=mode, compute_values=compute_values\n"
        "                )\n",
        "                outputs = self.sample_actions(\n"
        "                    observation,\n"
        "                    mode=mode,\n"
        "                    compute_values=compute_values,\n"
        "                    rtc_prev=rtc_prev,\n"
        "                    rtc_valid=rtc_valid,\n"
        "                )\n",
    ),
    (
        MODEL,
        '            "model_action": outputs["actions"]\n'
        "            .reshape(outputs[\"actions\"].shape[0], -1)\n"
        "            .contiguous(),\n"
        "        }\n",
        '            "model_action": outputs["actions"]\n'
        "            .reshape(outputs[\"actions\"].shape[0], -1)\n"
        "            .contiguous(),\n"
        "        }\n"
        "        if rtc_prev is not None:\n"
        '            forward_inputs["rtc_prev"] = rtc_prev\n'
        '            forward_inputs["rtc_valid"] = rtc_valid\n',
    ),
    (
        ROLLOUT,
        "        rlt_switch_flags: torch.Tensor | None = None,\n"
        "        intervene_requested: torch.Tensor | None = None,\n"
        "    ) -> tuple[torch.Tensor, dict[str, Any]]:\n"
        "        if self.rlt_feature_model is not None:\n",
        "        rlt_switch_flags: torch.Tensor | None = None,\n"
        "        intervene_requested: torch.Tensor | None = None,\n"
        "        rtc: dict[str, Any] | None = None,\n"
        "    ) -> tuple[torch.Tensor, dict[str, Any]]:\n"
        "        if self.rlt_feature_model is not None:\n",
    ),
    (
        ROLLOUT,
        "        return self.predict(env_obs, mode=mode)\n",
        "        return self.predict(env_obs, mode=mode, **(rtc or {}))\n",
    ),
    (
        ROLLOUT,
        "    def predict(\n"
        '        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "train"\n'
        "    ) -> tuple[torch.Tensor, dict[str, Any]]:\n",
        "    def predict(\n"
        '        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "train",\n'
        "        rtc_prev=None, rtc_valid=None,\n"
        "    ) -> tuple[torch.Tensor, dict[str, Any]]:\n",
    ),
    (
        ROLLOUT,
        "            else:\n"
        '                kwargs = {"mode": mode}\n',
        "            else:\n"
        '                kwargs = {"mode": mode}\n'
        "                if rtc_prev is not None:\n"
        '                    kwargs["rtc_prev"] = rtc_prev\n'
        '                    kwargs["rtc_valid"] = rtc_valid\n',
    ),
    # The rollout owns the tail: one Tail per worker, reset each epoch, advanced once per
    # executed chunk. Training and eval each reset it before their own epoch and never
    # interleave, so one holder serves both. The bootstrap value pass calls
    # _predict_rollout_actions without it, so it neither guides nor disturbs it.
    (
        ROLLOUT,
        "    def _build_policy_output(\n",
        "    def _rtc_tail(self):\n"
        '        """The tail holder, or None when this run denoises each chunk on its own."""\n'
        '        if not self.model_cfg.get("rtc", False):\n'
        "            return None\n"
        "        assert self.num_pipeline_stages == 1, (\n"
        '            "one tail per worker holds one batch of envs, and pipeline stages are "\n'
        '            "disjoint batches"\n'
        "        )\n"
        "        assert not self.enable_dagger, (\n"
        '            "the dagger branch of predict builds its own kwargs and drops the tail"\n'
        "        )\n"
        "        assert not self.env_decoupled_mode, (\n"
        '            "the decoupled eval loop calls predict without the tail"\n'
        "        )\n"
        "        assert SupportedModel(self.model_cfg.model_type) == SupportedModel.OPENPI, (\n"
        '            "the tail reaches the sampler through the openpi branch of predict"\n'
        "        )\n"
        '        if getattr(self, "_rtc", None) is None:\n'
        "            from rlinf.models.embodiment.openpi import vla_test_rtc\n"
        "\n"
        "            config = self.hf_model.config\n"
        "            self._rtc = vla_test_rtc.Tail(\n"
        "                config.action_horizon,\n"
        "                config.action_chunk,\n"
        "                config.action_dim,\n"
        "            )\n"
        "        return self._rtc\n"
        "\n"
        "    def _rtc_inputs(self, env_output):\n"
        "        tail = self._rtc_tail()\n"
        "        if tail is None:\n"
        "            return None\n"
        "        return tail.inputs(\n"
        "            self._infer_env_batch_size(env_output),\n"
        "            next(self.hf_model.parameters()).device,\n"
        "            torch.float32,\n"
        '            dones=env_output.get("dones"),\n'
        "        )\n"
        "\n"
        "    def _rtc_advance(self, result):\n"
        "        tail = self._rtc_tail()\n"
        "        if tail is not None:\n"
        '            tail.advance(result["model_actions"])\n'
        "\n"
        "    def _build_policy_output(\n",
    ),
    (
        ROLLOUT,
        "        self.update_dagger_beta()\n"
        "        for _ in range(self.n_train_chunk_steps):\n",
        "        self.update_dagger_beta()\n"
        "        tail = self._rtc_tail()\n"
        "        if tail is not None:\n"
        "            tail.reset()\n"
        "        for _ in range(self.n_train_chunk_steps):\n",
    ),
    (
        ROLLOUT,
        "                actions, result = self._predict_rollout_actions(\n"
        '                    env_output["obs"],\n'
        '                    final_obs=env_output.get("final_obs", None),\n'
        '                    rlt_switch_flags=env_output.get("rlt_switch_flags", None),\n'
        '                    intervene_requested=env_output.get("intervene_flags", None),\n'
        "                )\n"
        "\n"
        "                policy_output = self._build_policy_output(\n",
        "                actions, result = self._predict_rollout_actions(\n"
        '                    env_output["obs"],\n'
        '                    final_obs=env_output.get("final_obs", None),\n'
        '                    rlt_switch_flags=env_output.get("rlt_switch_flags", None),\n'
        '                    intervene_requested=env_output.get("intervene_flags", None),\n'
        "                    rtc=self._rtc_inputs(env_output),\n"
        "                )\n"
        "                self._rtc_advance(result)\n"
        "\n"
        "                policy_output = self._build_policy_output(\n",
    ),
    # The same for the check between iterations, so it scores the arm the rollout executes.
    (
        ROLLOUT,
        "            ):\n"
        "                for _ in range(self.n_eval_chunk_steps):\n"
        "                    for stage_id in range(self.num_pipeline_stages):\n"
        "                        env_output = await self.recv_from(\n",
        "            ):\n"
        "                tail = self._rtc_tail()\n"
        "                if tail is not None:\n"
        "                    tail.reset()\n"
        "                for _ in range(self.n_eval_chunk_steps):\n"
        "                    for stage_id in range(self.num_pipeline_stages):\n"
        "                        env_output = await self.recv_from(\n",
    ),
    (
        ROLLOUT,
        "                        actions, _ = self._predict_rollout_actions(\n"
        '                            env_output["obs"],\n'
        '                            mode="eval",\n'
        '                            final_obs=env_output.get("final_obs", None),\n'
        '                            rlt_switch_flags=env_output.get("rlt_switch_flags", None),\n'
        '                            intervene_requested=env_output.get("intervene_flags", None),\n'
        "                        )\n"
        "                        if isinstance(actions, torch.Tensor):\n"
        "                            actions = actions.detach().cpu().contiguous()\n"
        "                        self.send_to(\n",
        "                        actions, result = self._predict_rollout_actions(\n"
        '                            env_output["obs"],\n'
        '                            mode="eval",\n'
        '                            final_obs=env_output.get("final_obs", None),\n'
        '                            rlt_switch_flags=env_output.get("rlt_switch_flags", None),\n'
        '                            intervene_requested=env_output.get("intervene_flags", None),\n'
        "                            rtc=self._rtc_inputs(env_output),\n"
        "                        )\n"
        "                        self._rtc_advance(result)\n"
        "                        if isinstance(actions, torch.Tensor):\n"
        "                            actions = actions.detach().cpu().contiguous()\n"
        "                        self.send_to(\n",
    ),
    # Which envs finished during the chunk just executed, so the rollout worker tells a
    # reset env from a continuing one and starts a fresh episode unguided. Two edits: the
    # env worker sends it, and the merge every recv runs rebuilds the payload from a fixed
    # key list, so it has to name it too.
    (
        ENV,
        "        data = {\n"
        '            "obs": env_batch["obs"],\n'
        '            "final_obs": env_batch["final_obs"],\n'
        "        }\n",
        "        data = {\n"
        '            "obs": env_batch["obs"],\n'
        '            "final_obs": env_batch["final_obs"],\n'
        '            "dones": env_batch.get("dones"),\n'
        "        }\n",
    ),
    # The check between iterations resets its envs as it goes, so it sends the same flags
    # the training rollout does.
    (
        ENV,
        "        env_output = EnvOutput(\n"
        "            obs=extracted_obs,\n"
        "            final_obs=final_obs,\n"
        "            env_infos=infos if isinstance(infos, dict) else None,\n"
        "            rlt_switch_flags=rlt_switch_flags,\n"
        "        )\n",
        "        env_output = EnvOutput(\n"
        "            obs=extracted_obs,\n"
        "            final_obs=final_obs,\n"
        "            env_infos=infos if isinstance(infos, dict) else None,\n"
        "            rlt_switch_flags=rlt_switch_flags,\n"
        "            dones=chunk_dones,\n"
        "        )\n",
    ),
    (
        ROLLOUT,
        '            "intervene_flags": self._merge_optional_flag_tensors(\n'
        "                obs_dicts, intervene_flags_list\n"
        "            ),\n"
        "        }\n",
        '            "intervene_flags": self._merge_optional_flag_tensors(\n'
        "                obs_dicts, intervene_flags_list\n"
        "            ),\n"
        '            "dones": self._merge_optional_flag_tensors(\n'
        '                obs_dicts, [batch.get("dones") for batch in obs_batches]\n'
        "            ),\n"
        "        }\n",
    ),
    # `actor.recompute_prev_logprobs` evaluates the rollout's own chains through the actor's
    # weights before the first optimizer step, so the ratio the loss divides by starts at
    # exactly 1. The densities the rollout worker stores come from a second copy of these
    # weights in a second process, where a bf16 matmul reduces in a different order, and the
    # Gaussian exponent divides by a sigma near 0.02, so a small difference in the predicted
    # mean is a large one in the density. It is also one-signed: the action was drawn around
    # the rollout's mean, so the cross term averages away and the squared offset does not,
    # which is why `actor/ratio` reads 0.82 where symmetric noise would put it at or above 1.
    #
    # `torch.chunk` slices contiguously and the update splits the same shuffled batch the
    # same way, so chunk i here is the micro batch the update reads i-th. RLinf recomputes
    # for GR00T through that model's forward, and for the LLM actor in `run_inference`.
    (
        ACTOR,
        "        with torch.no_grad():\n"
        "            self.rollout_batch = process_nested_dict_for_train(\n"
        "                self.rollout_batch, shuffle_id\n"
        "            )\n",
        "        with torch.no_grad():\n"
        "            self.rollout_batch = process_nested_dict_for_train(\n"
        "                self.rollout_batch, shuffle_id\n"
        "            )\n"
        "\n"
        '        if self.cfg.actor.get("recompute_prev_logprobs", False):\n'
        "            recomputed = []\n"
        "            with torch.no_grad():\n"
        "                for batch in split_dict_to_chunk(\n"
        "                    self.rollout_batch,\n"
        '                    self.rollout_batch["prev_logprobs"].size(0)\n'
        "                    // self.cfg.actor.micro_batch_size,\n"
        "                ):\n"
        "                    batch = put_tensor_device(batch, self.device)\n"
        "                    with self.amp_context:\n"
        "                        output_dict = self.model(\n"
        '                            forward_inputs=batch.get("forward_inputs", None),\n'
        "                            compute_logprobs=True,\n"
        "                            compute_entropy=self.cfg.algorithm.entropy_bonus > 0,\n"
        '                            compute_values=self.cfg.algorithm.adv_type == "gae",\n'
        "                            use_cache=False,\n"
        "                        )\n"
        '                    recomputed.append(output_dict["logprobs"].detach().cpu())\n'
        '            self.rollout_batch["prev_logprobs"] = torch.cat(recomputed)\n',
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
