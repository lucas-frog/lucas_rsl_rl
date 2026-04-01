from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch

import rsl_rl
from rsl_rl.diffusion import (
    NULL_STYLE_ID,
    DiffusionScheduler,
    MotionEpsilonTransformer,
    SMPDiffusionSampler,
    SMPFeatureLayout,
    SMPGSIDecoder,
    SMPGSISampler,
    SMPReward,
    apply_classifier_free_guidance,
    build_g1_body_part_feature_masks,
    compose_style_predictions_with_body_masks,
    log_smp_noise_metrics,
)
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import store_code_state


def _cfg_get(container, key: str, default=None):
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


class SMPOnPolicyRunner(OnPolicyRunner):
    """在 PPO 训练时接入冻结 diffusion prior 的 SMP runner。"""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        super().__init__(env=env, train_cfg=train_cfg, log_dir=log_dir, device=device)
        self.smp_prior_cfg = train_cfg["smp_prior"]
        self.smp_reward_coef = float(train_cfg.get("smp_reward_coef", 1.0))
        self.task_reward_coef = float(train_cfg.get("task_reward_coef", 1.0))
        self.smp_obs_group = train_cfg.get("smp_obs_group", "smp_motion_window")
        self.log_histograms_every = int(_cfg_get(self.smp_prior_cfg, "log_histograms_every", 20))
        self.style_cfg = _cfg_get(self.smp_prior_cfg, "style_cfg", None)
        self.gsi_cfg = _cfg_get(train_cfg, "gsi_cfg", None)

        self.smp_scheduler = DiffusionScheduler(num_steps=int(self.smp_prior_cfg["num_diffusion_steps"]))
        self.smp_reward = SMPReward(
            num_diffusion_steps=int(self.smp_prior_cfg["num_diffusion_steps"]),
            timesteps_k=list(self.smp_prior_cfg["timesteps_k"]),
            reward_scale=float(self.smp_prior_cfg["reward_scale"]),
            adaptive_norm_decay=float(self.smp_prior_cfg.get("adaptive_norm_decay", 0.99)),
        )
        self.smp_prior = self._load_prior_model()
        self.style_program = self._resolve_style_program_from_cfg()
        self.gsi_sampler = self._build_gsi_sampler()
        self.git_status_repos.append(rsl_rl.__file__)

    def _load_prior_model(self) -> MotionEpsilonTransformer:
        checkpoint_path = self.smp_prior_cfg["checkpoint_path"]
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.smp_checkpoint = checkpoint
        self.smp_checkpoint_style_cfg = checkpoint.get("style_cfg", {})
        model_cfg = checkpoint.get("model_cfg", {})
        feature_dim = int(model_cfg.get("feature_dim", self.smp_prior_cfg["feature_dim"]))
        window_size = int(model_cfg.get("window_size", self.smp_prior_cfg["window_size"]))
        num_diffusion_steps = int(model_cfg.get("num_diffusion_steps", self.smp_prior_cfg["num_diffusion_steps"]))
        num_styles = int(model_cfg.get("num_styles", 0))
        hidden_dim = int(model_cfg.get("hidden_dim", checkpoint["model_state_dict"]["token_proj.weight"].shape[0]))
        num_layers = int(model_cfg.get("num_layers", 2))
        num_heads = int(model_cfg.get("num_heads", 8))

        model = MotionEpsilonTransformer(
            feature_dim=feature_dim,
            window_size=window_size,
            num_diffusion_steps=num_diffusion_steps,
            num_styles=num_styles,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def _resolve_style_name_to_id(self, style_name: str) -> int:
        style_to_id = self.smp_checkpoint_style_cfg.get("style_to_id", {})
        if style_name not in style_to_id:
            raise KeyError(f"Unknown style name '{style_name}' in prior checkpoint")
        return int(style_to_id[style_name])

    def _resolve_style_id_to_name(self, style_id: int) -> str | None:
        style_to_id = self.smp_checkpoint_style_cfg.get("style_to_id", {})
        for style_name, candidate_id in style_to_id.items():
            if int(candidate_id) == int(style_id):
                return str(style_name)
        return None

    def _resolve_style_program_from_cfg(self) -> dict[str, object]:
        if self.smp_prior.num_styles <= 0 or self.style_cfg is None:
            return {"mode": "unconditional", "guidance_scale": 1.0}

        mode = str(_cfg_get(self.style_cfg, "mode", "single_style"))
        guidance_scale = float(_cfg_get(self.style_cfg, "guidance_scale", 1.0))
        if mode == "single_style":
            target_style_id = _cfg_get(self.style_cfg, "target_style_id", None)
            target_style_name = _cfg_get(self.style_cfg, "target_style_name", None)
            if target_style_id is None:
                if target_style_name is None:
                    raise ValueError("single_style mode requires target_style_name or target_style_id")
                target_style_id = self._resolve_style_name_to_id(str(target_style_name))
            # 保留解析后的风格名称，方便日志和后续 GSI 复用同一份 style program。
            resolved_style_name = (
                str(target_style_name)
                if target_style_name is not None
                else self._resolve_style_id_to_name(int(target_style_id))
            )
            return {
                "mode": "single_style",
                "guidance_scale": guidance_scale,
                "target_style_id": int(target_style_id),
                "target_style_name": resolved_style_name,
            }

        if mode != "body_mask":
            raise ValueError(f"Unsupported style mode: {mode}")

        mask_name = str(_cfg_get(self.style_cfg, "mask_name", "g1_upper_lower"))
        feature_masks = build_g1_body_part_feature_masks(
            mask_name=mask_name,
            joint_name_order=list(_cfg_get(self.style_cfg, "joint_name_order", [])),
            ee_name_order=list(_cfg_get(self.style_cfg, "ee_name_order", [])),
            key_body_name_order=list(_cfg_get(self.style_cfg, "key_body_name_order", [])),
            feature_block_offsets=dict(_cfg_get(self.style_cfg, "feature_block_offsets", {})),
        )
        body_part_style_names = dict(_cfg_get(self.style_cfg, "body_part_style_names", {}))
        if "upper_body" not in body_part_style_names or "lower_body" not in body_part_style_names:
            raise ValueError("body_mask mode requires upper_body and lower_body style names")
        shared_style_name = _cfg_get(self.style_cfg, "shared_style_name", None)
        shared_body_defaulted = shared_style_name is None
        if shared_style_name is None:
            shared_style_name = body_part_style_names["lower_body"]
        # 记录补全后的名称映射，确保冻结 prior 后仍能从 cfg 还原完整风格程序语义。
        part_style_names = {
            "shared_body": str(shared_style_name),
            "upper_body": str(body_part_style_names["upper_body"]),
            "lower_body": str(body_part_style_names["lower_body"]),
        }
        part_style_ids = {
            part_name: self._resolve_style_name_to_id(style_name)
            for part_name, style_name in part_style_names.items()
        }
        stacked_masks = torch.stack(list(feature_masks.values()), dim=0)
        return {
            "mode": "body_mask",
            "guidance_scale": guidance_scale,
            "mask_name": mask_name,
            "feature_masks": feature_masks,
            "part_style_ids": part_style_ids,
            "part_style_names": part_style_names,
            "shared_body_defaulted": shared_body_defaulted,
            "coverage": float(stacked_masks.sum(dim=0).mean().item()),
        }

    def _build_gsi_sampler(self) -> SMPGSISampler | None:
        if not bool(_cfg_get(self.gsi_cfg, "enabled", False)):
            return None
        feature_block_offsets = dict(_cfg_get(self.style_cfg, "feature_block_offsets", {}))
        if not feature_block_offsets:
            return None
        feature_layout = SMPFeatureLayout.from_feature_block_offsets(feature_block_offsets)
        sampler = SMPDiffusionSampler(
            model=self.smp_prior,
            num_diffusion_steps=int(self.smp_prior_cfg["num_diffusion_steps"]),
            feature_dim=int(self.smp_prior_cfg["feature_dim"]),
            window_size=int(self.smp_prior_cfg["window_size"]),
            device=self.device,
        )
        decoder = SMPGSIDecoder(
            feature_layout=feature_layout,
            error_threshold=float(_cfg_get(self.gsi_cfg, "error_threshold", 1.0e-6)),
        )
        return SMPGSISampler(sampler=sampler, decoder=decoder)

    def _move_obs_to_device(self, obs):
        if hasattr(obs, "to"):
            return obs.to(self.device)
        if isinstance(obs, dict):
            return {key: value.to(self.device) if hasattr(value, "to") else value for key, value in obs.items()}
        return obs

    def _maybe_apply_gsi_reset(self, obs, dones):
        default_diag = {"reset_accept_rate": 0.0, "reset_resample_count": 0.0, "fallback_rate": 0.0}
        if self.gsi_sampler is None or not bool(_cfg_get(self.gsi_cfg, "sample_on_reset", True)):
            return obs, default_diag

        done_mask = dones > 0
        if done_mask.ndim > 1:
            done_mask = done_mask.view(done_mask.shape[0], -1).any(dim=1)
        reset_env_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
        if reset_env_ids.numel() == 0:
            return obs, default_diag

        from isaaclab_tasks.manager_based.locomotion.velocity.mdp.smp_reset import (
            apply_smp_reset_state,
            build_smp_reset_reference,
        )

        asset_name = str(_cfg_get(self.gsi_cfg, "asset_name", "robot"))
        max_resample_attempts = max(1, int(_cfg_get(self.gsi_cfg, "max_resample_attempts", 1)))
        guidance_scale = _cfg_get(self.gsi_cfg, "guidance_scale", None)
        if guidance_scale is None:
            guidance_scale = float(self.style_program.get("guidance_scale", 1.0))

        # reset 参考状态固定来自环境默认姿态，避免多次重采样期间语义漂移。
        reference_state = build_smp_reset_reference(self.env, env_ids=reset_env_ids, asset_name=asset_name)
        last_result = None
        for attempt in range(max_resample_attempts):
            last_result = self.gsi_sampler.sample_reset_state(
                batch_size=int(reset_env_ids.numel()),
                reference_state=reference_state,
                style_program=self.style_program,
                guidance_scale=float(guidance_scale),
            )
            if last_result.supports_reset_state:
                apply_smp_reset_state(self.env, env_ids=reset_env_ids, state=last_result.state, asset_name=asset_name)
                refreshed_obs = self._move_obs_to_device(self.env.get_observations())
                return refreshed_obs, {
                    "reset_accept_rate": 1.0,
                    "reset_resample_count": float(attempt),
                    "fallback_rate": 0.0,
                }

        if last_result is not None and not bool(_cfg_get(self.gsi_cfg, "fallback_to_default_reset", True)):
            apply_smp_reset_state(self.env, env_ids=reset_env_ids, state=last_result.state, asset_name=asset_name)
            refreshed_obs = self._move_obs_to_device(self.env.get_observations())
            return refreshed_obs, {
                "reset_accept_rate": 0.0,
                "reset_resample_count": float(max_resample_attempts - 1),
                "fallback_rate": 0.0,
            }

        return obs, {
            "reset_accept_rate": 0.0,
            "reset_resample_count": float(max_resample_attempts - 1),
            "fallback_rate": 1.0,
        }

    def _full_style_id(self, batch_size: int, style_id: int) -> torch.Tensor:
        return torch.full((batch_size,), int(style_id), device=self.device, dtype=torch.long)

    def _predict_prior_eps(self, xt: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
        if self.style_program["mode"] == "unconditional":
            return self.smp_prior(xt, t), {"cond_uncond_gap": 0.0}

        eps_uncond = self.smp_prior(xt, t, style_id=self._full_style_id(xt.shape[0], NULL_STYLE_ID))
        if self.style_program["mode"] == "single_style":
            eps_cond = self.smp_prior(xt, t, style_id=self._full_style_id(xt.shape[0], self.style_program["target_style_id"]))
            eps_prior = apply_classifier_free_guidance(
                eps_uncond,
                eps_cond,
                float(self.style_program["guidance_scale"]),
            )
            return eps_prior, {
                "cond_uncond_gap": float((eps_cond - eps_uncond).abs().mean().item()),
                "target_style_id": int(self.style_program["target_style_id"]),
            }

        part_to_eps = {}
        eps_cache: dict[int, torch.Tensor] = {}
        for part_name, style_id in self.style_program["part_style_ids"].items():
            if style_id not in eps_cache:
                eps_cache[style_id] = self.smp_prior(xt, t, style_id=self._full_style_id(xt.shape[0], style_id))
            part_to_eps[part_name] = eps_cache[style_id]
        eps_cond_comp = compose_style_predictions_with_body_masks(part_to_eps, self.style_program["feature_masks"])
        guidance_scale = float(self.style_program["guidance_scale"])
        eps_prior = eps_cond_comp if guidance_scale == 1.0 else apply_classifier_free_guidance(
            eps_uncond,
            eps_cond_comp,
            guidance_scale,
        )
        return eps_prior, {
            "cond_uncond_gap": float((eps_cond_comp - eps_uncond).abs().mean().item()),
            "part_style_ids": dict(self.style_program["part_style_ids"]),
            "coverage": float(self.style_program["coverage"]),
        }

    def _restore_smp_window(self, obs) -> torch.Tensor:
        if self.smp_obs_group not in obs:
            raise KeyError(f"SMP observation group '{self.smp_obs_group}' not found in observations")
        smp_window = obs[self.smp_obs_group]
        feature_dim = int(self.smp_prior_cfg["feature_dim"])
        window_size = int(self.smp_prior_cfg["window_size"])
        if smp_window.shape[-1] != window_size * feature_dim:
            raise ValueError(
                f"Expected flattened SMP dim {window_size * feature_dim}, got {smp_window.shape[-1]}"
            )
        return smp_window.view(smp_window.shape[0], window_size, feature_dim)

    def _compute_smp_metrics(self, obs) -> dict[str, object]:
        x0 = self._restore_smp_window(obs)
        eps = {}
        eps_hat = {}
        cond_uncond_gaps = []
        style_diag = {}
        with torch.inference_mode():
            for timestep in self.smp_reward.timesteps_k:
                t = torch.full((x0.shape[0],), timestep, device=self.device, dtype=torch.long)
                eps_t = torch.randn_like(x0)
                xt = self.smp_scheduler.q_sample(x0, t, eps_t)
                eps_hat_t, style_diag = self._predict_prior_eps(xt, t)
                eps[timestep] = eps_t
                eps_hat[timestep] = eps_hat_t
                cond_uncond_gaps.append(float(style_diag.get("cond_uncond_gap", 0.0)))
        metrics = self.smp_reward.compute(eps=eps, eps_hat=eps_hat)
        metrics["eps"] = eps
        metrics["eps_hat"] = eps_hat
        metrics["style_diag"] = style_diag
        metrics["cond_uncond_gap"] = statistics.mean(cond_uncond_gaps) if cond_uncond_gaps else 0.0
        return metrics

    def _combine_rewards(self, task_rewards: torch.Tensor, smp_rewards: torch.Tensor) -> torch.Tensor:
        if task_rewards.ndim == 2 and smp_rewards.ndim == 1:
            smp_rewards = smp_rewards.unsqueeze(-1)
        return self.task_reward_coef * task_rewards + self.smp_reward_coef * smp_rewards

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width=width, pad=pad)
        if "smp_mean_reward" not in locs:
            return

        self.writer.add_scalar("SMP/reward", locs["smp_mean_reward"], locs["it"])
        self.writer.add_scalar("SMP/cfg/noise_mse", float(locs["smp_noise_mse"]), locs["it"])
        self.writer.add_scalar("SMP/cfg/cond_uncond_gap", float(locs["smp_cfg_gap"]), locs["it"])
        self.writer.add_scalar("SMP/GSI/reset_accept_rate", float(locs.get("gsi_reset_accept_rate", 0.0)), locs["it"])
        self.writer.add_scalar("SMP/GSI/reset_resample_count", float(locs.get("gsi_reset_resample_count", 0.0)), locs["it"])
        self.writer.add_scalar("SMP/GSI/fallback_rate", float(locs.get("gsi_fallback_rate", 0.0)), locs["it"])

        mode = locs["style_program"]["mode"]
        mode_to_scalar = {"unconditional": -1.0, "single_style": 0.0, "body_mask": 1.0}
        self.writer.add_scalar("SMP/style/mode", mode_to_scalar.get(mode, -2.0), locs["it"])
        if mode == "single_style":
            self.writer.add_scalar("SMP/style/target_id", float(locs["style_program"]["target_style_id"]), locs["it"])
        elif mode == "body_mask":
            self.writer.add_scalar(
                "SMP/style_program/shared_body_style_id",
                float(locs["style_program"]["part_style_ids"]["shared_body"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "SMP/style_program/upper_body_style_id",
                float(locs["style_program"]["part_style_ids"]["upper_body"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "SMP/style_program/lower_body_style_id",
                float(locs["style_program"]["part_style_ids"]["lower_body"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "SMP/style_program/shared_body_defaulted",
                float(locs["style_program"].get("shared_body_defaulted", False)),
                locs["it"],
            )
            self.writer.add_scalar("SMP/style_mask/coverage", float(locs["style_program"]["coverage"]), locs["it"])

        if locs["it"] % self.log_histograms_every == 0:
            log_smp_noise_metrics(
                self.writer,
                global_step=locs["it"],
                noise_mse=locs["smp_noise_mse"],
                per_timestep_mse=locs["smp_per_timestep_mse"],
                eps=locs["smp_eps"],
                eps_hat=locs["smp_eps_hat"],
                prefix="SMP",
            )
            return

        self.writer.add_scalar("SMP/noise_mse", float(locs["smp_noise_mse"]), locs["it"])
        for timestep, mse in sorted(locs["smp_per_timestep_mse"].items()):
            self.writer.add_scalar(f"SMP/t{timestep}/noise_mse", float(mse), locs["it"])

    def train_mode(self):
        super().train_mode()
        self.smp_prior.eval()

    def eval_mode(self):
        super().eval_mode()
        self.smp_prior.eval()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        self._prepare_logging_writer()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self._move_obs_to_device(self.env.get_observations())
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            iter_smp_rewards = []
            iter_smp_noise = []
            iter_smp_cfg_gap = []
            iter_gsi_accept_rates = []
            iter_gsi_resample_counts = []
            iter_gsi_fallback_rates = []
            iter_smp_timestep_noise = {timestep: [] for timestep in self.smp_reward.timesteps_k}
            last_smp_eps = None
            last_smp_eps_hat = None
            last_style_diag = {}

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    obs, gsi_diag = self._maybe_apply_gsi_reset(obs, dones)

                    smp_metrics = self._compute_smp_metrics(obs)
                    rewards = self._combine_rewards(rewards, smp_metrics["reward"])
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    last_smp_eps = smp_metrics["eps"]
                    last_smp_eps_hat = smp_metrics["eps_hat"]
                    last_style_diag = smp_metrics["style_diag"]
                    iter_smp_rewards.append(float(smp_metrics["reward"].mean().item()))
                    iter_smp_noise.append(float(smp_metrics["noise_mse"].mean().item()))
                    iter_smp_cfg_gap.append(float(smp_metrics["cond_uncond_gap"]))
                    iter_gsi_accept_rates.append(float(gsi_diag["reset_accept_rate"]))
                    iter_gsi_resample_counts.append(float(gsi_diag["reset_resample_count"]))
                    iter_gsi_fallback_rates.append(float(gsi_diag["fallback_rate"]))
                    for timestep, mse in smp_metrics["per_timestep_mse"].items():
                        iter_smp_timestep_noise[int(timestep)].append(float(mse.mean().item()))

                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])

                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore[arg-type]
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards

                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None and not self.disable_logs:
                smp_per_timestep_mse = {
                    timestep: statistics.mean(values)
                    for timestep, values in iter_smp_timestep_noise.items()
                    if len(values) > 0
                }
                self.log(
                    {
                        **locals(),
                        "smp_mean_reward": statistics.mean(iter_smp_rewards) if len(iter_smp_rewards) > 0 else 0.0,
                        "smp_noise_mse": statistics.mean(iter_smp_noise) if len(iter_smp_noise) > 0 else 0.0,
                        "smp_cfg_gap": statistics.mean(iter_smp_cfg_gap) if len(iter_smp_cfg_gap) > 0 else 0.0,
                        "gsi_reset_accept_rate": statistics.mean(iter_gsi_accept_rates) if len(iter_gsi_accept_rates) > 0 else 0.0,
                        "gsi_reset_resample_count": statistics.mean(iter_gsi_resample_counts) if len(iter_gsi_resample_counts) > 0 else 0.0,
                        "gsi_fallback_rate": statistics.mean(iter_gsi_fallback_rates) if len(iter_gsi_fallback_rates) > 0 else 0.0,
                        "smp_per_timestep_mse": smp_per_timestep_mse,
                        "smp_eps": last_smp_eps,
                        "smp_eps_hat": last_smp_eps_hat,
                        "style_program": self.style_program,
                        "style_diag": last_style_diag,
                    }
                )
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for file_path in git_file_paths:
                        self.writer.save_file(file_path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
