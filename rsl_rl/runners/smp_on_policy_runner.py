from __future__ import annotations

import json
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
    """从 dict 或对象属性中读取配置项，未命中时返回默认值。"""
    # 统一兼容 dict 和对象属性两种配置载体，避免上层配置格式变化时到处分支判断。
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


class SMPOnPolicyRunner(OnPolicyRunner):
    """在 PPO 训练时接入冻结 diffusion prior 的 SMP runner。"""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        """初始化 SMP runner，并构建 prior、reward、style program 与可选 GSI 采样器。"""
        super().__init__(env=env, train_cfg=train_cfg, log_dir=log_dir, device=device)
        # 训练配置中与 SMP prior 相关的字段单独保存，后续所有 prior / style / GSI 逻辑都只依赖这一份配置。
        self.smp_prior_cfg = train_cfg["smp_prior"]
        # 主任务奖励和 SMP 奖励分别加权，便于在不改环境 reward 定义的前提下调整优化目标。
        self.smp_reward_coef = float(train_cfg.get("smp_reward_coef", 1.0))
        self.task_reward_coef = float(train_cfg.get("task_reward_coef", 1.0))
        self.smp_obs_group = train_cfg.get("smp_obs_group", "smp_motion_window")
        self.log_histograms_every = int(_cfg_get(self.smp_prior_cfg, "log_histograms_every", 20))
        # style_cfg 决定 prior 是无条件、单风格还是 body mask 组合风格；gsi_cfg 决定是否对 reset 做重采样。
        self.style_cfg = _cfg_get(self.smp_prior_cfg, "style_cfg", None)
        self.gsi_cfg = _cfg_get(train_cfg, "gsi_cfg", None)

        # 这里的 scheduler 和 reward 仅负责对扩散 prior 的噪声预测质量做度量，不参与参数更新。
        self.smp_scheduler = DiffusionScheduler(num_steps=int(self.smp_prior_cfg["num_diffusion_steps"]))
        fixed_normalizer_mse_by_timestep = None
        if str(_cfg_get(self.smp_prior_cfg, "reward_mode", "absolute")) == "fixed_normalizer":
            fixed_normalizer_mse_by_timestep = self._resolve_fixed_normalizer_mse_by_timestep()
        self.smp_reward = SMPReward(
            num_diffusion_steps=int(self.smp_prior_cfg["num_diffusion_steps"]),
            timesteps_k=list(self.smp_prior_cfg["timesteps_k"]),
            reward_scale=float(self.smp_prior_cfg["reward_scale"]),
            reward_mode=str(_cfg_get(self.smp_prior_cfg, "reward_mode", "absolute")),
            adaptive_norm_decay=float(self.smp_prior_cfg.get("adaptive_norm_decay", 0.999)),
            zscore_reward_center=float(_cfg_get(self.smp_prior_cfg, "zscore_reward_center", 0.8)),
            zscore_std_floor=float(_cfg_get(self.smp_prior_cfg, "zscore_std_floor", 1.0e-6)),
            fixed_normalizer_mse_by_timestep=fixed_normalizer_mse_by_timestep,
        )
        self.smp_prior = self._load_prior_model()
        self.style_program = self._resolve_style_program_from_cfg()
        if self.smp_reward.reward_mode == "target_vs_uncond" and self.style_program["mode"] == "unconditional":
            raise ValueError("reward_mode='target_vs_uncond' requires a conditional SMP style program")
        self.gsi_sampler = self._build_gsi_sampler()
        self.git_status_repos.append(rsl_rl.__file__)

    def _resolve_fixed_normalizer_mse_by_timestep(self) -> dict[int, float]:
        """解析 fixed_normalizer 模式使用的每个时间步参考 MSE。"""
        timesteps_k = [int(timestep) for timestep in _cfg_get(self.smp_prior_cfg, "timesteps_k", [])]
        inline_stats = dict(_cfg_get(self.smp_prior_cfg, "fixed_normalizer_mse_by_timestep", {}))
        if inline_stats:
            resolved = {int(timestep): float(value) for timestep, value in inline_stats.items()}
        else:
            stats_path = _cfg_get(self.smp_prior_cfg, "fixed_normalizer_stats_path", None)
            if stats_path is None:
                raise ValueError(
                    "reward_mode='fixed_normalizer' requires fixed_normalizer_mse_by_timestep "
                    "or fixed_normalizer_stats_path"
                )
            with open(stats_path, encoding="utf-8") as file:
                payload = json.load(file)
            if "fixed_normalizer_mse_by_timestep" in payload:
                source = payload["fixed_normalizer_mse_by_timestep"]
            elif "per_timestep_raw_mse_mean" in payload:
                source = payload["per_timestep_raw_mse_mean"]
            else:
                source = payload["groups"]["positive"]["per_timestep_raw_mse_mean"]
            resolved = {int(timestep): float(value) for timestep, value in dict(source).items()}

        missing_timesteps = [int(timestep) for timestep in timesteps_k if int(timestep) not in resolved]
        if missing_timesteps:
            raise ValueError(
                "fixed normalizer stats are missing timesteps "
                f"{missing_timesteps} required by timesteps_k={timesteps_k}"
            )
        return {int(timestep): float(resolved[int(timestep)]) for timestep in timesteps_k}

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        """执行 on-policy 训练主循环，在 rollout 中引入 SMP 指标与 GSI reset 机制。"""
        self._prepare_logging_writer()

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # 每轮迭代都统计 SMP 相关的均值，用来观察 prior 质量和 reset 重采样行为是否稳定。
            iter_task_rewards = []
            iter_task_rewards_scaled = []
            iter_smp_rewards = []
            iter_smp_rewards_scaled = []
            iter_combined_rewards = []
            iter_smp_noise = []
            iter_smp_cfg_gap = []
            iter_smp_reward_targets = []
            iter_smp_reward_unconds = []
            iter_smp_reward_gaps = []
            iter_smp_reward_finals = []
            iter_gsi_accept_rates = []
            iter_gsi_resample_counts = []
            iter_gsi_fallback_rates = []
            iter_smp_timestep_noise = {timestep: [] for timestep in self.smp_reward.timesteps_k}
            last_smp_eps = None
            last_smp_eps_hat = None
            last_style_diag = {}

            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # reward 使用 terminal-corrected 观测，policy 则继续消费 reset / GSI 后的观测。
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    reward_obs = self._build_smp_reward_obs(obs, dones, extras)
                    obs, gsi_diag = self._maybe_apply_gsi_reset(obs, dones)

                    smp_metrics = self._compute_smp_metrics(reward_obs)
                    reward_terms = self._decompose_rewards(rewards, smp_metrics["reward"])
                    rewards = reward_terms["combined"]
                    # process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    last_smp_eps = smp_metrics["eps"]
                    last_smp_eps_hat = smp_metrics["eps_hat"]
                    last_style_diag = smp_metrics["style_diag"]
                    iter_task_rewards.append(float(reward_terms["task_raw"].mean().item()))
                    iter_task_rewards_scaled.append(float(reward_terms["task_scaled"].mean().item()))
                    iter_smp_rewards.append(float(smp_metrics["reward"].mean().item()))
                    iter_smp_rewards_scaled.append(float(reward_terms["smp_scaled"].mean().item()))
                    iter_combined_rewards.append(float(reward_terms["combined"].mean().item()))
                    iter_smp_noise.append(float(smp_metrics["noise_mse"].mean().item()))
                    iter_smp_cfg_gap.append(float(smp_metrics["cond_uncond_gap"]))
                    if "reward_target" in smp_metrics:
                        iter_smp_reward_targets.append(float(smp_metrics["reward_target"].mean().item()))
                    if "reward_uncond" in smp_metrics:
                        iter_smp_reward_unconds.append(float(smp_metrics["reward_uncond"].mean().item()))
                    if "reward_gap" in smp_metrics:
                        iter_smp_reward_gaps.append(float(smp_metrics["reward_gap"].mean().item()))
                    if "reward" in smp_metrics:
                        iter_smp_reward_finals.append(float(smp_metrics["reward"].mean().item()))
                    iter_gsi_accept_rates.append(float(gsi_diag["reset_accept_rate"]))
                    iter_gsi_resample_counts.append(float(gsi_diag["reset_resample_count"]))
                    iter_gsi_fallback_rates.append(float(gsi_diag["fallback_rate"]))
                    for timestep, mse in smp_metrics["per_timestep_mse"].items():
                        iter_smp_timestep_noise[int(timestep)].append(float(mse.mean().item()))
                    
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    # book keeping
                    if self.log_dir is not None:
                        # 把 episode 级别信息累积到 buffer，供 logger 汇总平均 return / length。
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore[arg-type]
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                
                # compute returns
                self.alg.compute_returns(obs)

            # update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                smp_per_timestep_mse = {
                    timestep: statistics.mean(values)
                    for timestep, values in iter_smp_timestep_noise.items()
                    if len(values) > 0
                }
                # Log information
                self.log(
                    {
                        **locals(),
                        "smp_mean_reward": statistics.mean(iter_smp_rewards) if len(iter_smp_rewards) > 0 else 0.0,
                        "smp_task_reward_raw": statistics.mean(iter_task_rewards) if len(iter_task_rewards) > 0 else 0.0,
                        "smp_task_reward_scaled": (
                            statistics.mean(iter_task_rewards_scaled) if len(iter_task_rewards_scaled) > 0 else 0.0
                        ),
                        "smp_style_reward_raw": statistics.mean(iter_smp_rewards) if len(iter_smp_rewards) > 0 else 0.0,
                        "smp_style_reward_scaled": (
                            statistics.mean(iter_smp_rewards_scaled) if len(iter_smp_rewards_scaled) > 0 else 0.0
                        ),
                        "smp_combined_reward": (
                            statistics.mean(iter_combined_rewards) if len(iter_combined_rewards) > 0 else 0.0
                        ),
                        "smp_noise_mse": statistics.mean(iter_smp_noise) if len(iter_smp_noise) > 0 else 0.0,
                        "smp_cfg_gap": statistics.mean(iter_smp_cfg_gap) if len(iter_smp_cfg_gap) > 0 else 0.0,
                        "smp_reward_target": (
                            statistics.mean(iter_smp_reward_targets) if len(iter_smp_reward_targets) > 0 else None
                        ),
                        "smp_reward_uncond": (
                            statistics.mean(iter_smp_reward_unconds) if len(iter_smp_reward_unconds) > 0 else None
                        ),
                        "smp_reward_gap": (
                            statistics.mean(iter_smp_reward_gaps) if len(iter_smp_reward_gaps) > 0 else None
                        ),
                        "smp_reward_final": (
                            statistics.mean(iter_smp_reward_finals) if len(iter_smp_reward_finals) > 0 else None
                        ),
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
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for file_path in git_file_paths:
                        self.writer.save_file(file_path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """记录 PPO 与 SMP 相关日志，包括风格程序、噪声指标和 GSI 统计。"""
        super().log(locs, width=width, pad=pad)
        if "smp_mean_reward" not in locs:
            return

        # 这一组标量记录的是“训练过程的宏观健康度”：reward、噪声误差、CFG gap 和 GSI 重采样质量。
        # SMP 先验模块计算所得的模仿动作均值奖励
        self.writer.add_scalar("SMP/reward", locs["smp_mean_reward"], locs["it"])
        self.writer.add_scalar("SMP/reward_terms/task_raw", float(locs["smp_task_reward_raw"]), locs["it"])
        self.writer.add_scalar("SMP/reward_terms/task_scaled", float(locs["smp_task_reward_scaled"]), locs["it"])
        self.writer.add_scalar("SMP/reward_terms/style_raw", float(locs["smp_style_reward_raw"]), locs["it"])
        self.writer.add_scalar("SMP/reward_terms/style_scaled", float(locs["smp_style_reward_scaled"]), locs["it"])
        self.writer.add_scalar("SMP/reward_terms/combined", float(locs["smp_combined_reward"]), locs["it"])
        if locs.get("smp_reward_target") is not None:
            self.writer.add_scalar("SMP/diff/reward_target", float(locs["smp_reward_target"]), locs["it"])
        if locs.get("smp_reward_uncond") is not None:
            self.writer.add_scalar("SMP/diff/reward_uncond", float(locs["smp_reward_uncond"]), locs["it"])
        if locs.get("smp_reward_gap") is not None:
            self.writer.add_scalar("SMP/diff/reward_gap", float(locs["smp_reward_gap"]), locs["it"])
        if locs.get("smp_reward_final") is not None:
            self.writer.add_scalar("SMP/diff/reward_final", float(locs["smp_reward_final"]), locs["it"])
        # 扩散先验在各个采样时间步上预测噪声的总 MSE 误差均值
        self.writer.add_scalar("SMP/cfg/noise_mse", float(locs["smp_noise_mse"]), locs["it"])
        # CFG 引导机制下，有条件预测与无条件预测的差异量（体现了注入条件对动作的影响强度）
        self.writer.add_scalar("SMP/cfg/cond_uncond_gap", float(locs["smp_cfg_gap"]), locs["it"])
        # GSI重置阶段：生成的初始状态满足物理等约束要求、被直接接受的通过率
        self.writer.add_scalar("SMP/GSI/reset_accept_rate", float(locs.get("gsi_reset_accept_rate", 0.0)), locs["it"])
        # GSI重置阶段：为了找到有效状态平均需要的重新采样尝试次数
        self.writer.add_scalar("SMP/GSI/reset_resample_count", float(locs.get("gsi_reset_resample_count", 0.0)), locs["it"])
        # GSI重置阶段：受限于持续失败从而最终回退至预设静态默认姿态的比例
        self.writer.add_scalar("SMP/GSI/fallback_rate", float(locs.get("gsi_fallback_rate", 0.0)), locs["it"])

        mode = locs["style_program"]["mode"]
        mode_to_scalar = {"unconditional": -1.0, "single_style": 0.0, "body_mask": 1.0}
        # 当前日志记录周期使用的风格模式分类：无条件(-1.0)、全局单风格(0.0)、局部掩码组合(1.0)
        self.writer.add_scalar("SMP/style/mode", mode_to_scalar.get(mode, -2.0), locs["it"])
        if mode == "single_style":
            # 在单一风格模式下，配置中指定要求跟随的具体目标风格 ID
            self.writer.add_scalar("SMP/style/target_id", float(locs["style_program"]["target_style_id"]), locs["it"])
        elif mode == "body_mask":
            # 局部掩码模式下：分配给全身共享重叠区域部位的动作风格 ID
            self.writer.add_scalar(
                "SMP/style_program/shared_body_style_id",
                float(locs["style_program"]["part_style_ids"]["shared_body"]),
                locs["it"],
            )
            # 局部掩码模式下：分配给上半身对应的独立动作风格 ID
            self.writer.add_scalar(
                "SMP/style_program/upper_body_style_id",
                float(locs["style_program"]["part_style_ids"]["upper_body"]),
                locs["it"],
            )
            # 局部掩码模式下：分配给下半身对应的独立动作风格 ID
            self.writer.add_scalar(
                "SMP/style_program/lower_body_style_id",
                float(locs["style_program"]["part_style_ids"]["lower_body"]),
                locs["it"],
            )
            # 若缺失共享区域指定，标识其是否自动降级借用了其他部位(如下半身)的风格配置
            self.writer.add_scalar(
                "SMP/style_program/shared_body_defaulted",
                float(locs["style_program"].get("shared_body_defaulted", False)),
                locs["it"],
            )
            # 上下半身掩码叠加后，对整个机器人的关节动作特征维度占据的覆盖率
            self.writer.add_scalar("SMP/style_mask/coverage", float(locs["style_program"]["coverage"]), locs["it"])

        print(
            "SMP reward diagnostics:\n"
            f"{'Mean task reward raw:':>{pad}} {float(locs['smp_task_reward_raw']):.4f}\n"
            f"{'Mean task reward scaled:':>{pad}} {float(locs['smp_task_reward_scaled']):.4f}\n"
            f"{'Mean SMP reward raw:':>{pad}} {float(locs['smp_style_reward_raw']):.4f}\n"
            f"{'Mean SMP reward scaled:':>{pad}} {float(locs['smp_style_reward_scaled']):.4f}\n"
            f"{'Mean combined reward:':>{pad}} {float(locs['smp_combined_reward']):.4f}"
        )

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

        # 每轮迭代都常规上报噪声计算的总验证 MSE，维持监控曲线稳定
        self.writer.add_scalar("SMP/noise_mse", float(locs["smp_noise_mse"]), locs["it"])
        for timestep, mse in sorted(locs["smp_per_timestep_mse"].items()):
            # 将误差细分到每个特定的扩散采样时间步(timestep)，用于诊断模型对早晚期噪声的还原状况
            self.writer.add_scalar(f"SMP/t{timestep}/noise_mse", float(mse), locs["it"])

    def train_mode(self):
        """切换到训练模式，并保持 prior 始终处于 eval 状态。"""
        super().train_mode()
        self.smp_prior.eval()

    def eval_mode(self):
        """切换到评估模式，并保持 prior 始终处于 eval 状态。"""
        super().eval_mode()
        self.smp_prior.eval()

    def _load_prior_model(self) -> MotionEpsilonTransformer:
        """从 checkpoint 加载并冻结 diffusion prior，返回推理模式模型。"""
        checkpoint_path = self.smp_prior_cfg["checkpoint_path"]
        # prior checkpoint 同时携带模型权重和风格映射，训练时需要二者一起恢复，确保 style id 语义一致。
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.smp_checkpoint = checkpoint
        self.smp_checkpoint_style_cfg = checkpoint.get("style_cfg", {})
        model_cfg = checkpoint.get("model_cfg", {})
        feature_dim = int(model_cfg.get("feature_dim", self.smp_prior_cfg["feature_dim"]))
        window_size = int(model_cfg.get("window_size", self.smp_prior_cfg["window_size"]))
        num_diffusion_steps = int(model_cfg.get("num_diffusion_steps", self.smp_prior_cfg["num_diffusion_steps"]))
        self.smp_prior_feature_dim = feature_dim
        self.smp_prior_window_size = window_size
        self.smp_prior_feature_schema = str(model_cfg.get("feature_schema", _cfg_get(self.smp_prior_cfg, "feature_schema", "legacy_192")))
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
            # 冻结 prior，保证训练过程只更新 PPO policy，而不会破坏已学到的扩散先验。
            parameter.requires_grad_(False)
        return model

    def _resolve_style_name_to_id(self, style_name: str) -> int:
        """将风格名称映射为 checkpoint 中定义的整数 style id。"""
        style_to_id = self.smp_checkpoint_style_cfg.get("style_to_id", {})
        if style_name not in style_to_id:
            raise KeyError(f"Unknown style name '{style_name}' in prior checkpoint")
        return int(style_to_id[style_name])

    def _resolve_style_id_to_name(self, style_id: int) -> str | None:
        """根据 style id 反查风格名称，找不到时返回 None。"""
        style_to_id = self.smp_checkpoint_style_cfg.get("style_to_id", {})
        for style_name, candidate_id in style_to_id.items():
            if int(candidate_id) == int(style_id):
                return str(style_name)
        return None

    def _resolve_style_program_from_cfg(self) -> dict[str, object]:
        """解析 style 配置并生成统一的风格程序描述，用于 prior 推理与 GSI 复用。"""
        # 当 prior 不包含 style head 或用户没有提供风格配置时，直接退化为无条件 prior。
        if self.smp_prior.num_styles <= 0 or self.style_cfg is None:
            return {"mode": "unconditional", "guidance_scale": 1.0}

        mode = str(_cfg_get(self.style_cfg, "mode", "single_style"))
        guidance_scale = float(_cfg_get(self.style_cfg, "guidance_scale", 1.0))
        if mode == "single_style":
            # 单风格模式下，style id 可以直接来自配置，也可以先给出名字再回查 checkpoint 中的 id。
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
        # body_mask 模式要先把 feature block 的布局拆成不同身体部位，再把每个部位对应到一个风格 id。
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
        """按配置构建 GSI reset 采样器；条件不足时返回 None。"""
        if not bool(_cfg_get(self.gsi_cfg, "enabled", False)):
            return None
        feature_block_offsets = dict(_cfg_get(self.style_cfg, "feature_block_offsets", {}))
        if not feature_block_offsets:
            # 没有 feature block 布局就无法把 prior 的特征向量解码成可重采样的 reset state。
            return None
        feature_layout = SMPFeatureLayout.from_feature_block_offsets(feature_block_offsets)
        sampler = SMPDiffusionSampler(
            model=self.smp_prior,
            num_diffusion_steps=int(self.smp_prior_cfg["num_diffusion_steps"]),
            feature_dim=int(getattr(self, "smp_prior_feature_dim", self.smp_prior_cfg["feature_dim"])),
            window_size=int(getattr(self, "smp_prior_window_size", self.smp_prior_cfg["window_size"])),
            device=self.device,
        )
        decoder = SMPGSIDecoder(
            feature_layout=feature_layout,
            joint_axes=(
                torch.tensor(_cfg_get(self.style_cfg, "joint_axes", []), device=self.device, dtype=torch.float32)
                if _cfg_get(self.style_cfg, "joint_axes", None)
                else None
            ),
            error_threshold=float(_cfg_get(self.gsi_cfg, "error_threshold", 1.0e-6)),
        )
        return SMPGSISampler(sampler=sampler, decoder=decoder)

    def _move_obs_to_device(self, obs):
        """将观测（张量或字典）迁移到 runner 的训练设备。"""
        if hasattr(obs, "to"):
            return obs.to(self.device)
        if isinstance(obs, dict):
            return {key: value.to(self.device) if hasattr(value, "to") else value for key, value in obs.items()}
        return obs

    def _done_env_ids(self, dones: torch.Tensor) -> torch.Tensor:
        """将 done 张量归一化为一维环境索引。"""
        done_mask = dones > 0
        if done_mask.ndim > 1:
            done_mask = done_mask.view(done_mask.shape[0], -1).any(dim=1)
        return done_mask.nonzero(as_tuple=False).squeeze(-1)

    def _wrap_obs_like(self, obs_dict, obs_like):
        """尽量按输入观测的容器类型返回刷新后的观测。"""
        batch_size = getattr(obs_like, "batch_size", None)
        if batch_size is None:
            return obs_dict
        kwargs = {"batch_size": batch_size}
        device = getattr(obs_like, "device", None)
        if device is not None:
            kwargs["device"] = device
        try:
            return type(obs_like)(obs_dict, **kwargs)
        except Exception:
            return obs_dict

    def _compute_obs_term_value(self, env, term_cfg) -> torch.Tensor:
        """复用 ObservationManager 的单项观测后处理逻辑。"""
        obs = term_cfg.func(env, **term_cfg.params).clone()
        if term_cfg.modifiers is not None:
            for modifier in term_cfg.modifiers:
                obs = modifier.func(obs, **modifier.params)
        noise_cfg = term_cfg.noise
        noise_func = getattr(noise_cfg, "func", None)
        if noise_func is not None:
            try:
                obs = noise_func(obs, noise_cfg)
            except TypeError:
                obs = noise_func(obs)
        if term_cfg.clip:
            obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
        if term_cfg.scale is not None:
            obs = obs.mul_(term_cfg.scale)
        return obs

    def _refresh_reset_env_observations(self, obs_like, reset_env_ids: torch.Tensor):
        """在 GSI 改写 reset state 后，仅重建受影响环境的历史观测。"""
        target_env = getattr(self.env, "unwrapped", self.env)
        obs_manager = getattr(target_env, "observation_manager", None)
        if obs_manager is None or reset_env_ids.numel() == 0:
            return self._move_obs_to_device(self.env.get_observations())

        refreshed_obs = {}
        for group_name, group_term_names in obs_manager._group_obs_term_names.items():
            group_obs = dict.fromkeys(group_term_names, None)
            for term_name, term_cfg in zip(group_term_names, obs_manager._group_obs_term_cfgs[group_name]):
                term_obs = self._compute_obs_term_value(obs_manager._env, term_cfg)
                if term_cfg.history_length > 0:
                    circular_buffer = obs_manager._group_obs_term_history_buffer[group_name][term_name]
                    if circular_buffer._buffer is None:
                        repeat_dims = [1] * term_obs.ndim
                        circular_buffer._buffer = term_obs.unsqueeze(0).repeat(circular_buffer.max_length, *repeat_dims)
                        circular_buffer._pointer = circular_buffer.max_length - 1
                        circular_buffer._num_pushes[:] = 1
                    repeated_obs = term_obs[reset_env_ids].unsqueeze(0).expand(
                        circular_buffer.max_length, *term_obs[reset_env_ids].shape
                    )
                    circular_buffer._buffer[:, reset_env_ids] = repeated_obs
                    circular_buffer._num_pushes[reset_env_ids] = 1
                    if term_cfg.flatten_history_dim:
                        group_obs[term_name] = circular_buffer.buffer.reshape(target_env.num_envs, -1)
                    else:
                        group_obs[term_name] = circular_buffer.buffer
                else:
                    group_obs[term_name] = term_obs

            if obs_manager._group_obs_concatenate[group_name]:
                refreshed_obs[group_name] = torch.cat(
                    list(group_obs.values()), dim=obs_manager._group_obs_concatenate_dim[group_name]
                )
            else:
                refreshed_obs[group_name] = group_obs

        return self._move_obs_to_device(self._wrap_obs_like(refreshed_obs, obs_like))

    def _build_smp_reward_obs(self, obs, dones, extras):
        """为 SMP reward 构造终止态修正后的观测窗口。"""
        if self.smp_obs_group not in obs:
            return obs

        reward_window = obs[self.smp_obs_group].clone()
        reset_env_ids = self._done_env_ids(dones)
        if reset_env_ids.numel() == 0:
            return {self.smp_obs_group: reward_window}

        terminal_obs = extras.get("terminal_observation")
        if terminal_obs is None:
            return {self.smp_obs_group: reward_window}

        try:
            terminal_window = terminal_obs[self.smp_obs_group]
        except Exception:
            return {self.smp_obs_group: reward_window}

        reward_window[reset_env_ids] = terminal_window.to(reward_window.device)
        return {self.smp_obs_group: reward_window}

    def _maybe_apply_gsi_reset(self, obs, dones):
        """在环境 reset 时按需执行 GSI 重采样，并返回更新后的观测与诊断指标。"""
        default_diag = {"reset_accept_rate": 0.0, "reset_resample_count": 0.0, "fallback_rate": 0.0}
        if self.gsi_sampler is None or not bool(_cfg_get(self.gsi_cfg, "sample_on_reset", True)):
            return obs, default_diag

        # 只对刚刚结束 episode 的环境执行 reset 重采样，避免无关环境被误改写。
        reset_env_ids = self._done_env_ids(dones)
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
            # 每次尝试都以同一个 reference state 为起点，只有采样噪声和 prior 条件发生变化。
            last_result = self.gsi_sampler.sample_reset_state(
                batch_size=int(reset_env_ids.numel()),
                reference_state=reference_state,
                style_program=self.style_program,
                guidance_scale=float(guidance_scale),
            )
            if last_result.supports_reset_state:
                # 只有采样结果满足环境 reset 约束时，才真正写回仿真环境。
                apply_smp_reset_state(self.env, env_ids=reset_env_ids, state=last_result.state, asset_name=asset_name)
                refreshed_obs = self._refresh_reset_env_observations(obs, reset_env_ids)
                return refreshed_obs, {
                    "reset_accept_rate": 1.0,
                    "reset_resample_count": float(attempt),
                    "fallback_rate": 0.0,
                }

        if last_result is not None and not bool(_cfg_get(self.gsi_cfg, "fallback_to_default_reset", True)):
            apply_smp_reset_state(self.env, env_ids=reset_env_ids, state=last_result.state, asset_name=asset_name)
            refreshed_obs = self._refresh_reset_env_observations(obs, reset_env_ids)
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
        """生成长度为 batch_size 的 style id 张量，供 prior 批量前向使用。"""
        return torch.full((batch_size,), int(style_id), device=self.device, dtype=torch.long)

    def _predict_prior_eps(self, xt: torch.Tensor, t: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        """在给定 timestep 预测扩散噪声，并返回风格相关诊断信息。"""
        # 无条件模式最直接，prior 只依赖当前噪声状态和 timestep。
        if self.style_program["mode"] == "unconditional":
            eps = self.smp_prior(xt, t)
            return {"policy": eps, "target": eps, "uncond": eps}, {"cond_uncond_gap": 0.0}

        # classifier-free guidance 的常规做法是同时计算 uncond 和 cond，然后按 scale 融合。
        eps_uncond = self.smp_prior(xt, t, style_id=self._full_style_id(xt.shape[0], NULL_STYLE_ID))
        if self.style_program["mode"] == "single_style":
            eps_cond = self.smp_prior(xt, t, style_id=self._full_style_id(xt.shape[0], self.style_program["target_style_id"]))
            eps_prior = apply_classifier_free_guidance(
                eps_uncond,
                eps_cond,
                float(self.style_program["guidance_scale"]),
            )
            return {"policy": eps_prior, "target": eps_cond, "uncond": eps_uncond}, {
                "cond_uncond_gap": float((eps_cond - eps_uncond).abs().mean().item()),
                "target_style_id": int(self.style_program["target_style_id"]),
            }

        part_to_eps = {}
        eps_cache: dict[int, torch.Tensor] = {}
        for part_name, style_id in self.style_program["part_style_ids"].items():
            # 同一个 style id 可能被多个身体部位复用，因此用缓存避免重复前向。
            if style_id not in eps_cache:
                eps_cache[style_id] = self.smp_prior(xt, t, style_id=self._full_style_id(xt.shape[0], style_id))
            part_to_eps[part_name] = eps_cache[style_id]
        # 将不同身体部位的预测按 mask 拼接，得到一个完整的 conditional prediction。
        eps_cond_comp = compose_style_predictions_with_body_masks(part_to_eps, self.style_program["feature_masks"])
        guidance_scale = float(self.style_program["guidance_scale"])
        eps_prior = eps_cond_comp if guidance_scale == 1.0 else apply_classifier_free_guidance(
            eps_uncond,
            eps_cond_comp,
            guidance_scale,
        )
        return {"policy": eps_prior, "target": eps_cond_comp, "uncond": eps_uncond}, {
            "cond_uncond_gap": float((eps_cond_comp - eps_uncond).abs().mean().item()),
            "part_style_ids": dict(self.style_program["part_style_ids"]),
            "coverage": float(self.style_program["coverage"]),
        }

    def _restore_smp_window(self, obs) -> torch.Tensor:
        """从观测中恢复 SMP 时间窗张量，并校验展平维度是否匹配配置。"""
        if self.smp_obs_group not in obs:
            raise KeyError(f"SMP observation group '{self.smp_obs_group}' not found in observations")
        smp_window = obs[self.smp_obs_group]
        feature_dim = int(getattr(self, "smp_prior_feature_dim", self.smp_prior_cfg["feature_dim"]))
        window_size = int(getattr(self, "smp_prior_window_size", self.smp_prior_cfg["window_size"]))
        if smp_window.shape[-1] != window_size * feature_dim:
            raise ValueError(
                f"Expected flattened SMP dim {window_size * feature_dim}, got {smp_window.shape[-1]}"
            )
        return smp_window.view(smp_window.shape[0], window_size, feature_dim)

    def _compute_smp_metrics(self, obs) -> dict[str, object]:
        """计算 SMP 噪声重建指标与奖励，供 rollout 阶段融合到总奖励中。"""
        x0 = self._restore_smp_window(obs)
        eps = {}
        eps_hat = {}
        eps_hat_uncond = {} if self.smp_reward.reward_mode == "target_vs_uncond" else None
        cond_uncond_gaps = []
        style_diag = {}
        with torch.inference_mode():
            # 对多个固定 timestep 采样噪声并估计 prior 误差，用于监控 prior 是否仍然稳定。
            for timestep in self.smp_reward.timesteps_k:
                t = torch.full((x0.shape[0],), timestep, device=self.device, dtype=torch.long)
                eps_t = torch.randn_like(x0)
                xt = self.smp_scheduler.q_sample(x0, t, eps_t)
                eps_outputs, style_diag = self._predict_prior_eps(xt, t)
                eps[timestep] = eps_t
                if self.smp_reward.reward_mode == "target_vs_uncond":
                    eps_hat[timestep] = eps_outputs["target"]
                    assert eps_hat_uncond is not None
                    eps_hat_uncond[timestep] = eps_outputs["uncond"]
                else:
                    eps_hat[timestep] = eps_outputs["policy"]
                cond_uncond_gaps.append(float(style_diag.get("cond_uncond_gap", 0.0)))
        metrics = self.smp_reward.compute(eps=eps, eps_hat=eps_hat, eps_hat_uncond=eps_hat_uncond)
        metrics["eps"] = eps
        metrics["eps_hat"] = eps_hat
        if eps_hat_uncond is not None:
            metrics["eps_hat_uncond"] = eps_hat_uncond
        metrics["style_diag"] = style_diag
        metrics["cond_uncond_gap"] = statistics.mean(cond_uncond_gaps) if cond_uncond_gaps else 0.0
        return metrics

    def _decompose_rewards(self, task_rewards: torch.Tensor, smp_rewards: torch.Tensor) -> dict[str, torch.Tensor]:
        """拆分任务奖励、SMP 奖励及其加权后的组合结果，供训练与日志共用。"""
        # 保持 reward 形状与环境输出一致，只在必要时扩展维度后再做加权求和。
        if task_rewards.ndim == 2 and smp_rewards.ndim == 1:
            smp_rewards = smp_rewards.unsqueeze(-1)
        dt = self.env.unwrapped.step_dt
        task_scaled = self.task_reward_coef * task_rewards
        smp_scaled = self.smp_reward_coef * smp_rewards * dt
        return {
            "task_raw": task_rewards,
            "task_scaled": task_scaled,
            "smp_raw": smp_rewards,
            "smp_scaled": smp_scaled,
            "combined": task_scaled + smp_scaled,
        }

    def _combine_rewards(self, task_rewards: torch.Tensor, smp_rewards: torch.Tensor) -> torch.Tensor:
        """按配置系数融合任务奖励与 SMP 奖励，并保持输出形状一致。"""
        return self._decompose_rewards(task_rewards, smp_rewards)["combined"]
