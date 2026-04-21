from __future__ import annotations

import torch


class SMPReward:
    """根据扩散模型的噪声预测误差，计算每个样本的 SMP 奖励。

    可以把它理解成两种模式:
    1. absolute:
       - 在多个固定时间步上算预测噪声和真实噪声的 MSE。
       - 每个时间步单独做运行均值（EMA）归一化。
       - 再把时间步误差求平均，映射成奖励: r = exp(-scale * mse)。
    2. fixed_normalizer:
       - 使用离线估计好的每个时间步参考 MSE 作为固定归一化尺度。
       - 在线 PPO 阶段不更新这些尺度，避免 reward drift。
       - 再把时间步误差求平均，映射成奖励: r = exp(-scale * mse)。
    3. zscore:
       - 为每个时间步维护 MSE 的运行均值和方差。
       - 用 z-score 保留 batch 内动作好坏的相对差异。
       - 再用 sigmoid 把奖励限制在 [0, 1]，历史均值附近约为 0.8。
    4. target_vs_uncond:
       - 分别计算 target 条件预测与 unconditional 预测的逐样本噪声误差。
       - 不做归一化，先各自映射成 reward_target / reward_uncond。
       - 再把“相对 unconditional 的提升比例”映射到 [0, 1]:
         r = clamp((reward_target - reward_uncond) / (1 - reward_uncond), 0, 1)。

    输入约定:
    - eps[t]: 真实噪声，形状通常是 (B, ...)
    - eps_hat[t]: 预测噪声，形状与 eps[t] 相同
    - reward: 输出奖励，形状为 (B,)
    """

    def __init__(
        self,
        num_diffusion_steps: int,
        timesteps_k: list[int] | tuple[int, ...],
        reward_scale: float,
        reward_mode: str = "absolute",
        adaptive_norm_decay: float = 0.999,
        zscore_reward_center: float = 0.8,
        zscore_std_floor: float = 1.0e-6,
        fixed_normalizer_mse_by_timestep: dict[int, float] | None = None,
    ):
        # 参数提前校验，尽量把错误暴露在初始化阶段。
        if num_diffusion_steps <= 0:
            raise ValueError(f"num_diffusion_steps must be positive, got {num_diffusion_steps}")
        if len(timesteps_k) == 0:
            raise ValueError("timesteps_k must be non-empty")
        if reward_mode not in {"absolute", "fixed_normalizer", "zscore", "target_vs_uncond"}:
            raise ValueError(
                "reward_mode must be 'absolute', 'fixed_normalizer', 'zscore', or 'target_vs_uncond', "
                f"got {reward_mode}"
            )
        if not 0.0 < adaptive_norm_decay <= 1.0:
            raise ValueError(f"adaptive_norm_decay must lie in (0, 1], got {adaptive_norm_decay}")
        if not 0.0 < zscore_reward_center < 1.0:
            raise ValueError(f"zscore_reward_center must lie in (0, 1), got {zscore_reward_center}")
        if zscore_std_floor <= 0.0:
            raise ValueError(f"zscore_std_floor must be positive, got {zscore_std_floor}")
        if reward_mode == "fixed_normalizer":
            if not fixed_normalizer_mse_by_timestep:
                raise ValueError("reward_mode='fixed_normalizer' requires fixed_normalizer_mse_by_timestep")
            missing_timesteps = [
                int(timestep) for timestep in timesteps_k if int(timestep) not in fixed_normalizer_mse_by_timestep
            ]
            if missing_timesteps:
                raise ValueError(
                    "fixed_normalizer_mse_by_timestep is missing timesteps "
                    f"{missing_timesteps} required by timesteps_k={list(timesteps_k)}"
                )

        # 全局配置。
        self.num_diffusion_steps = num_diffusion_steps
        self.timesteps_k = [int(timestep) for timestep in timesteps_k]
        self.reward_scale = reward_scale
        self.reward_mode = reward_mode
        self.adaptive_norm_decay = adaptive_norm_decay
        self.zscore_reward_center = zscore_reward_center
        self.zscore_std_floor = zscore_std_floor
        self.zscore_reward_bias = torch.logit(torch.tensor(float(zscore_reward_center)))
        self.fixed_normalizer_mse_by_timestep = (
            None
            if fixed_normalizer_mse_by_timestep is None
            else {int(timestep): float(value) for timestep, value in fixed_normalizer_mse_by_timestep.items()}
        )

        # 为每个时间步维护一个 EMA 均值，用作该时间步的误差“尺度”。
        # running_mse[t] 是单个标量，不是每个样本各存一份。
        self.running_mse = {timestep: None for timestep in self.timesteps_k}
        self.running_mse_var = {timestep: None for timestep in self.timesteps_k}

    def _update_running_mean(self, timestep: int, mse: torch.Tensor) -> torch.Tensor:
        """更新并返回某个时间步的 MSE 运行均值。"""
        mse_mean = mse.detach().mean()
        running_mse = self.running_mse[timestep]
        if running_mse is None:
            running_mse = mse_mean
        else:
            running_mse = running_mse * self.adaptive_norm_decay + mse_mean * (1.0 - self.adaptive_norm_decay)
        self.running_mse[timestep] = running_mse
        return running_mse

    def _update_running_stats(self, timestep: int, mse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """更新并返回某个时间步的 MSE 运行均值和标准差。"""
        mse_detached = mse.detach()
        mse_mean = mse_detached.mean()
        mse_var = mse_detached.var(unbiased=False)
        running_mse = self.running_mse[timestep]
        running_mse_var = self.running_mse_var[timestep]
        if running_mse is None or running_mse_var is None:
            running_mse = mse_mean
            running_mse_var = mse_var
        else:
            running_mse = running_mse * self.adaptive_norm_decay + mse_mean * (1.0 - self.adaptive_norm_decay)
            running_mse_var = running_mse_var * self.adaptive_norm_decay + mse_var * (1.0 - self.adaptive_norm_decay)
        self.running_mse[timestep] = running_mse
        self.running_mse_var[timestep] = running_mse_var
        running_mse_std = running_mse_var.clamp_min(self.zscore_std_floor * self.zscore_std_floor).sqrt()
        return running_mse, running_mse_std

    def _normalize(self, timestep: int, mse: torch.Tensor) -> torch.Tensor:
        """把某个时间步的逐样本 MSE 转成可跨时间步比较的量。

        参数:
        - timestep: 当前扩散时间步
        - mse: 逐样本 MSE，shape 为 (B,)

        返回:
        - 归一化后的逐样本误差，shape 为 (B,)
        """
        # 按论文式 absolute reward 的语义，当前 batch 先用旧的运行均值做评分，
        # 再把当前 batch 的统计量写回 EMA，避免“当前 batch 参与定义自己的基准”。
        running_mse = self.running_mse[timestep]
        if running_mse is None:
            running_mse = mse.detach().mean()

        self._update_running_mean(timestep, mse)

        # clamp_min 防止分母太小，导致归一化值异常放大。
        return mse / running_mse.clamp_min(1.0e-6)

    def compute(
        self,
        eps: dict[int, torch.Tensor],
        eps_hat: dict[int, torch.Tensor],
        eps_hat_uncond: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, object]:
        """计算奖励主流程: 按 reward_mode 计算噪声误差并聚合成 reward。

        参数:
        - eps: 真实噪声字典，key 为时间步 t
        - eps_hat: 预测噪声字典，key 为时间步 t

        返回:
        - reward: 每个样本一个奖励，形状 (B,)
        - noise_mse: 聚合后的归一化误差，形状 (B,)
        - per_timestep_mse: 每个时间步的逐样本 MSE
        """
        per_timestep_mse = {}
        normalized_terms = []
        fixed_normalized_terms = []
        zscore_terms = []
        per_timestep_mse_uncond = {}

        # 对每个预设时间步分别计算误差并归一化。
        for timestep in self.timesteps_k:
            if timestep not in eps or timestep not in eps_hat:
                raise KeyError(f"Missing timestep {timestep} in eps or eps_hat")
            if eps[timestep].shape != eps_hat[timestep].shape:
                raise ValueError(
                    f"eps[{timestep}] and eps_hat[{timestep}] must share the same shape, "
                    f"got {eps[timestep].shape} and {eps_hat[timestep].shape}"
                )

            # 对每个样本计算该时间步上的噪声 MSE。
            # 输入可能是 (B, W, F) 或其他高维结构，统一展平后按样本求均值。
            mse = (eps_hat[timestep] - eps[timestep]).pow(2).flatten(start_dim=1).mean(dim=1)
            per_timestep_mse[timestep] = mse
            if self.reward_mode == "absolute":
                normalized_terms.append(self._normalize(timestep, mse))
            if self.reward_mode == "fixed_normalizer":
                assert self.fixed_normalizer_mse_by_timestep is not None
                fixed_scale = torch.as_tensor(
                    self.fixed_normalizer_mse_by_timestep[int(timestep)],
                    device=mse.device,
                    dtype=mse.dtype,
                )
                fixed_normalized_terms.append(mse / fixed_scale.clamp_min(1.0e-6))
            if self.reward_mode == "zscore":
                running_mse, running_mse_std = self._update_running_stats(timestep, mse)
                zscore_terms.append((mse - running_mse) / running_mse_std)

            if self.reward_mode == "target_vs_uncond":
                if eps_hat_uncond is None or timestep not in eps_hat_uncond:
                    raise KeyError(f"Missing timestep {timestep} in eps_hat_uncond for reward_mode='target_vs_uncond'")
                if eps[timestep].shape != eps_hat_uncond[timestep].shape:
                    raise ValueError(
                        f"eps[{timestep}] and eps_hat_uncond[{timestep}] must share the same shape, "
                        f"got {eps[timestep].shape} and {eps_hat_uncond[timestep].shape}"
                    )
                mse_uncond = (eps_hat_uncond[timestep] - eps[timestep]).pow(2).flatten(start_dim=1).mean(dim=1)
                per_timestep_mse_uncond[timestep] = mse_uncond

        if self.reward_mode == "absolute":
            # 跨时间步求平均后，用指数映射为奖励。
            # noise_mse 和 reward 的形状都为 (B,)。
            noise_mse = torch.stack(normalized_terms, dim=0).mean(dim=0)
            reward = torch.exp(-self.reward_scale * noise_mse)
            return {
                "reward": reward,
                "noise_mse": noise_mse,
                "per_timestep_mse": per_timestep_mse,
            }

        if self.reward_mode == "fixed_normalizer":
            noise_mse = torch.stack(fixed_normalized_terms, dim=0).mean(dim=0)
            reward = torch.exp(-self.reward_scale * noise_mse)
            return {
                "reward": reward,
                "noise_mse": noise_mse,
                "per_timestep_mse": per_timestep_mse,
            }

        if self.reward_mode == "zscore":
            noise_mse = torch.stack([per_timestep_mse[timestep] for timestep in self.timesteps_k], dim=0).mean(dim=0)
            noise_z = torch.stack(zscore_terms, dim=0).mean(dim=0)
            reward_bias = self.zscore_reward_bias.to(device=noise_z.device, dtype=noise_z.dtype)
            reward = torch.sigmoid(reward_bias - self.reward_scale * noise_z)
            return {
                "reward": reward,
                "noise_z": noise_z,
                "noise_mse": noise_mse,
                "per_timestep_mse": per_timestep_mse,
            }

        noise_mse = torch.stack([per_timestep_mse[timestep] for timestep in self.timesteps_k], dim=0).mean(dim=0)
        noise_mse_uncond = torch.stack(
            [per_timestep_mse_uncond[timestep] for timestep in self.timesteps_k], dim=0
        ).mean(dim=0)
        reward_target = torch.exp(-self.reward_scale * noise_mse)
        reward_uncond = torch.exp(-self.reward_scale * noise_mse_uncond)
        reward_gap = reward_target - reward_uncond
        reward = (reward_gap / (1.0 - reward_uncond).clamp_min(1.0e-6)).clamp(0.0, 1.0)
        return {
            "reward": reward,
            "noise_mse": noise_mse,
            "per_timestep_mse": per_timestep_mse,
            "noise_mse_uncond": noise_mse_uncond,
            "per_timestep_mse_uncond": per_timestep_mse_uncond,
            "reward_target": reward_target,
            "reward_uncond": reward_uncond,
            "reward_gap": reward_gap,
        }
