from __future__ import annotations

import torch


class SMPReward:
    """根据扩散模型的噪声预测误差，计算每个样本的 SMP 奖励。

    可以把它理解成两种模式:
    1. absolute:
       - 在多个固定时间步上算预测噪声和真实噪声的 MSE。
       - 每个时间步单独做运行均值（EMA）归一化。
       - 再把时间步误差求平均，映射成奖励: r = exp(-scale * mse)。
    2. target_vs_uncond:
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
        adaptive_norm_decay: float = 0.99,
        fixed_norm_mse: dict[int, float] | None = None,
    ):
        # 参数提前校验，尽量把错误暴露在初始化阶段。
        if num_diffusion_steps <= 0:
            raise ValueError(f"num_diffusion_steps must be positive, got {num_diffusion_steps}")
        if len(timesteps_k) == 0:
            raise ValueError("timesteps_k must be non-empty")
        if reward_mode not in {"absolute", "target_vs_uncond"}:
            raise ValueError(f"reward_mode must be 'absolute' or 'target_vs_uncond', got {reward_mode}")
        if not 0.0 < adaptive_norm_decay <= 1.0:
            raise ValueError(f"adaptive_norm_decay must lie in (0, 1], got {adaptive_norm_decay}")
        if fixed_norm_mse is not None and reward_mode != "absolute":
            raise ValueError("fixed_norm_mse is only supported when reward_mode='absolute'")

        normalized_fixed_mse = None
        if fixed_norm_mse is not None:
            normalized_fixed_mse = {int(timestep): float(value) for timestep, value in fixed_norm_mse.items()}
            missing_timesteps = [timestep for timestep in timesteps_k if int(timestep) not in normalized_fixed_mse]
            if missing_timesteps:
                raise ValueError(f"Missing fixed_norm_mse anchors for timesteps: {missing_timesteps}")
            invalid_timesteps = [
                int(timestep)
                for timestep, value in normalized_fixed_mse.items()
                if float(value) <= 0.0
            ]
            if invalid_timesteps:
                raise ValueError(f"fixed_norm_mse must be positive for all timesteps, got {invalid_timesteps}")

        # 全局配置。
        self.num_diffusion_steps = num_diffusion_steps
        self.timesteps_k = [int(timestep) for timestep in timesteps_k]
        self.reward_scale = reward_scale
        self.reward_mode = reward_mode
        self.adaptive_norm_decay = adaptive_norm_decay
        self.fixed_norm_mse = normalized_fixed_mse

        # 为每个时间步维护一个 EMA 均值，用作该时间步的误差“尺度”。
        # running_mse[t] 是单个标量，不是每个样本各存一份。
        self.running_mse = {
            timestep: (torch.tensor(float(self.fixed_norm_mse[timestep]), dtype=torch.float32) if self.fixed_norm_mse is not None else None)
            for timestep in self.timesteps_k
        }

    def _normalize(self, timestep: int, mse: torch.Tensor) -> torch.Tensor:
        """把某个时间步的逐样本 MSE 转成可跨时间步比较的量。

        参数:
        - timestep: 当前扩散时间步
        - mse: 逐样本 MSE，shape 为 (B,)

        返回:
        - 归一化后的逐样本误差，shape 为 (B,)
        """
        if self.fixed_norm_mse is not None:
            anchor_value = float(self.fixed_norm_mse[timestep])
            self.running_mse[timestep] = torch.tensor(anchor_value, dtype=torch.float32)
            return mse / mse.new_tensor(anchor_value).clamp_min(1.0e-6)

        # 用 EMA 估计“这个时间步通常有多大误差”。
        mse_mean = mse.detach().mean()
        running_mse = self.running_mse[timestep]
        if running_mse is None:
            # 首次出现该时间步时，直接用当前 batch 均值初始化。
            running_mse = mse_mean
        else:
            # EMA 更新公式: s <- decay * s + (1 - decay) * new_mean
            running_mse = running_mse * self.adaptive_norm_decay + mse_mean * (1.0 - self.adaptive_norm_decay)
        self.running_mse[timestep] = running_mse

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
