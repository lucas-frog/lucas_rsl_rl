from __future__ import annotations

import torch


class SMPReward:
    """基于固定扩散时间步集合计算 SMP 风格噪声一致性奖励。

    核心思路:
    1) 在多个扩散时间步 t 上计算噪声预测误差 MSE。
    2) 通过每个时间步各自的运行均值做自适应归一化，缓解不同 t 的尺度差异。
    3) 跨时间步聚合后映射为指数奖励: r = exp(-scale * mse)。

    记号:
    - eps[t]: 真实噪声，shape 通常为 (B, ...)
    - eps_hat[t]: 预测噪声，shape 与 eps[t] 相同
    - 输出 reward: shape 为 (B,)
    """

    def __init__(
        self,
        num_diffusion_steps: int,
        timesteps_k: list[int] | tuple[int, ...],
        reward_scale: float,
        adaptive_norm_decay: float = 0.99,
    ):
        # 基础参数校验，避免运行时出现非法时间步或退化归一化。
        if num_diffusion_steps <= 0:
            raise ValueError(f"num_diffusion_steps must be positive, got {num_diffusion_steps}")
        if len(timesteps_k) == 0:
            raise ValueError("timesteps_k must be non-empty")
        if not 0.0 < adaptive_norm_decay <= 1.0:
            raise ValueError(f"adaptive_norm_decay must lie in (0, 1], got {adaptive_norm_decay}")

        # 全局配置。
        self.num_diffusion_steps = num_diffusion_steps
        self.timesteps_k = [int(timestep) for timestep in timesteps_k]
        self.reward_scale = reward_scale
        self.adaptive_norm_decay = adaptive_norm_decay

        # 为每个时间步维护一个运行中的 MSE 均值，用于自适应归一化。
        # running_mse[t] 是标量张量（EMA 统计量），不是按样本存储。
        self.running_mse = {timestep: None for timestep in self.timesteps_k}

    def _normalize(self, timestep: int, mse: torch.Tensor) -> torch.Tensor:
        """将某一时间步的逐样本误差按其运行尺度归一化。

        参数:
        - timestep: 当前扩散时间步
        - mse: 逐样本 MSE，shape 为 (B,)

        返回:
        - 归一化后的逐样本误差，shape 为 (B,)
        """
        # 使用指数滑动平均估计该时间步的误差尺度，降低不同时间步量纲差异。
        mse_mean = mse.detach().mean()
        running_mse = self.running_mse[timestep]
        if running_mse is None:
            # 首次出现该时间步时，直接用当前 batch 均值初始化。
            running_mse = mse_mean
        else:
            # EMA 更新: s <- decay * s + (1 - decay) * new_mean
            running_mse = running_mse * self.adaptive_norm_decay + mse_mean * (1.0 - self.adaptive_norm_decay)
        self.running_mse[timestep] = running_mse

        # 归一化后可在不同时间步之间更稳定地做聚合。
        # clamp_min 防止极小分母导致数值爆炸。
        return mse / running_mse.clamp_min(1.0e-6)

    def compute(self, eps: dict[int, torch.Tensor], eps_hat: dict[int, torch.Tensor]) -> dict[str, object]:
        """计算逐时间步误差，并聚合为指数形式奖励。

        参数:
        - eps: 真实噪声字典，key 为时间步 t
        - eps_hat: 预测噪声字典，key 为时间步 t

        返回:
        - reward: shape (B,)，数值区间 (0, 1]
        - noise_mse: shape (B,)，跨时间步聚合后的归一化误差
        - per_timestep_mse: dict[t, Tensor(B,)]，每个 t 的逐样本 MSE
        """
        per_timestep_mse = {}
        normalized_terms = []

        # 遍历预定义时间步集合，分别统计并归一化误差。
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
            normalized_terms.append(self._normalize(timestep, mse))

        # 先跨时间步平均，再通过 exp(-scale * mse) 转成 [0, 1] 区间奖励。
        # noise_mse: (B,) ; reward: (B,)
        noise_mse = torch.stack(normalized_terms, dim=0).mean(dim=0)
        reward = torch.exp(-self.reward_scale * noise_mse)
        return {
            "reward": reward,
            "noise_mse": noise_mse,
            "per_timestep_mse": per_timestep_mse,
        }
