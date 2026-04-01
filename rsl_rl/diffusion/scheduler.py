from __future__ import annotations

import torch


class DiffusionScheduler:
    """封装前向扩散过程中的噪声日程与加噪计算。

    该类负责两件事:
    1) 构建离散时间步上的 beta/alpha/alpha_bar 日程。
    2) 提供 q(x_t | x_0) 的闭式采样: 给定 x0、t、eps 直接得到 x_t。

    记号约定:
    - beta_t: 第 t 步注入噪声强度
    - alpha_t = 1 - beta_t
    - alpha_bar_t = \prod_{i=0}^{t} alpha_i
    """

    def __init__(self, num_steps: int = 50, beta_start: float = 1.0e-4, beta_end: float = 2.0e-2):
        # 日程参数校验。这里采用线性 beta schedule，要求范围合法且递增。
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if beta_start <= 0.0 or beta_end <= 0.0:
            raise ValueError("beta_start and beta_end must be positive")
        if beta_end <= beta_start:
            raise ValueError("beta_end must be greater than beta_start")

        self.num_steps = num_steps
        # 线性 beta 日程：每一步注入的噪声强度。
        self.beta = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        # alpha_t = 1 - beta_t。
        self.alpha = 1.0 - self.beta
        # alpha_bar_t = ∏_{i<=t} alpha_i。
        # 在 DDPM 闭式公式中，x_t 可写为 x0 与标准高斯噪声 eps 的线性组合。
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        timesteps_k: list[int] | tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """按给定候选集合或全扩散步均匀采样时间步索引。

        参数:
        - batch_size: 返回的时间步数量（通常等于训练 batch 大小）
        - device: 输出张量设备
        - timesteps_k: 可选的候选时间步集合；若为空则在 [0, num_steps) 全范围采样

        返回:
        - shape 为 (batch_size,) 的 long 张量
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if timesteps_k is None:
            # 全时间步均匀采样：最常见的扩散训练策略。
            return torch.randint(0, self.num_steps, (batch_size,), device=device, dtype=torch.long)

        # 从固定候选时间步中采样，常用于训练时的稀疏时间步策略。
        choices = torch.as_tensor(timesteps_k, device=device, dtype=torch.long)
        if choices.ndim != 1 or choices.numel() == 0:
            raise ValueError("timesteps_k must be a non-empty 1D sequence")
        if torch.any(choices < 0) or torch.any(choices >= self.num_steps):
            raise ValueError(f"timesteps_k must stay within [0, {self.num_steps - 1}]")
        # 先在候选集合索引上采样，再映射回真实时间步值。
        choice_ids = torch.randint(0, choices.numel(), (batch_size,), device=device)
        return choices[choice_ids]

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """根据闭式公式生成加噪后的 x_t。

        公式:
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        其中 eps 通常来自 N(0, I)，与 x0 形状一致。
        """
        if x0.shape != eps.shape:
            raise ValueError(f"x0 and eps must share the same shape, got {x0.shape} and {eps.shape}")
        if t.ndim != 1 or t.shape[0] != x0.shape[0]:
            raise ValueError(f"t must have shape ({x0.shape[0]},), got {t.shape}")

        # 为每个样本取对应 alpha_bar_t，并转换到 x0 的 device/dtype。
        alpha_bar = self.alpha_bar.to(device=x0.device, dtype=x0.dtype)[t]
        # 将 (B,) 重塑为 (B,1,...,1) 以匹配任意维度输入的广播计算。
        alpha_bar = alpha_bar.view(-1, *([1] * (x0.ndim - 1)))
        # 返回与 x0 同形状的 x_t。
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * eps
