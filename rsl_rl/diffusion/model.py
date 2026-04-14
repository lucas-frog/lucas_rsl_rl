from __future__ import annotations

import math

import torch
import torch.nn as nn

from rsl_rl.diffusion.conditioning import StyleConditioner


class SinusoidalTimestepEmbedding(nn.Module):
    """将离散扩散时间步映射为正弦嵌入。"""

    def __init__(self, embedding_dim: int):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"Expected t to have shape (batch,), got {tuple(t.shape)}")

        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            return t.to(dtype=torch.float32).unsqueeze(-1)

        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=torch.float32) * exponent
        )
        angles = t.to(dtype=torch.float32).unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if self.embedding_dim % 2 == 1:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
        return embedding


class AdaLayerNorm(nn.Module):
    """使用条件向量调制 LayerNorm 的 scale/shift。"""

    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, hidden_dim * 2),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[0] != x.shape[0]:
            raise ValueError(
                f"Expected condition shape ({x.shape[0]}, D), got {tuple(condition.shape)}"
            )
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaTransformerEncoderBlock(nn.Module):
    """通过 AdaLN 注入扩散步与风格条件的 Transformer 编码块。"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = AdaLayerNorm(hidden_dim=hidden_dim, condition_dim=hidden_dim)
        self.ff_norm = AdaLayerNorm(hidden_dim=hidden_dim, condition_dim=hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(x, condition)
        attn_output, _ = self.self_attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.dropout(attn_output)

        ff_input = self.ff_norm(x, condition)
        x = x + self.dropout(self.ff(ff_input))
        return x


class MotionEpsilonTransformer(nn.Module):
    """基于 Transformer 的 epsilon 预测网络，用于扩散模型去噪。"""

    def __init__(
        self,
        feature_dim: int,
        window_size: int,
        num_diffusion_steps: int,
        num_styles: int = 0,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if num_diffusion_steps <= 0:
            raise ValueError(f"num_diffusion_steps must be positive, got {num_diffusion_steps}")
        if num_styles < 0:
            raise ValueError(f"num_styles must be non-negative, got {num_styles}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.feature_dim = feature_dim
        self.window_size = window_size
        self.num_diffusion_steps = num_diffusion_steps
        self.num_styles = num_styles
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.token_proj = nn.Linear(feature_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, window_size, hidden_dim))
        self.timestep_embedding = SinusoidalTimestepEmbedding(hidden_dim)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.style_conditioner = StyleConditioner(num_styles=num_styles, hidden_dim=hidden_dim) if num_styles > 0 else None
        self.blocks = nn.ModuleList(
            [
                AdaTransformerEncoderBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, feature_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, style_id: torch.Tensor | None = None) -> torch.Tensor:
        if xt.ndim != 3:
            raise ValueError(f"Expected xt shape (batch, window, feature), got {xt.shape}")
        if xt.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature dim {self.feature_dim}, got {xt.shape[-1]}")
        if xt.shape[1] > self.window_size:
            raise ValueError(f"Expected window size <= {self.window_size}, got {xt.shape[1]}")
        if t.ndim != 1 or t.shape[0] != xt.shape[0]:
            raise ValueError(f"Expected t shape ({xt.shape[0]},), got {t.shape}")
        if self.style_conditioner is None and style_id is not None:
            raise ValueError("style_id is provided but the model was created without style conditioning")

        t = t.to(device=xt.device, dtype=torch.long)

        hidden = self.token_proj(xt)
        hidden = hidden + self.pos_embedding[:, : xt.shape[1]]

        condition = self.timestep_mlp(self.timestep_embedding(t))
        if self.style_conditioner is not None:
            style_embed = self.style_conditioner(style_id=style_id, batch_size=xt.shape[0], device=xt.device)
            condition = condition + style_embed

        for block in self.blocks:
            hidden = block(hidden, condition)

        hidden = self.output_norm(hidden)
        return self.output_proj(hidden)
