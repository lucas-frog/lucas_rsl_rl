from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.diffusion.conditioning import StyleConditioner


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
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.feature_dim = feature_dim
        self.window_size = window_size
        self.num_diffusion_steps = num_diffusion_steps
        self.num_styles = num_styles
        self.hidden_dim = hidden_dim

        self.token_proj = nn.Linear(feature_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, window_size, hidden_dim))
        self.timestep_embedding = nn.Embedding(num_diffusion_steps, hidden_dim)
        self.timestep_mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.style_conditioner = StyleConditioner(num_styles=num_styles, hidden_dim=hidden_dim) if num_styles > 0 else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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
        hidden = hidden + self.timestep_mlp(self.timestep_embedding(t)).unsqueeze(1)
        if self.style_conditioner is not None:
            style_embed = self.style_conditioner(style_id=style_id, batch_size=xt.shape[0], device=xt.device)
            hidden = hidden + style_embed.unsqueeze(1)
        hidden = self.encoder(hidden)
        return self.output_proj(hidden)
