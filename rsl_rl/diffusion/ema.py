from __future__ import annotations

import torch


class ExponentialMovingAverage:
    """维护模型参数的指数滑动平均副本。"""

    def __init__(self, model, decay: float = 0.999):
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must lie in (0, 1], got {decay}")
        self.decay = decay
        self.shadow_state = {
            name: parameter.detach().clone()
            for name, parameter in model.state_dict().items()
        }

    def update(self, model):
        """用当前模型参数更新 EMA。"""
        with torch.no_grad():
            for name, parameter in model.state_dict().items():
                self.shadow_state[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model):
        """将 EMA 参数覆盖回模型。"""
        model.load_state_dict(self.shadow_state, strict=True)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "shadow_state": self.shadow_state}

    def load_state_dict(self, state_dict: dict[str, object]):
        self.decay = float(state_dict["decay"])
        self.shadow_state = {
            name: tensor.detach().clone()
            for name, tensor in state_dict["shadow_state"].items()
        }
