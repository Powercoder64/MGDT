"""Exponential moving average of model weights."""

import copy

import torch


class ModelEMA:
    """Keeps ``ema = decay * ema + (1 - decay) * weights`` for floating-point state entries."""

    def __init__(self, model, decay=0.9999):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        for key, ema_value in self.module.state_dict().items():
            value = model_state[key].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(value.to(ema_value.dtype), alpha=1.0 - self.decay)
            else:
                ema_value.copy_(value)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        self.module.load_state_dict(state)
