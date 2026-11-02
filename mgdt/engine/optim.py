"""AdamW parameter groups and learning-rate schedules."""

import math

import torch

_NO_DECAY_KEYWORDS = ("pos_embed", "time_embed", "log_kappa")


def param_groups(model, weight_decay):
    """Splits trainable parameters into decayed weights and non-decayed biases, norms and embeddings."""
    skip = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or name in skip or any(k in name for k in _NO_DECAY_KEYWORDS):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(model, cfg):
    """AdamW from an ``optim`` config section (lr, weight_decay, betas, eps)."""
    groups = param_groups(model, float(cfg.get("weight_decay", 0.05)))
    betas = tuple(cfg.get("betas", (0.9, 0.999)))
    return torch.optim.AdamW(groups, lr=float(cfg["lr"]), betas=betas, eps=float(cfg.get("eps", 1e-8)))


class LRSchedule:
    """Linear warm-up followed by cosine or polynomial decay, evaluated per optimiser step."""

    def __init__(self, base_lr, total_steps, warmup_steps=0, kind="cosine", min_lr=0.0, power=1.0):
        if kind not in ("cosine", "poly", "constant"):
            raise ValueError(f"Unknown schedule '{kind}'")
        self.base_lr = float(base_lr)
        self.total = max(1, int(total_steps))
        self.warmup = int(warmup_steps)
        self.kind = kind
        self.min_lr = float(min_lr)
        self.power = float(power)

    def __call__(self, step):
        if step < self.warmup:
            return self.base_lr * (step + 1) / self.warmup
        if self.kind == "constant":
            return self.base_lr
        progress = min(1.0, (step - self.warmup) / max(1, self.total - self.warmup))
        if self.kind == "cosine":
            return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * (1.0 - progress) ** self.power
