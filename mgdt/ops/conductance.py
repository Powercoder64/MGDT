"""Edge conductance functions with a learnable contrast parameter kappa."""

import math

import torch
import torch.nn as nn

FORMS = ("exp", "rational", "isotropic")
KAPPA_MODES = ("head", "layer", "fixed")


class Conductance(nn.Module):
    """Maps squared feature distances to conductances in (0, 1].

    Forms: ``exp`` g(s) = exp(-(s/kappa)^2), ``rational`` g(s) = 1 / (1 + (s/kappa)^2) and
    ``isotropic`` g(s) = 1. ``threshold`` (a multiple c of kappa) additionally zeroes every
    conductance with s >= c * kappa. With ``warmup`` enabled the sharpness 1/kappa ramps
    linearly from ``warmup_start`` times its value to its full value as training progress
    goes from 0 to 1 (progress is set by the trainer and defaults to 1).
    """

    def __init__(self, num_heads, form="exp", kappa_init=5.0, kappa_mode="head", warmup=True,
                 warmup_start=0.1, threshold=None):
        super().__init__()
        if form not in FORMS:
            raise ValueError(f"conductance must be one of {FORMS}, got {form}")
        if kappa_mode not in KAPPA_MODES:
            raise ValueError(f"kappa_mode must be one of {KAPPA_MODES}, got {kappa_mode}")
        self.form = form
        self.num_heads = num_heads
        self.kappa_mode = kappa_mode
        self.warmup = bool(warmup)
        self.warmup_start = float(warmup_start)
        self.threshold = None if threshold is None else float(threshold)
        shape = (num_heads,) if kappa_mode == "head" else (1,)
        init = torch.full(shape, math.log(float(kappa_init)))
        if kappa_mode == "fixed":
            self.register_buffer("log_kappa", init)
        else:
            self.log_kappa = nn.Parameter(init)
        self.register_buffer("progress", torch.ones(()), persistent=False)

    @property
    def is_isotropic(self):
        return self.form == "isotropic"

    def set_progress(self, value):
        """Sets warm-up progress in [0, 1]."""
        self.progress.fill_(float(min(max(value, 0.0), 1.0)))

    def sharpness_scale(self):
        if not self.warmup:
            return torch.ones((), device=self.log_kappa.device)
        return self.warmup_start + (1.0 - self.warmup_start) * self.progress

    def kappa(self):
        """Effective contrast parameter per head (or shared) after warm-up scaling."""
        return torch.exp(self.log_kappa) / self.sharpness_scale()

    def forward(self, sq_dist):
        """Conductance for squared distances of shape (B, H or 1, N, K)."""
        if self.is_isotropic:
            return torch.ones_like(sq_dist)
        inv_kappa_sq = (self.sharpness_scale() * torch.exp(-self.log_kappa)).pow(2)
        z = sq_dist * inv_kappa_sq.view(1, -1, 1, 1)
        g = torch.exp(-z) if self.form == "exp" else 1.0 / (1.0 + z)
        if self.threshold is not None:
            g = g * (z < self.threshold**2).to(g.dtype)
        return g

    def extra_repr(self):
        return (f"form={self.form}, kappa_mode={self.kappa_mode}, warmup={self.warmup}, "
                f"warmup_start={self.warmup_start}, threshold={self.threshold}")
