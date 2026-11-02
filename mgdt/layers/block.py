"""Pre-norm Transformer block with a configurable token mixer."""

import torch.nn as nn

from .attention import SoftmaxAttention
from .common import DropPath, Mlp
from .mgdt_mixer import MGDTMixer

MGDT_KEYS = {
    "k", "local_radius", "levels", "coarsening", "coarsen_factor", "hem_rounds", "kmeans_iters", "pre_smooth",
    "post_smooth", "coarse_steps", "tau", "normalization", "conductance", "threshold", "kappa_init", "kappa_mode",
    "kappa_warmup", "kappa_warmup_start", "similarity", "metric", "shared_dim", "clip_norm", "drop_threshold",
    "qkv_bias", "track_energy",
}
SOFTMAX_KEYS = {"shared_qk", "attn_drop", "qkv_bias"}


def build_mixer(dim, num_heads, cfg=None, proj_drop=0.0):
    """Creates an MGDT or softmax token mixer from a config dict with a ``type`` key."""
    cfg = dict(cfg or {})
    kind = cfg.pop("type", "mgdt")
    unknown = set(cfg) - MGDT_KEYS - SOFTMAX_KEYS
    if unknown:
        raise ValueError(f"Unknown mixer options: {sorted(unknown)}")
    if kind == "mgdt":
        return MGDTMixer(dim, num_heads, proj_drop=proj_drop, **{k: v for k, v in cfg.items() if k in MGDT_KEYS})
    if kind == "softmax":
        return SoftmaxAttention(dim, num_heads, proj_drop=proj_drop,
                                **{k: v for k, v in cfg.items() if k in SOFTMAX_KEYS})
    raise ValueError(f"Unknown mixer type '{kind}'")


class Block(nn.Module):
    """``x + DropPath(Mixer(LN(x)))`` followed by ``x + DropPath(MLP(LN(x)))``."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, drop_path=0.0, mixer=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = build_mixer(dim, num_heads, mixer, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, grid, token_mask=None):
        x = x + self.drop_path(self.mixer(self.norm1(x), grid, token_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
