"""Video classifier: joint space-time MGDT encoder followed by temporal MGDT aggregation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..layers.block import Block
from ..layers.common import init_weights
from .vit import resize_pos_embed


class MGDTVideoTransformer(nn.Module):
    """Tubelet embedding, joint space-time MGDT blocks, per-frame pooling, temporal MGDT blocks.

    The space-time blocks share their layout with :class:`MGDTVisionTransformer`, so they can
    be initialised from an image checkpoint with :meth:`load_image_state_dict`.
    """

    def __init__(self, num_classes=400, img_size=224, num_frames=32, tubelet_size=2, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.1,
                 head_drop=0.1, temporal_depth=10, temporal_heads=None, mixer=None, temporal_mixer=None,
                 grad_checkpointing=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.base_grid = (img_size // patch_size, img_size // patch_size)
        self.base_frames = num_frames // tubelet_size
        self.grad_checkpointing = grad_checkpointing
        self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=(tubelet_size, patch_size, patch_size),
                                     stride=(tubelet_size, patch_size, patch_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.base_grid[0] * self.base_grid[1], embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, self.base_frames, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        rates = torch.linspace(0, drop_path_rate, depth + temporal_depth).tolist()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop=drop_rate, drop_path=rates[i], mixer=mixer)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.temporal_blocks = nn.ModuleList([
            Block(embed_dim, temporal_heads or num_heads, mlp_ratio, drop=drop_rate, drop_path=rates[depth + i],
                  mixer=temporal_mixer if temporal_mixer is not None else mixer)
            for i in range(temporal_depth)
        ])
        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.head_drop = nn.Dropout(head_drop)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def no_weight_decay(self):
        return {"pos_embed", "time_embed"}

    def _run_block(self, block, x, grid):
        if self.grad_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, grid, use_reentrant=False)
        return block(x, grid)

    def _time_embed(self, frames):
        if frames == self.base_frames:
            return self.time_embed
        t = F.interpolate(self.time_embed.transpose(1, 2).float(), size=frames, mode="linear", align_corners=False)
        return t.transpose(1, 2).to(self.time_embed.dtype)

    def forward_features(self, video):
        """Clip embedding (B, C) for a video tensor (B, 3, T, H, W)."""
        x = self.patch_embed(video)
        B, C, T, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        pos = resize_pos_embed(self.pos_embed, self.base_grid, (h, w)).view(1, 1, h * w, C)
        pos = (pos + self._time_embed(T).view(1, T, 1, C)).reshape(1, T * h * w, C)
        x = self.pos_drop(x + pos)
        for block in self.blocks:
            x = self._run_block(block, x, (T, h, w))
        x = self.norm(x).view(B, T, h * w, C).mean(2)
        for block in self.temporal_blocks:
            x = self._run_block(block, x, (T,))
        return self.temporal_norm(x).mean(1)

    def forward(self, video):
        return self.head(self.head_drop(self.forward_features(video)))

    def load_image_state_dict(self, state_dict):
        """Initialises the space-time encoder from an image model state dict.

        The 2-D patch kernel is inflated over the tubelet depth (averaged); classifier
        weights are skipped. Returns the lists of missing and unexpected keys.
        """
        own = self.state_dict()
        mapped = {}
        for key, value in state_dict.items():
            if key.startswith("head."):
                continue
            if key == "patch_embed.weight" and value.dim() == 4:
                value = value.unsqueeze(2).repeat(1, 1, self.tubelet_size, 1, 1) / self.tubelet_size
            if key == "pos_embed" and key in own and value.shape != own[key].shape:
                src = int(round(value.shape[1] ** 0.5))
                value = resize_pos_embed(value, (src, src), self.base_grid)
            if key in own and own[key].shape == value.shape:
                mapped[key] = value
        result = self.load_state_dict(mapped, strict=False)
        return list(result.missing_keys), [k for k in state_dict if k not in mapped and not k.startswith("head.")]
