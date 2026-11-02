"""Isotropic image backbone built from MGDT blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..layers.block import Block
from ..layers.common import init_weights


def resize_pos_embed(pos_embed, src_grid, dst_grid):
    """Bicubic resize of (1, h*w, C) position embeddings from ``src_grid`` to ``dst_grid``."""
    if tuple(src_grid) == tuple(dst_grid):
        return pos_embed
    C = pos_embed.shape[-1]
    grid = pos_embed.reshape(1, src_grid[0], src_grid[1], C).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=tuple(dst_grid), mode="bicubic", align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(1, dst_grid[0] * dst_grid[1], C).to(pos_embed.dtype)


class MGDTVisionTransformer(nn.Module):
    """Patch embedding, learned position embeddings and a stack of MGDT blocks.

    Classification uses LayerNorm followed by global average pooling and a linear head.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.0, head_drop=0.0, mixer=None,
                 grad_checkpointing=False):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.base_grid = (img_size // patch_size, img_size // patch_size)
        self.grad_checkpointing = grad_checkpointing
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.base_grid[0] * self.base_grid[1], embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop=drop_rate, drop_path=rates[i], mixer=mixer)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head_drop = nn.Dropout(head_drop)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def no_weight_decay(self):
        return {"pos_embed"}

    def _run_block(self, block, x, grid):
        if self.grad_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, grid, use_reentrant=False)
        return block(x, grid)

    def forward_tokens(self, images, out_indices=()):
        """Returns final tokens (B, N, C), the token grid and tokens after ``out_indices`` blocks."""
        x = self.patch_embed(images)
        grid = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x + resize_pos_embed(self.pos_embed, self.base_grid, grid))
        wanted = set(out_indices)
        saved = {}
        for i, block in enumerate(self.blocks):
            x = self._run_block(block, x, grid)
            if i in wanted:
                saved[i] = x
        return x, grid, [saved[i] for i in out_indices]

    def forward_features(self, images):
        """Normalised output tokens (B, N, C)."""
        x, _, _ = self.forward_tokens(images)
        return self.norm(x)

    def forward(self, images):
        x = self.forward_features(images).mean(1)
        return self.head(self.head_drop(x))
