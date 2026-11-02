"""Semantic segmentation: MGDT backbone, simple feature pyramid and a UPerNet-style decoder."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.common import init_weights
from .vit import MGDTVisionTransformer


def group_norm(channels):
    """GroupNorm with up to 32 groups of at least two channels (independent of the batch size)."""
    groups = math.gcd(32, channels)
    while groups > 1 and channels // groups < 2:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Sequential):
    """Conv2d + GroupNorm + ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            group_norm(out_ch),
            nn.ReLU(inplace=True),
        )


class SimplePyramid(nn.Module):
    """Turns four 1/16-scale token maps into 1/4, 1/8, 1/16 and 1/32 feature maps."""

    def __init__(self, embed_dim, channels):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(4)])
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, channels, 2, stride=2),
            group_norm(channels),
            nn.GELU(),
            nn.ConvTranspose2d(channels, channels, 2, stride=2),
        )
        self.up2 = nn.ConvTranspose2d(embed_dim, channels, 2, stride=2)
        self.same = nn.Conv2d(embed_dim, channels, 1)
        self.down = nn.Sequential(nn.MaxPool2d(2, ceil_mode=True), nn.Conv2d(embed_dim, channels, 1))

    def forward(self, tokens, grid):
        maps = []
        for norm, t in zip(self.norms, tokens):
            B, N, C = t.shape
            maps.append(norm(t).transpose(1, 2).reshape(B, C, grid[0], grid[1]))
        return [self.up4(maps[0]), self.up2(maps[1]), self.same(maps[2]), self.down(maps[3])]


class UPerHead(nn.Module):
    """Pyramid pooling on the coarsest map, top-down FPN fusion and a pixel classifier."""

    def __init__(self, in_channels, channels, num_classes, pool_scales=(1, 2, 3, 6), dropout=0.1, fuse_kernel=3):
        super().__init__()
        self.ppm = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(s), ConvNormAct(in_channels, channels, 1)) for s in pool_scales
        ])
        self.bottleneck = ConvNormAct(in_channels + len(pool_scales) * channels, channels, 3)
        self.lateral = nn.ModuleList([ConvNormAct(in_channels, channels, 1) for _ in range(3)])
        self.fpn = nn.ModuleList([ConvNormAct(channels, channels, 3) for _ in range(3)])
        self.fuse = ConvNormAct(4 * channels, channels, fuse_kernel)
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    @staticmethod
    def _up(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, maps):
        top = maps[3]
        pooled = [top] + [self._up(p(top), top.shape[2:]) for p in self.ppm]
        lat = [lateral(m) for lateral, m in zip(self.lateral, maps[:3])] + [self.bottleneck(torch.cat(pooled, 1))]
        for i in range(3, 0, -1):
            lat[i - 1] = lat[i - 1] + self._up(lat[i], lat[i - 1].shape[2:])
        outs = [self.fpn[i](lat[i]) for i in range(3)] + [lat[3]]
        size = outs[0].shape[2:]
        fused = self.fuse(torch.cat([outs[0]] + [self._up(o, size) for o in outs[1:]], 1))
        return self.classifier(self.dropout(fused))


class MGDTSegmenter(nn.Module):
    """Dense predictor returning per-pixel logits at the input resolution."""

    def __init__(self, num_classes, backbone=None, out_indices=(3, 5, 7, 11), decoder_channels=256,
                 pool_scales=(1, 2, 3, 6), dropout=0.1, fuse_kernel=3):
        super().__init__()
        backbone = dict(backbone or {})
        backbone["num_classes"] = 0
        self.backbone = MGDTVisionTransformer(**backbone)
        self.backbone.norm = nn.Identity()
        self.out_indices = tuple(out_indices)
        self.neck = SimplePyramid(self.backbone.embed_dim, decoder_channels)
        self.decode_head = UPerHead(decoder_channels, decoder_channels, num_classes, tuple(pool_scales), dropout,
                                    fuse_kernel)
        self.neck.apply(init_weights)
        self.decode_head.apply(init_weights)

    def no_weight_decay(self):
        return {"backbone.pos_embed"}

    def forward(self, images):
        H, W = images.shape[-2:]
        p = self.backbone.patch_size
        pad_h, pad_w = (-H) % p, (-W) % p
        if pad_h or pad_w:
            images = F.pad(images, (0, pad_w, 0, pad_h))
        _, grid, feats = self.backbone.forward_tokens(images, self.out_indices)
        logits = self.decode_head(self.neck(feats, grid))
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return logits[..., :H, :W]
