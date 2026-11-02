"""Natural-language temporal grounding over pre-extracted clip features."""

import torch
import torch.nn as nn

from ..layers.attention import CrossAttention, SoftmaxAttention
from ..layers.block import Block
from ..layers.common import Mlp, init_weights, sinusoidal_positions


class TextEncoderLayer(nn.Module):
    """Pre-norm self-attention encoder layer for query words."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SoftmaxAttention(dim, num_heads, attn_drop=dropout, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=dropout)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), token_mask=mask)
        return x + self.mlp(self.norm2(x))


class FusionLayer(nn.Module):
    """Video tokens attend to query words, then pass through an MLP."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross = CrossAttention(dim, num_heads, attn_drop=dropout, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=dropout)

    def forward(self, x, words, word_mask):
        x = x + self.cross(self.norm1(x), words, word_mask)
        return x + self.mlp(self.norm2(x))


def _head(dim):
    return nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))


class MGDTGroundingHead(nn.Module):
    """Predicts start/end distributions and an in-moment score for every feature step."""

    def __init__(self, vocab_size, video_dim=768, d_model=256, num_heads=4, text_layers=2, fusion_layers=2,
                 temporal_depth=4, mlp_ratio=4.0, dropout=0.1, query_dropout=0.1, drop_path_rate=0.1, mixer=None):
        super().__init__()
        self.video_proj = nn.Sequential(nn.LayerNorm(video_dim), nn.Linear(video_dim, d_model), nn.Dropout(dropout))
        self.word_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.query_drop = nn.Dropout(query_dropout)
        self.text_layers = nn.ModuleList([TextEncoderLayer(d_model, num_heads, mlp_ratio, dropout)
                                          for _ in range(text_layers)])
        self.text_norm = nn.LayerNorm(d_model)
        self.fusion = nn.ModuleList([FusionLayer(d_model, num_heads, mlp_ratio, dropout)
                                     for _ in range(fusion_layers)])
        rates = torch.linspace(0, drop_path_rate, temporal_depth).tolist()
        self.temporal = nn.ModuleList([
            Block(d_model, num_heads, mlp_ratio, drop=dropout, drop_path=rates[i], mixer=mixer)
            for i in range(temporal_depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.start_head = _head(d_model)
        self.end_head = _head(d_model)
        self.inside_head = _head(d_model)
        self.apply(init_weights)

    def forward(self, video, video_mask, words, word_mask):
        """Returns start, end and inside logits of shape (B, T); padded steps are set to -1e4."""
        B, T, _ = video.shape
        d = self.word_embed.embedding_dim
        q = self.word_embed(words)
        q = self.query_drop(q + sinusoidal_positions(words.shape[1], d, q.device, q.dtype))
        for layer in self.text_layers:
            q = layer(q, word_mask)
        q = self.text_norm(q)
        x = self.video_proj(video)
        x = x + sinusoidal_positions(T, d, x.device, x.dtype)
        for layer in self.fusion:
            x = layer(x, q, word_mask)
        for block in self.temporal:
            x = block(x, (T,), video_mask)
        x = self.norm(x)
        outs = []
        for head in (self.start_head, self.end_head, self.inside_head):
            logits = head(x).squeeze(-1).float()
            outs.append(logits.masked_fill(~video_mask, -1e4))
        return tuple(outs)
