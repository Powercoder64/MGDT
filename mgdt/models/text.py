"""Byte-level long-sequence text classifier."""

import torch
import torch.nn as nn

from ..layers.block import Block
from ..layers.common import init_weights, sinusoidal_positions


class MGDTTextClassifier(nn.Module):
    """Byte embeddings, sinusoidal positions, MGDT blocks over a 1-D grid and masked mean pooling."""

    def __init__(self, num_classes=2, vocab_size=257, embed_dim=128, depth=6, num_heads=4, mlp_ratio=4.0,
                 dropout=0.1, drop_path_rate=0.0, mixer=None, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)
        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop=dropout, drop_path=rates[i], mixer=mixer)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(init_weights)

    def forward(self, tokens):
        mask = tokens != self.pad_id
        B, L = tokens.shape
        x = self.embed(tokens) + sinusoidal_positions(L, self.embed.embedding_dim, tokens.device, self.embed.weight.dtype)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, (L,), mask)
        x = self.norm(x)
        m = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(1) / m.sum(1).clamp_min(1.0)
        return self.head(pooled)
