"""Image captioning: MGDT visual encoder and an autoregressive Transformer decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.attention import CrossAttention, SoftmaxAttention, scaled_dot_product
from ..layers.common import Mlp, init_weights
from .vit import MGDTVisionTransformer


class DecoderLayer(nn.Module):
    """Pre-norm decoder layer: causal self-attention, cross-attention to image tokens, MLP."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = SoftmaxAttention(dim, num_heads, attn_drop=dropout, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, attn_drop=dropout, proj_drop=dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=dropout)

    def forward(self, x, memory, token_mask=None):
        x = x + self.self_attn(self.norm1(x), token_mask=token_mask, causal=True)
        x = x + self.cross_attn(self.norm2(x), memory)
        return x + self.mlp(self.norm3(x))

    def step(self, x, memory_kv, cache):
        """Processes one new token per sequence (B, 1, C), appending its keys/values to ``cache``."""
        attn = self.self_attn
        B = x.shape[0]
        h = self.norm1(x)
        q, k, v = attn._heads(attn.q_proj(h)), attn._heads(attn.k_proj(h)), attn._heads(attn.v_proj(h))
        if "k" in cache:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        cache["k"], cache["v"] = k, v
        out = scaled_dot_product(q, k, v, dropout=attn.attn_drop).transpose(1, 2).reshape(B, 1, -1)
        x = x + attn.proj_drop(attn.proj(out))
        x = x + self.cross_attn(self.norm2(x), None, memory_kv=memory_kv)
        return x + self.mlp(self.norm3(x))


class MGDTCaptioner(nn.Module):
    """Encoder-decoder captioner with tied input/output word embeddings."""

    def __init__(self, vocab_size, encoder=None, d_model=1024, depth=10, num_heads=16, mlp_ratio=4.0,
                 dropout=0.1, max_len=22, pad_id=0, bos_id=1, eos_id=2, unk_id=3):
        super().__init__()
        encoder = dict(encoder or {})
        encoder["num_classes"] = 0
        self.encoder = MGDTVisionTransformer(**encoder)
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = pad_id, bos_id, eos_id, unk_id
        self.max_len = max_len
        self.memory_proj = nn.Sequential(nn.Linear(self.encoder.embed_dim, d_model), nn.LayerNorm(d_model))
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        for module in (self.memory_proj, self.embed, self.layers, self.norm):
            module.apply(init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.lm_head.weight = self.embed.weight

    def no_weight_decay(self):
        return {"encoder.pos_embed", "pos_embed"}

    def encode(self, images):
        """Projected image tokens (B, N, d_model)."""
        return self.memory_proj(self.encoder.forward_features(images))

    def decode(self, tokens, memory):
        """Next-token logits (B, T, V) for input tokens (B, T)."""
        T = tokens.shape[1]
        x = self.drop(self.embed(tokens) + self.pos_embed[:, :T])
        mask = tokens != self.pad_id
        for layer in self.layers:
            x = layer(x, memory, mask)
        return self.lm_head(self.norm(x))

    def forward(self, images, tokens):
        return self.decode(tokens, self.encode(images))

    def start_decoding(self, images, beam_size=1):
        """Encodes images once and returns per-layer cross-attention keys/values and empty caches."""
        memory = self.encode(images)
        memory_kv = []
        for layer in self.layers:
            k, v = layer.cross_attn.memory_kv(memory)
            memory_kv.append((k.repeat_interleave(beam_size, 0), v.repeat_interleave(beam_size, 0)))
        return memory_kv, [dict() for _ in self.layers]

    def decode_step(self, tokens, position, memory_kv, caches):
        """Next-token logits (B, V) for the newest tokens (B,) placed at ``position``."""
        x = self.drop(self.embed(tokens)[:, None] + self.pos_embed[:, position:position + 1])
        for layer, kv, cache in zip(self.layers, memory_kv, caches):
            x = layer.step(x, kv, cache)
        return self.lm_head(self.norm(x))[:, 0]

    @torch.no_grad()
    def generate(self, images, beam_size=3, max_words=20):
        """Beam-search decoding; returns one list of word ids (without BOS/EOS) per image."""
        B, K = images.shape[0], int(beam_size)
        device = images.device
        memory_kv, caches = self.start_decoding(images, K)
        steps = min(int(max_words) + 1, self.max_len)
        tokens = torch.full((B * K,), self.bos_id, dtype=torch.long, device=device)
        seqs = torch.zeros(B * K, 0, dtype=torch.long, device=device)
        scores = torch.full((B, K), float("-inf"), device=device)
        scores[:, 0] = 0.0
        finished = torch.zeros(B * K, dtype=torch.bool, device=device)
        for t in range(steps):
            logp = F.log_softmax(self.decode_step(tokens, t, memory_kv, caches).float(), dim=-1)
            for banned in (self.pad_id, self.bos_id, self.unk_id):
                logp[:, banned] = float("-inf")
            if t == steps - 1:
                logp[~finished] = float("-inf")
                logp[~finished, self.eos_id] = 0.0
            logp[finished] = float("-inf")
            logp[finished, self.eos_id] = 0.0
            V = logp.shape[-1]
            scores, flat = (scores.view(B * K, 1) + logp).view(B, K * V).topk(K, dim=-1)
            origin = (torch.arange(B, device=device)[:, None] * K + flat // V).view(-1)
            tokens = (flat % V).view(-1)
            for cache in caches:
                cache["k"], cache["v"] = cache["k"][origin], cache["v"][origin]
            seqs = torch.cat([seqs[origin], tokens[:, None]], dim=1)
            finished = finished[origin] | (tokens == self.eos_id)
            if bool(finished.all()):
                break
        best = scores.argmax(-1)
        outputs = []
        for b in range(B):
            words = []
            for tok in seqs[b * K + int(best[b])].tolist():
                if tok == self.eos_id:
                    break
                words.append(tok)
            outputs.append(words[:max_words])
        return outputs
