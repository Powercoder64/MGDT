"""Softmax self- and cross-attention."""

import torch
import torch.nn as nn


def scaled_dot_product(q, k, v, key_mask=None, causal=False, dropout=None):
    """Softmax attention over the last two dims with optional key padding and causal masks."""
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q * scale, k.transpose(-2, -1))
    if key_mask is not None:
        scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
    if causal:
        Tq, Tk = scores.shape[-2], scores.shape[-1]
        future = torch.ones(Tq, Tk, dtype=torch.bool, device=scores.device).triu(Tk - Tq + 1)
        scores = scores.masked_fill(future, float("-inf"))
    attn = scores.float().softmax(-1).to(v.dtype)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, v)


class SoftmaxAttention(nn.Module):
    """Multi-head softmax self-attention used as a drop-in token mixer.

    With ``shared_qk`` the key projection also produces the queries, which gives the same
    parameter count as the diffusion mixer.
    """

    def __init__(self, dim, num_heads, qkv_bias=True, shared_qk=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.shared_qk = shared_qk
        if not shared_qk:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _heads(self, t):
        B, N, _ = t.shape
        return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, grid=None, token_mask=None, causal=False):
        B, N, C = x.shape
        k = self._heads(self.k_proj(x))
        q = k if self.shared_qk else self._heads(self.q_proj(x))
        v = self._heads(self.v_proj(x))
        out = scaled_dot_product(q, k, v, token_mask, causal, self.attn_drop)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class CrossAttention(nn.Module):
    """Multi-head attention from query tokens to a memory sequence."""

    def __init__(self, dim, num_heads, kv_dim=None, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        kv_dim = kv_dim or dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def memory_kv(self, memory):
        """Per-head keys and values (B, heads, M, head_dim) of a memory sequence."""
        B, M, _ = memory.shape
        k = self.k_proj(memory).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        return k, v

    def forward(self, x, memory, memory_mask=None, memory_kv=None):
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = memory_kv if memory_kv is not None else self.memory_kv(memory)
        out = scaled_dot_product(q, k, v, memory_mask, False, self.attn_drop)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))
