"""Sparse graphs with node masses and memory-efficient sparse-dense products."""

from dataclasses import dataclass
from typing import Optional

import torch

from ..utils.flops import record

_CHUNK_ELEMENTS = 1 << 24


def _edge_chunks(num_edges, width):
    step = max(1, _CHUNK_ELEMENTS // max(1, width))
    for start in range(0, num_edges, step):
        yield start, min(num_edges, start + step)


class _EdgeAggregate(torch.autograd.Function):
    """Computes ``out[dst] += weight * x[src]``; edge gathers are recomputed in backward."""

    @staticmethod
    def forward(ctx, weight, x, src, dst, num_nodes):
        out = x.new_zeros(num_nodes, x.shape[1])
        for s, e in _edge_chunks(src.numel(), x.shape[1]):
            out.index_add_(0, dst[s:e], x[src[s:e]] * weight[s:e, None])
        ctx.save_for_backward(weight, x, src, dst)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        weight, x, src, dst = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_weight = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_weight = torch.empty_like(weight)
            for s, e in _edge_chunks(src.numel(), x.shape[1]):
                grad_weight[s:e] = (grad_out[dst[s:e]] * x[src[s:e]]).sum(1)
        if ctx.needs_input_grad[1]:
            grad_x = torch.zeros_like(x)
            for s, e in _edge_chunks(src.numel(), x.shape[1]):
                grad_x.index_add_(0, src[s:e], grad_out[dst[s:e]] * weight[s:e, None])
        return grad_weight, grad_x, None, None, None


def safe_reciprocal(t, eps=1e-12):
    """Elementwise ``1 / t`` for non-negative ``t``, returning 0 where ``t <= eps``."""
    return torch.where(t > eps, 1.0 / t.clamp_min(eps), torch.zeros_like(t))


@dataclass
class SparseGraph:
    """Weighted edge list (``dst <- src``) over a flat node set, with node masses.

    ``weight`` holds the total conductance carried by each entry; a symmetric graph lists
    every undirected edge in both directions. ``mass`` is the number of fine tokens a node
    represents (0 for padding), and ``degree`` the summed incident conductance.
    """

    src: torch.Tensor
    dst: torch.Tensor
    weight: torch.Tensor
    num_nodes: int
    mass: torch.Tensor
    degree: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.degree is None:
            zeros = torch.zeros(self.num_nodes, dtype=self.weight.dtype, device=self.weight.device)
            self.degree = zeros.index_add(0, self.dst, self.weight)

    @property
    def num_edges(self):
        return self.src.numel()

    def matvec(self, x):
        """Returns ``W x`` for node features ``x`` of shape (num_nodes, D)."""
        record(self.num_edges * x.shape[1])
        return _EdgeAggregate.apply(self.weight, x, self.src, self.dst, self.num_nodes)
