"""Exact top-k neighbour search and per-edge squared distances with low-memory gradients."""

import torch

from ..utils.flops import record

_CHUNK_ELEMENTS = 1 << 24


def _row_step(groups, width):
    return max(1, _CHUNK_ELEMENTS // max(1, groups * width))


def _gather_rows(x, idx):
    """Gathers ``x[g, idx[g, i, k]]``: x (G, M, P), idx (G, n, K) -> (G, n, K, P)."""
    G, n, K = idx.shape
    P = x.shape[-1]
    flat = idx.reshape(G, n * K, 1).expand(G, n * K, P)
    return x.gather(1, flat).view(G, n, K, P)


def gather_weighted_sum(x, nbr, w):
    """Returns ``out[g, i] = sum_k w[g, i, k] * x[g, nbr[g, i, k]]``."""
    G, N, K = nbr.shape
    out = x.new_empty(G, N, x.shape[-1])
    step = _row_step(G, K * x.shape[-1])
    for s in range(0, N, step):
        e = min(N, s + step)
        out[:, s:e] = (w[:, s:e, :, None] * _gather_rows(x, nbr[:, s:e])).sum(2)
    return out


def scatter_weighted_sum(x, nbr, w, num_targets):
    """Returns ``out[g, j] = sum_{(i, k): nbr[g, i, k] = j} w[g, i, k] * x[g, i]``."""
    G, N, K = nbr.shape
    P = x.shape[-1]
    out = x.new_zeros(G, num_targets, P)
    step = _row_step(G, K * P)
    for s in range(0, N, step):
        e = min(N, s + step)
        n = e - s
        values = (w[:, s:e, :, None] * x[:, s:e, None, :]).reshape(G, n * K, P)
        index = nbr[:, s:e].reshape(G, n * K, 1).expand(G, n * K, P)
        out.scatter_add_(1, index, values)
    return out


class _EdgeSqDist(torch.autograd.Function):
    """Computes ``d[g, i, k] = ||a[g, i] - b[g, nbr[g, i, k]]||^2``."""

    @staticmethod
    def forward(ctx, a, b, nbr):
        G, N, K = nbr.shape
        out = a.new_empty(G, N, K)
        step = _row_step(G, K * a.shape[-1])
        for s in range(0, N, step):
            e = min(N, s + step)
            out[:, s:e] = (a[:, s:e, None, :] - _gather_rows(b, nbr[:, s:e])).pow(2).sum(-1)
        ctx.save_for_backward(a, b, nbr)
        return out

    @staticmethod
    def backward(ctx, grad):
        a, b, nbr = ctx.saved_tensors
        grad = grad.contiguous()
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = 2.0 * (a * grad.sum(-1, keepdim=True) - gather_weighted_sum(b, nbr, grad))
        if ctx.needs_input_grad[1]:
            G, M = b.shape[0], b.shape[1]
            col = torch.zeros(G, M, dtype=grad.dtype, device=grad.device)
            col.scatter_add_(1, nbr.reshape(G, -1), grad.reshape(G, -1))
            grad_b = 2.0 * (b * col[..., None] - scatter_weighted_sum(a, nbr, grad, M))
        return grad_a, grad_b, None


def edge_sq_dist(a, b, nbr):
    """Squared Euclidean distance between ``a[i]`` and each of its listed neighbours in ``b``."""
    record(nbr.numel() * a.shape[-1])
    return _EdgeSqDist.apply(a, b, nbr)


@torch.no_grad()
def knn(a, b, k, valid=None):
    """Exact k nearest rows of ``b`` for every row of ``a`` (self excluded).

    Args:
        a, b: (G, N, P) query and key features.
        k: number of neighbours (clamped to N - 1).
        valid: optional (G, N) bool mask; invalid rows of ``b`` are never selected.

    Returns:
        indices (G, N, k) and a bool mask (G, N, k) of usable entries.
    """
    G, N, _ = a.shape
    k = min(int(k), N - 1)
    if k <= 0:
        empty_idx = torch.zeros(G, N, 0, dtype=torch.long, device=a.device)
        return empty_idx, torch.zeros(G, N, 0, dtype=torch.bool, device=a.device)
    b_sq = b.pow(2).sum(-1)
    bt = b.transpose(1, 2)
    idx = torch.empty(G, N, k, dtype=torch.long, device=a.device)
    ok = torch.empty(G, N, k, dtype=torch.bool, device=a.device)
    step = _row_step(G, N)
    for s in range(0, N, step):
        e = min(N, s + step)
        d = b_sq[:, None, :] - 2.0 * torch.matmul(a[:, s:e], bt)
        rows = torch.arange(e - s, device=a.device)
        d[:, rows, rows + s] = float("inf")
        if valid is not None:
            d.masked_fill_(~valid[:, None, :], float("inf"))
        vals, ind = d.topk(k, dim=-1, largest=False)
        idx[:, s:e] = ind
        ok[:, s:e] = torch.isfinite(vals)
    return idx, ok
