"""Token-graph construction from local grid neighbours and top-k feature neighbours."""

import functools
import itertools
import math

import torch

from ..utils.flops import record
from .distance import edge_sq_dist, knn
from .sparse import SparseGraph


def _offsets(ndim, radius):
    if radius <= 0:
        return []
    r = range(-radius, radius + 1)
    if ndim == 1:
        return [(d,) for d in r if d != 0]
    if ndim == 2:
        return [o for o in itertools.product(r, repeat=2) if o != (0, 0)]
    if ndim == 3:
        spatial = [(0,) + o for o in itertools.product(r, repeat=2) if o != (0, 0)]
        temporal = [(d, 0, 0) for d in r if d != 0]
        return spatial + temporal
    raise ValueError(f"Token grids must have 1, 2 or 3 dimensions, got {ndim}")


@functools.lru_cache(maxsize=256)
def _local_table(grid, radius, device):
    n = math.prod(grid)
    offsets = _offsets(len(grid), radius)
    if not offsets:
        empty = torch.zeros(n, 0, dtype=torch.long)
        return empty.to(device), torch.zeros(n, 0, dtype=torch.bool).to(device)
    axes = [torch.arange(s) for s in grid]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(n, len(grid))
    offs = torch.tensor(offsets, dtype=torch.long)
    nb = coords[:, None, :] + offs[None, :, :]
    size = torch.tensor(grid, dtype=torch.long)
    valid = ((nb >= 0) & (nb < size)).all(-1)
    strides = torch.tensor([math.prod(grid[i + 1:]) for i in range(len(grid))], dtype=torch.long)
    idx = (nb * strides).sum(-1)
    idx = torch.where(valid, idx, torch.arange(n)[:, None].expand_as(idx))
    return idx.to(device), valid.to(device)


def local_neighbors(grid, radius, device):
    """Static neighbour table of a token grid: indices (N, K) and validity (N, K)."""
    return _local_table(tuple(int(s) for s in grid), int(radius), torch.device(device))


def num_local_neighbors(ndim, radius):
    """Number of local neighbour slots per token for a grid of ``ndim`` dimensions."""
    return len(_offsets(ndim, radius))


def build_token_graph(a, b, conductance, grid, radius, k, num_heads, token_mask=None,
                      symmetric=True, drop_threshold=1e-4):
    """Builds the sparse conductance graph of every (batch, head) token set.

    Each token is linked to its local grid neighbours and to its ``k`` nearest tokens in
    the similarity space; duplicate and self links are removed, conductances below
    ``drop_threshold`` are dropped and, if ``symmetric``, the graph is symmetrised as
    (W + W^T) / 2.

    Args:
        a: (B, Hf, N, P) similarity features, Hf in {1, num_heads}.
        b: optional (B, Hf, N, P) key features for asymmetric distances ``||a_i - b_j||``.
        conductance: a :class:`Conductance` module.
        grid: token grid shape with ``prod(grid) == N``.
        radius: local neighbourhood radius.
        k: number of feature-space neighbours.
        num_heads: number of diffusion heads H.
        token_mask: optional (B, N) bool mask of real (non-padding) tokens.

    Returns:
        :class:`SparseGraph` over ``B * H * N`` nodes (node index ``(b * H + h) * N + i``).
    """
    B, Hf, N, P = a.shape
    H = int(num_heads)
    Gf = B * Hf
    af = a.reshape(Gf, N, P)
    bf = af if b is None else b.reshape(Gf, N, P)
    device = a.device

    valid_tok = None
    if token_mask is not None:
        valid_tok = token_mask.to(torch.bool)[:, None, :].expand(B, Hf, N).reshape(Gf, N)

    loc_idx, loc_ok = local_neighbors(grid, radius, device)
    knn_idx, knn_ok = knn(af, bf, k, valid_tok)
    Kl = loc_idx.shape[1]
    nbr = torch.cat([loc_idx.unsqueeze(0).expand(Gf, N, Kl), knn_idx], -1)
    ok = torch.cat([loc_ok.unsqueeze(0).expand(Gf, N, Kl), knn_ok], -1)
    K = nbr.shape[-1]

    self_idx = torch.arange(N, device=device).view(1, N, 1)
    nbr = torch.where(ok, nbr, self_idx)
    nbr, _ = nbr.sort(-1)
    keep = nbr != self_idx
    keep[..., 1:] &= nbr[..., 1:] != nbr[..., :-1]
    if valid_tok is not None:
        keep &= valid_tok[:, :, None] & valid_tok.gather(1, nbr.reshape(Gf, N * K)).view(Gf, N, K)

    if conductance.is_isotropic:
        g = torch.ones(B, Hf, N, K, dtype=a.dtype, device=device)
    else:
        sq = edge_sq_dist(af, bf, nbr).view(B, Hf, N, K)
        g = conductance(sq)
    record(g.numel())
    g = g.expand(B, H, N, K)
    keep_f = keep.view(B, Hf, N, K).expand(B, H, N, K)
    w = g * keep_f.to(g.dtype)
    if drop_threshold and drop_threshold > 0:
        w = w * (g.detach() >= drop_threshold).to(g.dtype)

    G = B * H
    nbr_full = nbr.view(B, Hf, N, K).expand(B, H, N, K).reshape(G, N, K)
    base = (torch.arange(G, device=device) * N).view(G, 1, 1)
    dst = (torch.arange(N, device=device).view(1, N, 1) + base).expand(G, N, K).reshape(-1)
    src = (nbr_full + base).reshape(-1)
    w = w.reshape(-1)
    if symmetric:
        src, dst = torch.cat([src, dst]), torch.cat([dst, src])
        w = torch.cat([w, w]) * 0.5
    nz = w.detach() > 0

    if token_mask is not None:
        mass = token_mask.to(a.dtype)[:, None, :].expand(B, H, N).reshape(-1)
    else:
        mass = torch.ones(G * N, dtype=a.dtype, device=device)
    return SparseGraph(src=src[nz], dst=dst[nz], weight=w[nz], num_nodes=G * N, mass=mass)
