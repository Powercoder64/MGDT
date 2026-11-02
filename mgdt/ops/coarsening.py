"""Graph coarsening: heavy-edge matching, k-means clustering, restriction and coarse operators."""

import math

import torch

from ..utils.flops import record
from .sparse import SparseGraph, safe_reciprocal

_TIE_SALT = [0]


def set_tie_break_seed(seed):
    """Sets the salt of the deterministic tie-breaking hash used by heavy-edge matching."""
    _TIE_SALT[0] = int(seed) % 1000003


def _tie_break(i, j):
    """Symmetric pseudo-random value in [0, 1) for each node pair."""
    lo = torch.minimum(i, j).double()
    hi = torch.maximum(i, j).double()
    x = torch.sin(lo * 12.9898 + hi * 78.233 + _TIE_SALT[0] * 0.6180339887) * 43758.5453
    return x - torch.floor(x)


@torch.no_grad()
def heavy_edge_matching(graph, rounds=4):
    """Pairs nodes along their heaviest incident edges with parallel handshake matching.

    In each round every unmatched node proposes to its unmatched neighbour of largest
    conductance; mutual proposals are matched. Unmatched nodes remain singletons.

    Returns:
        assignment (num_nodes,) of coarse ids and the number of coarse nodes.
    """
    V = graph.num_nodes
    device = graph.weight.device
    src, dst = graph.src, graph.dst
    weight = graph.weight.detach()
    key = weight.double() + 1e-9 * _tie_break(src, dst)
    usable = (src != dst) & (weight > 0)
    ar = torch.arange(V, device=device)
    partner = ar.clone()
    matched = torch.zeros(V, dtype=torch.bool, device=device)
    for _ in range(int(rounds)):
        cand = usable & ~matched[src] & ~matched[dst]
        if not bool(cand.any()):
            break
        d, s, kk = dst[cand], src[cand], key[cand]
        best = torch.full((V,), float("-inf"), dtype=torch.float64, device=device)
        best = best.scatter_reduce(0, d, kk, reduce="amax", include_self=True)
        top = kk == best[d]
        choice = torch.full((V,), V, dtype=torch.long, device=device)
        choice = choice.scatter_reduce(0, d[top], s[top], reduce="amin", include_self=True)
        has = choice < V
        cand_i = ar[has]
        cand_j = choice[has]
        mutual = choice[cand_j] == cand_i
        partner[cand_i[mutual]] = cand_j[mutual]
        matched[cand_i[mutual]] = True
    is_rep = ar <= partner
    cid = torch.cumsum(is_rep.long(), 0) - 1
    assign = cid[torch.minimum(ar, partner)]
    return assign, int(is_rep.sum().item())


@torch.no_grad()
def kmeans_assignment(feats, mass, ratio=2, iters=5):
    """Clusters the nodes of every graph with Lloyd iterations into ``ceil(N / ratio)`` groups.

    Args:
        feats: (G, N, P) node features.
        mass: (G * N,) node masses; zero-mass nodes do not move centroids.

    Returns:
        assignment (G * N,) with fixed layout ``g * C + c`` and the number of coarse nodes G * C.
    """
    G, N, P = feats.shape
    C = max(1, math.ceil(N / ratio))
    w = mass.view(G, N).to(feats.dtype)
    init = torch.linspace(0, N - 1, C, device=feats.device).round().long()
    cent = feats[:, init].clone()
    f_sq = feats.pow(2).sum(-1)
    labels = None
    for it in range(int(iters) + 1):
        dist = f_sq[:, :, None] - 2.0 * torch.matmul(feats, cent.transpose(1, 2)) + cent.pow(2).sum(-1)[:, None, :]
        labels = dist.argmin(-1)
        if it == int(iters):
            break
        sums = torch.zeros(G, C, P, dtype=feats.dtype, device=feats.device)
        sums.scatter_add_(1, labels[..., None].expand(G, N, P), feats * w[..., None])
        cnt = torch.zeros(G, C, dtype=feats.dtype, device=feats.device).scatter_add_(1, labels, w)
        cent = torch.where(cnt[..., None] > 0, sums / cnt.clamp_min(1e-12)[..., None], cent)
    record(G * N * C * P * (int(iters) + 1))
    offsets = (torch.arange(G, device=feats.device) * C)[:, None]
    return (labels + offsets).reshape(-1), G * C


def coarse_mass(assign, mass, num_coarse):
    """Total fine mass of every coarse node."""
    return torch.zeros(num_coarse, dtype=mass.dtype, device=mass.device).index_add(0, assign, mass)


def restrict(x, assign, mass, mass_c):
    """Mass-weighted average of fine node values over each coarse node: (V, D) -> (Vc, D)."""
    record(x.numel())
    num = torch.zeros(mass_c.numel(), x.shape[1], dtype=x.dtype, device=x.device)
    num = num.index_add(0, assign, x * mass[:, None])
    return num * safe_reciprocal(mass_c)[:, None]


def prolong(x_c, assign):
    """Copies every coarse value to its fine members."""
    record(assign.numel() * x_c.shape[1])
    return x_c[assign]


def galerkin_coarsen(graph, assign, num_coarse, mass_c=None):
    """Coarse graph whose entry (I, J) is the total fine conductance between clusters I and J.

    Intra-cluster entries are removed and duplicate entries merged, so the operator
    restricted by mass-weighted averaging and prolonged by injection equals the coarse
    mass-weighted Laplacian.
    """
    if mass_c is None:
        mass_c = coarse_mass(assign, graph.mass, num_coarse)
    I = assign[graph.dst]
    J = assign[graph.src]
    keep = I != J
    keys = I[keep] * num_coarse + J[keep]
    w = graph.weight[keep]
    if keys.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=assign.device)
        return SparseGraph(empty, empty, w, num_coarse, mass_c)
    uniq, inverse = torch.unique(keys, return_inverse=True)
    w_c = torch.zeros(uniq.numel(), dtype=w.dtype, device=w.device).index_add(0, inverse, w)
    record(keys.numel())
    return SparseGraph(src=uniq % num_coarse, dst=uniq // num_coarse, weight=w_c,
                       num_nodes=num_coarse, mass=mass_c)


def coarsen(graph, method="hem", factor=2, rounds=4, feats=None, kmeans_iters=5):
    """Returns the fine-to-coarse assignment and coarse node count for one level.

    ``method='hem'`` matches node pairs (``factor=4`` matches twice); ``method='kmeans'``
    clusters ``feats`` (G, N, P) into ``ceil(N / factor)`` groups per graph.
    """
    if method == "hem":
        assign, num_coarse = heavy_edge_matching(graph, rounds)
        if factor == 4:
            with torch.no_grad():
                mass_c = coarse_mass(assign, graph.mass, num_coarse)
                detached = SparseGraph(graph.src, graph.dst, graph.weight.detach(), graph.num_nodes, graph.mass)
                inter = galerkin_coarsen(detached, assign, num_coarse, mass_c)
            assign2, num_coarse = heavy_edge_matching(inter, rounds)
            assign = assign2[assign]
        elif factor != 2:
            raise ValueError(f"heavy-edge matching supports factors 2 and 4, got {factor}")
        return assign, num_coarse
    if method == "kmeans":
        if feats is None:
            raise ValueError("k-means coarsening requires node features")
        return kmeans_assignment(feats, graph.mass, ratio=factor, iters=kmeans_iters)
    raise ValueError(f"Unknown coarsening method '{method}'")
