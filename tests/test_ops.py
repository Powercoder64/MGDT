import itertools

import pytest
import torch

from mgdt.ops import (Conductance, SolverConfig, SparseGraph, build_hierarchy, build_token_graph, coarse_mass,
                      coarsen, diffusion_energy, galerkin_coarsen, knn, smooth, v_cycle)
from mgdt.ops.distance import _EdgeSqDist
from mgdt.ops.sparse import _EdgeAggregate


def dense(graph):
    W = torch.zeros(graph.num_nodes, graph.num_nodes, dtype=graph.weight.dtype)
    W.index_put_((graph.dst, graph.src), graph.weight, accumulate=True)
    return W


def random_graph(B=2, H=3, side=6, P=8, k=5, kappa=3.0, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(B, H, side * side, P, dtype=torch.double)
    cond = Conductance(H, "exp", kappa_init=kappa).double()
    return a, build_token_graph(a, None, cond, (side, side), 1, k, H)


def test_edge_aggregate_gradcheck():
    torch.manual_seed(0)
    V, E, D = 7, 20, 3
    src, dst = torch.randint(0, V, (E,)), torch.randint(0, V, (E,))
    w = torch.rand(E, dtype=torch.double, requires_grad=True)
    x = torch.randn(V, D, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda w, x: _EdgeAggregate.apply(w, x, src, dst, V), (w, x))


def test_edge_sq_dist_gradcheck():
    torch.manual_seed(0)
    a = torch.randn(2, 6, 4, dtype=torch.double, requires_grad=True)
    b = torch.randn(2, 6, 4, dtype=torch.double, requires_grad=True)
    nbr = torch.randint(0, 6, (2, 6, 3))
    assert torch.autograd.gradcheck(lambda a, b: _EdgeSqDist.apply(a, b, nbr), (a, b))
    assert torch.autograd.gradcheck(lambda a: _EdgeSqDist.apply(a, a, nbr), (a,))


def test_knn_matches_brute_force():
    torch.manual_seed(0)
    a = torch.randn(3, 50, 8)
    idx, ok = knn(a, a, 5)
    d = torch.cdist(a, a)
    d.diagonal(dim1=1, dim2=2).fill_(float("inf"))
    ref = d.topk(5, largest=False).indices
    assert ok.all()
    assert torch.equal(idx.sort(-1).values, ref.sort(-1).values)


def test_token_graph_matches_dense_reference():
    a, g = random_graph()
    W = dense(g)
    assert torch.allclose(W, W.T)
    assert W.diagonal().abs().max() == 0
    N, side, kappa = 36, 6, 3.0
    af = a[0, 0]
    d = torch.cdist(af, af)
    d.fill_diagonal_(float("inf"))
    knn_idx = d.topk(5, largest=False).indices
    ref = torch.zeros(N, N, dtype=torch.double)
    for i in range(N):
        yi, xi = divmod(i, side)
        nbrs = {(yi + dy) * side + xi + dx for dy, dx in itertools.product((-1, 0, 1), repeat=2)
                if (dy, dx) != (0, 0) and 0 <= yi + dy < side and 0 <= xi + dx < side}
        for j in nbrs | set(knn_idx[i].tolist()):
            ref[i, j] = torch.exp(-(af[i] - af[j]).pow(2).sum() / kappa**2)
    ref = ref * (ref >= 1e-4)
    assert torch.allclose(W[:N, :N], 0.5 * (ref + ref.T))
    assert W[:N, N:].abs().max() == 0


@pytest.mark.parametrize("method,factor", [("hem", 2), ("hem", 4), ("kmeans", 2)])
def test_galerkin_consistency(method, factor):
    a, g = random_graph()
    feats = a.reshape(-1, a.shape[2], a.shape[3])
    assign, Vc = coarsen(g, method, factor, 4, feats)
    mass_c = coarse_mass(assign, g.mass, Vc)
    gc = galerkin_coarsen(g, assign, Vc, mass_c)
    W, Wc = dense(g), dense(gc)
    A = torch.diag(1 / g.mass) @ (torch.diag(W.sum(1)) - W)
    P = torch.zeros(g.num_nodes, Vc, dtype=torch.double)
    P[torch.arange(g.num_nodes), assign] = 1
    nz = mass_c > 0
    R = torch.diag(1 / mass_c[nz]) @ P[:, nz].T @ torch.diag(g.mass)
    Ac = torch.diag(1 / mass_c[nz]) @ (torch.diag(Wc.sum(1)) - Wc)[nz][:, nz]
    assert torch.allclose(R @ A @ P[:, nz], Ac)
    graph_id = torch.arange(g.num_nodes) // 36
    for c in range(Vc):
        members = (assign == c).nonzero().flatten()
        if len(members):
            assert len(set(graph_id[members].tolist())) == 1


def test_heavy_edge_matching_pairs_neighbours():
    _, g = random_graph()
    assign, Vc = coarsen(g, "hem", 2, 4)
    W = dense(g)
    sizes = torch.bincount(assign, minlength=Vc)
    assert sizes.max() <= 2 and sizes.min() >= 1
    for c in range(Vc):
        members = (assign == c).nonzero().flatten().tolist()
        if len(members) == 2:
            assert W[members[0], members[1]] > 0


def test_vcycle_reduces_energy_more_with_coarse_levels():
    _, g = random_graph()
    x = torch.randn(g.num_nodes, 4, dtype=torch.double)
    cfg = SolverConfig(1, 1, 2, 0.5, "degree", 1.0)
    e0 = diffusion_energy(g, x)
    energies = []
    for levels in (0, 1, 2):
        graphs, assigns = build_hierarchy(g, levels)
        energies.append(diffusion_energy(g, v_cycle(graphs, assigns, x, None, cfg)).item())
    assert energies[0] < e0.item()
    assert energies[1] < energies[0]
    assert energies[2] <= energies[1] + 1e-9


def test_unnormalised_vcycle_conserves_mass():
    _, g = random_graph()
    x = torch.randn(g.num_nodes, 4, dtype=torch.double)
    graphs, assigns = build_hierarchy(g, 2)
    y = v_cycle(graphs, assigns, x, None, SolverConfig(1, 1, 2, 0.5, "none", 1.0 / 13))
    assert torch.allclose(y.view(6, 36, 4).sum(1), x.view(6, 36, 4).sum(1), atol=1e-10)


def test_degree_normalised_step_conserves_degree_weighted_mass_and_is_stable():
    _, g = random_graph()
    x = torch.randn(g.num_nodes, 4, dtype=torch.double)
    cfg = SolverConfig(tau=1.0)
    y = smooth(g, x, None, cfg)
    d = g.degree[:, None]
    assert torch.allclose((d * y).view(6, 36, 4).sum(1), (d * x).view(6, 36, 4).sum(1), atol=1e-10)
    assert diffusion_energy(g, y) <= diffusion_energy(g, x)


def test_hard_threshold_preserves_boundary():
    N = 8
    u = torch.cat([torch.zeros(N // 2), torch.ones(N // 2)])
    src = torch.tensor([i for i in range(N) for j in range(N) if i != j])
    dst = torch.tensor([j for i in range(N) for j in range(N) if i != j])
    cond = Conductance(1, "exp", kappa_init=0.3, kappa_mode="fixed", warmup=False, threshold=2.0)
    w = cond(((u[src] - u[dst]) ** 2).view(1, 1, -1, 1)).view(-1)
    g = SparseGraph(src, dst, w, N, torch.ones(N))
    y = u[:, None]
    for _ in range(20):
        y = smooth(g, y, None, SolverConfig(tau=0.5))
    assert torch.allclose(y[:, 0], u)


def test_conductance_warmup_schedule():
    c = Conductance(2, "exp", kappa_init=4.0, warmup=True, warmup_start=0.1)
    c.set_progress(0.0)
    assert torch.allclose(c.kappa(), torch.full((2,), 40.0))
    c.set_progress(1.0)
    assert torch.allclose(c.kappa(), torch.full((2,), 4.0))
