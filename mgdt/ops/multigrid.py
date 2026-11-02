"""Graph hierarchies and the multigrid V-cycle for anisotropic graph diffusion."""

from typing import NamedTuple

import torch

from .coarsening import coarse_mass, coarsen, galerkin_coarsen, prolong, restrict
from .sparse import safe_reciprocal


class SolverConfig(NamedTuple):
    """Settings of one V-cycle."""

    pre_smooth: int = 1
    post_smooth: int = 1
    coarse_steps: int = 2
    tau: float = 0.5
    normalization: str = "degree"
    unnormalized_scale: float = 1.0


def diffusion_flux(graph, x):
    """Returns ``W x - D x`` (minus the unnormalised Laplacian applied to x)."""
    return graph.matvec(x) - graph.degree[:, None] * x


def smooth(graph, x, f, cfg):
    """One explicit diffusion step on ``A x = f`` with ``A = M^-1 (D - W)``.

    ``degree`` normalisation gives ``x + tau (M f + W x - D x) / D``; ``none`` gives
    ``x + tau * s * (M f + W x - D x) / M`` with a constant scale ``s``.
    """
    flux = diffusion_flux(graph, x)
    if f is not None:
        flux = flux + graph.mass[:, None] * f
    if cfg.normalization == "degree":
        return x + cfg.tau * flux * safe_reciprocal(graph.degree)[:, None]
    if cfg.normalization == "none":
        return x + (cfg.tau * cfg.unnormalized_scale) * flux * safe_reciprocal(graph.mass)[:, None]
    raise ValueError(f"Unknown normalization '{cfg.normalization}'")


def residual(graph, x, f):
    """Residual ``f - A x`` of the mass-weighted diffusion system."""
    r = diffusion_flux(graph, x) * safe_reciprocal(graph.mass)[:, None]
    return r if f is None else r + f


def diffusion_energy(graph, x):
    """Diffusion energy ``0.5 * x^T (D - W) x`` summed over feature channels."""
    x = x.to(graph.weight.dtype)
    return 0.5 * (x * (graph.degree[:, None] * x - graph.matvec(x))).sum()


def build_hierarchy(graph, levels, method="hem", factor=2, rounds=4, feats=None, kmeans_iters=5):
    """Builds ``levels`` successively coarser Galerkin graphs below ``graph``.

    Returns:
        list of graphs (finest first) and the list of fine-to-coarse assignments.
    """
    graphs, assigns = [graph], []
    current, current_feats = graph, feats
    for _ in range(int(levels)):
        if current.num_nodes <= 1:
            break
        assign, num_coarse = coarsen(current, method, factor, rounds, current_feats, kmeans_iters)
        mass_c = coarse_mass(assign, current.mass, num_coarse)
        coarse = galerkin_coarsen(current, assign, num_coarse, mass_c)
        if method == "kmeans":
            G = current_feats.shape[0]
            flat = current_feats.reshape(-1, current_feats.shape[-1])
            current_feats = restrict(flat, assign, current.mass, mass_c).view(G, -1, flat.shape[-1])
        graphs.append(coarse)
        assigns.append(assign)
        current = coarse
    return graphs, assigns


def v_cycle(graphs, assigns, x, f, cfg, level=0):
    """Applies one multigrid V-cycle to the diffusion system on ``graphs[level]``.

    Pre-smoothing, restriction of the residual, a recursive coarse correction, prolongation
    and post-smoothing; the coarsest level runs ``cfg.coarse_steps`` smoothing steps from zero.
    """
    graph = graphs[level]
    last = len(graphs) - 1
    if level == last and level > 0:
        for _ in range(cfg.coarse_steps):
            x = smooth(graph, x, f, cfg)
        return x
    for _ in range(cfg.pre_smooth):
        x = smooth(graph, x, f, cfg)
    if level < last:
        coarse = graphs[level + 1]
        assign = assigns[level]
        f_c = restrict(residual(graph, x, f), assign, graph.mass, coarse.mass)
        e_c = v_cycle(graphs, assigns, torch.zeros_like(f_c), f_c, cfg, level + 1)
        x = x + prolong(e_c, assign)
    for _ in range(cfg.post_smooth):
        x = smooth(graph, x, f, cfg)
    return x
