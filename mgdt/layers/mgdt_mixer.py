"""Multigrid anisotropic diffusion token mixer."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.conductance import Conductance
from ..ops.graph import build_token_graph, num_local_neighbors
from ..ops.multigrid import SolverConfig, build_hierarchy, diffusion_energy, v_cycle

SIMILARITY_SPACES = ("key", "query", "shared", "qk", "asym")
METRICS = ("l2", "cos")


class MGDTMixer(nn.Module):
    """Replaces softmax attention by anisotropic diffusion of value features on a token graph.

    Per head, tokens are linked to their local grid neighbours and their ``k`` nearest
    tokens in a learned similarity space; edge conductances ``g(||u_i - u_j||)`` come from
    :class:`Conductance`. Values are diffused with one multigrid V-cycle (degree-normalised
    explicit Euler smoothing, heavy-edge-matching or k-means coarsening), then heads are
    concatenated and projected. Graph construction and diffusion run in FP32.

    Similarity spaces: ``key`` / ``query`` (a per-head linear projection), ``shared`` (a small
    MLP projector shared by all heads), ``qk`` (projection of concatenated per-head query and
    key) and ``asym`` (non-symmetric distances ``||q_i - k_j||``, graph not symmetrised).
    """

    def __init__(self, dim, num_heads, k=16, local_radius=1, levels=1, coarsening="hem", coarsen_factor=2,
                 hem_rounds=4, kmeans_iters=5, pre_smooth=1, post_smooth=1, coarse_steps=2, tau=0.5,
                 normalization="degree", conductance="exp", threshold=None, kappa_init=5.0, kappa_mode="head",
                 kappa_warmup=True, kappa_warmup_start=0.1, similarity="key", metric="l2", shared_dim=None,
                 clip_norm=5.0, drop_threshold=1e-4, qkv_bias=True, proj_drop=0.0, track_energy=False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if similarity not in SIMILARITY_SPACES:
            raise ValueError(f"similarity must be one of {SIMILARITY_SPACES}")
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        if coarsening not in ("hem", "kmeans"):
            raise ValueError("coarsening must be 'hem' or 'kmeans'")
        if normalization not in ("degree", "none"):
            raise ValueError("normalization must be 'degree' or 'none'")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.k = int(k)
        self.local_radius = int(local_radius)
        self.levels = int(levels)
        self.coarsening = coarsening
        self.coarsen_factor = int(coarsen_factor)
        self.hem_rounds = int(hem_rounds)
        self.kmeans_iters = int(kmeans_iters)
        self.similarity = similarity
        self.metric = metric
        self.clip_norm = None if clip_norm is None else float(clip_norm)
        self.drop_threshold = float(drop_threshold or 0.0)
        self.track_energy = bool(track_energy)
        self.solver = SolverConfig(int(pre_smooth), int(post_smooth), int(coarse_steps), float(tau), normalization, 1.0)
        self.last_energy_reduction = None

        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        if similarity == "key":
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        elif similarity == "query":
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        elif similarity == "shared":
            sd = int(shared_dim or self.head_dim)
            self.phi = nn.Sequential(nn.Linear(dim, 4 * sd), nn.GELU(), nn.Linear(4 * sd, sd))
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            if similarity == "qk":
                self.phi = nn.Linear(2 * self.head_dim, self.head_dim)
        self.conductance = Conductance(num_heads, conductance, kappa_init, kappa_mode, kappa_warmup,
                                       kappa_warmup_start, threshold)
        if self.conductance.is_isotropic:
            for name, param in self.named_parameters():
                if not name.startswith("v_proj"):
                    param.requires_grad_(False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _heads(self, t):
        B, N, _ = t.shape
        return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def similarity_features(self, x):
        """Returns (a, b): similarity features (B, H or 1, N, P) and optional asymmetric keys."""
        if self.similarity == "key":
            return self._heads(self.k_proj(x)), None
        if self.similarity == "query":
            return self._heads(self.q_proj(x)), None
        if self.similarity == "shared":
            return self.phi(x).unsqueeze(1), None
        q = self._heads(self.q_proj(x))
        k = self._heads(self.k_proj(x))
        if self.similarity == "qk":
            return self.phi(torch.cat([q, k], dim=-1)), None
        return q, k

    def _prepare(self, f):
        f = f.float()
        if self.metric == "cos":
            return F.normalize(f, dim=-1, eps=1e-6)
        if self.clip_norm is not None:
            norm = f.norm(dim=-1, keepdim=True)
            f = f * (self.clip_norm / norm.clamp_min(self.clip_norm))
        return f

    def forward(self, x, grid, token_mask=None):
        B, N, C = x.shape
        if math.prod(grid) != N:
            raise ValueError(f"grid {tuple(grid)} does not match {N} tokens")
        H, Dh = self.num_heads, self.head_dim
        v = self._heads(self.v_proj(x))
        a, b = self.similarity_features(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            a = self._prepare(a)
            b = None if b is None else self._prepare(b)
            graph = build_token_graph(a, b, self.conductance, grid, self.local_radius, self.k, H,
                                      token_mask=token_mask, symmetric=self.similarity != "asym",
                                      drop_threshold=self.drop_threshold)
            feats = None
            if self.levels > 0 and self.coarsening == "kmeans":
                feats = a.detach().expand(B, H, N, a.shape[-1]).reshape(B * H, N, a.shape[-1])
            graphs, assigns = build_hierarchy(graph, self.levels, self.coarsening, self.coarsen_factor,
                                              self.hem_rounds, feats, self.kmeans_iters)
            slots = num_local_neighbors(len(grid), self.local_radius) + min(self.k, N - 1)
            solver = self.solver._replace(unnormalized_scale=1.0 / max(1, slots))
            u = v.float().reshape(B * H * N, Dh)
            out = v_cycle(graphs, assigns, u, None, solver)
            if self.track_energy:
                with torch.no_grad():
                    e0 = diffusion_energy(graph, u.detach())
                    e1 = diffusion_energy(graph, out.detach())
                    self.last_energy_reduction = float((1.0 - e1 / e0.clamp_min(1e-12)).item())
        out = out.view(B, H, N, Dh).transpose(1, 2).reshape(B, N, C).to(x.dtype)
        return self.proj_drop(self.proj(out))
