"""Toy step-edge experiment: anisotropic diffusion, isotropic diffusion and softmax attention.

A noisy 32x32 image (left half 0, right half 1) is treated as 1024 scalar tokens. Diffusion
rows run degree-normalised explicit diffusion steps on the 4-connected pixel graph with
exponential (anisotropic) or constant (isotropic) conductance; the softmax row applies
repeated dot-product self-attention. The figure shows the field and the left/right means.

python tools/visualize_diffusion.py --output outputs/toy_step_edge.png
"""

import argparse
import os

import _bootstrap  # noqa: F401
import torch

from mgdt.ops import Conductance, SolverConfig, SparseGraph, local_neighbors, smooth


def grid_graph(size, values, conductance):
    """4-connected pixel graph with conductance g(|u_i - u_j|) from the current values."""
    idx, ok = local_neighbors((size, size), 1, values.device)
    four = torch.tensor([abs(dy) + abs(dx) == 1 for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
    idx, ok = idx[:, four], ok[:, four]
    N = size * size
    dst = torch.arange(N)[:, None].expand_as(idx)[ok]
    src = idx[ok]
    sq = (values[dst] - values[src]).pow(2)
    w = conductance(sq.view(1, 1, -1, 1)).view(-1)
    return SparseGraph(src=src, dst=dst, weight=w, num_nodes=N, mass=torch.ones(N))


def diffuse(values, size, conductance, steps, tau):
    cfg = SolverConfig(tau=tau)
    u = values.clone()
    out = {0: u.clone()}
    for t in range(1, steps + 1):
        graph = grid_graph(size, u, conductance)
        u = smooth(graph, u[:, None], None, cfg)[:, 0]
        out[t] = u.clone()
    return out


def attend(values, steps, beta):
    u = values.clone()
    out = {0: u.clone()}
    for t in range(1, steps + 1):
        attn = torch.softmax(beta * u[:, None] * u[None, :], dim=-1)
        u = attn @ u
        out[t] = u.clone()
    return out


def main():
    parser = argparse.ArgumentParser(description="Toy step-edge diffusion comparison")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--kappa", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0, help="softmax temperature for scalar tokens")
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 1, 3, 5, 10])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/toy_step_edge.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    S = args.size
    clean = torch.zeros(S, S)
    clean[:, S // 2:] = 1.0
    noisy = (clean + args.noise * torch.randn(S, S)).view(-1)
    T = max(args.steps)
    aniso = Conductance(1, "exp", kappa_init=args.kappa, kappa_mode="fixed", warmup=False)
    iso = Conductance(1, "isotropic", warmup=False)
    with torch.no_grad():
        rows = [
            ("Anisotropic\ndiffusion (MGDT)", diffuse(noisy, S, aniso, T, args.tau)),
            ("Isotropic\ndiffusion", diffuse(noisy, S, iso, T, args.tau)),
            ("Softmax\nattention", attend(noisy, T, args.beta)),
        ]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(2.2 * len(args.steps) + 1.5, 9))
    outer = fig.add_gridspec(len(rows) * 2, len(args.steps), height_ratios=[3, 1.2] * len(rows),
                             hspace=0.45, wspace=0.08)
    image = None
    for r, (name, traj) in enumerate(rows):
        for c, t in enumerate(args.steps):
            ax = fig.add_subplot(outer[2 * r, c])
            image = ax.imshow(traj[t].view(S, S).numpy(), cmap="coolwarm", vmin=0, vmax=1)
            ax.axvline(S / 2 - 0.5, color="k", linestyle="--", linewidth=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t = {t}")
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
        ax = fig.add_subplot(outer[2 * r + 1, :])
        ts = list(range(T + 1))
        left = [traj[t].view(S, S)[:, : S // 2].mean().item() for t in ts]
        right = [traj[t].view(S, S)[:, S // 2:].mean().item() for t in ts]
        ax.plot(ts, left, "o-", color="#2c6e9b", markersize=3, label="Left half mean")
        ax.plot(ts, right, "s-", color="#e67e22", markersize=3, label="Right half mean")
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel("Average value", fontsize=8)
        ax.legend(fontsize=7, loc="center right")
        if r == len(rows) - 1:
            ax.set_xlabel("Diffusion step / layer")
    fig.colorbar(image, ax=fig.axes, fraction=0.015, pad=0.01, label="Feature value")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    for name, traj in rows:
        contrast = (traj[T].view(S, S)[:, S // 2:].mean() - traj[T].view(S, S)[:, : S // 2].mean()).item()
        print(f"{name.replace(chr(10), ' ')}: mean {traj[T].mean():.4f} (initial {traj[0].mean():.4f}), "
              f"edge contrast after {T} steps {contrast:.3f}")
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
