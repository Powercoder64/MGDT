"""Latency and peak-memory scaling of the MGDT mixer versus softmax attention.

Both operators use the same width, heads, batch size and precision; MGDT latency includes
graph construction and every V-cycle operation. Results are averaged over ``--repeats`` runs
and written as CSV, a Markdown table and a log-log plot.

python tools/benchmark_scaling.py --tokens 256 512 1024 2048 4096 8192 --output-dir outputs/scaling
"""

import argparse
import csv
import os
import time

import _bootstrap  # noqa: F401
import torch

from mgdt.layers import build_mixer
from mgdt.utils import load_config, seed_everything


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_once(module, x, grid, device, amp, backward):
    """Returns (latency in ms, peak memory in GB or None) for one forward (+backward) pass."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    start = time.perf_counter()
    ctx = torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                         enabled=amp)
    if backward:
        with ctx:
            out = module(x, grid)
        out.float().sum().backward()
    else:
        with torch.no_grad(), ctx:
            module(x, grid)
    _sync(device)
    latency = (time.perf_counter() - start) * 1000.0
    memory = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    return latency, memory


def measure(module, tokens, args, device):
    x = torch.randn(args.batch_size, tokens, args.dim, device=device, requires_grad=args.backward)
    grid = (tokens,)
    try:
        for _ in range(args.warmup):
            run_once(module, x, grid, device, args.amp, args.backward)
        lat, mem = [], []
        for _ in range(args.repeats):
            ms, gb = run_once(module, x, grid, device, args.amp, args.backward)
            lat.append(ms)
            mem.append(gb)
        return sum(lat) / len(lat), (max(mem) if mem[0] is not None else None)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def main():
    parser = argparse.ArgumentParser(description="MGDT vs softmax attention scaling benchmark")
    parser.add_argument("--config", default="configs/imagenet1k.yaml", help="config whose MGDT mixer is used")
    parser.add_argument("--tokens", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="run under mixed precision")
    parser.add_argument("--backward", action="store_true", help="time forward + backward")
    parser.add_argument("--output-dir", default="outputs/scaling")
    parser.add_argument("--opts", nargs="*", default=[])
    args = parser.parse_args()

    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config, args.opts)
    mixer_cfg = dict(cfg.model.get("mixer") or cfg.model.get("backbone", {}).get("mixer"))
    mgdt = build_mixer(args.dim, args.heads, dict(mixer_cfg, type="mgdt")).to(device)
    softmax = build_mixer(args.dim, args.heads, {"type": "softmax"}).to(device)

    rows = []
    for n in args.tokens:
        s_ms, s_gb = measure(softmax, n, args, device)
        m_ms, m_gb = measure(mgdt, n, args, device)
        rows.append({
            "tokens": n,
            "softmax_ms": s_ms, "mgdt_ms": m_ms,
            "speedup": (s_ms / m_ms) if (s_ms and m_ms) else None,
            "softmax_gb": s_gb, "mgdt_gb": m_gb,
            "memory_reduction_pct": (100.0 * (1 - m_gb / s_gb)) if (s_gb and m_gb) else None,
        })
        print(rows[-1])

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "scaling.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def fmt(v, spec):
        return "OOM" if v is None else format(v, spec)

    lines = ["| Tokens | Softmax (ms) | MGDT (ms) | Speedup | Softmax (GB) | MGDT (GB) | Reduction |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        speed = "-" if r["speedup"] is None else f"{r['speedup']:.2f}x"
        red = "-" if r["memory_reduction_pct"] is None else f"{r['memory_reduction_pct']:.1f}%"
        s_gb = "n/a" if device.type != "cuda" else fmt(r["softmax_gb"], ".2f")
        m_gb = "n/a" if device.type != "cuda" else fmt(r["mgdt_gb"], ".2f")
        lines.append(f"| {r['tokens']} | {fmt(r['softmax_ms'], '.1f')} | {fmt(r['mgdt_ms'], '.1f')} | {speed} | "
                     f"{s_gb} | {m_gb} | {red} |")
    table = "\n".join(lines)
    with open(os.path.join(args.output_dir, "scaling.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(table)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("softmax_ms", "mgdt_ms", "Latency (ms)", "Runtime scaling")]
    if device.type == "cuda":
        panels.append(("softmax_gb", "mgdt_gb", "Peak GPU memory (GB)", "Memory scaling"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4), squeeze=False)
    for ax, (s_key, m_key, ylabel, title) in zip(axes[0], panels):
        for key, label, color, marker in ((s_key, "Softmax attention", "#c0392b", "o"),
                                          (m_key, "MGDT", "#2c6e9b", "s")):
            pts = [(r["tokens"], r[key]) for r in rows if r[key] is not None]
            if pts:
                ax.plot(*zip(*pts), label=label, color=color, marker=marker, linewidth=2)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("Number of tokens")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "scaling.png"), dpi=200)
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
