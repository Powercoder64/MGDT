"""Runs and summarises the ablation studies defined in ``configs/ablations/*.yaml``.

Each spec lists benchmarks and named variants; a variant's ``mixer`` entries override the
benchmark's token-mixer config and its ``opts`` entries are absolute ``key: value`` overrides.

python tools/run_ablation.py --spec configs/ablations/mechanism.yaml --benchmark imagenet1k \
    --variant isotropic --nproc 8
python tools/run_ablation.py --spec configs/ablations/mechanism.yaml --summarize
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import _bootstrap  # noqa: F401
import numpy as np
import yaml

ROOT = _bootstrap.ROOT
BENCHMARKS = {
    "imagenet1k": ("configs/imagenet1k.yaml", "model.mixer"),
    "kinetics400": ("configs/kinetics400.yaml", "model.mixer"),
    "kinetics700": ("configs/kinetics700.yaml", "model.mixer"),
    "ade20k": ("configs/ade20k.yaml", "model.backbone.mixer"),
    "cityscapes": ("configs/cityscapes.yaml", "model.backbone.mixer"),
    "coco_captions": ("configs/coco_captions.yaml", "model.encoder.mixer"),
    "ego4d_nlq": ("configs/ego4d_nlq.yaml", "model.mixer"),
    "lra_text": ("configs/lra_text.yaml", "model.mixer"),
}


def load_spec(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def overrides_for(spec, benchmark, variant):
    _, mixer_key = BENCHMARKS[benchmark]
    v = spec["variants"][variant] or {}
    opts = [f"{mixer_key}.{k}={json.dumps(val)}" for k, val in (v.get("mixer") or {}).items()]
    common = dict(spec.get("opts") or {})
    common.update(v.get("opts") or {})
    for key, val in (spec.get("benchmark_opts", {}).get(benchmark) or {}).items():
        common[key] = val
    for key, val in ((v.get("benchmark_opts") or {}).get(benchmark) or {}).items():
        common[key] = val
    opts += [f"{k}={json.dumps(val)}" for k, val in common.items()]
    return opts


def run_dir(spec_path, benchmark, variant, seed):
    name = os.path.splitext(os.path.basename(spec_path))[0]
    return os.path.join("outputs", "ablations", name, benchmark, variant, f"seed{seed}")


def launch(spec_path, spec, benchmark, variant, seed, nproc, dry_run):
    config, _ = BENCHMARKS[benchmark]
    out = run_dir(spec_path, benchmark, variant, seed)
    if nproc > 1:
        cmd = ["torchrun", f"--nproc_per_node={nproc}", "tools/train.py"]
    else:
        cmd = [sys.executable, "tools/train.py"]
    cmd += ["--config", config, "--seed", str(seed), "--output-dir", out, "--opts"]
    cmd += overrides_for(spec, benchmark, variant)
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=ROOT)


def summarize(spec_path, spec):
    for benchmark in spec["benchmarks"]:
        print(f"\n## {benchmark}")
        keys = None
        for variant in spec["variants"]:
            files = sorted(glob.glob(os.path.join(ROOT, run_dir(spec_path, benchmark, variant, "*"), "results.json")))
            if not files:
                print(f"{variant:28s} (no results)")
                continue
            results = [json.load(open(p, encoding="utf-8")) for p in files]
            finished = [r for r in results if not r.get("diverged")]
            parts = [f"runs {len(results)}", f"diverged {100.0 * (len(results) - len(finished)) / len(results):.1f}%"]
            if finished:
                keys = keys or [k for k, v in finished[0].items() if isinstance(v, (int, float))
                                and k not in ("step", "epoch", "num_samples", "diverged", "best")]
                for k in keys:
                    vals = np.array([r[k] for r in finished if k in r], dtype=float)
                    if len(vals):
                        parts.append(f"{k} {vals.mean():.3f}±{vals.std(ddof=1) if len(vals) > 1 else 0.0:.3f}")
            print(f"{variant:28s} " + " | ".join(parts))


def main():
    parser = argparse.ArgumentParser(description="Run MGDT ablations")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--benchmark", help="one of the spec's benchmarks (default: all)")
    parser.add_argument("--variant", help="one of the spec's variants (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--nproc", type=int, default=1, help="GPUs per run (torchrun when > 1)")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    parser.add_argument("--summarize", action="store_true", help="print a summary of finished runs")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    if args.summarize:
        summarize(args.spec, spec)
        return
    benchmarks = [args.benchmark] if args.benchmark else list(spec["benchmarks"])
    variants = [args.variant] if args.variant else list(spec["variants"])
    for b in benchmarks:
        if b not in spec["benchmarks"]:
            raise ValueError(f"Benchmark '{b}' is not part of {args.spec}")
        for v in variants:
            if v not in spec["variants"]:
                raise ValueError(f"Variant '{v}' is not part of {args.spec}")
            for seed in args.seeds:
                launch(args.spec, spec, b, v, seed, args.nproc, args.dry_run)


if __name__ == "__main__":
    main()
