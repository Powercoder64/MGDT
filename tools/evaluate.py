"""Evaluate a trained checkpoint with the benchmark's evaluation protocol.

python tools/evaluate.py --config configs/ade20k.yaml --checkpoint outputs/ade20k/final.pth
"""

import argparse

import _bootstrap  # noqa: F401

from mgdt.engine import evaluate
from mgdt.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate an MGDT checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ema", action="store_true", help="evaluate the EMA weights stored in the checkpoint")
    parser.add_argument("--track-energy", action="store_true",
                        help="also report the mean per-cycle diffusion-energy reduction")
    parser.add_argument("--output-dir", help="where eval_results.json is written (default: config output_dir)")
    parser.add_argument("--opts", nargs="*", default=[], help="config overrides as key=value")
    args = parser.parse_args()
    cfg = load_config(args.config, args.opts)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    cfg.model.pretrained = None
    metrics = evaluate(cfg, args.checkpoint, use_ema=args.ema, track_energy=args.track_energy)
    print(metrics)


if __name__ == "__main__":
    main()
