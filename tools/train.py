"""Train MGDT on one benchmark.

Single GPU:  python tools/train.py --config configs/imagenet1k.yaml
Multi-GPU:   torchrun --nproc_per_node=8 tools/train.py --config configs/imagenet1k.yaml
"""

import argparse

import _bootstrap  # noqa: F401

from mgdt.engine import train
from mgdt.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Train MGDT")
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--output-dir", help="overrides output_dir")
    parser.add_argument("--seed", type=int, help="overrides seed")
    parser.add_argument("--opts", nargs="*", default=[], help="config overrides as key=value (YAML values)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.opts)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = args.seed
    train(cfg)


if __name__ == "__main__":
    main()
