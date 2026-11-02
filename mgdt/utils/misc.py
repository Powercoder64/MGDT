"""Seeding, logging and small helpers."""

import json
import logging
import os
import random
import sys

import numpy as np
import torch


def seed_everything(seed, deterministic=False):
    """Seeds Python, NumPy and PyTorch RNGs and optionally enables deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    from ..ops.coarsening import set_tie_break_seed

    set_tie_break_seed(seed)


def worker_init_fn(worker_id):
    """Seeds NumPy and Python RNGs of a DataLoader worker from its torch seed."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def setup_logger(output_dir=None, rank=0, name="mgdt"):
    """Creates a logger that prints on rank 0 and mirrors to ``output_dir/log.txt``."""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if output_dir is not None and rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(output_dir, "log.txt"))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def save_json(obj, path):
    """Writes ``obj`` as indented JSON, creating parent directories."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def count_parameters(model, trainable_only=False):
    """Returns the number of (trainable) parameters, counting tied tensors once."""
    seen = set()
    total = 0
    for p in model.parameters():
        if trainable_only and not p.requires_grad:
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


class AverageMeter:
    """Tracks a running weighted average."""

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(1, self.count)
