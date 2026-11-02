"""DataLoader construction for single-process and distributed runs."""

import logging
import math

import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Sampler, SequentialSampler

from ..utils.misc import worker_init_fn

logger = logging.getLogger("mgdt")


class ShardSampler(Sampler):
    """Deterministic, non-padded split of a dataset across processes (rank::world_size)."""

    def __init__(self, dataset, rank, world_size):
        self.indices = list(range(rank, len(dataset), world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def per_process_batch(global_batch, dist_info, accum_steps=1):
    """Per-process micro-batch size for a global batch size split over processes and accumulation."""
    denom = dist_info.world_size * max(1, int(accum_steps))
    size = max(1, int(global_batch) // denom)
    if size * denom != int(global_batch):
        logger.warning("Global batch %d is not divisible by %d processes x %d accumulation steps; using %d",
                       global_batch, dist_info.world_size, accum_steps, size * denom)
    return size


def make_loader(dataset, batch_size, train, dist_info, num_workers=4, collate_fn=None, seed=0):
    """Training loaders shuffle and drop the last incomplete batch; evaluation loaders keep order."""
    if train:
        if dist_info.enabled:
            sampler = DistributedSampler(dataset, num_replicas=dist_info.world_size, rank=dist_info.rank,
                                         shuffle=True, seed=seed, drop_last=True)
        else:
            generator = torch.Generator()
            generator.manual_seed(seed)
            sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = ShardSampler(dataset, dist_info.rank, dist_info.world_size) if dist_info.enabled \
            else SequentialSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )
    if train and len(loader) == 0:
        raise RuntimeError(f"Training set of {len(dataset)} samples is smaller than one batch of {batch_size}")
    return loader, sampler


def steps_per_epoch(loader, accum_steps):
    """Optimiser steps per pass over a loader."""
    return max(1, math.floor(len(loader) / max(1, int(accum_steps))))
