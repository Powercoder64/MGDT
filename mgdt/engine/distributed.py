"""Process-group setup and collective helpers that also work without distributed training."""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main(self):
        return self.rank == 0


def init_distributed():
    """Initialises torch.distributed from torchrun environment variables when WORLD_SIZE > 1."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistInfo()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return DistInfo(True, rank, world_size, local_rank)


def get_device(info):
    """CUDA device of this process, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda", info.local_rank)
    return torch.device("cpu")


def is_enabled():
    return dist.is_available() and dist.is_initialized()


def barrier():
    if is_enabled():
        dist.barrier()


def all_reduce_sum(tensor):
    """In-place sum across processes; returns the tensor."""
    if is_enabled():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_reduce_min(tensor):
    if is_enabled():
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return tensor


def all_gather_objects(obj):
    """Gathers a picklable object from every process into a list."""
    if not is_enabled():
        return [obj]
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def broadcast_object(obj, src=0):
    if not is_enabled():
        return obj
    holder = [obj]
    dist.broadcast_object_list(holder, src=src)
    return holder[0]


def cleanup():
    if is_enabled():
        dist.destroy_process_group()
