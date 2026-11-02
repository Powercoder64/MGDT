"""Common task interface used by the trainer."""

import torch

from ...models import build_model


def move_to(batch, device):
    """Recursively moves tensors in nested lists/tuples/dicts to ``device``."""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to(v, device) for v in batch)
    return batch


class Task:
    """A benchmark: datasets, model construction, loss and evaluation protocol."""

    primary_metric = "loss"
    higher_is_better = True
    sync_batchnorm = False

    def __init__(self, cfg, device, dist_info, logger):
        self.cfg = cfg
        self.device = device
        self.dist = dist_info
        self.logger = logger

    def model_kwargs(self):
        """Extra constructor arguments derived from the data (e.g. vocabulary size)."""
        return {}

    def build_model(self):
        return build_model(self.cfg.model, **self.model_kwargs())

    def init_from_pretrained(self, model, state_dict):
        """Loads pretrained weights; returns (missing_keys, unexpected_keys)."""
        result = model.load_state_dict(state_dict, strict=False)
        return list(result.missing_keys), list(result.unexpected_keys)

    def build_train_loader(self):
        raise NotImplementedError

    def build_eval_loader(self):
        raise NotImplementedError

    def train_step(self, model, batch):
        """Returns (loss, dict of scalar logs) for one micro-batch already on the device."""
        raise NotImplementedError

    def evaluate(self, model, loader):
        """Returns a dict of metrics identical on every process."""
        raise NotImplementedError

    def autocast(self):
        amp = self.cfg.train.get("amp", "none")
        if amp in (None, "none", False):
            return torch.autocast(device_type=self.device.type, enabled=False)
        dtype = torch.float16 if amp == "fp16" and self.device.type == "cuda" else torch.bfloat16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def eval_batch_size(self, default=64):
        """Per-process evaluation batch size."""
        total = int(self.cfg.train.get("eval_batch_size", default))
        return max(1, total // self.dist.world_size)

    @property
    def num_workers(self):
        return int(self.cfg.data.get("num_workers", 8))
