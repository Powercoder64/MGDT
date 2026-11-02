"""Long Range Arena byte-level text classification."""

import torch
import torch.nn.functional as F

from ...data import IMDBByteDataset
from ..distributed import all_reduce_sum
from ..loader import make_loader, per_process_batch
from .base import Task


class TextClassificationTask(Task):
    primary_metric = "accuracy"

    def _dataset(self, split):
        d = self.cfg.data
        return IMDBByteDataset(d.root, split, max_len=int(d.max_len))

    def build_train_loader(self):
        bs = per_process_batch(self.cfg.train.batch_size, self.dist, self.cfg.train.get("accum_steps", 1))
        return make_loader(self._dataset(self.cfg.data.get("train_split", "train")), bs, True, self.dist,
                           self.num_workers, seed=int(self.cfg.seed))

    def build_eval_loader(self):
        loader, _ = make_loader(self._dataset(self.cfg.data.get("eval_split", "test")), self.eval_batch_size(),
                                False, self.dist, self.num_workers)
        return loader

    def train_step(self, model, batch):
        tokens, labels = batch
        loss = F.cross_entropy(model(tokens).float(), labels)
        return loss, {}

    @torch.no_grad()
    def evaluate(self, model, loader):
        model.eval()
        stats = torch.zeros(2, dtype=torch.float64, device=self.device)
        for tokens, labels in loader:
            tokens = tokens.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with self.autocast():
                logits = model(tokens)
            stats += torch.tensor([(logits.argmax(-1) == labels).sum().item(), labels.numel()],
                                  dtype=torch.float64, device=self.device)
        all_reduce_sum(stats)
        return {"accuracy": 100.0 * float(stats[0]) / max(1.0, float(stats[1])), "num_samples": int(stats[1])}
