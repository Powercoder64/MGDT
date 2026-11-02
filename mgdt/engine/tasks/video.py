"""Kinetics-400/700 action recognition with multi-view evaluation."""

import torch
import torch.nn.functional as F

from ...data import KineticsDataset
from ...metrics import topk_correct
from ..checkpoint import unwrap
from ..distributed import all_reduce_sum
from ..loader import make_loader, per_process_batch
from .base import Task


class KineticsTask(Task):
    primary_metric = "top1"

    def init_from_pretrained(self, model, state_dict):
        return unwrap(model).load_image_state_dict(state_dict)

    def _dataset(self, train):
        d = self.cfg.data
        return KineticsDataset(
            root=d.root,
            list_file=d.train_list if train else d.val_list,
            train=train,
            num_frames=int(d.num_frames),
            frame_stride=int(d.frame_stride),
            img_size=int(d.img_size),
            repeated_aug=int(d.get("repeated_aug", 1)) if train else 1,
            num_clips=int(d.get("test_clips", 4)),
            num_crops=int(d.get("test_crops", 3)),
            scale=tuple(d.get("crop_scale", (0.08, 1.0))),
        )

    def build_train_loader(self):
        repeats = int(self.cfg.data.get("repeated_aug", 1))
        clips = per_process_batch(self.cfg.train.batch_size, self.dist, self.cfg.train.get("accum_steps", 1))
        videos = max(1, clips // repeats)
        return make_loader(self._dataset(True), videos, True, self.dist, self.num_workers, seed=int(self.cfg.seed))

    def build_eval_loader(self):
        loader, _ = make_loader(self._dataset(False), self.eval_batch_size(8), False, self.dist, self.num_workers)
        return loader

    def train_step(self, model, batch):
        clips, labels = batch
        B, R = clips.shape[:2]
        clips = clips.flatten(0, 1)
        labels = labels.repeat_interleave(R)
        logits = model(clips)
        loss = F.cross_entropy(logits.float(), labels,
                               label_smoothing=float(self.cfg.train.get("label_smoothing", 0.0)))
        return loss, {}

    @torch.no_grad()
    def evaluate(self, model, loader):
        model.eval()
        stats = torch.zeros(3, dtype=torch.float64, device=self.device)
        for views, labels in loader:
            views = views.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            probs = 0
            for v in range(views.shape[1]):
                with self.autocast():
                    logits = model(views[:, v])
                probs = probs + logits.float().softmax(-1)
            c1, c5 = topk_correct(probs, labels, (1, 5))
            stats += torch.tensor([c1, c5, labels.numel()], dtype=torch.float64, device=self.device)
        all_reduce_sum(stats)
        n = max(1.0, float(stats[2]))
        return {"top1": 100.0 * float(stats[0]) / n, "top5": 100.0 * float(stats[1]) / n, "num_samples": int(n)}
