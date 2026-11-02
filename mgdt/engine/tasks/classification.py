"""ImageNet-1K image classification."""

import torch
import torch.nn.functional as F

from ...data import build_imagenet, imagenet_eval_transform, imagenet_train_transform
from ...metrics import topk_correct
from ..distributed import all_reduce_sum
from ..loader import make_loader, per_process_batch
from .base import Task


class ImageNetTask(Task):
    primary_metric = "top1"

    def __init__(self, cfg, device, dist_info, logger):
        super().__init__(cfg, device, dist_info, logger)
        mix = cfg.data.get("mixup") or {}
        self.mixup_fn = None
        if mix and (mix.get("mixup_alpha", 0) > 0 or mix.get("cutmix_alpha", 0) > 0):
            from timm.data import Mixup

            self.mixup_fn = Mixup(
                mixup_alpha=float(mix.get("mixup_alpha", 0.8)),
                cutmix_alpha=float(mix.get("cutmix_alpha", 1.0)),
                prob=float(mix.get("prob", 1.0)),
                switch_prob=float(mix.get("switch_prob", 0.5)),
                mode="batch",
                label_smoothing=float(cfg.train.get("label_smoothing", 0.1)),
                num_classes=int(cfg.model.num_classes),
            )

    def build_train_loader(self):
        d = self.cfg.data
        aug = d.get("train_aug", {})
        transform = imagenet_train_transform(
            img_size=int(d.img_size),
            rand_augment=aug.get("rand_augment", "rand-m9-mstd0.5-inc1"),
            random_erasing=float(aug.get("random_erasing", 0.25)),
            interpolation=aug.get("interpolation", "bicubic"),
            color_jitter=aug.get("color_jitter", 0.4),
        )
        dataset = build_imagenet(d.root, d.get("train_split", "train"), transform)
        bs = per_process_batch(self.cfg.train.batch_size, self.dist, self.cfg.train.get("accum_steps", 1))
        return make_loader(dataset, bs, True, self.dist, self.num_workers, seed=int(self.cfg.seed))

    def build_eval_loader(self):
        d = self.cfg.data
        dataset = build_imagenet(d.root, d.get("val_split", "val"),
                                 imagenet_eval_transform(int(d.img_size), float(d.get("crop_pct", 0.875))))
        loader, _ = make_loader(dataset, self.eval_batch_size(), False, self.dist, self.num_workers)
        return loader

    def train_step(self, model, batch):
        images, target = batch
        if self.mixup_fn is not None:
            images, soft = self.mixup_fn(images, target)
            logits = model(images)
            loss = torch.sum(-soft * F.log_softmax(logits.float(), dim=-1), dim=-1).mean()
        else:
            logits = model(images)
            loss = F.cross_entropy(logits.float(), target,
                                   label_smoothing=float(self.cfg.train.get("label_smoothing", 0.0)))
        return loss, {}

    @torch.no_grad()
    def evaluate(self, model, loader):
        model.eval()
        stats = torch.zeros(3, dtype=torch.float64, device=self.device)
        for images, target in loader:
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            with self.autocast():
                logits = model(images)
            c1, c5 = topk_correct(logits.float(), target, (1, 5))
            stats += torch.tensor([c1, c5, target.numel()], dtype=torch.float64, device=self.device)
        all_reduce_sum(stats)
        n = max(1.0, float(stats[2]))
        return {"top1": 100.0 * float(stats[0]) / n, "top5": 100.0 * float(stats[1]) / n, "num_samples": int(n)}
