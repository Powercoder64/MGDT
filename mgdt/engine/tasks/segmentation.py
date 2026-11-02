"""ADE20K and Cityscapes semantic segmentation."""

import torch
import torch.nn.functional as F

from ...data import IGNORE_INDEX, ADE20KDataset, CityscapesDataset, SegEvalTransform, SegTrainTransform
from ...metrics import boundary_f_score, boundary_stats, confusion_matrix, miou_from_confusion
from ..checkpoint import unwrap
from ..distributed import all_reduce_sum
from ..loader import make_loader, per_process_batch
from .base import Task

DATASETS = {"ade20k": ADE20KDataset, "cityscapes": CityscapesDataset}


def slide_inference(model, image, num_classes, crop_size, stride):
    """Averages logits of overlapping crops; image (1, 3, H, W) -> logits (1, C, H, W)."""
    _, _, H, W = image.shape
    ch, cw = crop_size
    sh, sw = stride
    rows = max(H - ch + sh - 1, 0) // sh + 1
    cols = max(W - cw + sw - 1, 0) // sw + 1
    logits = image.new_zeros(1, num_classes, H, W, dtype=torch.float32)
    count = image.new_zeros(1, 1, H, W, dtype=torch.float32)
    for r in range(rows):
        for c in range(cols):
            y1, x1 = r * sh, c * sw
            y2, x2 = min(y1 + ch, H), min(x1 + cw, W)
            y1, x1 = max(y2 - ch, 0), max(x2 - cw, 0)
            crop = image[:, :, y1:y2, x1:x2]
            h, w = crop.shape[-2:]
            if h < ch or w < cw:
                crop = F.pad(crop, (0, cw - w, 0, ch - h))
            out = model(crop)[:, :, :h, :w].float()
            logits[:, :, y1:y2, x1:x2] += out
            count[:, :, y1:y2, x1:x2] += 1
    return logits / count


class SegmentationTask(Task):
    primary_metric = "mIoU"

    def __init__(self, cfg, device, dist_info, logger):
        super().__init__(cfg, device, dist_info, logger)
        name = cfg.data.dataset
        if name not in DATASETS:
            raise ValueError(f"Unknown segmentation dataset '{name}'")
        self.dataset_cls = DATASETS[name]

    def model_kwargs(self):
        return {"num_classes": self.dataset_cls.num_classes}

    def init_from_pretrained(self, model, state_dict):
        backbone = unwrap(model).backbone
        state = {k: v for k, v in state_dict.items() if not k.startswith(("head.", "norm."))}
        result = backbone.load_state_dict(state, strict=False)
        return list(result.missing_keys), list(result.unexpected_keys)

    def build_train_loader(self):
        d = self.cfg.data
        transform = SegTrainTransform(scale=tuple(d.scale), ratio_range=tuple(d.ratio_range),
                                      crop_size=tuple(d.crop_size), cat_max_ratio=float(d.get("cat_max_ratio", 0.75)))
        dataset = self.dataset_cls(d.root, d.train_split, transform)
        bs = per_process_batch(self.cfg.train.batch_size, self.dist, self.cfg.train.get("accum_steps", 1))
        return make_loader(dataset, bs, True, self.dist, self.num_workers, seed=int(self.cfg.seed))

    def build_eval_loader(self):
        d = self.cfg.data
        test = d.get("test", {})
        scale = test.get("scale")
        dataset = self.dataset_cls(d.root, d.val_split, SegEvalTransform(None if scale is None else tuple(scale)))
        loader, _ = make_loader(dataset, 1, False, self.dist, self.num_workers)
        return loader

    def train_step(self, model, batch):
        images, labels = batch
        logits = model(images)
        loss = F.cross_entropy(logits.float(), labels, ignore_index=IGNORE_INDEX)
        return loss, {}

    @torch.no_grad()
    def evaluate(self, model, loader):
        model.eval()
        test = self.cfg.data.get("test", {})
        mode = test.get("mode", "whole")
        C = self.dataset_cls.num_classes
        conf = torch.zeros(C, C, dtype=torch.float64, device=self.device)
        bstats = torch.zeros(4, dtype=torch.float64, device=self.device)
        tolerance = int(test.get("boundary_tolerance", 3))
        for image, label in loader:
            image = image.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)[0]
            with self.autocast():
                if mode == "slide":
                    logits = slide_inference(model, image, C, tuple(test.crop_size), tuple(test.stride))
                else:
                    logits = model(image).float()
            logits = F.interpolate(logits.float(), size=label.shape[-2:], mode="bilinear", align_corners=False)
            pred = logits.argmax(1)[0]
            conf += confusion_matrix(pred, label, C, IGNORE_INDEX).double()
            bstats += boundary_stats(pred, label, IGNORE_INDEX, tolerance)
        all_reduce_sum(conf)
        all_reduce_sum(bstats)
        miou, acc, _ = miou_from_confusion(conf)
        return {"mIoU": miou, "aAcc": acc, "Fb": boundary_f_score(bstats)}
