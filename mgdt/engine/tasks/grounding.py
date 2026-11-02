"""Ego4D Natural Language Queries temporal grounding."""

import torch
import torch.nn.functional as F

from ...data import Ego4DNLQDataset, build_query_vocab, collate_nlq, load_nlq_annotations
from ...metrics import recall_at_1, temporal_iou
from ..distributed import all_gather_objects
from ..loader import make_loader, per_process_batch
from .base import Task


def decode_spans(start_logits, end_logits, mask, max_span=None):
    """Most likely (start, end) step pair with start <= end; logits (B, T), mask (B, T)."""
    B, T = start_logits.shape
    neg = torch.finfo(torch.float32).min
    ls = F.log_softmax(start_logits.float().masked_fill(~mask, neg), dim=-1)
    le = F.log_softmax(end_logits.float().masked_fill(~mask, neg), dim=-1)
    scores = ls[:, :, None] + le[:, None, :]
    i = torch.arange(T, device=scores.device)
    valid = i[None, :] >= i[:, None]
    if max_span is not None:
        valid &= (i[None, :] - i[:, None]) < int(max_span)
    valid = valid[None] & mask[:, :, None] & mask[:, None, :]
    flat = scores.masked_fill(~valid, float("-inf")).view(B, -1).argmax(-1)
    return flat // T, flat % T


class GroundingTask(Task):
    primary_metric = "R1@0.3"

    def __init__(self, cfg, device, dist_info, logger):
        super().__init__(cfg, device, dist_info, logger)
        d = cfg.data
        self.train_samples = load_nlq_annotations(d.train_json)
        self.val_samples = load_nlq_annotations(d.val_json)
        self.vocab = build_query_vocab(self.train_samples, int(d.get("min_word_count", 1)))
        logger.info("NLQ queries: %d train, %d val; vocabulary %d words",
                    len(self.train_samples), len(self.val_samples), len(self.vocab))

    def model_kwargs(self):
        return {"vocab_size": len(self.vocab), "video_dim": int(self.cfg.data.feature_dim)}

    def _dataset(self, train):
        d = self.cfg.data
        return Ego4DNLQDataset(self.train_samples if train else self.val_samples, d.feature_dir, self.vocab,
                               float(d.feature_fps), train, max_len=int(d.max_len),
                               jitter=float(d.get("temporal_jitter", 0.1)),
                               max_query_len=int(d.get("max_query_len", 32)))

    def build_train_loader(self):
        bs = per_process_batch(self.cfg.train.batch_size, self.dist, self.cfg.train.get("accum_steps", 1))
        return make_loader(self._dataset(True), bs, True, self.dist, self.num_workers, collate_fn=collate_nlq,
                           seed=int(self.cfg.seed))

    def build_eval_loader(self):
        loader, _ = make_loader(self._dataset(False), self.eval_batch_size(16), False, self.dist,
                                self.num_workers, collate_fn=collate_nlq)
        return loader

    def train_step(self, model, batch):
        start_logits, end_logits, inside_logits = model(batch["video"], batch["video_mask"], batch["words"],
                                                        batch["word_mask"])
        loss_start = F.cross_entropy(start_logits, batch["start"])
        loss_end = F.cross_entropy(end_logits, batch["end"])
        steps = torch.arange(start_logits.shape[1], device=start_logits.device)[None]
        inside = ((steps >= batch["start"][:, None]) & (steps <= batch["end"][:, None])).float()
        mask = batch["video_mask"]
        loss_inside = F.binary_cross_entropy_with_logits(inside_logits[mask], inside[mask])
        weight = float(self.cfg.train.get("inside_loss_weight", 1.0))
        loss = loss_start + loss_end + weight * loss_inside
        return loss, {"loss_span": float((loss_start + loss_end).detach()), "loss_inside": float(loss_inside.detach())}

    @torch.no_grad()
    def evaluate(self, model, loader):
        model.eval()
        fps = float(self.cfg.data.feature_fps)
        max_span = self.cfg.data.get("max_span")
        local = []
        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with self.autocast():
                s_logits, e_logits, _ = model(batch["video"], batch["video_mask"], batch["words"], batch["word_mask"])
            s, e = decode_spans(s_logits, e_logits, batch["video_mask"], max_span)
            for idx, si, ei in zip(batch["index"].tolist(), s.tolist(), e.tolist()):
                local.append((idx, si / fps, (ei + 1) / fps))
        gathered = sorted(item for part in all_gather_objects(local) for item in part)
        preds, gts = [], []
        for idx, start, end in gathered:
            sample = self.val_samples[idx]
            preds.append([start, min(end, sample["duration"])])
            gts.append([sample["start"], sample["end"]])
        ious = temporal_iou(torch.tensor(preds, dtype=torch.float64), torch.tensor(gts, dtype=torch.float64))
        metrics = recall_at_1(ious, (0.3, 0.5))
        metrics["num_samples"] = len(gathered)
        return metrics
