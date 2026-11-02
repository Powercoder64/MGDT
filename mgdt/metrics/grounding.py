"""Temporal grounding metrics."""

import torch


def temporal_iou(pred, gt):
    """IoU of time intervals; pred and gt are (N, 2) tensors of [start, end]."""
    inter = (torch.minimum(pred[:, 1], gt[:, 1]) - torch.maximum(pred[:, 0], gt[:, 0])).clamp_min(0)
    union = torch.maximum(pred[:, 1], gt[:, 1]) - torch.minimum(pred[:, 0], gt[:, 0])
    return inter / union.clamp_min(1e-6)


def recall_at_1(ious, thresholds=(0.3, 0.5)):
    """Percentage of queries whose top-1 prediction reaches each tIoU threshold, plus mean IoU."""
    out = {f"R1@{t}": 100.0 * float((ious >= t).float().mean().item()) for t in thresholds}
    out["mIoU"] = 100.0 * float(ious.mean().item())
    return out


__all__ = ["recall_at_1", "temporal_iou"]
