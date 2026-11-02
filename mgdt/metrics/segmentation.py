"""Semantic segmentation metrics: confusion-matrix mIoU and boundary F-score."""

import torch
import torch.nn.functional as F


def confusion_matrix(pred, target, num_classes, ignore_index=255):
    """(C, C) matrix with rows = ground truth and columns = prediction, ignoring ``ignore_index``."""
    valid = target != ignore_index
    idx = target[valid].long() * num_classes + pred[valid].long()
    return torch.bincount(idx, minlength=num_classes * num_classes).view(num_classes, num_classes)


def miou_from_confusion(conf):
    """Returns (mIoU %, pixel accuracy %, per-class IoU) averaging over classes present in GT or prediction."""
    conf = conf.double()
    tp = conf.diag()
    union = conf.sum(0) + conf.sum(1) - tp
    iou = tp / union
    present = union > 0
    miou = 100.0 * float(iou[present].mean().item()) if bool(present.any()) else 0.0
    acc = 100.0 * float((tp.sum() / conf.sum().clamp_min(1)).item())
    return miou, acc, iou


def _boundaries(label):
    """Pixels whose 4-neighbourhood contains a different label; label (H, W)."""
    b = torch.zeros_like(label, dtype=torch.bool)
    b[1:, :] |= label[1:, :] != label[:-1, :]
    b[:-1, :] |= label[:-1, :] != label[1:, :]
    b[:, 1:] |= label[:, 1:] != label[:, :-1]
    b[:, :-1] |= label[:, :-1] != label[:, 1:]
    return b


def _dilate(mask, radius):
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(mask[None, None].float(), k, stride=1, padding=radius)[0, 0] > 0


def boundary_stats(pred, target, ignore_index=255, tolerance=3):
    """Counts for the boundary F-score of one image.

    Returns a tensor ``[matched_pred, total_pred, matched_gt, total_gt]``; a predicted boundary
    pixel is matched if a ground-truth boundary lies within ``tolerance`` pixels and vice versa.
    """
    valid = target != ignore_index
    gt_b = _boundaries(target) & valid
    pred_b = _boundaries(pred) & valid
    matched_pred = (pred_b & _dilate(gt_b, tolerance)).sum()
    matched_gt = (gt_b & _dilate(pred_b, tolerance)).sum()
    return torch.stack([matched_pred, pred_b.sum(), matched_gt, gt_b.sum()]).double()


def boundary_f_score(stats):
    """Boundary F-score (%) from summed :func:`boundary_stats` counts."""
    precision = stats[0] / stats[1].clamp_min(1)
    recall = stats[2] / stats[3].clamp_min(1)
    return 100.0 * float((2 * precision * recall / (precision + recall).clamp_min(1e-12)).item())


__all__ = ["boundary_f_score", "boundary_stats", "confusion_matrix", "miou_from_confusion"]
