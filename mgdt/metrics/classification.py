"""Classification accuracy."""


def topk_correct(logits, target, ks=(1, 5)):
    """Number of samples whose target is within the top-k predictions, for each k."""
    maxk = min(max(ks), logits.shape[-1])
    pred = logits.topk(maxk, dim=-1).indices
    hits = pred.eq(target.view(-1, 1))
    return [int(hits[:, :min(k, maxk)].any(dim=1).sum().item()) for k in ks]


def accuracy(logits, target):
    """Top-1 accuracy in percent."""
    return 100.0 * float((logits.argmax(-1) == target).float().mean().item())


__all__ = ["accuracy", "topk_correct"]
