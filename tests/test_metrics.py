import random

import pytest
import torch

from mgdt.engine.tasks import decode_spans
from mgdt.metrics import (CiderD, boundary_f_score, boundary_stats, confusion_matrix, miou_from_confusion,
                          recall_at_1, temporal_iou, topk_correct)


def test_topk():
    logits = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
    target = torch.tensor([1, 2, 1])
    assert topk_correct(logits, target, (1, 2)) == [1, 2]


def test_miou_and_boundary():
    target = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [255, 2, 2, 2]])
    pred = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1], [0, 2, 2, 2]])
    conf = confusion_matrix(pred, target, 3)
    miou, acc, iou = miou_from_confusion(conf)
    assert torch.allclose(iou, torch.tensor([3 / 4, 4 / 5, 1.0], dtype=torch.float64))
    assert abs(miou - 100 * (0.75 + 0.8 + 1.0) / 3) < 1e-9
    assert boundary_f_score(boundary_stats(target.clone().masked_fill(target == 255, 2), target)) == pytest.approx(100.0)


def test_cider_matches_reference_implementation():
    pytest.importorskip("pycocoevalcap")
    from pycocoevalcap.cider.cider import Cider

    random.seed(0)
    words = "a man woman dog cat sitting on the table with red blue car street".split()
    refs, hyps = {}, {}
    for i in range(30):
        refs[i] = [" ".join(random.choices(words, k=random.randint(6, 12))) for _ in range(5)]
        hyps[i] = " ".join(random.choices(words, k=random.randint(5, 12)))
    ref_score, _ = Cider().compute_score(refs, {k: [v] for k, v in hyps.items()})
    ours, _ = CiderD().compute_score(refs, hyps)
    assert ours == pytest.approx(ref_score, abs=1e-12)


def test_grounding_metrics_and_decoding():
    ious = temporal_iou(torch.tensor([[0.0, 2.0], [1.0, 3.0]]), torch.tensor([[0.0, 2.0], [2.0, 6.0]]))
    assert torch.allclose(ious, torch.tensor([1.0, 0.2]))
    assert recall_at_1(ious)["R1@0.3"] == 50.0
    start = torch.full((1, 6), -10.0)
    end = torch.full((1, 6), -10.0)
    start[0, 4], end[0, 2], end[0, 5] = 5.0, 9.0, 4.0
    s, e = decode_spans(start, end, torch.ones(1, 6, dtype=torch.bool))
    assert (s.item(), e.item()) == (4, 5)
