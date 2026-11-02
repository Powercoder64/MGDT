"""Ego4D Natural Language Queries over pre-extracted clip features."""

import json
import math
import os
import random

import torch
from torch.utils.data import Dataset

from .vocab import Vocabulary, tokenize


def load_nlq_annotations(json_path):
    """Flattens an Ego4D NLQ annotation file into one record per language query."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for video in data["videos"]:
        for clip in video["clips"]:
            clip_uid = clip["clip_uid"]
            duration = float(clip["video_end_sec"]) - float(clip["video_start_sec"])
            for a_idx, ann in enumerate(clip.get("annotations", [])):
                ann_uid = ann.get("annotation_uid", f"{clip_uid}_{a_idx}")
                for q_idx, query in enumerate(ann.get("language_queries", [])):
                    text = query.get("query")
                    if not text or "clip_start_sec" not in query or "clip_end_sec" not in query:
                        continue
                    start = max(0.0, float(query["clip_start_sec"]))
                    end = min(duration, float(query["clip_end_sec"]))
                    if end < start:
                        continue
                    samples.append({
                        "query_id": f"{ann_uid}_{q_idx}",
                        "clip_uid": clip_uid,
                        "query": text,
                        "start": start,
                        "end": end,
                        "duration": duration,
                    })
    if not samples:
        raise RuntimeError(f"No language queries found in {json_path}")
    return samples


def build_query_vocab(samples, min_count=1):
    """Vocabulary of training query words."""
    return Vocabulary.build([tokenize(s["query"]) for s in samples], min_count=min_count)


class Ego4DNLQDataset(Dataset):
    """Returns features, ground-truth start/end feature indices and query word ids.

    Feature ``j`` of a clip covers time ``[j / fps, (j + 1) / fps)``. Training crops a window of
    at most ``max_len`` steps that contains the moment and applies temporal jitter that removes
    up to ``jitter`` of the steps outside the moment.
    """

    def __init__(self, samples, feature_dir, vocab, feature_fps, train, max_len=1024, jitter=0.1,
                 max_query_len=32):
        self.samples = samples
        self.feature_dir = feature_dir
        self.vocab = vocab
        self.fps = float(feature_fps)
        self.train = train
        self.max_len = int(max_len)
        self.jitter = float(jitter)
        self.max_query_len = int(max_query_len)
        missing = sorted({s["clip_uid"] for s in samples
                          if not os.path.isfile(os.path.join(feature_dir, s["clip_uid"] + ".pt"))})
        if missing:
            raise FileNotFoundError(f"{len(missing)} feature files missing in {feature_dir}, e.g. {missing[0]}.pt")

    def __len__(self):
        return len(self.samples)

    def _window(self, T, s_idx, e_idx):
        if T > self.max_len:
            lo = max(0, e_idx - self.max_len + 1)
            hi = min(s_idx, T - self.max_len)
            w0 = random.randint(lo, max(lo, hi))
            return w0, w0 + self.max_len
        cut = int(self.jitter * T)
        w0 = random.randint(0, min(s_idx, cut))
        w1 = T - random.randint(0, min(T - 1 - e_idx, cut))
        return w0, w1

    def __getitem__(self, index):
        s = self.samples[index]
        feats = torch.load(os.path.join(self.feature_dir, s["clip_uid"] + ".pt"), map_location="cpu",
                           weights_only=True).float()
        T = feats.shape[0]
        s_idx = min(T - 1, int(math.floor(s["start"] * self.fps)))
        e_idx = min(T - 1, max(s_idx, int(math.floor(s["end"] * self.fps))))
        w0, w1 = (0, T)
        if self.train:
            w0, w1 = self._window(T, s_idx, e_idx)
            feats = feats[w0:w1]
        start = min(max(0, s_idx - w0), w1 - w0 - 1)
        end = min(max(start, e_idx - w0), w1 - w0 - 1)
        words = self.vocab.encode(tokenize(s["query"]))[: self.max_query_len] or [self.vocab.unk_id]
        return {
            "video": feats,
            "start": start,
            "end": end,
            "words": torch.tensor(words, dtype=torch.long),
            "index": index,
        }


def collate_nlq(batch):
    """Pads features and query words of a batch and builds their masks."""
    B = len(batch)
    T = max(b["video"].shape[0] for b in batch)
    L = max(b["words"].numel() for b in batch)
    D = batch[0]["video"].shape[1]
    video = torch.zeros(B, T, D)
    video_mask = torch.zeros(B, T, dtype=torch.bool)
    words = torch.zeros(B, L, dtype=torch.long)
    word_mask = torch.zeros(B, L, dtype=torch.bool)
    for i, b in enumerate(batch):
        t, n = b["video"].shape[0], b["words"].numel()
        video[i, :t] = b["video"]
        video_mask[i, :t] = True
        words[i, :n] = b["words"]
        word_mask[i, :n] = True
    return {
        "video": video,
        "video_mask": video_mask,
        "words": words,
        "word_mask": word_mask,
        "start": torch.tensor([b["start"] for b in batch], dtype=torch.long),
        "end": torch.tensor([b["end"] for b in batch], dtype=torch.long),
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }
