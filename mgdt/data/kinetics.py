"""Kinetics-400/700 video classification from a CSV list (``path,label``)."""

import csv
import logging
import os
import random

import torch
from torch.utils.data import Dataset

from .video import VideoEvalTransform, VideoTrainTransform, clip_indices, decode_video, temporal_starts

logger = logging.getLogger("mgdt")


def read_video_list(list_file):
    """Reads ``path,label`` rows (header optional) into a list of (relative_path, int_label)."""
    samples = []
    with open(list_file, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row or row[0] == "path":
                continue
            samples.append((row[0], int(row[1])))
    if not samples:
        raise RuntimeError(f"No samples found in {list_file}")
    return samples


class KineticsDataset(Dataset):
    """Returns ``(clips, label)``.

    Training: ``repeated_aug`` independently augmented clips, shape (R, 3, T, S, S).
    Evaluation: ``num_clips`` temporal x ``num_crops`` spatial views, shape (V, 3, T, S, S).
    """

    def __init__(self, root, list_file, train, num_frames=32, frame_stride=2, img_size=224, repeated_aug=1,
                 num_clips=4, num_crops=3, scale=(0.08, 1.0), max_retries=10):
        self.root = root
        self.samples = read_video_list(list_file)
        self.train = train
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.repeated_aug = max(1, int(repeated_aug))
        self.num_clips = num_clips
        self.max_retries = max_retries
        self.span = (num_frames - 1) * frame_stride + 1
        self.train_tf = VideoTrainTransform(img_size, scale=scale)
        self.eval_tf = VideoEvalTransform(img_size, num_crops=num_crops)

    def __len__(self):
        return len(self.samples)

    def _frames(self, frames, start):
        idx = clip_indices(len(frames), self.num_frames, self.frame_stride, start)
        return torch.from_numpy(frames[idx]).permute(0, 3, 1, 2)

    def __getitem__(self, index):
        if not self.train:
            path, label = self.samples[index]
            frames = decode_video(os.path.join(self.root, path))
            views = []
            for start in temporal_starts(len(frames), self.span, self.num_clips):
                views.extend(self.eval_tf(self._frames(frames, start)))
            return torch.stack(views), label
        for _ in range(self.max_retries):
            path, label = self.samples[index]
            try:
                frames = decode_video(os.path.join(self.root, path))
            except Exception as exc:
                logger.warning("Failed to decode %s (%s); sampling another video", path, exc)
                index = random.randrange(len(self.samples))
                continue
            clips = []
            for _ in range(self.repeated_aug):
                start = random.randint(0, max(0, len(frames) - self.span))
                clips.append(self.train_tf(self._frames(frames, start)))
            return torch.stack(clips), label
        raise RuntimeError(f"Failed to decode {self.max_retries} consecutive videos")
