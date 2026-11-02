"""Video decoding with PyAV and clip-level spatial transforms."""

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, RandomResizedCrop

from .imagenet import IMAGENET_MEAN, IMAGENET_STD


def decode_video(path):
    """Decodes every frame of a video file into a uint8 array (T, H, W, 3)."""
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames)


def clip_indices(num_total, num_frames, stride, start):
    """Frame indices ``start + i * stride`` clamped to the video length."""
    idx = start + np.arange(num_frames) * stride
    return np.clip(idx, 0, num_total - 1)


def temporal_starts(num_total, span, num_clips):
    """Uniformly spaced clip start frames covering the whole video."""
    max_start = max(0, num_total - span)
    if num_clips == 1:
        return [max_start // 2]
    return np.linspace(0, max_start, num_clips).round().astype(int).tolist()


def _normalize(clip):
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (clip.float() / 255.0 - mean) / std


class VideoTrainTransform:
    """Random resized crop (shared by all frames), horizontal flip and normalisation.

    Input (T, 3, H, W) uint8; output (3, T, size, size) float.
    """

    def __init__(self, size=224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), flip_prob=0.5):
        self.size = size
        self.scale = tuple(scale)
        self.ratio = tuple(ratio)
        self.flip_prob = flip_prob

    def __call__(self, clip):
        i, j, h, w = RandomResizedCrop.get_params(clip, self.scale, self.ratio)
        clip = TF.resized_crop(clip, i, j, h, w, [self.size, self.size],
                               interpolation=InterpolationMode.BILINEAR, antialias=True)
        if random.random() < self.flip_prob:
            clip = clip.flip(-1)
        return _normalize(clip).permute(1, 0, 2, 3).contiguous()


class VideoEvalTransform:
    """Short-side resize and ``num_crops`` square crops along the long side (1 = centre, 3 = both ends + centre)."""

    def __init__(self, size=224, num_crops=3):
        self.size = size
        self.num_crops = num_crops

    def __call__(self, clip):
        clip = TF.resize(clip, self.size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        H, W = clip.shape[-2:]
        if self.num_crops == 1:
            offsets = [((H - self.size) // 2, (W - self.size) // 2)]
        elif H >= W:
            offsets = [(o, (W - self.size) // 2) for o in (0, (H - self.size) // 2, H - self.size)]
        else:
            offsets = [((H - self.size) // 2, o) for o in (0, (W - self.size) // 2, W - self.size)]
        views = []
        for top, left in offsets:
            crop = clip[..., top:top + self.size, left:left + self.size]
            views.append(_normalize(crop).permute(1, 0, 2, 3).contiguous())
        return views
