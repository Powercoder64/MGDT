"""ADE20K and Cityscapes semantic segmentation datasets and joint image/label transforms."""

import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .imagenet import IMAGENET_MEAN, IMAGENET_STD

IGNORE_INDEX = 255

CITYSCAPES_ID_TO_TRAIN_ID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}
_CITYSCAPES_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _k, _v in CITYSCAPES_ID_TO_TRAIN_ID.items():
    _CITYSCAPES_LUT[_k] = _v


def _rescale_size(w, h, scale):
    """Keep-ratio size fitting inside ``scale`` (long edge <= max(scale), short edge <= min(scale))."""
    factor = min(max(scale) / max(w, h), min(scale) / min(w, h))
    return int(w * factor + 0.5), int(h * factor + 0.5)


def _to_tensor(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


class SegTrainTransform:
    """Random rescale, category-balanced random crop, horizontal flip, normalisation and padding."""

    def __init__(self, scale=(2048, 512), ratio_range=(0.5, 2.0), crop_size=(512, 512), cat_max_ratio=0.75,
                 flip_prob=0.5, ignore_index=IGNORE_INDEX):
        self.scale = tuple(scale)
        self.ratio_range = tuple(ratio_range)
        self.crop_h, self.crop_w = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.flip_prob = flip_prob
        self.ignore_index = ignore_index

    def __call__(self, img, label):
        ratio = random.uniform(*self.ratio_range)
        scale = (self.scale[0] * ratio, self.scale[1] * ratio)
        new_w, new_h = _rescale_size(img.width, img.height, scale)
        img = np.asarray(img.resize((new_w, new_h), Image.BILINEAR))
        label = np.asarray(Image.fromarray(label).resize((new_w, new_h), Image.NEAREST))
        y0 = x0 = 0
        for _ in range(10):
            y0 = random.randint(0, max(0, new_h - self.crop_h))
            x0 = random.randint(0, max(0, new_w - self.crop_w))
            crop = label[y0:y0 + self.crop_h, x0:x0 + self.crop_w]
            values, counts = np.unique(crop, return_counts=True)
            counts = counts[values != self.ignore_index]
            if self.cat_max_ratio >= 1.0 or (len(counts) > 1 and counts.max() / counts.sum() < self.cat_max_ratio):
                break
        img = img[y0:y0 + self.crop_h, x0:x0 + self.crop_w]
        label = label[y0:y0 + self.crop_h, x0:x0 + self.crop_w]
        if random.random() < self.flip_prob:
            img = img[:, ::-1]
            label = label[:, ::-1]
        img_t = _to_tensor(np.ascontiguousarray(img))
        label_t = torch.from_numpy(np.ascontiguousarray(label).astype(np.int64))
        pad_h, pad_w = self.crop_h - img_t.shape[1], self.crop_w - img_t.shape[2]
        if pad_h > 0 or pad_w > 0:
            img_t = torch.nn.functional.pad(img_t, (0, pad_w, 0, pad_h), value=0.0)
            label_t = torch.nn.functional.pad(label_t, (0, pad_w, 0, pad_h), value=self.ignore_index)
        return img_t, label_t


class SegEvalTransform:
    """Optional keep-ratio rescale of the image; the label keeps its original resolution."""

    def __init__(self, scale=None):
        self.scale = None if scale is None else tuple(scale)

    def __call__(self, img, label):
        if self.scale is not None:
            img = img.resize(_rescale_size(img.width, img.height, self.scale), Image.BILINEAR)
        return _to_tensor(img), torch.from_numpy(label.astype(np.int64))


class ADE20KDataset(Dataset):
    """ADEChallengeData2016: ``images/{training,validation}/*.jpg`` and ``annotations/.../*.png``.

    Label 0 (other) becomes the ignore index and classes 1-150 map to 0-149.
    """

    num_classes = 150

    def __init__(self, root, split, transform):
        img_dir = os.path.join(root, "images", split)
        ann_dir = os.path.join(root, "annotations", split)
        images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        if not images:
            raise FileNotFoundError(f"No images found in {img_dir}")
        self.items = []
        for path in images:
            ann = os.path.join(ann_dir, os.path.splitext(os.path.basename(path))[0] + ".png")
            if not os.path.isfile(ann):
                raise FileNotFoundError(f"Missing annotation {ann}")
            self.items.append((path, ann))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img_path, ann_path = self.items[index]
        img = Image.open(img_path).convert("RGB")
        raw = np.asarray(Image.open(ann_path), dtype=np.int64)
        label = (raw - 1).astype(np.int64)
        label[raw == 0] = IGNORE_INDEX
        return self.transform(img, label.astype(np.uint8))


class CityscapesDataset(Dataset):
    """Cityscapes fine annotations: ``leftImg8bit/<split>/<city>/*_leftImg8bit.png`` and
    ``gtFine/<split>/<city>/*_gtFine_labelIds.png`` (label ids are mapped to the 19 train ids)."""

    num_classes = 19

    def __init__(self, root, split, transform):
        images = sorted(glob.glob(os.path.join(root, "leftImg8bit", split, "*", "*_leftImg8bit.png")))
        if not images:
            raise FileNotFoundError(f"No images found under {os.path.join(root, 'leftImg8bit', split)}")
        self.items = []
        for path in images:
            city = os.path.basename(os.path.dirname(path))
            stem = os.path.basename(path)[: -len("_leftImg8bit.png")]
            ann = os.path.join(root, "gtFine", split, city, stem + "_gtFine_labelIds.png")
            if not os.path.isfile(ann):
                raise FileNotFoundError(f"Missing annotation {ann}")
            self.items.append((path, ann))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img_path, ann_path = self.items[index]
        img = Image.open(img_path).convert("RGB")
        label = _CITYSCAPES_LUT[np.asarray(Image.open(ann_path), dtype=np.uint8)]
        return self.transform(img, label)
