"""COCO Captions with the Karpathy split (``dataset_coco.json``)."""

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .imagenet import IMAGENET_MEAN, IMAGENET_STD
from .vocab import Vocabulary

TRAIN_SPLITS = ("train", "restval")


def caption_train_transform(img_size=224, scale=(0.5, 1.0)):
    """Random resized crop and horizontal flip."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=tuple(scale), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def caption_eval_transform(img_size=224):
    """Resize the full image to a square input."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_karpathy(json_path):
    """Returns the list of image records of a Karpathy-split JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)["images"]


def build_caption_vocab(records, min_count=5):
    """Vocabulary of the training captions (train + restval)."""
    tokens = [s["tokens"] for r in records if r["split"] in TRAIN_SPLITS for s in r["sentences"]]
    return Vocabulary.build(tokens, min_count=min_count)


def _image_path(root, record):
    return os.path.join(root, record["filepath"], record["filename"])


class CocoCaptionTrain(Dataset):
    """Every (image, caption) pair of the training splits; captions become [BOS] w1..wn [EOS] [PAD]..."""

    def __init__(self, image_root, records, vocab, transform, max_words=20):
        self.image_root = image_root
        self.vocab = vocab
        self.transform = transform
        self.max_words = max_words
        self.pairs = [(_image_path(image_root, r), s["tokens"]) for r in records if r["split"] in TRAIN_SPLITS
                      for s in r["sentences"]]
        if not self.pairs:
            raise RuntimeError("No training captions found in the Karpathy split file")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        path, tokens = self.pairs[index]
        image = self.transform(Image.open(path).convert("RGB"))
        ids = [self.vocab.bos_id] + self.vocab.encode(tokens[: self.max_words]) + [self.vocab.eos_id]
        seq = torch.full((self.max_words + 2,), self.vocab.pad_id, dtype=torch.long)
        seq[: len(ids)] = torch.tensor(ids, dtype=torch.long)
        return image, seq


class CocoCaptionEval(Dataset):
    """Images of the ``val`` or ``test`` split with their reference captions."""

    def __init__(self, image_root, records, split, transform):
        self.records = [r for r in records if r["split"] == split]
        if not self.records:
            raise RuntimeError(f"No images with split '{split}' in the Karpathy split file")
        self.image_root = image_root
        self.transform = transform
        self.references = [[" ".join(s["tokens"]) for s in r["sentences"]] for r in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image = self.transform(Image.open(_image_path(self.image_root, self.records[index])).convert("RGB"))
        return image, index
