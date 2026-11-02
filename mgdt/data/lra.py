"""Long Range Arena byte-level text classification (IMDB reviews)."""

import glob
import os

import torch
from torch.utils.data import Dataset


class IMDBByteDataset(Dataset):
    """Reads ``aclImdb/{train,test}/{neg,pos}/*.txt`` and encodes reviews as UTF-8 bytes.

    Byte ``b`` maps to token ``b + 1``; token 0 is padding. Sequences are truncated and
    padded to ``max_len``.
    """

    def __init__(self, root, split, max_len=4096):
        self.max_len = int(max_len)
        self.items = []
        for label, name in ((0, "neg"), (1, "pos")):
            files = sorted(glob.glob(os.path.join(root, split, name, "*.txt")))
            self.items.extend((path, label) for path in files)
        if not self.items:
            raise FileNotFoundError(f"No reviews found under {os.path.join(root, split)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        with open(path, "rb") as f:
            data = f.read()[: self.max_len]
        tokens = torch.zeros(self.max_len, dtype=torch.long)
        if data:
            tokens[: len(data)] = torch.frombuffer(bytearray(data), dtype=torch.uint8).long() + 1
        return tokens, label
