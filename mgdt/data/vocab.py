"""Word vocabulary with special tokens."""

import json
import re
from collections import Counter

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = (PAD, BOS, EOS, UNK)

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text):
    """Lower-cases text and splits it into alphanumeric word tokens."""
    return _WORD_RE.findall(text.lower())


class Vocabulary:
    """Maps words to ids; ids 0-3 are <pad>, <bos>, <eos> and <unk>."""

    def __init__(self, words):
        self.itos = list(SPECIALS) + [w for w in words if w not in SPECIALS]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @property
    def pad_id(self):
        return 0

    @property
    def bos_id(self):
        return 1

    @property
    def eos_id(self):
        return 2

    @property
    def unk_id(self):
        return 3

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, token_lists, min_count=1):
        """Keeps words seen at least ``min_count`` times, ordered by frequency then alphabetically."""
        counts = Counter(t for tokens in token_lists for t in tokens)
        words = sorted((w for w, c in counts.items() if c >= min_count), key=lambda w: (-counts[w], w))
        return cls(words)

    def encode(self, tokens):
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids if i >= len(SPECIALS) or i == self.unk_id]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f)[len(SPECIALS):])
