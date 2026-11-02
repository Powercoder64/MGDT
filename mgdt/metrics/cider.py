"""CIDEr-D captioning metric (n = 4, sigma = 6, clipped tf-idf, length penalty, x10)."""

import math
from collections import defaultdict

import numpy as np


def _ngrams(words, n=4):
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            counts[tuple(words[i:i + k])] += 1
    return counts


class CiderD:
    """Corpus-level CIDEr-D with document frequencies computed over the evaluated references."""

    def __init__(self, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma

    def _vec(self, counts, doc_freq, log_num_docs):
        vec = [defaultdict(float) for _ in range(self.n)]
        norm = [0.0] * self.n
        length = 0
        for gram, tf in counts.items():
            k = len(gram) - 1
            df = math.log(max(1.0, doc_freq[gram]))
            vec[k][gram] = float(tf) * (log_num_docs - df)
            norm[k] += vec[k][gram] ** 2
            if k == 1:
                length += tf
        return vec, [math.sqrt(x) for x in norm], length

    def _sim(self, vh, vr, nh, nr, lh, lr):
        delta = float(lh - lr)
        val = np.zeros(self.n)
        for k in range(self.n):
            for gram in vh[k]:
                val[k] += min(vh[k][gram], vr[k][gram]) * vr[k][gram]
            if nh[k] != 0 and nr[k] != 0:
                val[k] /= nh[k] * nr[k]
            val[k] *= math.exp(-(delta ** 2) / (2 * self.sigma ** 2))
        return val

    def compute_score(self, references, hypotheses):
        """Scores hypotheses against references.

        Args:
            references: dict id -> list of tokenised reference strings.
            hypotheses: dict id -> single tokenised hypothesis string.

        Returns:
            (corpus CIDEr-D, per-id scores as an array in ``sorted`` id order).
        """
        ids = sorted(hypotheses)
        refs = [[_ngrams(r.split(), self.n) for r in references[i]] for i in ids]
        hyps = [_ngrams(hypotheses[i].split(), self.n) for i in ids]
        doc_freq = defaultdict(float)
        for ref_counts in refs:
            for gram in {g for rc in ref_counts for g in rc}:
                doc_freq[gram] += 1
        log_num_docs = math.log(float(len(refs)))
        scores = []
        for hyp, ref_counts in zip(hyps, refs):
            vh, nh, lh = self._vec(hyp, doc_freq, log_num_docs)
            total = np.zeros(self.n)
            for rc in ref_counts:
                vr, nr, lr = self._vec(rc, doc_freq, log_num_docs)
                total += self._sim(vh, vr, nh, nr, lh, lr)
            scores.append(float(np.mean(total)) / len(ref_counts) * 10.0)
        scores = np.array(scores)
        return float(scores.mean()), scores


__all__ = ["CiderD"]
