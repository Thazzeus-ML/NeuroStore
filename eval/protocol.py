"""Shared experiment helpers: masked cues, assembly->concept naming, decoders."""
from __future__ import annotations
import numpy as np
from neurostore.readout import select_assemblies, complete_pattern
from eval.metrics import set_f1


def mask_cues(codes, frac, seed, tag=0):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 9000, int(round(frac * 100)), tag]))
    cues, masked = [], []
    for c in codes:
        k = len(c); nm = int(round(frac * k)); perm = rng.permutation(k)
        masked.append(np.sort(c[perm[:nm]])); cues.append(np.sort(c[perm[nm:]]))
    return cues, masked


class ConceptMap:
    """Names assemblies after the ground-truth concepts of the training events they win.
    (The only place labels touch NeuroStore; the references use labels the same way.)"""

    def __init__(self, asm, codes, concepts, C):
        self.counts = np.zeros((asm.A, C)); self.n = np.zeros(asm.A)
        for c, cs in zip(codes, concepts):
            if len(cs) == 0:
                continue
            cos = asm.vote(c)
            if cos.max() <= 0:
                continue
            a = int(cos.argmax()); self.n[a] += 1
            for q in cs:
                self.counts[a, q] += 1

    def concepts_of(self, a, thr):
        if self.n[a] == 0:
            return ()
        return tuple(np.flatnonzero(self.counts[a] / self.n[a] >= thr))

    def purity(self):
        m = self.n > 0
        return float((self.counts.max(1)[m] / self.n[m]).mean()) if m.any() else 0.0

    def concepts_covered(self, thr=0.5):
        return len({q for a in range(len(self.n)) for q in self.concepts_of(a, thr)})


def ns_decode(cos, cmap, rho, cos_min, thr):
    sel = select_assemblies(cos, rho, cos_min)
    out = set()
    for a in sel:
        out |= set(map(int, cmap.concepts_of(int(a), thr)))
    return tuple(sorted(out))


def f1_mean(preds, truths):
    return float(np.mean([set_f1(p, t) for p, t in zip(preds, truths)]))


def prec_mean(comps, masked):
    ps = []
    for comp, ms in zip(comps, masked):
        m = len(ms)
        ps.append(len(set(map(int, comp[:m])) & set(map(int, ms))) / m if m else 1.0)
    return float(np.mean(ps))


def recall_at_k_batch(pred_ranked_list, nexts, k=32):
    return float(np.mean([len(set(map(int, p[:k])) & set(map(int, n))) / len(n) for p, n in zip(pred_ranked_list, nexts)]))


def topk_rows(S, k):
    """rows of S -> top-k column indices (ties broken by index)"""
    top = np.argpartition(-S, k - 1, axis=1)[:, :k]
    return top


class Bigram:
    """first-order unit bigram over consecutive codes"""

    def __init__(self, N):
        self.N = N; self.B = np.zeros((N, N), np.float32); self.n = 0

    def fit(self, codes):
        X = np.zeros((len(codes), self.N), np.float32); X[np.arange(len(codes))[:, None], codes] = 1
        self.B = X[:-1].T @ X[1:]
        self.rown = self.B / np.maximum(self.B.sum(1, keepdims=True), 1)
        return self

    def scores(self, cur_codes, normalized=True):
        M = self.rown if normalized else self.B
        return np.stack([M[c].sum(0) for c in cur_codes])
