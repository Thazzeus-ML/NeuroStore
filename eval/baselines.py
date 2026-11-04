"""Reference systems for the experiments."""
from __future__ import annotations
from collections import Counter
import numpy as np


class ExactMatch:
    """Hash of the full code -> labels. A masked cue is a different code, so it almost never hits."""

    def fit(self, codes, concepts, N):
        self.table: dict = {}
        for c, cs in zip(codes, concepts):
            self.table.setdefault(np.asarray(c, np.int32).tobytes(), []).append(cs)
        self.global_rank = np.argsort(-np.bincount(np.asarray(codes).ravel(), minlength=N), kind="stable")
        single = [cs[0] for cs in concepts if len(cs) == 1]
        self.majority = (Counter(single).most_common(1)[0][0],)
        return self

    def query(self, cue, m):
        hit = self.table.get(np.asarray(cue, np.int32).tobytes())
        pred = Counter(hit).most_common(1)[0][0] if hit else self.majority
        cue_set = set(map(int, cue))
        comp = [u for u in self.global_rank if int(u) not in cue_set][:m]      # no pattern information
        return pred, np.array(comp, np.int64), bool(hit)

    def nbytes(self):
        return 8 * len(self.table)           # one 64-bit hash per distinct code (labels excluded)


class JaccardNN:
    """Brute-force Jaccard nearest neighbours over all stored codes."""

    def fit(self, codes, N):
        codes = np.asarray(codes)
        self.n, self.k, self.N = len(codes), codes.shape[1], N
        self.T = np.zeros((self.n, N), np.float32)
        self.T[np.arange(self.n)[:, None], codes] = 1.0
        self.codes = codes
        return self

    def topk(self, cues, K=10, chunk=256):
        idx = np.empty((len(cues), K), np.int64); sims = np.empty((len(cues), K), np.float32)
        for s in range(0, len(cues), chunk):
            cs = cues[s:s + chunk]
            Q = np.zeros((len(cs), self.N), np.float32)
            for j, c in enumerate(cs):
                Q[j, c] = 1.0
            inter = Q @ self.T.T
            union = Q.sum(1, keepdims=True) + self.k - inter
            jac = inter / np.maximum(union, 1)
            top = np.argpartition(-jac, K - 1, axis=1)[:, :K]
            tj = np.take_along_axis(jac, top, 1)
            o = np.argsort(-tj, axis=1, kind="stable")
            idx[s:s + len(cs)] = np.take_along_axis(top, o, 1); sims[s:s + len(cs)] = np.take_along_axis(tj, o, 1)
        return idx, sims

    def nbytes_packed(self):
        return self.n * self.N // 8

    def nbytes_dense(self):
        return self.T.nbytes
