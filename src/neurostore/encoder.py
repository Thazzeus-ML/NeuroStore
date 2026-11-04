"""Sparse encoder: raw features -> exactly k of N units.

Each raw feature projects (fixed random hash, seeded) to d units. A unit's candidate score
is the number of active features projecting to it. The k units with the largest margin
(score - theta_u + tie-break jitter) fire. theta is the per-unit adaptive threshold that
homeostasis moves. encode() is a pure function of (features, theta): it never mutates state.
"""
from __future__ import annotations
import numpy as np


def normalize_features(feats, F: int) -> np.ndarray:
    """Dedupe, sort, drop out-of-range ids. Empty events are allowed."""
    f = np.unique(np.asarray(feats, dtype=np.int64))
    return f[(f >= 0) & (f < F)].astype(np.int32)


def q8(x) -> np.ndarray:
    return np.clip(np.rint(np.asarray(x, np.float64) * 255.0), 0, 255).astype(np.uint8)


class SparseEncoder:
    def __init__(self, F: int, N: int, k: int, d: int, seed: int):
        assert 0 < k <= N and 0 < d <= N
        self.F, self.N, self.k, self.d = F, N, k, d
        rng = np.random.default_rng(np.random.SeedSequence([seed, 77]))
        self.fmap = np.argpartition(rng.random((F, N)), d - 1, axis=1)[:, :d].astype(np.int32)
        self.jitter = (rng.random(N) * 1e-3).astype(np.float32)
        self.theta = np.zeros(N, np.float32)       # adaptive thresholds (state)

    def scores(self, feats: np.ndarray) -> np.ndarray:
        if len(feats) == 0:
            return np.zeros(self.N, np.float32)
        return np.bincount(self.fmap[feats].ravel(), minlength=self.N).astype(np.float32)

    def encode(self, feats: np.ndarray, theta: np.ndarray | None = None):
        """-> (units sorted int32[k], strength float32[k] in [0,1])."""
        th = self.theta if theta is None else theta
        s = self.scores(feats)
        margin = s - th + self.jitter
        idx = np.argpartition(-margin, self.k - 1)[:self.k]
        units = np.sort(idx).astype(np.int32)
        smax = max(float(s.max()), 1.0)
        return units, (s[units] / smax).astype(np.float32)

    def encode_raw(self, feats, theta=None):
        return self.encode(normalize_features(feats, self.F), theta)
