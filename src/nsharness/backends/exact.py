"""Exact (brute-force) cosine search: the reference every other backend is measured against."""
from __future__ import annotations
import numpy as np
from ..core import Backend
from .._util import topk_desc, iter_chunks


class ExactBackend(Backend):
    """metric="cosine": cosine similarity (stores unit vectors + norms). metric="l2": similarity = -(mean squared difference per
    feature), i.e. textbook Euclidean kNN (stores the vectors + squared norms). With a feature mask only observed features count
    (cosine over observed features / mean squared difference over observed features)."""
    name = "exact"
    supports_mask = True
    supports_reconstruct = True

    def __init__(self, metric="cosine", beta=None, **extra):
        assert metric in ("cosine", "l2"), metric
        super().__init__(metric=metric, beta=beta, **extra)

    def fit(self, X, seed: int = 0):
        X = np.asarray(X, np.float32)
        self.n, self.d = X.shape
        if self.cfg["metric"] == "l2":
            self.X = X.copy()
            self.xsq = (X * X).sum(1).astype(np.float32)
        else:
            self.norm = np.maximum(np.linalg.norm(X, axis=1), 1e-12).astype(np.float32)
            self.Xn = (X / self.norm[:, None]).astype(np.float32)
        return self

    def _rows(self) -> int:
        return int(max(1, min(1024, 5e7 // max(self.n, 1))))

    def sims(self, Q, mask=None):
        """(q, n) similarities (higher = more similar). With a feature mask, only observed features enter."""
        Q = np.asarray(Q, np.float32)
        if self.cfg["metric"] == "l2":
            if mask is None:
                return -((Q * Q).sum(1, keepdims=True) - 2.0 * (Q @ self.X.T) + self.xsq[None, :]) / self.d
            m = np.asarray(mask, np.float32)
            Qm = Q * m
            return -((Qm * Qm).sum(1, keepdims=True) - 2.0 * (Qm @ self.X.T) + m @ (self.X * self.X).T) / np.maximum(m.sum(1, keepdims=True), 1.0)
        if mask is None:
            return (Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)) @ self.Xn.T
        m = np.asarray(mask, np.float32)
        Qm = Q * m
        den_x = np.sqrt(np.maximum(m @ (self.Xn * self.Xn).T, 1e-12))
        den_q = np.maximum(np.linalg.norm(Qm, axis=1, keepdims=True), 1e-12)
        return (Qm @ self.Xn.T) / (den_x * den_q)

    def search(self, Q, k, mask=None):
        Q = np.asarray(Q, np.float32)
        k = min(int(k), self.n)
        oi = np.empty((len(Q), k), np.int64)
        os_ = np.empty((len(Q), k), np.float32)
        for a, b in iter_chunks(len(Q), self._rows()):
            S = self.sims(Q[a:b], None if mask is None else mask[a:b])
            i, s = topk_desc(S, k)
            oi[a:b], os_[a:b] = i, s
        return oi, os_

    def raw(self, idx):
        """Stored feature vectors (de-normalised) of the given item ids (any shape)."""
        idx = np.asarray(idx)
        if self.cfg["metric"] == "l2":
            return self.X[idx]
        return self.Xn[idx] * self.norm[idx][..., None]

    def reconstruct(self, Q, k, mask=None):
        idx, sim = self.search(Q, k, mask)
        w = self.weights(sim)
        w = w / np.maximum(w.sum(1, keepdims=True), 1e-300)
        return (w[..., None] * self.raw(idx)).sum(1).astype(np.float32)

    def memory_report(self):
        if self.cfg["metric"] == "l2":
            return {"vectors_f32": self.X.nbytes, "sq_norms_f32": self.xsq.nbytes}
        return {"vectors_f32": self.Xn.nbytes, "norms_f32": self.norm.nbytes}
