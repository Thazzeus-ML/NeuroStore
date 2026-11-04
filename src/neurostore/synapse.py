"""Synapse graph / lattice: directed edges (pre, post, weight, typical_lag, last_update).

  * immutable CSR per epoch (layouts.csr.CSRGraph) + per-unit delta buffers (COO chunks, indexed
    lazily by pre unit) that the consolidator merges.
  * lazy decay: w_i <- w_i - lambda*w_i is applied per epoch (epoch = one consolidation round) but
    never by rewriting edges: each edge carries the epoch of its last update and is read as
    w * (1-lambda)^(epoch_now - edge_epoch) via a lookup table.
  * reads = lattice gather + delta gather; merge() folds delta into the lattice WITHOUT changing the
    decayed values, so reads before and after a merge agree (property-tested).
  * merge() works on dense NxN scratch arrays (fine at N=1024, O(N^2); see REPORT).
"""
from __future__ import annotations
import numpy as np
from .layouts.csr import CSRGraph


class SynapseLattice:
    def __init__(self, N, lam=0.03, prune_R=128, eps=1e-3, max_age=255):
        self.N, self.lam, self.R, self.eps = N, lam, prune_R, eps
        self.dtable = (1.0 - lam) ** np.arange(max_age + 1)
        self.epoch = 0
        self.csr = CSRGraph(N, [], [], [], [], [])
        self._chunks: list = []
        self.n_delta = 0
        self._dcsr = None

    # ------------------------------------------------------------ writes
    def tick(self):
        self.epoch += 1

    def add_delta(self, pre, post, inc, incdt=None):
        """inc: Hebbian increment eta*a_i*a_j*K(dt); incdt: inc*dt (for the typical-lag estimate)."""
        pre = np.asarray(pre, np.int64)
        if len(pre) == 0:
            return
        inc = np.asarray(inc, np.float32)
        incdt = np.zeros_like(inc) if incdt is None else np.asarray(incdt, np.float32)
        self._chunks.append((pre, np.asarray(post, np.int64), inc, incdt, np.full(len(pre), self.epoch, np.int64)))
        self.n_delta += len(pre); self._dcsr = None

    def _delta_arrays(self):
        c = self._chunks
        return tuple(np.concatenate([x[i] for x in c]) for i in range(5))

    def _delta_graph(self):
        if self._dcsr is None and self.n_delta:
            pre, post, inc, incdt, dep = self._delta_arrays()
            lag = incdt / np.maximum(inc, 1e-12)
            self._dcsr = CSRGraph(self.N, pre, post, inc, lag, dep)
            self._dcsr.ep = dep[np.argsort(pre, kind="stable")].astype(np.int64)   # keep full-width epochs
        return self._dcsr

    # ------------------------------------------------------------ reads
    def gather(self, units, lag_min=None):
        w1 = self.csr.gather(units, self.dtable, self.epoch, lag_min)
        d = self._delta_graph()
        if d is None:
            return w1
        w2 = d.gather(units, self.dtable, self.epoch, lag_min)
        return tuple(np.concatenate([a, b]) for a, b in zip(w1, w2))

    def accumulate(self, units, x=None, normalize=False, lag_min=None):
        """The one read kernel: take active units, read their rows, add into a score array."""
        units = np.asarray(units, np.int64)
        x = np.ones(len(units), np.float32) if x is None else np.asarray(x, np.float32)
        which, post, w = self.gather(units, lag_min)
        if normalize and len(w):
            rs = np.bincount(which, weights=w, minlength=len(units))
            w = w / np.maximum(rs[which], 1e-9)
        return np.bincount(post, weights=w * x[which], minlength=self.N).astype(np.float32)

    def spread(self, units, x=None, hops=2, hop_decay=0.5, keep=96, normalize=True, lagged_only=False, lag_min=0.05):
        """Spreading activation: weight times per-hop decay, bounded hops, top-`keep` units per hop."""
        units = np.asarray(units, np.int64)
        fx = np.ones(len(units), np.float32) if x is None else np.asarray(x, np.float32)
        total = np.zeros(self.N, np.float32)
        lm = lag_min if lagged_only else None
        for _ in range(hops):
            nxt = self.accumulate(units, fx, normalize, lm) * hop_decay
            if keep and (nxt > 0).sum() > keep:
                thr = np.partition(nxt, -keep)[-keep]
                nxt[nxt < thr] = 0.0
            total += nxt
            units = np.flatnonzero(nxt); fx = nxt[units]
            if len(units) == 0:
                break
        return total

    def in_edges(self, v):
        return self.csr.in_edges(v)

    # ------------------------------------------------------------ merge
    def merge(self, prune=True):
        N, E, dt = self.N, self.epoch, self.dtable
        mx = len(dt) - 1
        pre_old = np.repeat(np.arange(N), np.diff(self.csr.indptr))
        key_old = pre_old * N + self.csr.post.astype(np.int64)
        W = np.zeros(N * N, np.float32); EP = np.zeros(N * N, np.int64); LAG = np.zeros(N * N, np.float32)
        W[key_old] = self.csr.w; EP[key_old] = self.csr.ep; LAG[key_old] = self.csr.lag
        if self.n_delta:
            pre, post, inc, incdt, dep = self._delta_arrays()
            dec = dt[np.minimum(E - dep, mx)]
            key = pre * N + post
            touched, inv = np.unique(key, return_inverse=True)
            inc_sum = np.bincount(inv, weights=inc * dec, minlength=len(touched))
            dt_sum = np.bincount(inv, weights=incdt * dec, minlength=len(touched))
            old_eff = W[touched] * dt[np.minimum(E - EP[touched], mx)]
            new_w = old_eff + inc_sum
            W[touched] = new_w
            LAG[touched] = (old_eff * LAG[touched] + dt_sum) / np.maximum(new_w, 1e-12)
            EP[touched] = E
        if prune:
            eff = (W * dt[np.minimum(E - EP, mx)].astype(np.float32)).reshape(N, N)
            keep = (W.reshape(N, N) > 0) & (eff >= self.eps)
            if self.R is not None and self.R < N:
                thr = -np.partition(-eff, self.R - 1, axis=1)[:, self.R - 1]
                keep &= eff >= np.maximum(thr, self.eps)[:, None]
            keep = keep.reshape(-1)
        else:
            keep = W > 0
        idx = np.flatnonzero(keep)
        self.csr = CSRGraph(N, idx // N, idx % N, W[idx], LAG[idx], EP[idx], assume_sorted=True)
        self._chunks, self.n_delta, self._dcsr = [], 0, None

    # ------------------------------------------------------------ introspection
    def effective_matrix(self) -> np.ndarray:
        """dense effective weights (lattice + delta), for tests"""
        M = np.zeros((self.N, self.N), np.float64)
        pre, post, w = self.csr.edges(self.dtable, self.epoch)
        np.add.at(M, (pre, post), w)
        if self.n_delta:
            p, q, inc, _, dep = self._delta_arrays()
            np.add.at(M, (p, q), inc * self.dtable[np.minimum(self.epoch - dep, len(self.dtable) - 1)])
        return M

    def nbytes(self):
        d = self.n_delta * (8 + 8 + 4 + 4 + 8)
        return self.csr.nbytes() + d
