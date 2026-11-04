"""Prediction cache + surprise.

Before an event is written, predict the units expected from the PREVIOUS event:
  * context key = the (<=2) assemblies that best match the previous event;
  * expected units = lattice spread (1 hop, row-normalised) from those assemblies' member units,
    keeping units with score >= rho * max (at most k);
  * results are cached per context key and invalidated whenever the lattice epoch or the
    assembly round changes. Cold start (no assemblies / no lattice) falls back to a direct
    spread from the previous event's own units, or to no expectation.
Surprise S = (|fired \\ expected| + |expected \\ fired|) / (|fired| + |expected|), in [0,1].
"""
from __future__ import annotations
import numpy as np
from .readout import select_assemblies


class Predictor:
    def __init__(self, store):
        self.s = store
        self.cache: dict = {}
        self.tag = None
        self.hits = self.misses = 0

    def _from_scores(self, sc):
        cfg = self.s.cfg
        mx = float(sc.max()) if len(sc) else 0.0
        if mx <= 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        idx = np.flatnonzero(sc >= cfg.pred_rho * mx)
        if len(idx) > cfg.k:
            idx = idx[np.argsort(-sc[idx])[:cfg.k]]
        idx = np.sort(idx)
        return idx, (sc[idx] / mx).astype(np.float32)

    def expected(self, prev_units, prev_cos):
        s = self.s
        lat, asm = s.lattice, s.asm
        if s.cfg.pred_mode == "assembly" and asm is not None and prev_cos is not None and prev_cos.max() >= 0.2:
            a = int(prev_cos.argmax())
            tag = ("asm", asm.round)
            if tag != self.tag:
                self.cache.clear(); self.tag = tag
            if a in self.cache:
                self.hits += 1
                return self.cache[a]
            self.misses += 1
            sc = np.zeros(s.cfg.N, np.float32); sc[asm.member_ids[a]] = asm.member_w[a]
            idx = np.argsort(-sc)[:s.cfg.k]; idx = np.sort(idx[sc[idx] > 0])
            self.cache[a] = (idx, np.ones(len(idx), np.float32))
            return self.cache[a]
        if prev_units is None or lat is None or (lat.csr.n_edges == 0 and lat.n_delta == 0):
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        tag = (lat.epoch, asm.round if asm is not None else 0)
        if tag != self.tag:
            self.cache.clear(); self.tag = tag
        key = None
        if asm is not None and asm.n_alive and prev_cos is not None and prev_cos.max() >= 0.2:
            key = tuple(sorted(int(a) for a in select_assemblies(prev_cos, 0.7, 0.2, 2)))
        if key is not None and key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        if key:
            seeds = np.zeros(s.cfg.N, np.float32)
            for a in key:
                seeds[asm.member_ids[a]] += asm.member_w[a]
            u = np.flatnonzero(seeds); x = seeds[u]
        else:
            u, x = prev_units, None
        out = self._from_scores(lat.spread(u, x, hops=1, normalize=True, keep=0))
        if key is not None:
            self.cache[key] = out
        return out


def surprise(fired, exp_units, exp_scores, N):
    """-> (event surprise S, per-spike surprise for each fired unit)"""
    e = np.zeros(N, np.float32)
    if len(exp_units):
        e[exp_units] = exp_scores
    inter = int((e[fired] > 0).sum())
    a, b = len(fired) - inter, len(exp_units) - inter
    S = (a + b) / (len(fired) + len(exp_units)) if (len(fired) + len(exp_units)) else 1.0
    return float(S), (1.0 - e[fired]).astype(np.float32)
