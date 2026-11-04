"""Baseline C: dict-of-dicts adjacency {pre: {post: [w, lag, epoch]}} with python objects."""
from __future__ import annotations
import numpy as np


class DictGraph:
    name = "dict"

    def __init__(self, N, pre, post, w, lag, ep):
        self.N = N
        self.adj: dict = {}
        for a, b, x, l, e in zip(np.asarray(pre).tolist(), np.asarray(post).tolist(), np.asarray(w, np.float32).tolist(),
                                 np.asarray(lag, np.float16).tolist(), np.asarray(ep).tolist()):
            self.adj.setdefault(a, {})[b] = [x, l, int(e)]

    @classmethod
    def from_csr(cls, g):
        pre = np.repeat(np.arange(g.N), np.diff(g.indptr))
        return cls(g.N, pre, g.post.astype(np.int64), g.w, g.lag, g.ep)

    @property
    def n_edges(self):
        return sum(len(r) for r in self.adj.values())

    def out_edges(self, u):
        r = self.adj.get(int(u), {})
        ks = sorted(r)
        return (np.array(ks, np.int64), np.array([r[k][0] for k in ks], np.float32),
                np.array([r[k][1] for k in ks], np.float16), np.array([r[k][2] for k in ks]))

    def in_edges(self, v):
        """no transposed copy: full scan over every row"""
        out = [(a, r[v]) for a, r in self.adj.items() if v in r]
        out.sort()
        return (np.array([a for a, _ in out], np.int64), np.array([e[0] for _, e in out], np.float32),
                np.array([e[1] for _, e in out], np.float16), np.array([e[2] for _, e in out]))

    def gather(self, units, dtable, epoch, lag_min=None):
        which, post, w = [], [], []
        for i, u in enumerate(np.asarray(units).tolist()):
            for b, (x, l, e) in self.adj.get(u, {}).items():
                if lag_min is not None and not (l > lag_min):
                    continue
                which.append(i); post.append(b); w.append(x * dtable[min(epoch - e, len(dtable) - 1)])
        return np.array(which, np.int64), np.array(post, np.int64), np.array(w, np.float32)

    def edges(self, dtable=None, epoch=0):
        pre, post, w = [], [], []
        for a in sorted(self.adj):
            for b in sorted(self.adj[a]):
                x, _, e = self.adj[a][b]
                pre.append(a); post.append(b)
                w.append(x if dtable is None else x * dtable[min(epoch - e, len(dtable) - 1)])
        return np.array(pre, np.int64), np.array(post, np.int64), np.array(w, np.float32)
