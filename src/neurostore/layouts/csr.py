"""Synapse lattice storage: CSR by pre unit with post[], weight[], lag[], epoch[] as separate
arrays, plus a transposed index (post -> pre) that references the same weight array.
Immutable once built. Lazy decay: effective weight = w * dtable[epoch_now - edge_epoch]."""
from __future__ import annotations
import numpy as np
from ..util import gather_ranges


class CSRGraph:
    name = "csr"

    def __init__(self, N, pre, post, w, lag, ep, assume_sorted=False):
        self.N = N
        pre = np.asarray(pre, np.int64)
        if not assume_sorted and len(pre):
            o = np.argsort(pre, kind="stable"); pre = pre[o]; post = np.asarray(post)[o]
            w = np.asarray(w)[o]; lag = np.asarray(lag)[o]; ep = np.asarray(ep)[o]
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(pre, minlength=N))]).astype(np.int64)
        self.post = np.asarray(post).astype(np.int16)
        self.w = np.asarray(w, np.float32)
        self.lag = np.asarray(lag, np.float16)
        self.ep = np.asarray(ep).astype(np.uint16)
        self._t = None

    @property
    def n_edges(self) -> int:
        return len(self.post)

    # ---- transposed copy (post -> pre), built on first use
    def _transpose(self):
        if self._t is None:
            pre = np.repeat(np.arange(self.N), np.diff(self.indptr))
            o = np.argsort(self.post, kind="stable")
            tptr = np.concatenate([[0], np.cumsum(np.bincount(self.post, minlength=self.N))]).astype(np.int64)
            self._t = (tptr, pre[o].astype(np.int16), o.astype(np.int64))
        return self._t

    def out_edges(self, u):
        s, e = self.indptr[u], self.indptr[u + 1]
        return self.post[s:e].astype(np.int64), self.w[s:e], self.lag[s:e], self.ep[s:e]

    def in_edges(self, v):
        tptr, tpre, tpos = self._transpose()
        s, e = tptr[v], tptr[v + 1]
        pos = tpos[s:e]
        return tpre[s:e].astype(np.int64), self.w[pos], self.lag[pos], self.ep[pos]

    def gather(self, units, dtable, epoch, lag_min=None):
        """Sparse-accumulate gather: -> (which, post, effective weight) for all out-edges of `units`."""
        offs, which = gather_ranges(self.indptr, units)
        if len(offs) == 0:
            return which, offs, np.zeros(0, np.float32)
        age = np.minimum(epoch - self.ep[offs].astype(np.int64), len(dtable) - 1)
        weff = self.w[offs] * dtable[age]
        post = self.post[offs].astype(np.int64)
        if lag_min is not None:
            m = self.lag[offs] > lag_min
            which, post, weff = which[m], post[m], weff[m]
        return which, post, weff

    def edges(self, dtable=None, epoch=0):
        """all edges as (pre, post, effective weight), sorted by (pre, post)"""
        pre = np.repeat(np.arange(self.N), np.diff(self.indptr))
        w = self.w if dtable is None else self.w * dtable[np.minimum(epoch - self.ep.astype(np.int64), len(dtable) - 1)]
        o = np.lexsort((self.post, pre))
        return pre[o], self.post[o].astype(np.int64), w[o]

    def nbytes(self):
        t = 0 if self._t is None else sum(a.nbytes for a in self._t)
        return int(self.indptr.nbytes + self.post.nbytes + self.w.nbytes + self.lag.nbytes + self.ep.nbytes + t)
