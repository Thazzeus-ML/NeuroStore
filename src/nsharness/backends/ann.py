"""IVF approximate search: spherical k-means coarse quantiser, nprobe inverted lists, optional 8-bit scalar quantisation."""
from __future__ import annotations
import math
import numpy as np
from ..core import Backend
from .._util import topk_desc, kmeans, seed_stream, iter_chunks


class IVFBackend(Backend):
    name = "ann"
    supports_mask = False            # a masked query is simply zero-filled
    supports_reconstruct = False     # the index sits on top of an exact store; completion is measured on that store

    def __init__(self, nlist=None, nprobe=8, sq8=False, beta=None, iters=10, train_cap=40000, list_factor=2.0):
        super().__init__(nlist=nlist, nprobe=nprobe, sq8=sq8, beta=beta, iters=iters, train_cap=train_cap, list_factor=list_factor)
        self.last_scanned = float("nan")

    @property
    def nprobe(self):
        return int(self.cfg["nprobe"])

    def set_nprobe(self, nprobe: int):
        self.cfg["nprobe"] = int(nprobe)
        return self

    def fit(self, X, seed: int = 0):
        rng = seed_stream(seed, 11)
        X = np.asarray(X, np.float32)
        self.n, self.d = X.shape
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        nlist = self.cfg["nlist"] or int(max(1, round(self.cfg["list_factor"] * math.sqrt(self.n))))
        nlist = int(min(nlist, self.n))
        cap = int(self.cfg["train_cap"])
        sample = Xn if self.n <= cap else Xn[rng.choice(self.n, cap, replace=False)]
        self.C, _ = kmeans(sample, nlist, rng, iters=int(self.cfg["iters"]), spherical=True)
        self.C = self.C.astype(np.float32)
        self.nlist = len(self.C)
        assign = np.empty(self.n, np.int32)
        for a, b in iter_chunks(self.n, 8192):
            assign[a:b] = np.argmax(Xn[a:b] @ self.C.T, axis=1)
        order = np.argsort(assign, kind="stable")
        counts = np.bincount(assign, minlength=self.nlist)
        self.ptr = np.zeros(self.nlist + 1, np.int64)
        np.cumsum(counts, out=self.ptr[1:])
        self.order = order.astype(np.int32)
        Xl = np.ascontiguousarray(Xn[order])
        if self.cfg["sq8"]:
            self.scale = (np.maximum(np.abs(Xl).max(axis=0), 1e-9) / 127.0).astype(np.float32)
            self.Xq = np.clip(np.round(Xl / self.scale), -127, 127).astype(np.int8)
            self.Xl = None
        else:
            self.Xl, self.Xq, self.scale = Xl, None, None
        return self

    def _list_vectors(self, lo, hi):
        if self.Xl is not None:
            return self.Xl[lo:hi]
        return self.Xq[lo:hi].astype(np.float32) * self.scale

    def search(self, Q, k, mask=None):
        Q = np.asarray(Q, np.float32)
        Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
        q = len(Q)
        k = min(int(k), self.n)
        npb = min(self.nprobe, self.nlist)
        probes, _ = topk_desc(Q @ self.C.T, npb)
        best_s = np.full((q, k), -np.inf, np.float32)
        best_i = np.full((q, k), -1, np.int64)
        flat_l = probes.ravel()
        flat_q = np.repeat(np.arange(q), npb)
        o = np.argsort(flat_l, kind="stable")
        fl, fq = flat_l[o], flat_q[o]
        uniq, starts = np.unique(fl, return_index=True)
        ends = np.append(starts[1:], len(fl))
        scanned = 0
        for l, a, b in zip(uniq, starts, ends):
            lo, hi = int(self.ptr[l]), int(self.ptr[l + 1])
            if hi == lo:
                continue
            qs = fq[a:b]
            S = Q[qs] @ self._list_vectors(lo, hi).T
            scanned += S.size
            ti, tv = topk_desc(S, min(k, hi - lo))
            cs = np.concatenate([best_s[qs], tv], axis=1)
            ci = np.concatenate([best_i[qs], self.order[lo:hi][ti]], axis=1)
            j, v = topk_desc(cs, k)
            best_s[qs] = v
            best_i[qs] = np.take_along_axis(ci, j, axis=1)
        self.last_scanned = scanned / max(q, 1) + self.nlist
        short = np.where(best_i[:, -1] < 0)[0]          # fewer than k candidates in the probed lists: fall back to a full scan
        if len(short):
            full = np.concatenate([self._list_vectors(int(self.ptr[l]), int(self.ptr[l + 1])) for l in range(self.nlist)])
            ids = self.order
            ti, tv = topk_desc(Q[short] @ full.T, k)
            best_s[short], best_i[short] = tv, ids[ti]
            self.last_scanned = float(self.n + self.nlist)
        return best_i, best_s

    def memory_report(self):
        rep = {"centroids_f32": self.C.nbytes, "list_ptr_i64": self.ptr.nbytes, "item_ids_i32": self.order.nbytes}
        if self.Xl is not None:
            rep["vectors_f32"] = self.Xl.nbytes
        else:
            rep["vectors_i8"] = self.Xq.nbytes
            rep["sq8_scale_f32"] = self.scale.nbytes
        return rep
