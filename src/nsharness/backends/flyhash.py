"""FlyHash: sparse random expansion + winner-take-all, similarity = overlap of active units, searched through an inverted index."""
from __future__ import annotations
import math
import numpy as np
from ..core import Backend
from .._util import topk_desc, csr_gather, seed_stream, iter_chunks


class FlyHashBackend(Backend):
    name = "flyhash"
    supports_mask = False            # masked queries are zero-filled (a z-scored zero is the mean)
    supports_reconstruct = False     # the code has no decoder: only the active-unit set is stored

    def __init__(self, m=2048, frac=0.05, fan_frac=0.1, signed=False, beta=None):
        super().__init__(m=int(m), frac=float(frac), fan_frac=float(fan_frac), signed=bool(signed), beta=beta)

    def _encode(self, X):
        X = np.asarray(X, np.float32)
        out = np.empty((len(X), self.k_act), np.int32)
        for a, b in iter_chunks(len(X), 2048):
            A = X[a:b] @ self._P
            part = np.argpartition(-A, self.k_act - 1, axis=1)[:, : self.k_act]
            out[a:b] = np.sort(part, axis=1)
        return out

    def fit(self, X, seed: int = 0):
        rng = seed_stream(seed, 21)
        X = np.asarray(X, np.float32)
        self.n, self.d = X.shape
        m = self.cfg["m"]
        f = int(max(1, min(self.d, round(self.cfg["fan_frac"] * self.d))))
        idx = np.stack([rng.choice(self.d, f, replace=False) for _ in range(m)])
        self.proj_idx = idx.astype(np.uint8 if self.d <= 256 else np.uint16)
        sign = rng.choice(np.array([-1.0, 1.0], np.float32), size=(m, f)) if self.cfg["signed"] else np.ones((m, f), np.float32)
        self._P = np.zeros((self.d, m), np.float32)
        self._P[idx, np.arange(m)[:, None]] = sign
        self.k_act = int(max(1, round(self.cfg["frac"] * m)))
        act = self._encode(X)
        flat = act.ravel().astype(np.int64)
        order = np.argsort(flat, kind="stable")
        self.post = np.repeat(np.arange(self.n, dtype=np.int32), self.k_act)[order]
        ptr = np.zeros(m + 1, np.int64)
        np.cumsum(np.bincount(flat, minlength=m), out=ptr[1:])
        self.ptr = ptr.astype(np.int32 if ptr[-1] < 2 ** 31 else np.int64)
        h = (np.arange(self.n, dtype=np.uint64) * np.uint64(2654435761) + np.uint64(int(seed) * 40503 + 1)) & np.uint64(0xFFFFFFFF)
        self._tie = (h.astype(np.float64) / 2.0 ** 32 * 0.5).astype(np.float32)     # tie-break in [0, 0.5): never outranks one more overlap
        return self

    def search(self, Q, k, mask=None):
        act = self._encode(Q)
        q, k = len(act), min(int(k), self.n)
        oi = np.empty((q, k), np.int64)
        os_ = np.empty((q, k), np.float32)
        for i in range(q):
            cnt = np.bincount(self.post[csr_gather(self.ptr, act[i])], minlength=self.n)
            idx, _ = topk_desc((cnt + self._tie)[None, :], k)
            oi[i] = idx[0]
            os_[i] = cnt[idx[0]] / self.k_act
        return oi, os_

    def memory_report(self):
        m, k, n = self.cfg["m"], self.k_act, self.n
        return {
            "inverted_index_postings_i32": self.post.nbytes,
            "inverted_index_ptr": self.ptr.nbytes,
            "projection_indices": self.proj_idx.nbytes + (self.proj_idx.size // 8 if self.cfg["signed"] else 0),
            "alt:bitmap_packed": int(n * math.ceil(m / 8)),
            "alt:index_list_u16": int(n * k * 2),
            "alt:entropy_bound": int(math.ceil(n * (math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1)) / math.log(2) / 8)),
            "cache:dense_projection_f32": self._P.nbytes,
            "cache:tie_break_f32": self._tie.nbytes,
        }
