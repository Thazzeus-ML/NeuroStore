"""Vector-symbolic (hyperdimensional) memory: every item is ONE D-bit hypervector, the majority bundle of
role (XOR) level-vector bindings of its features. Search = packed popcount Hamming; decode = unbind a bundle of neighbours."""
from __future__ import annotations
import numpy as np
from ..core import Backend
from .._util import topk_desc, popcount64, seed_stream, iter_chunks


class VSABackend(Backend):
    name = "vsa"
    supports_mask = True            # missing features are simply left out of the query bundle
    supports_reconstruct = True     # decode(bundle of retrieved neighbours); nothing but hypervectors is stored

    def __init__(self, D=4096, levels=32, lo=-3.0, hi=3.0, beta=None, chunk=256):
        assert D % 64 == 0 and 2 <= levels <= 255
        super().__init__(D=int(D), levels=int(levels), lo=float(lo), hi=float(hi), beta=beta, chunk=int(chunk))

    # ---- codebooks and encoding
    def _levels(self, F):
        c = self.cfg
        t = (np.clip(np.asarray(F, np.float32), c["lo"], c["hi"]) - c["lo"]) / (c["hi"] - c["lo"]) * (c["levels"] - 1)
        return np.rint(t).astype(np.uint8)

    def _make_codebooks(self, d, rng):
        D, L = self.cfg["D"], self.cfg["levels"]
        base = rng.integers(0, 2, D, dtype=np.uint8)
        perm = rng.permutation(D)
        flips = np.rint(np.arange(L) * (D / 2) / (L - 1)).astype(int)       # level 0 and level L-1 differ in D/2 bits
        lev = np.empty((L, D), np.uint8)
        for l in range(L):
            v = base.copy()
            v[perm[: flips[l]]] ^= 1
            lev[l] = v
        role = rng.integers(0, 2, (d, D), dtype=np.uint8)
        tie = rng.integers(0, 2, D, dtype=np.uint8)
        return lev, role, tie

    def _encode(self, lvl, mask=None):
        D = self.cfg["D"]
        n, d = lvl.shape
        out = np.empty((n, D // 64), np.uint64)
        for a, b in iter_chunks(n, self.cfg["chunk"]):
            cnt = np.zeros((b - a, D), np.int16)
            if mask is None:
                for j in range(d):
                    cnt += self._table[j][lvl[a:b, j]]
                nobs = np.full((b - a, 1), d, np.int16)
            else:
                m = np.asarray(mask[a:b], bool)
                for j in range(d):
                    cnt += self._table[j][lvl[a:b, j]] * m[:, j, None]
                nobs = m.sum(1, keepdims=True).astype(np.int16)
            twice = 2 * cnt
            bits = np.where(twice > nobs, 1, np.where(twice < nobs, 0, self._tie_bits[None, :])).astype(np.uint8)
            out[a:b] = np.packbits(bits, axis=1, bitorder="little").view(np.uint64)
        return out

    def fit(self, X, seed: int = 0):
        rng = seed_stream(seed, 31)
        X = np.asarray(X, np.float32)
        self.n, self.d = X.shape
        self._lev, self._role, self._tie_bits = self._make_codebooks(self.d, rng)
        self._table = self._role[:, None, :] ^ self._lev[None, :, :]                 # (d, L, D) bound role (+) level
        self._lev_pm = (1.0 - 2.0 * self._lev).astype(np.float32)                   # bipolar: XOR == product
        self._role_pm = (1.0 - 2.0 * self._role).astype(np.float32)
        self.H = self._encode(self._levels(X), None)
        return self

    # ---- search
    def _hamming(self, qbits):
        out = np.empty(self.n, np.int32)
        for a, b in iter_chunks(self.n, 16384):
            out[a:b] = popcount64(self.H[a:b] ^ qbits).sum(axis=1)
        return out

    def search(self, Q, k, mask=None):
        qb = self._encode(self._levels(Q), mask)
        q, k = len(qb), min(int(k), self.n)
        oi = np.empty((q, k), np.int64)
        os_ = np.empty((q, k), np.float32)
        D = self.cfg["D"]
        for i in range(q):
            sim = 1.0 - 2.0 * self._hamming(qb[i]).astype(np.float32) / D
            idx, vals = topk_desc(sim[None, :], k)
            oi[i], os_[i] = idx[0], vals[0]
        return oi, os_

    # ---- decode
    def reconstruct(self, Q, k, mask=None):
        c = self.cfg
        idx, sim = self.search(Q, k, mask)
        w = self.weights(sim)
        w = w / np.maximum(w.sum(1, keepdims=True), 1e-300)
        vals = c["lo"] + np.arange(c["levels"], dtype=np.float32) * (c["hi"] - c["lo"]) / (c["levels"] - 1)
        out = np.empty((len(idx), self.d), np.float32)
        for i in range(len(idx)):
            bits = np.unpackbits(self.H[idx[i]].view(np.uint8), axis=1, bitorder="little")      # (k, D)
            bundle = (w[i][:, None] * (1.0 - 2.0 * bits)).sum(0).astype(np.float32)            # weighted bipolar bundle
            scores = (self._role_pm * bundle[None, :]) @ self._lev_pm.T                          # (d, L): unbind role j, match levels
            out[i] = vals[scores.argmax(1)]
        return out

    def memory_report(self):
        D, L, d = self.cfg["D"], self.cfg["levels"], self.d
        return {
            "hypervectors_packed": self.H.nbytes,
            "level_codebook_packed": L * D // 8,
            "role_codebook_packed": d * D // 8,
            "tie_bits_packed": D // 8,
            "cache:bind_table_u8": self._table.nbytes,
            "cache:codebooks_f32": self._lev_pm.nbytes + self._role_pm.nbytes,
        }
