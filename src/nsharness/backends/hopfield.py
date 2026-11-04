"""Modern (dense-associative) Hopfield memory: softmax attention over the stored patterns, energy-based familiarity, clamped completion."""
from __future__ import annotations
import numpy as np
from .exact import ExactBackend
from .._util import iter_chunks, logsumexp_rows, softmax_rows


class HopfieldBackend(ExactBackend):
    """Same stored patterns as ExactBackend (so the same bytes); what differs is how they are read.
    readout: attention over ALL patterns with weights softmax(beta * cos) (set full_attention=False for top-k only).
    familiarity: the Hopfield negative energy  lse(beta * cos) / beta  (a smoothed, density-aware max-similarity).
    reconstruct: `steps` clamped attention updates  xi <- X^T softmax(beta X xi)  that keep observed entries fixed."""
    name = "hopfield"

    def __init__(self, beta=30.0, steps=1, full_attention=True):
        super().__init__(beta=float(beta), steps=int(steps), full_attention=bool(full_attention))

    def _rows(self):
        return int(max(1, min(512, 2e7 // max(self.n, 1))))

    def _attention(self, S):
        return softmax_rows(self.cfg["beta"] * S).astype(np.float64)

    def readout(self, Q, Y, k, mask=None):
        if not self.cfg["full_attention"]:
            return super().readout(Q, Y, k, mask)
        Q = np.asarray(Q, np.float32)
        Yf = np.asarray(Y, np.float64)
        out = []
        for a, b in iter_chunks(len(Q), self._rows()):
            A = self._attention(self.sims(Q[a:b], None if mask is None else mask[a:b]))
            out.append(A @ Yf)
        return np.concatenate(out)

    def familiarity(self, Q, mask=None):
        base = super().familiarity(Q, mask)
        Q = np.asarray(Q, np.float32)
        lse = [logsumexp_rows(self.cfg["beta"] * self.sims(Q[a:b], None if mask is None else mask[a:b])) / self.cfg["beta"]
               for a, b in iter_chunks(len(Q), self._rows())]
        base["neg_energy"] = np.concatenate(lse)
        return base

    def reconstruct(self, Q, k, mask=None):
        if not self.cfg["full_attention"]:
            return super().reconstruct(Q, k, mask)
        Q = np.asarray(Q, np.float32)
        raw = self.Xn * self.norm[:, None]
        obs = None if mask is None else np.asarray(mask, bool)
        xi = Q.copy()
        for t in range(self.cfg["steps"]):
            new = np.empty_like(xi)
            for a, b in iter_chunks(len(xi), self._rows()):
                S = self.sims(xi[a:b], obs[a:b] if (obs is not None and t == 0) else None)
                new[a:b] = (self._attention(S) @ raw).astype(np.float32)
            xi = new if obs is None else np.where(obs, Q, new)
        return xi
