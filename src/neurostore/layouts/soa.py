"""Baseline B: plain SoA over spikes (one array per field, per-spike event_id and t).
normalized=True is a control ("soa_event"): event_id and t stored once per event, spike
arrays only hold unit/strength/surprise. It isolates what the volley layout gains merely
from not repeating event_id/t on every spike."""
from __future__ import annotations
import numpy as np
from .base import SpikeStore


class SoAStore(SpikeStore):
    def __init__(self, k, cap_events=1024, normalized=False):
        super().__init__(k)
        self.normalized = normalized
        self.name = "soa_event" if normalized else "soa"
        c = cap_events
        self.unit = np.empty(c * k, "<u2"); self.st = np.empty(c * k, "u1"); self.su = np.empty(c * k, "u1")
        if normalized:
            self.eid = np.empty(c, "<u4"); self.t = np.empty(c, "<f4")
        else:
            self.eid = np.empty(c * k, "<u4"); self.t = np.empty(c * k, "<f4")

    def _grow(self):
        for nm in ("unit", "st", "su", "eid", "t"):
            a = getattr(self, nm); b = np.empty(2 * len(a), a.dtype); b[:len(a)] = a; setattr(self, nm, b)

    def append(self, event_id, t, units, strength, surprise):
        k, n = self.k, self.n
        if (n + 1) * k > len(self.unit):
            self._grow()
        s = slice(n * k, (n + 1) * k)
        self.unit[s] = units; self.st[s] = strength; self.su[s] = surprise
        if self.normalized:
            self.eid[n] = event_id; self.t[n] = t
            b = k * 4 + 8
        else:
            self.eid[s] = event_id; self.t[s] = t
            b = k * 12
        self.n += 1
        return b

    def event(self, i):
        k = self.k; s = slice(i * k, (i + 1) * k)
        j = i if self.normalized else i * k
        return int(self.eid[j]), float(self.t[j]), self.unit[s].astype(np.int32), self.st[s].copy(), self.su[s].copy()

    def gather(self, idx):
        idx = np.asarray(idx, np.int64)
        f = idx[:, None] * self.k + np.arange(self.k)
        return self.unit[f].astype(np.int32), self.st[f], self.su[f]

    def event_ids(self):
        return (self.eid[:self.n] if self.normalized else self.eid[:self.n * self.k:self.k]).astype(np.uint32)

    def times(self):
        return (self.t[:self.n] if self.normalized else self.t[:self.n * self.k:self.k]).astype(np.float32)

    def eids_at(self, idx):
        i = np.asarray(idx, np.int64)
        return self.eid[i if self.normalized else i * self.k].astype(np.uint32)

    def times_at(self, idx):
        i = np.asarray(idx, np.int64)
        return self.t[i if self.normalized else i * self.k].astype(np.float32)

    def _chunks(self, field):
        a = {"unit": self.unit, "strength": self.st, "surprise": self.su}[field]
        yield 0, a[:self.n * self.k]

    def nbytes(self):
        nsp = self.n * self.k
        b = nsp * (2 + 1 + 1)
        return b + (self.n * 8 if self.normalized else nsp * 8)
