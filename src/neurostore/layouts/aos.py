"""Baseline A: row-per-spike array of structs (packed 12-byte records)."""
from __future__ import annotations
import numpy as np
from .base import SpikeStore

DT = np.dtype([("eid", "<u4"), ("unit", "<u2"), ("t", "<f4"), ("strength", "u1"), ("surprise", "u1")])
assert DT.itemsize == 12


class AoSStore(SpikeStore):
    name = "aos"

    def __init__(self, k, cap_events=1024):
        super().__init__(k)
        self.rows = np.empty(cap_events * k, DT)

    def append(self, event_id, t, units, strength, surprise):
        k, n = self.k, self.n
        if (n + 1) * k > len(self.rows):
            new = np.empty(2 * len(self.rows), DT); new[:n * k] = self.rows[:n * k]; self.rows = new
        r = self.rows[n * k:(n + 1) * k]
        r["eid"] = event_id; r["unit"] = units; r["t"] = t
        r["strength"] = strength; r["surprise"] = surprise
        self.n += 1
        return k * DT.itemsize

    def event(self, i):
        r = self.rows[i * self.k:(i + 1) * self.k]
        return int(r["eid"][0]), float(r["t"][0]), r["unit"].astype(np.int32), r["strength"].copy(), r["surprise"].copy()

    def gather(self, idx):
        idx = np.asarray(idx, np.int64)
        r = self.rows[(idx[:, None] * self.k + np.arange(self.k))]
        return r["unit"].astype(np.int32), r["strength"], r["surprise"]

    def event_ids(self):
        return self.rows["eid"][:self.n * self.k:self.k].astype(np.uint32)

    def times(self):
        return self.rows["t"][:self.n * self.k:self.k].astype(np.float32)

    def _chunks(self, field):
        name = "unit" if field == "unit" else field
        yield 0, self.rows[name][:self.n * self.k]

    def nbytes(self):
        return self.n * self.k * DT.itemsize
