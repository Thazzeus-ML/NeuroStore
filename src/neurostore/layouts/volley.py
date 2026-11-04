"""Volley tiles (AoSoA). One event = one fixed-width volley of k slots. A tile holds V volleys
with separate field arrays: event_id[V], t[V], unit[V*k], strength[V*k], surprise[V*k].
Tiles are allocated in slabs of TS tiles so growth never copies sealed data."""
from __future__ import annotations
import numpy as np
from .base import SpikeStore


class VolleyStore(SpikeStore):
    name = "volley"

    def __init__(self, k, V=64, tiles_per_slab=64):
        super().__init__(k)
        self.V, self.TS = V, tiles_per_slab
        self.SV = V * tiles_per_slab          # events per slab
        self.slabs: list[dict] = []

    def _new_slab(self):
        TS, V, k = self.TS, self.V, self.k
        self.slabs.append({"eid": np.zeros((TS, V), "<u4"), "t": np.zeros((TS, V), "<f4"),
                           "unit": np.zeros((TS, V, k), "<u2"), "st": np.zeros((TS, V, k), "u1"),
                           "su": np.zeros((TS, V, k), "u1")})

    def append(self, event_id, t, units, strength, surprise):
        n = self.n
        s, loc = divmod(n, self.SV)
        if s == len(self.slabs):
            self._new_slab()
        tile, v = divmod(loc, self.V)
        sl = self.slabs[s]
        sl["eid"][tile, v] = event_id; sl["t"][tile, v] = t
        sl["unit"][tile, v] = units; sl["st"][tile, v] = strength; sl["su"][tile, v] = surprise
        self.n += 1
        return 8 + 4 * self.k

    def event(self, i):
        s, loc = divmod(i, self.SV); tile, v = divmod(loc, self.V); sl = self.slabs[s]
        return (int(sl["eid"][tile, v]), float(sl["t"][tile, v]), sl["unit"][tile, v].astype(np.int32),
                sl["st"][tile, v].copy(), sl["su"][tile, v].copy())

    def gather(self, idx):
        idx = np.asarray(idx, np.int64); B = len(idx)
        u = np.empty((B, self.k), np.int32); st = np.empty((B, self.k), np.uint8); su = np.empty((B, self.k), np.uint8)
        s_of, loc = idx // self.SV, idx % self.SV
        for s in np.unique(s_of):
            m = s_of == s; sl = self.slabs[s]
            u[m] = sl["unit"].reshape(-1, self.k)[loc[m]]
            st[m] = sl["st"].reshape(-1, self.k)[loc[m]]
            su[m] = sl["su"].reshape(-1, self.k)[loc[m]]
        return u, st, su

    def _flat(self, name):
        out = []
        for s, sl in enumerate(self.slabs):
            cnt = min(self.SV, self.n - s * self.SV)
            if cnt > 0:
                out.append(sl[name].reshape(-1)[:cnt] if sl[name].ndim == 2 else sl[name].reshape(-1)[:cnt * self.k])
        return out

    def event_ids(self):
        return np.concatenate(self._flat("eid")) if self.n else np.zeros(0, np.uint32)

    def times(self):
        return np.concatenate(self._flat("t")) if self.n else np.zeros(0, np.float32)

    def _chunks(self, field):
        name = {"unit": "unit", "strength": "st", "surprise": "su"}[field]
        for s, a in enumerate(self._flat(name)):
            yield s * self.SV * self.k, a

    def nbytes(self):
        return self.n * (8 + 4 * self.k)
