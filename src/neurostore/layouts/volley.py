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

    def _gather_hdr(self, name, idx):
        idx = np.asarray(idx, np.int64); out = np.empty(len(idx), self.slabs[0][name].dtype if self.slabs else "f4")
        s_of, loc = idx // self.SV, idx % self.SV
        for s in np.unique(s_of):
            m = s_of == s
            out[m] = self.slabs[s][name].reshape(-1)[loc[m]]
        return out

    def eids_at(self, idx):
        return self._gather_hdr("eid", idx).astype(np.uint32)

    def times_at(self, idx):
        return self._gather_hdr("t", idx).astype(np.float32)

    def build_postings(self):
        """unit-major lists of (tile, slot): tile = global tile id (int32), slot = v*k + j within the tile (uint16)"""
        flat = np.concatenate([a for _, a in self._chunks("unit")]) if self.n else np.zeros(0, "<u2")
        order = np.argsort(flat, kind="stable")
        self._pp_ptr = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=65536))]).astype(np.int64)
        rows = order.astype(np.int64)
        ev, j = rows // self.k, rows % self.k
        self._pp_tile = (ev // self.V).astype(np.int32)
        self._pp_slot = ((ev % self.V) * self.k + j).astype(np.uint16)
        self._pp_n = self.n

    def unit_history_indexed(self, u):
        s, e = self._pp_ptr[u], self._pp_ptr[u + 1]
        ev = self._pp_tile[s:e].astype(np.int64) * self.V + self._pp_slot[s:e].astype(np.int64) // self.k
        return self.eids_at(ev), self.times_at(ev)

    def postings_nbytes(self):
        return int(self._pp_tile.nbytes + self._pp_slot.nbytes + self._pp_ptr.nbytes)

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
