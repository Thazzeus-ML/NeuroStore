"""Spike-store interface. Every layout stores identical logical content:
event i (log position) = (event_id u32, t f32, units u16[k], strength u8[k], surprise u8[k]).
Layouts differ only in physical arrangement."""
from __future__ import annotations
import numpy as np

HEADER_BYTES = 8   # event_id u32 + t f32


class SpikeStore:
    name = "base"

    def __init__(self, k: int):
        self.k, self.n = k, 0

    # -- writes
    def append(self, event_id, t, units, strength, surprise) -> int:
        raise NotImplementedError

    # -- event-local reads
    def event(self, i: int):
        raise NotImplementedError

    def gather(self, idx):
        """-> units[B,k] int32, strength[B,k] uint8, surprise[B,k] uint8"""
        raise NotImplementedError

    def event_ids(self) -> np.ndarray:
        raise NotImplementedError

    def times(self) -> np.ndarray:
        raise NotImplementedError

    # -- scans
    def _chunks(self, field):
        """yield (spike_offset, 1-D array over that chunk of spikes in log order)"""
        raise NotImplementedError

    def field_sum(self, field: str) -> int:
        return int(sum(int(a.sum(dtype=np.int64)) for _, a in self._chunks(field)))

    def eids_at(self, idx):
        raise NotImplementedError

    def times_at(self, idx):
        raise NotImplementedError

    def unit_history(self, u: int):
        """(event_ids, times) of every event in which unit u fired (full scan, no index)."""
        pos = [np.flatnonzero(a == u) + off for off, a in self._chunks("unit")]
        pos = np.concatenate(pos) if pos else np.zeros(0, np.int64)
        ev = pos // self.k
        return self.eids_at(ev), self.times_at(ev)

    # ---- postings: unit-major lists, rebuilt from sealed data by compaction
    def build_postings(self):
        units = np.concatenate([a for _, a in self._chunks("unit")]) if self.n else np.zeros(0, "<u2")
        order = np.argsort(units, kind="stable")
        self._pp_ptr = np.concatenate([[0], np.cumsum(np.bincount(units, minlength=65536))]).astype(np.int64)
        self._pp_rows = order.astype(np.int32)          # spike row ids (log order within a unit)
        self._pp_n = self.n

    def unit_history_indexed(self, u: int):
        rows = self._pp_rows[self._pp_ptr[u]:self._pp_ptr[u + 1]].astype(np.int64)
        ev = rows // self.k
        return self.eids_at(ev), self.times_at(ev)

    def postings_nbytes(self) -> int:
        return int(self._pp_rows.nbytes + self._pp_ptr.nbytes)

    def nbytes(self) -> int:
        """logical bytes of the filled portion"""
        raise NotImplementedError


def check_event(units, strength, surprise, k):
    assert len(units) == k and len(strength) == k and len(surprise) == k
