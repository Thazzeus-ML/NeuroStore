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

    def unit_history(self, u: int):
        """(event_ids, times) of every event in which unit u fired (full scan)."""
        pos = [np.flatnonzero(a == u) + off for off, a in self._chunks("unit")]
        pos = np.concatenate(pos) if pos else np.zeros(0, np.int64)
        ev = pos // self.k
        return self.event_ids()[ev], self.times()[ev]

    def nbytes(self) -> int:
        """logical bytes of the filled portion"""
        raise NotImplementedError


def check_event(units, strength, surprise, k):
    assert len(units) == k and len(strength) == k and len(surprise) == k
