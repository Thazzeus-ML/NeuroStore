"""Append-only spike log. Records are (event_id, unit_id, t, strength, surprise), stored as
fixed-width volleys by default. Fully expected events are logged as a count only."""
from __future__ import annotations
import numpy as np
from .layouts import make_store


class SpikeLog:
    def __init__(self, k: int, layout: str = "volley", **kw):
        self.k = k
        self.store = make_store(layout, k, **kw)
        self.bytes_written = 0
        self.n_count_only = 0
        self.count_only_t: list = []        # not stored per-event; only a running count + last t
        self.last_count_t = None

    @property
    def n_stored(self) -> int:
        return self.store.n

    def append(self, event_id, t, units, strength_q, surprise_q) -> int:
        assert len(units) == self.k
        b = self.store.append(event_id, t, units, strength_q, surprise_q)
        self.bytes_written += b
        return b

    def count_only(self, t=None):
        self.n_count_only += 1
        self.last_count_t = t

    def pattern(self, i):
        return self.store.event(i)

    def gather(self, idx):
        return self.store.gather(idx)

    def times(self):
        return self.store.times()

    def event_ids(self):
        return self.store.event_ids()
