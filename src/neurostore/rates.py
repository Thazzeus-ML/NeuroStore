"""Per-unit rate counters: cumulative, tumbling windows (closed windows are kept) and a
sliding window of n_sub sub-buckets (bucket granularity)."""
from __future__ import annotations
import numpy as np


class RateCounters:
    def __init__(self, N: int, window: float = 50.0, n_sub: int = 10):
        self.N, self.window, self.n_sub = N, window, n_sub
        self.bl = window / n_sub
        self.total = np.zeros(N, np.int64)
        self.n_events = 0
        self.ring = np.zeros((n_sub, N), np.int32)
        self.ring_ev = np.zeros(n_sub, np.int64)
        self.head = None
        self.cur = np.zeros(N, np.int32); self.cur_ev = 0; self.cur_w = None
        self.closed: list = []              # (window_index, counts, n_events)
        self.last_t = -np.inf

    def update(self, t: float, units: np.ndarray):
        assert t >= self.last_t, "time must be non-decreasing"
        self.last_t = t
        b = int(t // self.bl)
        if self.head is None:
            self.head = b
        if b > self.head:
            for j in range(1, min(b - self.head, self.n_sub) + 1):
                self.ring[(self.head + j) % self.n_sub] = 0; self.ring_ev[(self.head + j) % self.n_sub] = 0
            self.head = b
        self.ring[b % self.n_sub, units] += 1; self.ring_ev[b % self.n_sub] += 1
        w = int(t // self.window)
        if self.cur_w is None:
            self.cur_w = w
        if w != self.cur_w:
            self.closed.append((self.cur_w, self.cur.copy(), self.cur_ev))
            self.cur[:] = 0; self.cur_ev = 0; self.cur_w = w
        self.cur[units] += 1; self.cur_ev += 1
        self.total[units] += 1; self.n_events += 1

    def sliding(self):
        """(counts[N], n_events) over the last n_sub buckets"""
        return self.ring.sum(0), int(self.ring_ev.sum())

    def tumbling_current(self):
        return self.cur.copy(), self.cur_ev

    def rate(self) -> np.ndarray:
        """cumulative firing probability per unit per event"""
        return self.total / max(self.n_events, 1)
