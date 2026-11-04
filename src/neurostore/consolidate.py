"""Consolidator: replays hot events from the spike log into assemblies (and, from phase 3, the
synapse lattice). Runs in rounds; one round = one epoch."""
from __future__ import annotations
import numpy as np


class Consolidator:
    def __init__(self, store):
        self.s = store
        self.rounds = 0
        self.replayed = 0

    def select(self):
        s, cfg = self.s, self.s.cfg
        if not s.hot:
            return np.zeros(0, np.int64)
        pr = np.array([p for p, _ in s.hot]); pos = np.array([q for _, q in s.hot], np.int64)
        if len(pos) > cfg.replay_budget:
            # highest priority first; ties -> most recent
            order = np.lexsort((-pos, -pr))[:cfg.replay_budget]
            pos = pos[order]
        return np.sort(pos)

    def run_round(self, positions=None):
        s = self.s
        pos = self.select() if positions is None else np.asarray(positions, np.int64)
        if len(pos):
            units, st, su = s.log.gather(pos)
            if s.asm is not None:
                for u in units:
                    s.asm.learn(u)
            self._synapses(pos, units, st)
        if s.asm is not None:
            s.asm.maintain()
        s.hot.clear()
        s.since_round = 0
        self.rounds += 1; self.replayed += len(pos)

    def _synapses(self, pos, units, st):
        pass   # phase 3
