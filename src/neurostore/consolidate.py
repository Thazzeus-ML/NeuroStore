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
        """Hebbian update w_ij += eta * a_i * a_j * K(dt) for the replayed events, written to the
        lattice's delta buffers. a = stored strength/255. Within-event pairs have dt = 0 (K = 1, both
        directions); pre-before-post pairs use the previous stored events inside lag_window with
        K(dt) = exp(-dt/tau_k). Pair sums over the batch are computed with matrix products."""
        s, cfg = self.s, self.s.cfg
        lat = s.lattice
        if lat is None or not len(pos):
            return
        N, B = cfg.N, len(pos)
        lat.tick()
        A = np.zeros((B, N), np.float32)
        A[np.arange(B)[:, None], units] = st.astype(np.float32) / 255.0
        inc = A.T @ A if cfg.within_event else np.zeros((N, N), np.float32)
        np.fill_diagonal(inc, 0.0)
        incdt = np.zeros((N, N), np.float32)
        times = s.log.times().astype(np.float64)
        for j in range(1, cfg.n_lags + 1):
            pp = pos - j
            ok = pp >= 0
            dt = np.where(ok, times[pos] - times[np.maximum(pp, 0)], np.inf)
            ok &= (dt > 0) & (dt <= cfg.lag_window)
            if not ok.any():
                continue
            pu, pst, _ = s.log.gather(pp[ok])
            P = np.zeros((int(ok.sum()), N), np.float32)
            P[np.arange(len(P))[:, None], pu] = pst.astype(np.float32) / 255.0
            K = np.exp(-dt[ok] / cfg.tau_k).astype(np.float32)
            Aok = A[ok]
            inc += (P * K[:, None]).T @ Aok
            incdt += (P * (K * dt[ok].astype(np.float32))[:, None]).T @ Aok
        inc *= cfg.eta; incdt *= cfg.eta
        pre, post = np.nonzero(inc > 0)
        lat.add_delta(pre, post, inc[pre, post], incdt[pre, post])
        if self.rounds % cfg.merge_every == cfg.merge_every - 1:
            lat.merge()
