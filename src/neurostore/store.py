"""NeuroStore: orchestrates write path, read path and the background consolidator.

Write path: normalize -> sparse encode -> predict -> gate by surprise -> append -> update rates -> mark hot.
"""
from __future__ import annotations
import numpy as np
from .config import Config
from .encoder import SparseEncoder, normalize_features, q8
from .spikelog import SpikeLog
from .rates import RateCounters
from .assembly import AssemblyIndex
from .consolidate import Consolidator
from .synapse import SynapseLattice
from .homeostasis import Homeostasis
from .predict import Predictor, surprise


class NeuroStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enc = SparseEncoder(cfg.F, cfg.N, cfg.k, cfg.d, cfg.seed)
        self.log = SpikeLog(cfg.k, cfg.layout)
        self.rates = RateCounters(cfg.N, cfg.rate_window, cfg.rate_sub)
        self.asm = AssemblyIndex(cfg) if cfg.assemblies else None
        self.lattice = SynapseLattice(cfg.N, cfg.lam, cfg.prune_R, cfg.eps) if cfg.hebbian else None
        self.pred = Predictor(self)
        self.homeo = Homeostasis(self.enc, cfg.homeo_alpha, cfg.homeo_eta, cfg.theta_clip) if cfg.homeo else None
        self.prev_cos = None
        self.s_hist: list = []       # surprise of every written event (stored or not)
        self.cons = Consolidator(self)
        self.hot: list = []          # (priority, log position)
        self.eid = 0
        self.since_round = 0
        self.prev_units = None
        self.n_events = 0

    # ------------------------------------------------------------------ write path
    def write(self, t: float, feats) -> dict:
        cfg = self.cfg
        f = normalize_features(feats, cfg.F)                                  # normalize
        units, s_raw = self.enc.encode(f)                                     # sparse encode
        if self.homeo is not None:
            self.homeo.update(units)                                          # thresholds drift toward target rate
        exp_u, exp_s = self.pred.expected(self.prev_units, self.prev_cos)     # predict
        S, per = surprise(units, exp_u, exp_s, cfg.N)
        stored = (not cfg.gate) or (S > cfg.gate_tau)                         # gate by surprise
        scale = max(cfg.strength_floor, S) if cfg.gate else 1.0
        pos = -1
        if stored:                                                            # append
            pos = self.log.n_stored
            self.log.append(self.eid, t, units, q8(s_raw * scale), q8(per))
        else:
            self.log.count_only(t)
        self.rates.update(t, units)                                           # update rates
        cos = self.asm.vote(units) if (self.asm is not None and self.asm.n_alive) else None
        if stored:                                                            # mark hot
            hot, prio = S >= cfg.hot_tau, S
            if cos is not None:
                b = int(cos.argmax())
                if cos[b] >= 0.2:
                    self.asm.wmatch[b] += 1
                    if self.asm.wmatch[b] >= cfg.repeat_hot:
                        hot, prio = True, prio + 0.1
            if hot:
                self.hot.append((prio, pos))
        self.prev_units, self.prev_cos = units, cos
        self.s_hist.append(S)
        self.eid += 1; self.n_events += 1; self.since_round += 1
        if cfg.consolidate == "batch" and self.since_round >= cfg.consolidate_every:
            self.cons.run_round()
        return {"stored": stored, "surprise": S, "units": units, "pos": pos}

    def flush(self):
        if self.cfg.consolidate == "batch" and self.since_round > 0:
            self.cons.run_round()

    def ingest(self, stream, lo=0, hi=None):
        hi = len(stream) if hi is None else hi
        for i in range(lo, hi):
            self.write(float(stream.t[i]), stream.event(i))

    # ------------------------------------------------------------------ read path
    def encode_cue(self, feats):
        return self.enc.encode(normalize_features(feats, self.cfg.F))
