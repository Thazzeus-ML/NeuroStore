"""Synthetic world with hidden structure and known ground truth.

C latent concepts, each a fixed hidden set of M member features among F raw features.
An event draws 1-3 concepts, emits each member with probability p, adds Poisson noise
features. The *primary* concept follows a Markov chain (stay / preferred successors /
popularity background); extra concepts are drawn from the popularity distribution.

Documented statistics (checked by tests and by bench/phase0_generator.py):
  * E[#concepts]            = sum_i i * n_concept_probs[i-1]
  * E[#features | 1 concept] = M*(1-(1-p)(1-q)) + (F-M)*q,  q = 1-exp(-noise_mean/F)
    (noise = Poisson(noise_mean) uniform draws with replacement, i.e. independent
     Poisson(noise_mean/F) per feature, so the formula is exact up to sampling error)
  * primary concept frequency -> stationary distribution of P (World.stationary())
  * P(primary_{t+1} == primary_t) -> sum_i pi_i P_ii
  * drift: exactly round(drift_frac*M) members of EVERY concept are replaced
Every random draw comes from a numpy Generator seeded by SeedSequence([seed, stream, idx]).
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np

_STREAM_IDS = {"world": 0, "train": 1, "val": 2, "test": 3, "extra": 4}


@dataclass(frozen=True)
class WorldConfig:
    F: int = 800                 # raw features
    C: int = 40                  # latent concepts
    M: int = 20                  # members per concept
    p: float = 0.7               # member emission probability
    noise_mean: float = 3.0      # Poisson mean of noise draws per event
    n_concept_probs: tuple = (0.6, 0.3, 0.1)   # P(1), P(2), P(3) concepts per event
    stay: float = 0.5            # chain: P(primary repeats)
    n_succ: int = 3              # chain: preferred successors per concept
    succ_mass: float = 0.3       # chain: mass on preferred successors; rest = popularity
    zipf_s: float = 0.0          # popularity skew exponent (0 = uniform)
    drift: bool = False          # membership changes midway through train streams
    drift_frac: float = 0.4      # fraction of members replaced at the drift point
    anomaly_rate: float = 0.0    # fraction of events that are pure-noise anomalies
    anomaly_size: int = 20       # features in an anomaly event
    dt_mean: float = 1.0         # mean inter-event time (exponential)

    def with_(self, **kw) -> "WorldConfig":
        return replace(self, **kw)


@dataclass
class Stream:
    t: np.ndarray            # float64 event times (strictly increasing)
    indptr: np.ndarray       # int64, len n+1
    feats: np.ndarray        # int32, concatenated sorted-unique feature ids per event
    concepts: list           # per event: sorted tuple of true concept ids (() for anomalies)
    primary: np.ndarray      # int16, chain state (-1 for anomalies)
    anomaly: np.ndarray      # bool
    drift_index: int = -1    # first event generated under post-drift membership (-1: none)

    def __len__(self) -> int:
        return len(self.t)

    def event(self, i: int) -> np.ndarray:
        return self.feats[self.indptr[i]:self.indptr[i + 1]]

    def slice(self, a: int, b: int) -> "Stream":
        o0, o1 = self.indptr[a], self.indptr[b]
        di = -1 if self.drift_index < 0 else max(self.drift_index - a, 0)   # 0: slice entirely post-drift
        if self.drift_index >= b:
            di = -1
        return Stream(self.t[a:b], self.indptr[a:b + 1] - o0, self.feats[o0:o1],
                      self.concepts[a:b], self.primary[a:b], self.anomaly[a:b], di)


class World:
    def __init__(self, cfg: WorldConfig, seed: int):
        assert cfg.M <= cfg.F and abs(sum(cfg.n_concept_probs) - 1) < 1e-9
        assert cfg.stay + cfg.succ_mass <= 1.0
        self.cfg, self.seed = cfg, seed
        rng = self._rng("world", 0)
        C, F, M = cfg.C, cfg.F, cfg.M
        # membership before / after drift
        m0 = np.sort(np.argsort(rng.random((C, F)), axis=1)[:, :M], axis=1)
        m1 = m0.copy()
        if cfg.drift:
            n_rep = int(round(cfg.drift_frac * M))
            for c in range(C):
                keep = rng.permutation(M)[n_rep:]
                pool = np.setdiff1d(np.arange(F), m0[c])
                new = rng.choice(pool, n_rep, replace=False)
                m1[c] = np.sort(np.concatenate([m0[c][keep], new]))
        self.members = (m0.astype(np.int32), m1.astype(np.int32))
        # popularity (Zipf over a random concept permutation) and chain
        ranks = rng.permutation(C) + 1
        pop = ranks.astype(float) ** (-cfg.zipf_s)
        self.pop = pop / pop.sum()
        P = np.zeros((C, C))
        for i in range(C):
            others = np.setdiff1d(np.arange(C), [i])
            succ = rng.choice(others, cfg.n_succ, replace=False)
            P[i, i] += cfg.stay
            P[i, succ] += cfg.succ_mass / cfg.n_succ
            P[i] += (1.0 - cfg.stay - cfg.succ_mass) * self.pop
        self.P = P
        self._cumP = np.cumsum(P, axis=1)
        self._cumP[:, -1] = 1.0

    def _rng(self, split: str, idx: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, _STREAM_IDS[split], idx]))

    # ---- analytic quantities (ground truth for the documented statistics) ----
    def stationary(self) -> np.ndarray:
        w, v = np.linalg.eig(self.P.T)
        pi = np.real(v[:, np.argmax(np.real(w))])
        return pi / pi.sum()

    def expected_self_transition(self) -> float:
        return float(np.sum(self.stationary() * np.diag(self.P)))

    def expected_mean_concepts(self) -> float:
        return float(sum((i + 1) * q for i, q in enumerate(self.cfg.n_concept_probs)))

    def expected_features_single(self) -> float:
        c = self.cfg
        q = 1.0 - np.exp(-c.noise_mean / c.F)
        return c.M * (1 - (1 - c.p) * (1 - q)) + (c.F - c.M) * q

    # ---- stream generation ----
    def stream(self, split: str, n: int, phase: int | None = None, idx: int = 0) -> Stream:
        """phase None: drift midway if cfg.drift (else phase 0); 0/1: fixed membership."""
        cfg = self.cfg
        rng = self._rng(split, idx)
        C, F, M = cfg.C, cfg.F, cfg.M
        drift_index = (n // 2) if (phase is None and cfg.drift) else -1
        nconc = rng.choice(len(cfg.n_concept_probs), size=n, p=cfg.n_concept_probs) + 1
        u_chain = rng.random(n)
        masks = rng.random((n, 3, M)) < cfg.p
        nn = rng.poisson(cfg.noise_mean, size=n)
        noise_flat = rng.integers(0, F, size=int(nn.sum()))
        noff = np.concatenate([[0], np.cumsum(nn)])
        anom = rng.random(n) < cfg.anomaly_rate
        dts = rng.exponential(cfg.dt_mean, size=n)
        extra_u = rng.random((n, 2))
        cur = int(rng.choice(C, p=self.pop))
        primary = np.empty(n, np.int16)
        concepts: list = []
        chunks: list = []
        indptr = np.zeros(n + 1, np.int64)
        cumpop = np.cumsum(self.pop)
        for i in range(n):
            if anom[i]:
                primary[i] = -1
                concepts.append(())
                f = np.sort(rng.choice(F, cfg.anomaly_size, replace=False)).astype(np.int32)
            else:
                cur = int(np.searchsorted(self._cumP[cur], u_chain[i], side="right")) if i else cur
                cur = min(cur, C - 1)
                primary[i] = cur
                chosen = [cur]
                for j in range(1, nconc[i]):
                    # inverse-CDF draw from popularity restricted to unchosen concepts
                    w = self.pop.copy(); w[chosen] = 0.0
                    cw = np.cumsum(w); r = extra_u[i, j - 1] * cw[-1]
                    chosen.append(int(min(np.searchsorted(cw, r, side="right"), C - 1)))
                concepts.append(tuple(sorted(chosen)))
                ph = (1 if (drift_index >= 0 and i >= drift_index) else 0) if phase is None else phase
                mem = self.members[ph]
                parts = [mem[c][masks[i, j]] for j, c in enumerate(chosen)]
                parts.append(noise_flat[noff[i]:noff[i + 1]].astype(np.int32))
                f = np.unique(np.concatenate(parts)).astype(np.int32)
            chunks.append(f)
            indptr[i + 1] = indptr[i] + len(f)
        feats = np.concatenate(chunks) if chunks else np.zeros(0, np.int32)
        return Stream(np.cumsum(dts), indptr, feats, concepts, primary, anom, drift_index)
