"""Domain-neutral data containers and the Backend interface."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np

from ._util import topk_desc, softmax_rows


@dataclass
class Split:
    """A set of items. Z: (n, T, C) float32 windows (T = 1 for non-temporal data); y: (n,) targets; groups: (n,) group ids
    (e.g. engine id) used for grouped bootstrap and grouped splits; meta: aligned per-item arrays (stage, regime, ...)."""
    Z: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.Z = np.asarray(self.Z, np.float32)
        assert self.Z.ndim == 3, "Z must be (n, T, C)"
        n = len(self.Z)
        self.y, self.groups = np.asarray(self.y), np.asarray(self.groups)
        assert len(self.y) == n and len(self.groups) == n
        for k, v in self.meta.items():
            assert len(v) == n, f"meta[{k}] has wrong length"

    def __len__(self):
        return len(self.Z)

    @property
    def T(self):
        return self.Z.shape[1]

    @property
    def C(self):
        return self.Z.shape[2]

    def take(self, idx) -> "Split":
        idx = np.asarray(idx)
        return Split(self.Z[idx], self.y[idx], self.groups[idx], {k: np.asarray(v)[idx] for k, v in self.meta.items()})


@dataclass
class DomainData:
    name: str
    target_kind: str                       # "regression" | "classification"
    memory: Split                          # what gets stored
    queries: dict                          # name -> Split (in-distribution evaluation sets)
    ood: dict = field(default_factory=dict)        # name -> Split (out-of-distribution families supplied by the domain)
    pair_sampler: Callable | None = None   # (rng, n, gap_lo, gap_hi) -> (ZA, ZB, gap): source windows for chimeras
    info: dict = field(default_factory=dict)


class Backend:
    """Common interface. All similarities are 'higher = more similar'. Subclasses implement fit/search and may override the rest."""
    name = "base"
    supports_mask = False
    supports_reconstruct = True

    def __init__(self, **cfg):
        self.cfg = dict(cfg)
        self.n = 0
        self.d = 0

    # ---- required
    def fit(self, X: np.ndarray, seed: int = 0) -> "Backend":
        raise NotImplementedError

    def search(self, Q: np.ndarray, k: int, mask=None):
        """-> (idx (q,k) int64, sim (q,k) float32)"""
        raise NotImplementedError

    def memory_report(self) -> dict:
        """Persistent bytes of the arrays a deployment must keep, by component."""
        raise NotImplementedError

    # ---- defaults built on search()
    def weights(self, sim: np.ndarray, beta=None) -> np.ndarray:
        """Retrieval weights of the k returned items: uniform, or softmax(beta * sim) when beta (default cfg['beta']) is set.
        Items with sim == -inf (padding) get weight 0."""
        sim = np.asarray(sim, np.float64)
        valid = np.isfinite(sim)
        beta = self.cfg.get("beta") if beta is None else beta
        if not beta:
            return valid.astype(np.float64)
        top = np.where(valid, sim, -np.inf).max(axis=1, keepdims=True)
        return np.where(valid, np.exp(beta * (np.where(valid, sim, 0.0) - top)), 0.0)

    def readout(self, Q, Y, k, mask=None):
        """Weighted aggregate of the stored targets Y (n,) or (n, m) of the k retrieved items."""
        idx, sim = self.search(Q, k, mask)
        return self.readout_from(idx, sim, Y, k)

    def readout_from(self, idx, sim, Y, k, beta=None):
        """Same aggregate from an already-computed search result (lets a tuner reuse one search for many (k, beta))."""
        idx, sim = idx[:, :k], sim[:, :k]
        w = self.weights(sim, beta)
        Yk = np.asarray(Y, np.float64)[np.where(idx >= 0, idx, 0)]
        tot = np.maximum(w.sum(1), 1e-300)
        if Yk.ndim == 3:
            return (w[..., None] * Yk).sum(1) / tot[:, None]
        return (w * Yk).sum(1) / tot

    def familiarity(self, Q, mask=None) -> dict:
        """Scores where HIGHER = MORE FAMILIAR (looks like something stored)."""
        _, sim = self.search(Q, 10, mask)
        return {"max_sim": sim[:, 0].astype(np.float64), "margin": (sim[:, 0] - sim[:, 1]).astype(np.float64),
                "mean_top10": sim.mean(1).astype(np.float64)}

    def reconstruct(self, Q, k, mask=None):
        raise NotImplementedError(f"{self.name} has no feature-space reconstruction")

    # ---- accounting
    def index_bytes(self) -> int:
        """Deployed bytes = components WITHOUT a prefix. 'alt:' keys are alternative encodings of the same content and
        'cache:' keys are implementation conveniences that a deployment would not keep; both are excluded from the total."""
        return int(sum(v for k, v in self.memory_report().items() if not k.startswith(("alt:", "cache:"))))

    def params(self) -> dict:
        return {"backend": self.name, **self.cfg}


def sim_topk_from_dense(S: np.ndarray, k: int):
    idx, vals = topk_desc(S, k)
    return idx, vals.astype(np.float32)
