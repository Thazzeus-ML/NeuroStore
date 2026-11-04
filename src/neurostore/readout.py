"""Read-out decoders. The cue is encoded by the same encoder; population vote answers
'what is this', rate readout 'how much / how often', temporal readout 'what comes next',
surprise readout 'what is unusual'."""
from __future__ import annotations
import numpy as np


def select_assemblies(cos: np.ndarray, rho: float, cos_min: float, max_sel: int = 3) -> np.ndarray:
    best = float(cos.max()) if len(cos) else 0.0
    if best < cos_min:
        return np.zeros(0, np.int64)
    ids = np.flatnonzero(cos >= max(cos_min, rho * best))
    return ids[np.argsort(-cos[ids])][:max_sel]


def complete_pattern(asm, cos: np.ndarray, sel, cue_units, m: int) -> np.ndarray:
    """Pattern completion from the selected assemblies: ranked units (cue units excluded)."""
    if len(sel) == 0 or m <= 0:
        return np.zeros(0, np.int64)
    score = np.zeros(asm.N, np.float32)
    for a in sel:
        score[asm.member_ids[a]] += cos[a] * asm.member_w[a]
    score[np.asarray(cue_units, np.int64)] = 0.0
    nz = int((score > 0).sum())
    if nz == 0:
        return np.zeros(0, np.int64)
    m = min(m, nz)
    top = np.argpartition(-score, m - 1)[:m]
    return top[np.argsort(-score[top])]


def rate_readout(asm, counts: np.ndarray, n_events: int) -> np.ndarray:
    """How often does each assembly occur in a window? Least-squares fit of the window's unit
    counts to each assembly's firing profile: est_a = sum_u w_au * count_u / sum_u w_au^2."""
    W = asm.W
    den = (W ** 2).sum(1)
    return np.where(den > 0, (W @ counts.astype(np.float32)) / np.where(den > 0, den, 1.0), 0.0)


def temporal_readout(lattice, units, strength=None, hops=2, hop_decay=0.5, keep=96, m=32, exclude=None,
                     normalize=True, lagged_only=False):
    """What comes next: spread activation along synapses from the current active units."""
    sc = lattice.spread(units, strength, hops=hops, hop_decay=hop_decay, keep=keep, normalize=normalize,
                        lagged_only=lagged_only)
    if exclude is not None:
        sc[exclude] = 0.0
    nz = int((sc > 0).sum())
    if nz == 0:
        return np.zeros(0, np.int64), sc
    m = min(m, nz)
    top = np.argpartition(-sc, m - 1)[:m]
    return top[np.argsort(-sc[top])], sc


def surprise_readout(surprises: np.ndarray, top: int | None = None):
    """What is unusual: events ranked by stored surprise (descending)."""
    order = np.argsort(-np.asarray(surprises), kind="stable")
    return order if top is None else order[:top]
