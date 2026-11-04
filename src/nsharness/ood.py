"""Out-of-distribution and chimera tests: can the memory tell that a query is not like anything it stores?

Familiarity scores (higher = more familiar) are compared between in-distribution (ID) queries and an OOD set with AUROC
(ID is the positive class), with a bootstrap that resamples whole groups (engines), plus the OOD detection rate at a 5 % ID false-alarm rate."""
from __future__ import annotations
import numpy as np
from .core import Split
from .stats import auroc, auroc_group_bootstrap, rank_average

SYNTHETIC_FAMILIES = ("gauss", "shuffle_time", "permute_channels")


def make_gauss(Z, rng):
    return rng.normal(0.0, 1.0, Z.shape).astype(np.float32)


def shuffle_time(Z, rng):
    out = Z.copy()
    for i in range(len(out)):
        out[i] = out[i][rng.permutation(out.shape[1])]
    return out


def permute_channels(Z, rng):
    out = Z.copy()
    for i in range(len(out)):
        out[i] = out[i][:, rng.permutation(out.shape[2])]
    return out


def synth_ood(Z, family: str, seed: int):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), sum(map(ord, family))]))
    return {"gauss": make_gauss, "shuffle_time": shuffle_time, "permute_channels": permute_channels}[family](Z, rng)


def chimera(ZA, ZB, rng, kind: str = "channel"):
    """Splice two real windows into one that is locally plausible but globally inconsistent.
    channel: a random half of the channels comes from A, the rest from B.  time: the first half of the window is A, the second half B."""
    n, T, C = ZA.shape
    if kind == "channel":
        take = np.zeros((n, C), bool)
        order = np.argsort(rng.random((n, C)), axis=1)[:, : C // 2]
        take[np.arange(n)[:, None], order] = True
        return np.where(take[:, None, :], ZA, ZB).astype(np.float32)
    if kind == "time":
        assert T >= 2
        cut = T // 2
        out = ZB.copy()
        out[:, :cut] = ZA[:, :cut]
        return out.astype(np.float32)
    raise ValueError(kind)


def target_gap_pairs(split: Split, rng, n: int, gap_lo: float, gap_hi: float, space: str = "percentile"):
    """Pairs (A, B) of items of `split` whose targets differ by a gap in [gap_lo, gap_hi]. Domain-neutral: for run-to-failure data the
    target is RUL, so the gap is the difference in life stage. space="percentile": gap between target percentiles (ties share one
    percentile, so a plateau of capped targets is ONE stage); space="value": gap in target units (e.g. cycles of RUL).
    B is drawn uniformly among all items inside the band, so tied plateaus do not collapse onto a single partner.
    Returns (iA, iB, realised_gap); fewer than n pairs are returned if the band is (nearly) empty."""
    y = np.asarray(split.y, float)
    u = (rank_average(y) - 0.5) / len(y) if space == "percentile" else y
    order = np.argsort(u, kind="stable")
    us = u[order]
    eps = 1e-12
    iA, iB, gap = [], [], []
    tries = 0
    while len(iA) < n and tries < 20 * n:
        tries += 1
        a = int(rng.integers(len(y)))
        l1, r1 = np.searchsorted(us, u[a] + gap_lo - eps, "left"), np.searchsorted(us, u[a] + gap_hi + eps, "right")
        l2, r2 = np.searchsorted(us, u[a] - gap_hi - eps, "left"), np.searchsorted(us, u[a] - gap_lo + eps, "right")
        s1, s2 = max(int(r1 - l1), 0), max(int(r2 - l2), 0)
        if s1 + s2 == 0:
            continue
        pos = rng.integers(l1, r1) if rng.random() < s1 / (s1 + s2) else rng.integers(l2, r2)
        b = int(order[pos])
        if b == a:
            continue
        iA.append(a); iB.append(b); gap.append(abs(u[a] - u[b]))
    return np.array(iA, int), np.array(iB, int), np.array(gap)


def make_chimeras(split: Split, rng, n: int, gap_lo: float, gap_hi: float, kind: str = "channel", space: str = "percentile") -> Split:
    iA, iB, gap = target_gap_pairs(split, rng, n, gap_lo, gap_hi, space)
    Z = chimera(split.Z[iA], split.Z[iB], rng, kind)
    return Split(Z, split.y[iA], split.groups[iA], {"gap": gap})


def familiarity_auroc(fam_id: dict, groups_id, fam_ood: dict, groups_ood, n_boot: int = 300, seed: int = 0) -> dict:
    """AUROC (ID positive) per familiarity score with a group bootstrap CI, and the share of OOD items flagged at a 5 % ID false-alarm rate."""
    out = {}
    for name, pos in fam_id.items():
        neg = fam_ood[name]
        a, lo, hi = auroc_group_bootstrap(pos, groups_id, neg, groups_ood, n_boot=n_boot, seed=seed)
        thr = np.percentile(pos, 5.0)
        out[name] = {"auroc": float(a), "lo": float(lo), "hi": float(hi), "ood_flagged_at_fpr5": float(np.mean(neg < thr)),
                     "n_id": int(len(pos)), "n_ood": int(len(neg))}
    return out
