"""Statistics: intervals, paired tests, AUROC, engine-grouped bootstrap, and the analytic flat-search recall used to
validate the harness against theory. Mirrors the helper cell of the Colab notebook (a test keeps them in sync)."""
from __future__ import annotations
import math
import numpy as np

NAN = float("nan")


def wilson_ci(k, n, z=1.96):
    if n <= 0:
        return (NAN, NAN)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(b, c):
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0) for i in range(k + 1)]
    return float(min(1.0, 2.0 * sum(math.exp(v) for v in logs)))


def rank_average(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    _, inv, counts = np.unique(xs, return_inverse=True, return_counts=True)
    avg = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = np.empty(len(x))
    ranks[order] = avg[inv]
    return ranks


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie); higher score = more 'positive'."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return NAN
    r = rank_average(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _group_index(groups):
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(np.bincount(inv, minlength=len(uniq)))])
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]


def group_bootstrap(stat_fn, groups, n_boot=1000, seed=0, alpha=0.05):
    """Resample whole groups (e.g. engines) with replacement. stat_fn(idx) -> float for an index array into the
    original rows. Returns (point, lo, hi, boot_values). Rows inside a group are correlated, so resampling rows
    would understate the uncertainty."""
    gi = _group_index(groups)
    rng = np.random.default_rng(seed)
    point = float(stat_fn(np.arange(len(np.asarray(groups)))))
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(gi), len(gi))
        vals[b] = stat_fn(np.concatenate([gi[j] for j in pick]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi), vals


def paired_group_bootstrap(stat_fn_a, stat_fn_b, groups, n_boot=1000, seed=0, alpha=0.05):
    """CI of stat_a - stat_b using the SAME group resamples for both (paired)."""
    gi = _group_index(groups)
    rng = np.random.default_rng(seed)
    allidx = np.arange(len(np.asarray(groups)))
    point = float(stat_fn_a(allidx) - stat_fn_b(allidx))
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([gi[j] for j in rng.integers(0, len(gi), len(gi))])
        vals[b] = stat_fn_a(idx) - stat_fn_b(idx)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi), vals


def auroc_group_bootstrap(pos, pos_groups, neg, neg_groups, n_boot=300, seed=0, alpha=0.05):
    """AUROC with a two-sample group bootstrap (known and OOD groups resampled independently)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    gp, gn = _group_index(pos_groups), _group_index(neg_groups)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        ip = np.concatenate([gp[j] for j in rng.integers(0, len(gp), len(gp))])
        in_ = np.concatenate([gn[j] for j in rng.integers(0, len(gn), len(gn))])
        vals[b] = auroc(pos[ip], neg[in_])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return auroc(pos, neg), float(lo), float(hi)


# ---- analytic expectation for exact flat search over random +-1 patterns (validation of the harness) --------------
_LG = {}


def _lg(n):
    if n not in _LG:
        _LG[n] = np.array([math.lgamma(i + 1) for i in range(n + 1)])
    return _LG[n]


def _binom_pmf(n, p):
    k = np.arange(n + 1)
    if p <= 0.0:
        o = np.zeros(n + 1); o[0] = 1.0; return o
    if p >= 1.0:
        o = np.zeros(n + 1); o[n] = 1.0; return o
    lg = _lg(n)
    return np.exp(lg[n] - lg[k] - lg[n - k] + k * math.log(p) + (n - k) * math.log1p(-p))


def analytic_flat_recall(D, N, p_flip):
    """Exact P(top-1 correct): query = stored pattern with each of D bits flipped w.p. p_flip, N-1 other independent
    random +-1 patterns, ties broken uniformly at random."""
    D, N = int(D), int(N)
    pmf_t, pmf_o = _binom_pmf(D, 1.0 - float(p_flip)), _binom_pmf(D, 0.5)
    cdf_o = np.cumsum(pmf_o)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(cdf_o > 0, pmf_o / cdf_o, 0.0)
        log_all_le = (N - 1) * np.log(cdf_o)
        tie = np.where(q > 1e-300, -np.expm1(N * np.log1p(-q)) / (N * q), 1.0)
    return float(np.sum(pmf_t * np.where(cdf_o > 0, np.exp(log_all_le) * tie, 0.0)))
