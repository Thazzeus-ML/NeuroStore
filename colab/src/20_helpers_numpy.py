# %% [markdown] id=helpers-md
# ## A. NumPy-only helpers: statistics and analytic references
#
# Everything in this cell is pure NumPy/`math` so it can be unit-tested without JAX (see `tests/test_colab_helpers.py`).
#
# * **Wilson / bootstrap intervals** and an **exact McNemar test** replace bare point estimates. A recall measured on
#   128 queries has a 95% interval of roughly ±5 percentage points, so differences of 1–3 points between two methods
#   are noise unless a paired test says otherwise.
# * **`analytic_flat_recall(D, N, p)`** is the exact probability that exact nearest-neighbour search finds the target
#   when the query is the target with each bit flipped with probability `p`, among `N` random ±1 patterns of
#   dimension `D`. The benchmark's measured recall must agree with it; this validates data generation, noise
#   injection and the retrieval kernels in a single check.
# * The same formula covers VSA bundle decoding (a majority bundle of `k` terms behaves like a query with
#   effective flip probability `(1 - c_k) / 2`).

# %% id=helpers-numpy
import math
import numpy as np

NAN = float("nan")


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k/n."""
    if n <= 0:
        return (NAN, NAN)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts (b: A right/B wrong, c: A wrong/B right)."""
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0) for i in range(k + 1)]
    return float(min(1.0, 2.0 * sum(math.exp(v) for v in logs)))


def paired_diff_ci(a_correct, b_correct, z=1.96):
    """Mean of (a - b) over paired 0/1 outcomes with a normal-approximation interval."""
    d = np.asarray(a_correct, float) - np.asarray(b_correct, float)
    n = len(d)
    if n < 2:
        return (NAN, NAN, NAN)
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    return (m, m - z * se, m + z * se)


def bootstrap_ci(x, stat=np.mean, n_boot=1000, alpha=0.05, seed=0):
    """Percentile bootstrap interval of `stat` over the rows of x."""
    x = np.asarray(x)
    if len(x) == 0:
        return (NAN, NAN)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    if stat is np.mean:
        s = x[idx].mean(axis=1)
    else:
        s = np.array([stat(x[i]) for i in idx])
    lo, hi = np.quantile(s, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def rank_average(x):
    """1-based ranks with ties replaced by their average rank."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    _, inv, counts = np.unique(xs, return_inverse=True, return_counts=True)
    avg = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = np.empty(len(x))
    ranks[order] = avg[inv]
    return ranks


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie), via rank sums (O(n log n), tie aware)."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return NAN
    r = rank_average(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def tpr_at_threshold(pos, thr):
    return float(np.mean(np.asarray(pos) >= thr))


_LG_CACHE = {}


def _lgamma_table(n):
    if n not in _LG_CACHE:
        _LG_CACHE[n] = np.array([math.lgamma(i + 1) for i in range(n + 1)])
    return _LG_CACHE[n]


def _binom_pmf(n, p):
    k = np.arange(n + 1)
    if p <= 0.0:
        out = np.zeros(n + 1); out[0] = 1.0; return out
    if p >= 1.0:
        out = np.zeros(n + 1); out[n] = 1.0; return out
    lg = _lgamma_table(n)
    return np.exp(lg[n] - lg[k] - lg[n - k] + k * math.log(p) + (n - k) * math.log1p(-p))


def analytic_flat_recall(D, N, p_flip):
    """Exact P(top-1 correct) of flat dot-product search.

    Target query = stored pattern with each of D bits flipped independently with probability p_flip; the other N-1
    stored patterns are independent random +-1 vectors, so their agreement count with the query is Bin(D, 1/2).
    Ties are broken uniformly at random, which equals the expectation of lowest-index tie-breaking when the target
    index is uniform. Uses E[1/(1+Bin(n,q))] = (1-(1-q)^(n+1)) / ((n+1) q).
    """
    D, N = int(D), int(N)
    pmf_t = _binom_pmf(D, 1.0 - float(p_flip))
    pmf_o = _binom_pmf(D, 0.5)
    cdf_o = np.cumsum(pmf_o)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(cdf_o > 0, pmf_o / cdf_o, 0.0)
        log_all_le = (N - 1) * np.log(cdf_o)
        tie = np.where(q > 1e-300, -np.expm1(N * np.log1p(-q)) / (N * q), 1.0)
    pc = np.where(cdf_o > 0, np.exp(log_all_le) * tie, 0.0)
    return float(np.sum(pmf_t * pc))


def solve_noise_for_recall(D, N, target, tol=1e-4):
    """Flip probability in [0, 0.5] at which analytic flat recall equals `target` (recall decreases with noise)."""
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if analytic_flat_recall(D, N, mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def vsa_bundle_corr(k):
    """Correlation between a +-1 majority bundle of k (odd) random terms and any one of its terms.

    P(bundle bit == term bit) = 1/2 + 1/2 P(sum of the other k-1 terms is zero), so the correlation equals
    P(sum of k-1 coin flips is zero) = C(k-1, (k-1)/2) / 2^(k-1).
    """
    if k % 2 != 1:
        raise ValueError("k must be odd")
    if k <= 1000:                                   # exact integers / float powers stay in range
        return math.comb(k - 1, (k - 1) // 2) / 2.0 ** (k - 1)
    return math.exp(math.lgamma(k) - 2.0 * math.lgamma((k + 1) / 2.0) - (k - 1) * math.log(2.0))


def analytic_vsa_decode(D, n_fillers, k):
    """Expected accuracy of decoding one filler from a k-term bundle by nearest-codebook cleanup."""
    return analytic_flat_recall(D, n_fillers, (1.0 - vsa_bundle_corr(k)) / 2.0)


def ideal_sparse_code_bytes(m, k):
    """Information-theoretic size of a k-of-m binary code: log2 C(m, k) bits."""
    return (math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1)) / math.log(2.0) / 8.0


def loglog_slope(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    if len(x) < 2:
        return NAN
    return float(np.polyfit(x, y, 1)[0])
