"""Closed-form predictions used to *validate the benchmark itself*.

A benchmark that cannot be checked against theory can silently measure the wrong thing.
Each function here is an exact (or tight, clearly-stated approximate) analytic model of
something the framework measures empirically; the test-suite and the reports compare the two.

All models assume i.i.d. uniform random bipolar hypervectors, which is exactly how the
synthetic workloads generate their data.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .common import norm_cdf, norm_sf

# --------------------------------------------------------------------------------------
# Nearest-neighbour identification of a noisy key (Workload A)
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _binomial_half_pmf(dim: int) -> np.ndarray:
    """pmf of Binomial(dim, 1/2) as a float64 array (exact big-integer arithmetic)."""
    denom = 1 << dim
    return np.array([math.comb(dim, h) / denom for h in range(dim + 1)], dtype=np.float64)


def p_identify_bruteforce(dim: int, n_items: int, n_flips: int) -> float:
    """Exact P[argmin-Hamming recovers the stored key] for a query with ``n_flips`` flipped bits.

    The true key is at Hamming distance ``n_flips``; each of the ``n_items - 1`` independent
    random impostors is at Binomial(dim, 1/2) distance. Ties are broken uniformly at random
    (equivalent to ``argmin`` on a random key order). Closed form::

        P = ((b + c)^N - c^N) / (N b),   b = P[h = n],  c = P[h > n]

    which reduces to ``c^(N-1)`` when ties are impossible.
    """
    if n_items <= 1:
        return 1.0
    pmf = _binomial_half_pmf(dim)
    n = int(min(max(n_flips, 0), dim))
    b = float(pmf[n])
    c = float(pmf[n + 1 :].sum())
    big_n = n_items
    if b < 1e-12:
        return c ** (big_n - 1)
    a_free = b + c  # P[impostor not strictly closer]
    return (a_free**big_n - c**big_n) / (big_n * b)


def identification_curve(dim: int, n_items: int, noise_rates) -> list[float]:
    """:func:`p_identify_bruteforce` at each noise rate (flips = round(dim * rate))."""
    return [p_identify_bruteforce(dim, n_items, int(round(dim * r))) for r in noise_rates]


def noise_threshold_bruteforce(dim: int, n_items: int, target: float = 0.99) -> dict[str, float]:
    """Largest flip count / rate at which brute-force identification stays >= ``target``."""
    best = -1
    for n in range(0, dim // 2 + 1):
        if p_identify_bruteforce(dim, n_items, n) >= target:
            best = n
        else:
            break
    if best < 0:
        return {"flips": -1, "rate": 0.0}
    return {"flips": best, "rate": best / dim}


# --------------------------------------------------------------------------------------
# Linear (Hebbian) associative matrix memory (capacity experiment)
# --------------------------------------------------------------------------------------


def linear_memory_ber(dim: int, n_items: int) -> float:
    """Raw bit error rate of ``sign(W k_i)`` for ``W = V^T K / dim`` with N random pairs.

    Crosstalk on one output bit is ~ N(0, (N-1)/dim) against a signal of 1, so
    ``BER = Q(sqrt(dim / (N - 1)))`` (Gaussian approximation, accurate for dim >~ 100).
    """
    if n_items <= 1:
        return 0.0
    return float(norm_sf(math.sqrt(dim / (n_items - 1))))


def q_inverse(p: float) -> float:
    """Inverse of the Gaussian tail function ``Q`` by bisection (``0 < p < 0.5``)."""
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(norm_sf(mid)) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def linear_memory_capacity(dim: int, ber: float = 1e-3) -> int:
    """Largest N with ``linear_memory_ber(dim, N) <= ber``."""
    z = q_inverse(ber)
    return int(math.floor(1 + dim / (z * z)))


# --------------------------------------------------------------------------------------
# Superposition / holographic memories with cleanup (Workloads B, D and capacity)
# --------------------------------------------------------------------------------------


def holographic_p_correct(dim: int, n_terms: int, n_candidates: int, grid: int = 4001) -> float:
    """P[top-1 cleanup returns the right item] after unbinding one of ``n_terms`` superposed terms.

    After unbinding, the target has normalised similarity ``X ~ N(mu, sigma^2)`` with
    ``mu = sqrt(dim / n_terms)`` and ``sigma^2 = (n_terms - 1) / n_terms`` (units of
    ``1/sqrt(dim)``), each of the ``n_candidates - 1`` impostors is standard normal, so::

        P = E_X[ Phi(X)^(n_candidates - 1) ]

    evaluated by trapezoidal quadrature. Valid for MAP (bipolar) and unitary-HRR algebras.
    """
    if n_candidates <= 1:
        return 1.0
    mu = math.sqrt(dim / n_terms)
    sigma = math.sqrt(max(n_terms - 1, 0) / n_terms)
    if sigma < 1e-9:
        return 1.0
    x = np.linspace(mu - 9 * sigma, mu + 9 * sigma, grid)
    pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    cdf = norm_cdf(x)
    sf = norm_sf(x)
    log_phi = np.where(x > 0, np.log1p(-np.minimum(sf, 1 - 1e-300)), np.log(np.maximum(cdf, 1e-300)))
    integrand = pdf * np.exp((n_candidates - 1) * log_phi)
    dx = x[1] - x[0]
    return float(np.sum((integrand[1:] + integrand[:-1]) * 0.5) * dx)


def holographic_capacity(dim: int, n_candidates: int, target: float = 0.99, t_max: int = 1 << 20) -> int:
    """Largest number of superposed terms with top-1 cleanup accuracy >= ``target``."""
    if holographic_p_correct(dim, 1, n_candidates) < target:
        return 0
    lo, hi = 1, 2
    while hi < t_max and holographic_p_correct(dim, hi, n_candidates) >= target:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if holographic_p_correct(dim, mid, n_candidates) >= target:
            lo = mid
        else:
            hi = mid
    return lo
