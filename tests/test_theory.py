"""The closed-form models used to validate the benchmark must themselves be right.

Each model is compared with a direct Monte-Carlo simulation that shares no code with it.
"""
import math

import numpy as np
import pytest

from neurostore import theory
from neurostore.common import flip_exact, make_rng, random_bipolar


def _mc_identification(dim, n_items, flips, trials, rng):
    ok = 0
    for _ in range(trials):
        keys = random_bipolar(rng, n_items, dim).astype(np.float32)
        i = int(rng.integers(n_items))
        q = flip_exact(rng, keys[i], flips).astype(np.float32)
        s = keys @ q
        best = np.flatnonzero(s == s.max())
        ok += int(rng.choice(best) == i)  # uniform tie-break, as in the closed form
    return ok / trials


@pytest.mark.slow
@pytest.mark.parametrize("dim,n_items", [(128, 30), (256, 100)])
def test_identification_formula_matches_monte_carlo(dim, n_items):
    rng = make_rng(2024)
    # pick flip counts spanning P ~ 0.97 ... 0.2
    cands = [f for f in range(0, dim // 2) if 0.15 < theory.p_identify_bruteforce(dim, n_items, f) < 0.99]
    flips = [cands[0], cands[len(cands) // 2], cands[-1]]
    for f in flips:
        th = theory.p_identify_bruteforce(dim, n_items, f)
        emp = _mc_identification(dim, n_items, f, 1500, rng)
        se = math.sqrt(th * (1 - th) / 1500)
        assert abs(emp - th) < 4 * se + 0.01, (dim, n_items, f, emp, th)


def test_identification_formula_edge_cases_and_monotonicity():
    assert theory.p_identify_bruteforce(512, 100, 0) == pytest.approx(1.0)
    assert theory.p_identify_bruteforce(512, 1, 200) == 1.0
    ps = [theory.p_identify_bruteforce(256, 50, f) for f in range(0, 129, 8)]
    assert all(a >= b - 1e-12 for a, b in zip(ps, ps[1:]))
    # With EXACTLY D/2 flips the query sits at the median distance from its own key, while the nearest of
    # N-1 random impostors sits ~2.5 sigma below the median: nearest-neighbour search almost never returns
    # the truth (~0.5^(N-1)), which is worse than the 1/N of an independent random query.
    assert theory.p_identify_bruteforce(512, 100, 256) < 1e-6
    assert theory.p_identify_bruteforce(512, 100, 256) < theory.p_identify_bruteforce(512, 100, 240)
    # more impostors can only hurt
    assert theory.p_identify_bruteforce(256, 1000, 80) < theory.p_identify_bruteforce(256, 100, 80)


def test_noise_threshold_known_value_and_scaling():
    t = theory.noise_threshold_bruteforce(512, 100, 0.99)
    assert t["flips"] == 214 and t["rate"] == pytest.approx(214 / 512)
    # higher dimension -> tolerates a larger fraction; more items -> less
    assert theory.noise_threshold_bruteforce(2048, 100)["rate"] > t["rate"]
    assert theory.noise_threshold_bruteforce(512, 10000)["rate"] < t["rate"]
    # the returned flip count is the *last* one passing
    assert theory.p_identify_bruteforce(512, 100, t["flips"]) >= 0.99 > theory.p_identify_bruteforce(512, 100, t["flips"] + 1)


def test_noise_threshold_approaches_one_half_as_dimension_grows():
    r = [theory.noise_threshold_bruteforce(d, 100)["rate"] for d in (256, 1024, 4096)]
    assert r[0] < r[1] < r[2] < 0.5 and r[2] > 0.46


@pytest.mark.slow
def test_linear_memory_ber_matches_monte_carlo():
    rng = make_rng(7)
    dim = 256
    for n in (30, 60, 120):
        errs = bits = 0
        for _ in range(12):
            k, v = random_bipolar(rng, n, dim).astype(np.float32), random_bipolar(rng, n, dim).astype(np.float32)
            out = np.sign(k @ (v.T @ k / dim).T)
            errs += int((out != v).sum())
            bits += v.size
        th = theory.linear_memory_ber(dim, n)
        se = math.sqrt(th * (1 - th) / bits)
        assert abs(errs / bits - th) < 5 * se + 2e-4, (n, errs / bits, th)


def test_linear_memory_capacity_is_consistent_with_ber():
    for dim in (256, 512, 4096):
        c = theory.linear_memory_capacity(dim, 1e-3)
        assert theory.linear_memory_ber(dim, c) <= 1e-3 < theory.linear_memory_ber(dim, c + 1)
    assert theory.linear_memory_capacity(4096) == pytest.approx(8 * theory.linear_memory_capacity(512), rel=0.02)  # linear in D
    assert theory.linear_memory_ber(512, 1) == 0.0


def test_q_inverse_roundtrip():
    for p in (0.4, 0.1, 1e-3, 1e-6):
        assert float(theory.norm_sf(theory.q_inverse(p))) == pytest.approx(p, rel=1e-6)


@pytest.mark.slow
def test_holographic_formula_matches_monte_carlo_for_a_bundle():
    rng = make_rng(99)
    dim, v = 512, 100
    book = random_bipolar(rng, v, dim).astype(np.float32)
    for n in (16, 24, 36):
        err = tot = 0
        while tot < 12000:
            roles = random_bipolar(rng, n, dim).astype(np.float32)
            idx = rng.integers(0, v, n)
            trace = (roles * book[idx]).sum(axis=0)
            err += int((((roles * trace) @ book.T).argmax(axis=1) != idx).sum())
            tot += n
        emp, th = err / tot, 1 - theory.holographic_p_correct(dim, n, v)
        se = math.sqrt(max(th, 1e-4) * (1 - th) / tot)
        assert abs(emp - th) < 4 * se + 2e-3, (n, emp, th)


def test_holographic_formula_properties():
    assert theory.holographic_p_correct(1024, 1, 100) == pytest.approx(1.0, abs=1e-6)
    assert theory.holographic_p_correct(1024, 50, 1) == 1.0
    ps = [theory.holographic_p_correct(1024, t, 100) for t in (10, 40, 80, 160, 320)]
    assert all(a > b for a, b in zip(ps, ps[1:]))
    # more candidates (larger codebook) and fewer dimensions both hurt
    assert theory.holographic_p_correct(1024, 80, 1000) < theory.holographic_p_correct(1024, 80, 100)
    assert theory.holographic_p_correct(512, 80, 100) < theory.holographic_p_correct(1024, 80, 100)


def test_holographic_capacity_is_the_exact_boundary_and_scales_with_dimension():
    for dim, e in ((512, 100), (2048, 100), (2048, 1000)):
        c = theory.holographic_capacity(dim, e, 0.99)
        assert theory.holographic_p_correct(dim, c, e) >= 0.99 > theory.holographic_p_correct(dim, c + 1, e)
    c1, c2 = theory.holographic_capacity(1024, 100), theory.holographic_capacity(4096, 100)
    assert c2 == pytest.approx(4 * c1, rel=0.1)  # capacity ~ D / (const depending on log E)
    assert theory.holographic_capacity(2048, 10000) < theory.holographic_capacity(2048, 100)
