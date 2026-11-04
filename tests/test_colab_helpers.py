"""Pure-NumPy helpers of the Colab benchmark notebook (colab/neurostore_colab_v2.py, cell `helpers-numpy`)."""
import math, sys
from pathlib import Path
import numpy as np
import pytest
from hypothesis import given, strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "colab"))
import nbtools  # noqa: E402

CELLS = nbtools.load_pct(Path(__file__).resolve().parent.parent / "colab" / "src")
H = nbtools.exec_cells(CELLS, ["helpers-numpy"])


def test_wilson_known_values():
    lo, hi = H["wilson_ci"](0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 1e-3
    lo, hi = H["wilson_ci"](50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    assert all(math.isnan(v) for v in H["wilson_ci"](0, 0))


@given(k=st.integers(0, 200), n=st.integers(1, 200))
def test_wilson_contains_point_estimate(k, n):
    k = min(k, n)
    lo, hi = H["wilson_ci"](k, n)
    eps = 1e-12
    assert 0.0 <= lo <= k / n + eps and k / n - eps <= hi <= 1.0


def test_mcnemar_exact():
    assert H["mcnemar_exact"](0, 0) == 1.0
    assert H["mcnemar_exact"](5, 5) == 1.0
    assert abs(H["mcnemar_exact"](3, 0) - 0.25) < 1e-12
    assert abs(H["mcnemar_exact"](10, 0) - 2 / 1024) < 1e-12
    assert H["mcnemar_exact"](2, 8) == pytest.approx(H["mcnemar_exact"](8, 2))


@given(pos=st.lists(st.integers(0, 6), min_size=1, max_size=25), neg=st.lists(st.integers(0, 6), min_size=1, max_size=25))
def test_auroc_matches_pairwise_definition(pos, neg):
    p, n = np.array(pos, float), np.array(neg, float)
    brute = np.mean(p[:, None] > n[None, :]) + 0.5 * np.mean(p[:, None] == n[None, :])
    assert H["auroc"](p, n) == pytest.approx(brute, abs=1e-12)


def test_rank_average():
    assert list(H["rank_average"]([10, 20, 20, 30])) == [1.0, 2.5, 2.5, 4.0]


def test_bootstrap_ci_brackets_mean():
    x = np.random.default_rng(0).normal(5, 1, 400)
    lo, hi = H["bootstrap_ci"](x)
    assert lo < x.mean() < hi and hi - lo < 0.3


def _simulate_flat(D, N, p, trials, seed):
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(trials // 500):
        mem = rng.integers(0, 2, size=(500, N, D), dtype=np.int8) * 2 - 1
        q = mem[:, 0, :].copy()
        flip = rng.random(q.shape) < p
        q[flip] *= -1
        s = np.einsum("bd,bnd->bn", q.astype(np.int32), mem.astype(np.int32)).astype(float)
        s += rng.random(s.shape) * 0.5          # uniform random tie-breaking (scores are integers of equal parity)
        ok += int((s.argmax(1) == 0).sum())
    n = (trials // 500) * 500
    return ok / n, n


@pytest.mark.parametrize("D,N,p", [(64, 32, 0.30), (64, 128, 0.40), (128, 16, 0.45), (32, 64, 0.20)])
def test_analytic_flat_recall_matches_simulation(D, N, p):
    sim, n = _simulate_flat(D, N, p, 40000, seed=D * 1000 + N)
    ana = H["analytic_flat_recall"](D, N, p)
    se = math.sqrt(max(ana * (1 - ana), 1e-6) / n)
    assert abs(sim - ana) < 4.5 * se + 1e-3, (sim, ana, se)


def test_analytic_flat_recall_limits_and_monotonicity():
    f = H["analytic_flat_recall"]
    assert f(512, 128, 0.0) > 0.999999
    assert f(512, 128, 0.5) == pytest.approx(1 / 128, rel=1e-6)
    vals = [f(512, 512, p) for p in (0.0, 0.2, 0.3, 0.4, 0.45, 0.5)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    assert f(512, 128, 0.4) > f(512, 2048, 0.4)           # more distractors, lower recall
    assert f(1024, 512, 0.4) > f(512, 512, 0.4)           # more bits, higher recall


def test_solve_noise_for_recall_inverts_analytic():
    for tgt in (0.5, 0.9, 0.97):
        p = H["solve_noise_for_recall"](512, 1024, tgt)
        assert abs(H["analytic_flat_recall"](512, 1024, p) - tgt) < 2e-3


def test_vsa_bundle_corr_matches_simulation():
    assert H["vsa_bundle_corr"](1) == 1.0 and H["vsa_bundle_corr"](3) == 0.5 and H["vsa_bundle_corr"](5) == 0.375
    rng = np.random.default_rng(1)
    for k in (3, 5, 9):
        x = rng.integers(0, 2, size=(200000, k), dtype=np.int8) * 2 - 1
        b = np.sign(x.sum(1))
        assert abs((b * x[:, 0]).mean() - H["vsa_bundle_corr"](k)) < 0.01
    with pytest.raises(ValueError):
        H["vsa_bundle_corr"](4)


def test_ideal_sparse_code_bytes_and_slope():
    assert H["ideal_sparse_code_bytes"](8, 1) == pytest.approx(3 / 8)
    assert H["loglog_slope"]([1, 2, 4, 8], [3, 6, 12, 24]) == pytest.approx(1.0)
