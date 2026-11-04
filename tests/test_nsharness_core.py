"""nsharness primitives: utilities, statistics, metrics, featurizer, standardized corruption."""
import math
import sys
from pathlib import Path
import numpy as np
import pytest
from hypothesis import given, strategies as st

from nsharness import _util, stats, metrics, corruption
from nsharness.features import Featurizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "colab"))
import nbtools  # noqa: E402

COLAB = nbtools.exec_cells(nbtools.load_pct(Path(__file__).resolve().parent.parent / "colab" / "src"), ["helpers-numpy"])


# ---------------------------------------------------------------- utilities
@pytest.mark.parametrize("k", [1, 3, 7, 40])
def test_topk_desc_matches_full_sort(k):
    S = np.random.default_rng(0).normal(size=(6, 40)).astype(np.float32)
    idx, vals = _util.topk_desc(S, k)
    want = -np.sort(-S, axis=1)[:, :k]
    assert np.allclose(vals, want) and np.allclose(np.take_along_axis(S, idx, 1), want)
    assert all(len(set(r)) == k for r in idx.tolist())


def test_csr_gather_known_and_empty_rows():
    ptr = np.array([0, 2, 2, 5])
    assert list(_util.csr_gather(ptr, np.array([2, 0]))) == [2, 3, 4, 0, 1]
    assert len(_util.csr_gather(ptr, np.array([1]))) == 0


def test_popcount_matches_python():
    x = np.random.default_rng(1).integers(0, 2 ** 63, 50, dtype=np.uint64)
    assert list(_util.popcount64(x)) == [bin(int(v)).count("1") for v in x]


@pytest.mark.parametrize("spherical", [False, True])
def test_kmeans_recovers_separated_clusters(spherical):
    rng = np.random.default_rng(2)
    centers = np.array([[8, 0, 0], [0, 8, 0], [0, 0, 8]], np.float32)
    lab = rng.integers(0, 3, 300)
    X = centers[lab] + 0.2 * rng.normal(size=(300, 3)).astype(np.float32)
    if spherical:
        X = _util.l2_normalize(X)
    _, a = _util.kmeans(X, 3, np.random.default_rng(0), spherical=spherical, n_init=2)
    for c in range(3):                              # every true cluster maps to a single k-means cluster
        assert len(set(a[lab == c].tolist())) == 1


def test_seed_stream_is_reproducible_and_independent():
    a, b = _util.seed_stream(1, 2).random(4), _util.seed_stream(1, 2).random(4)
    assert np.array_equal(a, b) and not np.array_equal(a, _util.seed_stream(1, 3).random(4))


# ---------------------------------------------------------------- statistics
@given(pos=st.lists(st.integers(0, 6), min_size=1, max_size=20), neg=st.lists(st.integers(0, 6), min_size=1, max_size=20))
def test_auroc_matches_pairwise_definition(pos, neg):
    p, n = np.array(pos, float), np.array(neg, float)
    brute = np.mean(p[:, None] > n[None, :]) + 0.5 * np.mean(p[:, None] == n[None, :])
    assert stats.auroc(p, n) == pytest.approx(brute, abs=1e-12)


def test_group_bootstrap_resamples_groups_not_rows():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(20), 50)                              # 20 groups of 50 identical rows
    x = np.repeat(rng.normal(size=20), 50)
    pt, lo, hi, _ = stats.group_bootstrap(lambda i: x[i].mean(), g, n_boot=400, seed=1)
    pt_row, lo_row, hi_row, _ = stats.group_bootstrap(lambda i: x[i].mean(), np.arange(len(x)), n_boot=400, seed=1)
    assert pt == pytest.approx(x.mean()) and lo <= pt <= hi
    assert (hi - lo) > 3 * (hi_row - lo_row)                      # ignoring the grouping would badly understate uncertainty


def test_paired_bootstrap_is_zero_for_identical_statistics():
    g = np.repeat(np.arange(10), 5)
    x = np.random.default_rng(3).normal(size=50)
    pt, lo, hi, _ = stats.paired_group_bootstrap(lambda i: x[i].mean(), lambda i: x[i].mean(), g, n_boot=100)
    assert pt == 0.0 and lo == 0.0 and hi == 0.0


def test_auroc_group_bootstrap_brackets_point():
    rng = np.random.default_rng(4)
    pos, neg = rng.normal(1, 1, 200), rng.normal(0, 1, 200)
    a, lo, hi = stats.auroc_group_bootstrap(pos, np.repeat(np.arange(20), 10), neg, np.repeat(np.arange(20), 10), n_boot=200)
    assert lo < a < hi and 0.6 < a < 0.85


@pytest.mark.parametrize("D,N,p", [(64, 16, 0.2), (128, 512, 0.35), (512, 1024, 0.4), (256, 3, 0.45)])
def test_harness_analytic_recall_is_the_colab_helper(D, N, p):
    assert stats.analytic_flat_recall(D, N, p) == pytest.approx(COLAB["analytic_flat_recall"](D, N, p), abs=1e-12)


def test_analytic_recall_limits():
    assert stats.analytic_flat_recall(128, 100, 0.0) == pytest.approx(1.0)
    assert stats.analytic_flat_recall(128, 1, 0.3) == pytest.approx(1.0)
    assert stats.analytic_flat_recall(128, 50, 0.5) == pytest.approx(1 / 50, abs=1e-9)      # a query independent of its target is chance


# ---------------------------------------------------------------- metrics
def test_metrics_known_values():
    assert metrics.rmse([0, 0], [3, 4]) == pytest.approx(math.sqrt(12.5))
    assert metrics.mae([1, 2], [2, 4]) == pytest.approx(1.5)
    assert metrics.nasa_score([10], [20]) == pytest.approx(math.e - 1)             # late by 10: exp(10/10) - 1
    assert metrics.nasa_score([20], [10]) == pytest.approx(math.exp(10 / 13) - 1)  # early by 10 is penalised less
    assert metrics.nasa_score([10], [20]) > metrics.nasa_score([20], [10])
    assert metrics.neighbor_recall([[1, 2], [3, 4]], [[1, 5], [3, 4]]) == pytest.approx(0.75)
    t = np.zeros((2, 3)); r = np.ones((2, 3)); miss = np.array([[1, 0, 0], [0, 1, 1]], bool)
    assert metrics.completion_nmse(t, r, miss) == 1.0 and math.isnan(metrics.completion_nmse(t, r, np.zeros((2, 3), bool)))


# ---------------------------------------------------------------- featurizer
def test_stats_features_on_a_ramp():
    T, C = 10, 3
    Z = (np.arange(T, dtype=np.float32)[None, :, None] * np.array([1, 2, 0], np.float32)[None, None, :]).repeat(4, 0)
    f = Featurizer("stats", standardize=False).fit(Z)
    F = f.transform(Z)
    assert F.shape == (4, 12) and f.n_features == 12
    assert np.allclose(F[0, :3], [9, 18, 0])                         # last
    assert np.allclose(F[0, 3:6], [4.5, 9, 0])                       # mean
    assert np.allclose(F[0, 6:9], [1, 2, 0], atol=1e-5)              # least-squares slope per cycle
    assert np.allclose(F[0, 9:], Z[0].std(0), atol=1e-5)             # std


def test_featurizer_standardises_and_aligns_channels():
    Z = np.random.default_rng(0).normal(3, 2, size=(200, 8, 5)).astype(np.float32)
    f = Featurizer("stats").fit(Z)
    F = f.transform(Z)
    assert np.allclose(F.mean(0), 0, atol=1e-4) and np.allclose(F.std(0), 1, atol=1e-3)
    obs = np.ones((3, 5), bool); obs[:, 2] = False
    fm = f.feature_mask(obs)
    assert fm.shape == (3, 20) and list(np.where(~fm[0])[0]) == [2, 7, 12, 17]
    assert f.param_bytes() == f.mu.nbytes + f.sd.nbytes


def test_featurizer_flatten_pca_and_errors():
    Z = np.random.default_rng(1).normal(size=(60, 1, 12)).astype(np.float32)
    assert Featurizer("flatten", standardize=False).fit_transform(Z).shape == (60, 12)
    f = Featurizer("flatten", pca=4).fit(Z)
    assert f.transform(Z).shape == (60, 4) and f.feature_mask(np.ones((60, 12), bool)) is None
    with pytest.raises(ValueError):
        Featurizer("stats").fit(Z)                                    # a single time step has no slope


# ---------------------------------------------------------------- corruption
@pytest.fixture
def windows():
    return np.random.default_rng(5).normal(size=(40, 12, 8)).astype(np.float32)


@pytest.mark.parametrize("fam", corruption.FAMILIES)
def test_corruption_contract(windows, fam):
    Z0 = windows.copy()
    ident, obs0 = corruption.corrupt(windows, fam, 0.0, 1)
    assert np.array_equal(ident, windows) and ident is not windows and obs0.all()
    a, oa = corruption.corrupt(windows, fam, 0.5, 7)
    b, ob = corruption.corrupt(windows, fam, 0.5, 7)
    c, _ = corruption.corrupt(windows, fam, 0.5, 8)
    assert np.array_equal(a, b) and np.array_equal(oa, ob)
    if fam != "quantize":                                          # quantisation has no random component
        assert not np.array_equal(a, c)
    assert a.shape == windows.shape and a.dtype == np.float32 and np.isfinite(a).all()
    assert np.array_equal(windows, Z0), "input must never be modified"


def test_dropout_loses_exactly_ceil_sC_channels(windows):
    for s, want in [(0.1, 1), (0.25, 2), (0.5, 4), (1.0, 8)]:
        Zc, obs = corruption.corrupt(windows, "dropout", s, 3)
        assert (~obs).sum(1).tolist() == [want] * 40
        assert np.all(Zc[~obs[:, None, :].repeat(Zc.shape[1], 1)] == 0)          # lost channels are zero-filled (the z-scored mean)
    _, silent = corruption.corrupt(windows, "stuck", 0.5, 3)
    assert silent.all(), "stuck sensors are silent corruptions: they are not flagged as missing"


def test_stuck_flatlines_at_first_value(windows):
    Zc, _ = corruption.corrupt(windows, "stuck", 0.5, 11)
    stuck = np.ptp(Zc, axis=1) == 0                              # exactly flat over the window
    assert (stuck.sum(1) == 4).all()                             # ceil(0.5 * 8) channels per window
    i, c = np.argwhere(stuck)[0]
    assert np.allclose(Zc[i, :, c], windows[i, 0, c])


def test_noise_spikes_quantize_drift_have_documented_size():
    Z = np.zeros((2000, 20, 6), np.float32)
    n, _ = corruption.corrupt(Z, "noise", 0.4, 1)
    assert n.std() == pytest.approx(0.4, rel=0.03)
    sp, _ = corruption.corrupt(Z, "spikes", 0.5, 1)
    assert (sp != 0).mean() == pytest.approx(0.05, rel=0.1) and set(np.unique(np.abs(sp))) == {0.0, 5.0}
    Zr = np.random.default_rng(0).uniform(-3, 3, (500, 10, 4)).astype(np.float32)
    q, _ = corruption.corrupt(Zr, "quantize", 1.0, 1)
    assert len(np.unique(q)) <= 2 ** 2                              # severity 1 -> 2 bits
    d, _ = corruption.corrupt(Z, "drift", 1.0, 1)
    assert np.allclose(np.abs(d[:, -1, :]).max(1), 2.0) and (np.abs(d[:, 0, :]) == 0).all()
    assert ((np.abs(d[:, -1, :]) > 0).sum(1) == round(0.3 * 6)).all()


@pytest.mark.parametrize("fam", ["noise", "dropout", "drift", "spikes", "quantize", "combined"])
def test_damage_grows_with_severity(windows, fam):
    dev = [float(np.mean((corruption.corrupt(windows, fam, s, 2)[0] - windows) ** 2)) for s in (0.1, 0.4, 0.8)]
    assert dev[0] < dev[1] < dev[2]


def test_calibrate_severity_bisects_and_reports_unreachable():
    s, r, ok = corruption.calibrate_severity(lambda s: 1 + 4 * s, 3.0)
    assert ok and s == pytest.approx(0.5, abs=0.01) and r >= 3.0
    s, r, ok = corruption.calibrate_severity(lambda s: 1 + 0.2 * s, 3.0)
    assert not ok and s == 1.0 and r == pytest.approx(1.2)
    with pytest.raises(KeyError):
        corruption.corrupt(np.zeros((1, 2, 2), np.float32), "lightning", 0.5, 0)


def test_describe_severity_matches_what_corrupt_does(windows):
    d = corruption.describe_severity("dropout", 0.25, 8)
    assert d == {"channels_lost": 2, "of": 8} and (~corruption.corrupt(windows, "dropout", 0.25, 1)[1]).sum(1).tolist() == [2] * 40
    assert corruption.describe_severity("quantize", 1.0, 8) == {"adc_bits": 2} and corruption.describe_severity("quantize", 0.0, 8) == {"adc_bits": 8}
    assert corruption.describe_severity("combined", 0.5, 10)["channels_lost"] == 2 and corruption.describe_severity("drift", 0.5, 10)["channels_drifting"] == 3
    for fam in corruption.FAMILIES:
        assert isinstance(corruption.describe_severity(fam, 0.3, 14), dict)


def test_featurizer_clip_and_level_quantisation():
    Z = np.random.default_rng(3).normal(0, 2.5, size=(300, 8, 4)).astype(np.float32)
    plain = Featurizer("stats").fit(Z).transform(Z)
    assert np.abs(plain).max() > 3.5                                              # the raw z-scores exceed the bound
    clipped = Featurizer("stats", clip=3.0).fit(Z).transform(Z)
    assert np.abs(clipped).max() <= 3.0 + 1e-6 and np.allclose(clipped, np.clip(plain, -3, 3), atol=1e-6)
    q = Featurizer("stats", clip=3.0, levels=9).fit(Z)
    Fq = q.transform(Z)
    grid = np.linspace(-3, 3, 9)
    assert np.abs(Fq[..., None] - grid).min(-1).max() < 1e-5                      # every value sits on one of the 9 levels
    assert np.allclose(q.transform(Z), Fq) and q.n_features == 16
    with pytest.raises(AssertionError):
        Featurizer("stats", levels=8)                                             # a quantiser needs its span
