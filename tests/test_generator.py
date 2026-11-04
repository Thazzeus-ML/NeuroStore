import numpy as np
from hypothesis import given, strategies as st
from sim.generator import World, WorldConfig
from eval.metrics import tv_distance, auc, set_f1, precision_at_m


def same(a, b):
    return (np.array_equal(a.t, b.t) and np.array_equal(a.feats, b.feats)
            and np.array_equal(a.indptr, b.indptr) and a.concepts == b.concepts
            and np.array_equal(a.primary, b.primary) and np.array_equal(a.anomaly, b.anomaly))


def test_deterministic_per_seed():
    for seed in range(3):
        a = World(WorldConfig(drift=True, zipf_s=1.0, anomaly_rate=0.05), seed).stream("train", 500)
        b = World(WorldConfig(drift=True, zipf_s=1.0, anomaly_rate=0.05), seed).stream("train", 500)
        assert same(a, b)
    a = World(WorldConfig(), 0).stream("train", 300)
    b = World(WorldConfig(), 1).stream("train", 300)
    assert not same(a, b)


def test_splits_are_independent_streams():
    w = World(WorldConfig(), 3)
    tr, te = w.stream("train", 400), w.stream("test", 400)
    assert not np.array_equal(tr.feats[:200], te.feats[:200])
    assert same(w.stream("test", 400), te)
    assert not same(w.stream("test", 400, idx=1), te)


@given(F=st.integers(60, 400), C=st.integers(4, 12), M=st.integers(3, 20),
       p=st.floats(0.2, 1.0), zipf=st.floats(0.0, 2.0), seed=st.integers(0, 50),
       drift=st.booleans(), anom=st.floats(0, 0.3))
def test_stream_invariants(F, C, M, p, zipf, seed, drift, anom):
    cfg = WorldConfig(F=F, C=C, M=min(M, F), p=p, zipf_s=zipf, drift=drift, anomaly_rate=anom,
                      n_succ=min(3, C - 1), anomaly_size=min(20, F))
    s = World(cfg, seed).stream("train", 120)
    assert len(s) == 120 and np.all(np.diff(s.t) > 0)
    for i in range(len(s)):
        f = s.event(i)
        assert np.all(np.diff(f) > 0) and (len(f) == 0 or (f[0] >= 0 and f[-1] < F))
        assert len(s.concepts[i]) in ((0,) if s.anomaly[i] else (1, 2, 3))
        assert all(0 <= c < C for c in s.concepts[i])
        if not s.anomaly[i]:
            assert s.primary[i] in s.concepts[i]


def test_drift_replaces_exact_count_and_test_phases_differ():
    cfg = WorldConfig(drift=True, drift_frac=0.4)
    w = World(cfg, 2)
    n_rep = round(0.4 * cfg.M)
    for c in range(cfg.C):
        assert len(np.setdiff1d(w.members[1][c], w.members[0][c])) == n_rep
        assert len(set(w.members[1][c])) == cfg.M
    s = w.stream("train", 200)
    assert s.drift_index == 100
    assert w.stream("test", 50, phase=0).drift_index == -1
    # no drift configured -> membership identical
    w2 = World(WorldConfig(drift=False), 2)
    assert np.array_equal(w2.members[0], w2.members[1])


def test_zipf_skews_stationary_distribution():
    flat = World(WorldConfig(zipf_s=0.0), 0).stationary()
    skew = World(WorldConfig(zipf_s=1.5), 0).stationary()
    assert skew.max() > 2 * flat.max() and abs(flat.sum() - 1) < 1e-9
    assert np.allclose(World(WorldConfig(), 0).P.sum(1), 1.0)


def test_documented_statistics_quick():
    w = World(WorldConfig(zipf_s=1.0), 0)
    s = w.stream("test", 20000)
    mc = np.mean([len(c) for c in s.concepts])
    assert abs(mc - w.expected_mean_concepts()) / w.expected_mean_concepts() < 0.03
    emp = np.bincount(s.primary, minlength=w.cfg.C).astype(float)
    assert tv_distance(emp, w.stationary()) < 0.05
    single = [len(s.event(i)) for i in range(len(s)) if len(s.concepts[i]) == 1]
    assert abs(np.mean(single) - w.expected_features_single()) / w.expected_features_single() < 0.04


def test_anomalies_have_no_concepts():
    s = World(WorldConfig(anomaly_rate=0.2), 1).stream("val", 500)
    assert 0.1 < s.anomaly.mean() < 0.3
    assert all(c == () for c, a in zip(s.concepts, s.anomaly) if a)


def test_slice_roundtrip():
    s = World(WorldConfig(drift=True), 0).stream("train", 100)
    sl = s.slice(20, 70)
    assert len(sl) == 50 and np.array_equal(sl.event(0), s.event(20)) and sl.drift_index == 30


def test_metrics():
    assert abs(auc([.1, .4, .35, .8], [0, 0, 1, 1]) - 0.75) < 1e-9
    assert set_f1([1, 2], [1, 2]) == 1 and set_f1([3], [1]) == 0
    assert precision_at_m([5, 6, 7], {5, 7}, 2) == 0.5
    assert abs(tv_distance([1, 1], [1, 3]) - 0.25) < 1e-9
