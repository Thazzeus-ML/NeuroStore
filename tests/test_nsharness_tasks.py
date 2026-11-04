"""Evaluation layer: metric bundles, paired comparisons, OOD/chimera construction, group-held-out tuning, benchmarks."""
import numpy as np
import pytest

from nsharness import tasks, ood, tune, bench
from nsharness.core import Split
from nsharness.pipeline import Pipeline


def run_to_failure_world(n_eng=24, seed=0, W=8):
    """Toy turbofan: 4 channels drift with age; target = capped RUL."""
    rng = np.random.default_rng(seed)
    Zs, ys, gs, ts = [], [], [], []
    for e in range(n_eng):
        L = int(rng.integers(50, 90)); t = np.arange(1, L + 1)
        base = rng.normal(0, 0.2, 4)
        sig = np.stack([base[c] + (t / L) ** 2 * (c + 1) * 0.8 + rng.normal(0, 0.1, L) for c in range(4)], 1)
        pad = np.concatenate([np.repeat(sig[:1], W - 1, 0), sig])
        Zs.append(np.stack([pad[i:i + W] for i in range(L)])); ys.append(np.minimum(L - t, 30)); gs.append(np.full(L, e)); ts.append(t)
    Z = np.concatenate(Zs).astype(np.float32)
    Z = (Z - Z.reshape(-1, 4).mean(0)) / Z.reshape(-1, 4).std(0)
    return Split(Z, np.concatenate(ys).astype(np.float32), np.concatenate(gs), {"t": np.concatenate(ts)})


# ---------------------------------------------------------------- metrics bundle
def test_metric_bundle_known_values():
    y = np.arange(40.0); g = np.arange(40)
    m = tasks.metric_bundle(y, y + 2.0, g, n_boot=100)
    assert m["rmse"] == pytest.approx(2.0) and m["rmse_lo"] <= 2.0 <= m["rmse_hi"] and m["mae"] == pytest.approx(2.0)
    two = tasks.metric_bundle(y, np.stack([y + 1, y + 3]), g, n_boot=50)                 # two seeds with errors 1 and 3
    assert two["rmse"] == pytest.approx(np.sqrt(5.0)) and two["rmse_per_seed"] == pytest.approx([1.0, 3.0])
    assert tasks.metric_bundle(y, y, g, n_boot=30)["rmse_hi"] == 0.0
    capped = tasks.metric_bundle(y, y + 2.0, g, n_boot=30, y_capped=y + 1.0)
    assert capped["rmse_capped"] == pytest.approx(1.0) and capped["nasa_sum"] == pytest.approx(40 * (np.exp(0.2) - 1))


def test_paired_rmse_diff():
    rng = np.random.default_rng(0); y = rng.normal(size=300); g = np.repeat(np.arange(30), 10)
    good, bad = y + 0.1 * rng.normal(size=300), y + 0.5 * rng.normal(size=300)
    d = tasks.paired_rmse_diff(y, good, bad, g, n_boot=200)
    assert d["diff"] < 0 and d["hi"] < 0 and d["excludes_zero"]
    same = tasks.paired_rmse_diff(y, good, good, g, n_boot=50)
    assert same["diff"] == 0 and not same["excludes_zero"]


# ---------------------------------------------------------------- chimeras / OOD
def test_channel_and_time_chimeras_are_exact_splices():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 10, 6)).astype(np.float32); B = rng.normal(size=(30, 10, 6)).astype(np.float32)
    ch = ood.chimera(A, B, rng, "channel")
    from_a = np.array([[np.array_equal(ch[i, :, c], A[i, :, c]) for c in range(6)] for i in range(30)])
    from_b = np.array([[np.array_equal(ch[i, :, c], B[i, :, c]) for c in range(6)] for i in range(30)])
    assert (from_a | from_b).all() and (from_a.sum(1) == 3).all()
    tm = ood.chimera(A, B, rng, "time")
    assert np.array_equal(tm[:, :5], A[:, :5]) and np.array_equal(tm[:, 5:], B[:, 5:])


def test_target_gap_pairs_respect_the_requested_gap():
    world = run_to_failure_world()
    rng = np.random.default_rng(1)
    for lo, hi in [(0.0, 0.05), (0.2, 0.4), (0.5, 0.9)]:
        iA, iB, gap = ood.target_gap_pairs(world, rng, 200, lo, hi)
        assert len(iA) == len(iB) == len(gap) > 50 and (iA != iB).all()
        assert gap.min() >= lo - 1e-9 and gap.max() <= hi + 1e-9
    iA, iB, gap = ood.target_gap_pairs(world, rng, 200, 10, 25, space="value")                  # gap in target units (cycles of RUL)
    assert len(iA) > 50 and np.allclose(gap, np.abs(world.y[iA] - world.y[iB])) and gap.min() >= 10 and gap.max() <= 25
    same = ood.target_gap_pairs(world, rng, 300, 0, 0, space="value")                           # same stage: the capped plateau is one stage
    assert len(set(same[1].tolist())) > 20, "partners must not collapse onto one tied item"
    ch = ood.make_chimeras(world, rng, 50, 0.2, 0.9, "channel")
    assert 0 < len(ch) <= 50 and ch.Z.shape[1:] == world.Z.shape[1:] and (ch.meta["gap"] >= 0.2 - 1e-9).all()


def test_synthetic_ood_families_keep_shape_and_change_structure():
    Z = run_to_failure_world().Z[:20]
    for fam in ood.SYNTHETIC_FAMILIES:
        O = ood.synth_ood(Z, fam, 3)
        assert O.shape == Z.shape and not np.array_equal(O, Z)
        assert np.array_equal(O, ood.synth_ood(Z, fam, 3))
    assert np.allclose(np.sort(ood.synth_ood(Z, "shuffle_time", 3), axis=1), np.sort(Z, axis=1))
    assert np.allclose(np.sort(ood.synth_ood(Z, "permute_channels", 3), axis=2), np.sort(Z, axis=2))


def test_familiarity_auroc_detects_ood_and_is_chance_for_same_distribution():
    world = run_to_failure_world(n_eng=30)
    plan = tune.cv_plan(world, 3, 20, 0)
    tr, qi = plan[0]
    pipe = Pipeline("exact", {}, {"kind": "stats"}, k=5).fit(world.take(tr), 0)
    ids = world.take(qi)
    gauss = Split(ood.synth_ood(ids.Z, "gauss", 1), ids.y, ids.groups)
    res = tasks.ood_eval(pipe, ids, {"gauss": gauss}, n_boot=60)["gauss"]
    assert res["max_sim"]["auroc"] > 0.9 and res["max_sim"]["lo"] > 0.8
    # same-distribution control: two disjoint halves of the held-out engines (neither is in the memory) must be indistinguishable
    eng = np.unique(ids.groups)
    half_a, half_b = ids.take(np.where(np.isin(ids.groups, eng[::2]))[0]), ids.take(np.where(np.isin(ids.groups, eng[1::2]))[0])
    res0 = tasks.ood_eval(pipe, half_a, {"same": half_b}, n_boot=200)["same"]["max_sim"]
    assert res0["lo"] < 0.5 < res0["hi"] and 0.3 < res0["auroc"] < 0.7
    # and a memory item is MORE familiar than an unseen one (AUROC of unseen-as-positive against stored items is ~0)
    stored = world.take(tr[:300])
    assert tasks.ood_eval(pipe, ids, {"stored": stored}, n_boot=50)["stored"]["max_sim"]["auroc"] < 0.1


# ---------------------------------------------------------------- tuning / CV
def test_cv_plan_has_no_group_leakage_and_covers_every_group_once():
    world = run_to_failure_world()
    plan = tune.cv_plan(world, 4, 5, 0, ("t", 8))
    seen = []
    for tr, qi in plan:
        gtr, gq = set(world.groups[tr].tolist()), set(world.groups[qi].tolist())
        assert not (gtr & gq) and (world.meta["t"][qi] >= 8).all()
        seen += sorted(gq)
    assert sorted(seen) == sorted(set(world.groups.tolist()))


def test_tune_prefers_the_better_setting_and_reports_sorted_rows():
    world = run_to_failure_world(n_eng=30)
    plan = tune.cv_plan(world, 3, 12, 0, ("t", 8))
    rows = tune.tune("exact", [{}], [(1, None), (10, 30.0), (len(world) // 3, None)], world, plan, {"kind": "stats"})
    assert [r["cv_rmse"] for r in rows] == sorted(r["cv_rmse"] for r in rows)
    assert rows[0]["k"] == 10 and rows[-1]["k"] > 100                                   # averaging everything is the worst readout
    grid = tune.expand_grid(m=[256, 512], frac=[0.05, 0.1])
    assert len(grid) == 4 and {"m": 512, "frac": 0.1} in grid
    hop = tune.tune("hopfield", [{"beta": 5.0}], [(1, 5.0), (1, 60.0)], world, plan, {"kind": "stats"})
    assert hop[0]["beta"] == 60.0                                                       # sharper attention wins on this task


# ---------------------------------------------------------------- tasks on pipelines
def test_completion_eval_beats_mean_imputation_and_flags_nothing_when_nothing_dropped():
    world = run_to_failure_world(n_eng=30)
    plan = tune.cv_plan(world, 3, 20, 0)
    tr, qi = plan[0]
    pipe = Pipeline("exact", {"beta": 20.0}, {"kind": "stats"}, k=5, use_mask=True).fit(world.take(tr), 0)
    r = tasks.completion_eval(pipe, world.Z[qi], 0.3, 1, give_mask=True)
    assert r["nmse"] < 0.8 and r["missing_frac"] == pytest.approx(0.5)            # ceil(0.3 * 4 channels) = 2 channels lost
    assert np.isnan(tasks.completion_eval(pipe, world.Z[qi], 0.0, 1, give_mask=True)["nmse"])
    with pytest.raises(ValueError):
        tasks.completion_eval(Pipeline("exact", {}, {"kind": "stats", "pca": 3}).fit(world.take(tr), 0), world.Z[qi], 0.3, 1, False)


def test_recall_vs_exact_and_corruption_prediction_paths():
    world = run_to_failure_world()
    plan = tune.cv_plan(world, 3, 10, 0)
    tr, qi = plan[0]
    ex = Pipeline("exact", {}, {"kind": "stats"}, k=5).fit(world.take(tr), 0)
    ann = Pipeline("ann", {"nprobe": 1000}, {"kind": "stats"}, k=5).fit(world.take(tr), 0)
    assert tasks.recall_vs_exact(ex, ex, world.Z[qi]) == 1.0 and tasks.recall_vs_exact(ann, ex, world.Z[qi]) > 0.99
    clean = tasks.predict_under_corruption(ex, world.Z[qi], "noise", 0.0, 0)
    assert np.array_equal(clean, ex.predict(world.Z[qi]))
    noisy = tasks.predict_under_corruption(ex, world.Z[qi], "noise", 0.8, 0)
    err = lambda p: np.sqrt(np.mean((p - world.y[qi]) ** 2))
    assert err(noisy) > err(clean)


# ---------------------------------------------------------------- bench
def test_latency_percentiles_are_gated_by_sample_count():
    Q = np.random.default_rng(0).normal(size=(50, 4)).astype(np.float32)
    f = lambda q: q @ q.T
    a = bench.latency_profile(f, Q, n_samples=30, warmup=2)
    assert a["p50_ms"] is not None and a["p95_ms"] is None and a["p99_ms"] is None
    b = bench.latency_profile(f, Q, n_samples=150, warmup=2)
    assert b["p95_ms"] is not None and b["p99_ms"] is None
    c = bench.latency_profile(f, Q, n_samples=1000, warmup=2)
    assert c["p99_ms"] is not None and c["min_ms"] <= c["p50_ms"] <= c["p95_ms"] <= c["p99_ms"]
    assert bench.latency_profile(f, Q, n_samples=10, warmup=0)["p50_ms"] is None


def test_throughput_build_profile_and_environment():
    Q = np.random.default_rng(0).normal(size=(64, 8)).astype(np.float32)
    t = bench.throughput(lambda q: q @ q.T, Q, batch=16, repeats=2, min_seconds=0.02)
    assert t["qps_median"] > 0 and len(t["qps_all"]) == 2 and t["batch"] == 16
    out, prof = bench.build_profile(lambda: np.ones((1000, 1000), np.float32))
    assert out.shape == (1000, 1000) and prof["tracemalloc_peak_bytes"] >= 4_000_000 and prof["seconds"] >= 0
    assert bench.search_peak_bytes(lambda q: q @ q.T, np.ones((200, 64), np.float32)) >= 200 * 200 * 4
    env = bench.environment()
    assert env["numpy"] == np.__version__ and env["cpus"] >= 1


def test_memory_table_ratios():
    world = run_to_failure_world()
    pipe = Pipeline("exact", {}, {"kind": "stats"}).fit(world, 0)
    n, raw_w = len(world), world.Z.nbytes
    raw_f = n * pipe.feat.n_features * 4
    tab = bench.memory_table(pipe, n, raw_w, raw_f)
    assert tab["deployed_bytes"] == pipe.deployed_bytes() and tab["ratio_vs_raw_features"] > 1.0     # exact stores features + norms + labels
    assert tab["bytes_per_item"] == pytest.approx(tab["deployed_bytes"] / n) and tab["ratio_vs_raw_windows"] < tab["ratio_vs_raw_features"]
