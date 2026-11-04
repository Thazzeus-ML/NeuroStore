"""End-to-end workload behaviour on the quick profile: schema, determinism and the measured physics."""
import json

import numpy as np
import pytest

from conftest import requires_faiss, requires_redis, requires_rocksdict
from neurostore.profiles import PROFILES, get_profile
from neurostore.workloads import WORKLOADS, a_telemetry, b_entity, c_prompt_cache, capacity, cim_emulation, d_graph_walk

KEYS = ["a", "b", "c", "d", "capacity", "cim"]


def rows(res, table, **where):
    return [r for r in res.tables[table] if all(r.get(k) == v for k, v in where.items())]


# ----------------------------------------------------------------------------- profiles
def test_profiles_define_the_same_knobs_for_every_workload():
    assert set(PROFILES) == {"quick", "standard", "full"}
    for wl in KEYS:
        assert set(PROFILES["quick"][wl]) == set(PROFILES["standard"][wl]) == set(PROFILES["full"][wl]), wl


def test_get_profile_returns_independent_copies_and_applies_overrides():
    a = get_profile("quick")
    a["a"]["trials"] = 1
    assert get_profile("quick")["a"]["trials"] != 1
    assert get_profile("quick", {"a": {"trials": 7}})["a"]["trials"] == 7
    with pytest.raises(KeyError):
        get_profile("nope")
    with pytest.raises(KeyError):
        get_profile("quick", {"zzz": {}})


# ----------------------------------------------------------------------------- schema (all workloads)
@pytest.mark.parametrize("key", KEYS)
def test_each_workload_returns_a_complete_json_serialisable_result(quick_suite, key):
    res = quick_suite.get(key)
    assert res is not None and res.workload == key and res.title == WORKLOADS[key].TITLE
    assert res.tables and all(isinstance(v, list) and v for v in res.tables.values()), key
    assert res.elapsed_s > 0 and res.profile == "quick" and res.seed == 1
    assert not any("CRASHED" in n for n in res.notes)
    json.dumps(res.to_dict(), allow_nan=False)  # strict JSON: no NaN/inf anywhere
    for c in res.checks:
        assert set(c) >= {"name", "target", "measured", "passed", "kind"} and c["kind"] in ("spec", "selfcheck", "info")
    for h in res.headlines:
        assert set(h) >= {"metric", "unit", "higher_is_better", "context", "systems"} and h["systems"]
        assert all(set(s) == {"value", "family"} for s in h["systems"].values())
    assert res.skipped == []  # include_external=False in the fixture, and every internal system must run


def test_benchmark_self_checks_pass_on_the_quick_profile(quick_suite):
    failed = [(w.workload, c["name"], c["measured"]) for w in quick_suite.workloads for c in w.checks if c["kind"] == "selfcheck" and c["passed"] is False]
    assert not failed, failed
    # every workload that has a closed-form or equivalence oracle must actually carry a self-check
    n = {w.workload: sum(c["kind"] == "selfcheck" for c in w.checks) for w in quick_suite.workloads}
    assert all(n[k] >= 1 for k in ("a", "c", "d", "capacity", "cim")), n


# ----------------------------------------------------------------------------- determinism
def test_results_are_reproducible_from_the_seed_and_change_with_it(quick):
    def acc(seed):
        return [(r["system"], r["dim"], r["K"], r["corrupt"], r.get("unbind_cos")) for r in b_entity.run(quick["b"], seed=seed).tables["accuracy"]]

    assert acc(5) == acc(5) and acc(5) != acc(6)
    a1, a2 = a_telemetry.run(quick["a"], seed=3, include_external=False), a_telemetry.run(quick["a"], seed=3, include_external=False)
    key = lambda res: [(r["system"], r["noise"], r["rate"]) for r in res.tables["recall_vs_noise"]]  # noqa: E731
    assert key(a1) == key(a2)


# ----------------------------------------------------------------------------- A
def test_workload_a_measures_what_it_claims(quick_suite, quick):
    res = quick_suite.get("a")
    d0 = quick["a"]["dims"][0]
    get = lambda system, noise: next(r for r in res.tables["recall_vs_noise"] if r["dim"] == d0 and r["system"] == system and abs(r["noise"] - noise) < 1e-9)  # noqa: E731
    hop = "hopfield_b20"
    assert get(hop, 0.0)["rate"] == 1.0 and get("python_dict", 0.0)["rate"] == 1.0
    assert get("python_dict", 0.1)["rate"] == 0.0 and get("python_dict", 0.5)["rate"] == 0.0  # exact hash dies at the first flip
    assert get(hop, 0.2)["rate"] > get("lsh_k8_L32", 0.2)["rate"] - 1e-9
    assert get("bruteforce_hamming", 0.5)["rate"] < 0.2  # information-theoretically hopeless
    thr = {(r["dim"], r["system"]): r for r in res.tables["thresholds"]}
    assert thr[(d0, "python_dict")]["last_ok"] == 0.0
    assert thr[(d0, hop)]["last_ok"] > 0.1
    assert thr[(d0, "theory_bruteforce_ident_exact")]["last_ok"] > 0.3
    assert {r["system"] for r in res.tables["latency_memory"]} >= {hop, "python_dict", "bruteforce_hamming", "lsh_k8_L32"}
    for r in res.tables["latency_memory"]:
        assert r["read_p50_us"] > 0 and r["write_p50_us"] > 0 and 0 < r["eta"] <= 1.0 and r["memory_bytes"] > 0


def test_workload_a_iterated_attractor_is_at_least_as_robust_as_the_single_step(quick_suite):
    res = quick_suite.get("a")
    for t in res.tables["beta_thresholds"]:
        assert t["value_last_ok_s3"] >= t["value_last_ok_s1"]
    assert any(s["system"] == "hopfield_b20_s3" for s in res.tables["thresholds"])


def test_hnsw_ef_search_is_a_per_instance_parameter():
    pytest.importorskip("faiss")
    from neurostore.baselines import FaissBinaryHNSWSystem

    assert FaissBinaryHNSWSystem().name == "faiss_binary_hnsw" and FaissBinaryHNSWSystem(64).name == "faiss_binary_hnsw"  # 64 is the default
    s = FaissBinaryHNSWSystem(512)
    assert s.name == "faiss_binary_hnsw_ef512" and s.efs == 512 and FaissBinaryHNSWSystem.efs == 64  # instance state, not class state


def test_workload_a_scale_sweep_shows_tolerance_falling_as_the_store_grows(quick_suite):
    th = [r for r in quick_suite.get("a").tables["scale_thresholds"] if r["system"] == "theory_bruteforce_ident_exact"]
    by_n = {r["n_items"]: r["last_ok"] for r in th}
    assert by_n[40] > by_n[400]


@pytest.mark.external
@requires_faiss
@requires_rocksdict
@requires_redis
def test_workload_a_runs_every_external_baseline(quick):
    res = a_telemetry.run(quick["a"], seed=1, profile="quick", include_external=True)
    names = {r["system"] for r in res.tables["recall_vs_noise"]}
    assert {"rocksdb", "redis", "faiss_binary_flat", "faiss_binary_hnsw", "qdrant_local", "datasketch_minhash_lsh"} <= names
    assert not [s for s in res.skipped if s["system"] in names]
    # HNSW is also run at a higher efSearch in the store-size sweep and the latency table (recall/latency trade-off)
    ef = f"faiss_binary_hnsw_ef{quick['a']['hnsw_ef_sweep'][0]}"
    assert ef in {r["system"] for r in res.tables["scale_thresholds"]} and ef in {r["system"] for r in res.tables["latency_memory"]}


# ----------------------------------------------------------------------------- B
def test_workload_b_exact_methods_fail_on_corrupted_cues_and_vsa_does_not(quick_suite):
    res = quick_suite.get("b")
    acc = res.tables["accuracy"]
    for name in ("inverted_index", "linear_scan"):
        assert all(r["ident_rate"] == 1.0 for r in acc if r["system"] == name and r["corrupt"] == 0)
        assert all(r["ident_rate"] <= 0.05 for r in acc if r["system"] == name and r["corrupt"] == 1)
    corrupted = [r for r in acc if r["system"] == "vsa_map" and r["corrupt"] == 1 and r["K"] == 3]
    assert corrupted and all(r["ident_rate"] > 0.5 for r in corrupted)
    assert not [r for r in acc if r["K"] - r["corrupt"] < 2]  # unanswerable combinations are excluded


def test_workload_b_vsa_improves_with_dimension_and_exposes_its_costs(quick_suite):
    res = quick_suite.get("b")
    low, high = (next(r for r in res.tables["accuracy"] if r["system"] == "vsa_map" and r["dim"] == d and r["K"] == 2 and r["corrupt"] == 0) for d in (256, 1024))
    assert high["recon_rate"] >= low["recon_rate"] and high["unbind_cos"] == pytest.approx(low["unbind_cos"], abs=0.05)  # cosine is D-independent
    lat = {r["system"]: r for r in res.tables["latency_memory"]}
    assert lat["vsa_map"]["bytes_per_record"] > 20 * lat["inverted_index"]["bytes_per_record"]  # one hypervector per record vs a few bytes
    assert lat["vsa_map(bit-packed, ideal)"]["bytes_per_record"] < lat["vsa_map"]["bytes_per_record"] / 20
    assert all(c["frac_cos_gt_0p6"] == 0 for c in res.tables["reference_method_control"])


# ----------------------------------------------------------------------------- C
def test_workload_c_ranks_the_systems_as_expected(quick_suite):
    res = quick_suite.get("c")
    for n in (300, 1000):
        s = {r["system"]: r for r in res.tables["systems"] if r["n_cache"] == n}
        assert s["exact_hash"]["recall_at_1"] == 0.0 and s["exact_hash"]["tpr_at_fpr"] == 0.0
        assert s["bruteforce_cosine"]["recall_at_1"] >= 0.99
        assert s["flyhash_scan_1024_32"]["recall_at_1"] > 0.9 and s["flyhash_inv_1024_32"]["recall_at_1"] == s["flyhash_scan_1024_32"]["recall_at_1"]
        assert s["bruteforce_cosine"]["ann_agreement"] == 1.0
        assert s["exact_hash"]["p50_us"] == min(r["p50_us"] for r in s.values())  # a dict lookup is the latency floor
        assert 0 <= s["flyhash_scan_1024_32"]["realised_fpr"] <= 0.15
        assert s["flyhash_scan_1024_32"]["bytes_per_prompt"] == 128 and s["simhash_1024"]["bytes_per_prompt"] == 128
    sweep = res.tables["hardness_sweep"]
    bf = {r["para_cos"]: r["recall_at_1"] for r in sweep if r["system"] == "bruteforce_cosine"}
    assert bf[0.9] >= bf[0.7]


def test_REGRESSION_c_sweeps_share_the_corpus_and_the_topics_of_the_main_evaluation(quick):
    """The hardness sweep once regenerated topic centres, so its novel prompts were far easier negatives than the
    main evaluation's. Sweep datasets must differ only in their queries."""
    cfg = {**quick["c"], "n_cache": [300]}
    main = c_prompt_cache._dataset(cfg, 7, 300, 0.9, tag=0)
    for tag, cos in ((101, 0.7), (102, 0.6)):
        sweep = c_prompt_cache._dataset(cfg, 7, 300, cos, tag=tag)
        assert np.array_equal(main["cache"], sweep["cache"])  # identical corpus
        assert not np.array_equal(main["neg_q"], sweep["neg_q"]) and not np.array_equal(main["pos_q"], sweep["pos_q"])  # new queries
        # same difficulty of negatives: nearest cached prompt is equally similar to a novel prompt in both
        hardness = lambda d: float((d["neg_q"] @ d["cache"].T).max(axis=1).mean())  # noqa: E731
        assert hardness(sweep) == pytest.approx(hardness(main), abs=0.05)
        assert np.allclose((sweep["pos_q"] * sweep["cache"][sweep["pos_src"]]).sum(axis=1), cos, atol=1e-4)
    # topic structure is real: novel prompts are far closer to the cache than independent random vectors would be
    assert hardness(main) > 0.5


def test_workload_c_flyhash_sweep_compares_against_simhash_at_equal_bits(quick_suite):
    rows = quick_suite.get("c").tables["flyhash_sweep"]
    assert {r["kind"] for r in rows} == {"flyhash", "simhash"}
    assert {r["bits"] for r in rows if r["kind"] == "simhash"} == {r["bits"] for r in rows if r["kind"] == "flyhash"}
    assert all(0 <= r["recall_at_1"] <= 1 and 0 <= r["tpr_at_fpr"] <= 1 for r in rows)
    names = [c["name"] for c in quick_suite.get("c").checks]
    assert any("hardest paraphrases" in n for n in names)


def test_workload_c_accepts_user_supplied_embeddings(quick, tmp_path):
    rng = np.random.default_rng(0)
    cache = rng.standard_normal((200, 32)).astype(np.float32)
    src = rng.integers(0, 200, 60)
    para = cache[src] + 0.3 * rng.standard_normal((60, 32)).astype(np.float32)
    novel = rng.standard_normal((80, 32)).astype(np.float32)
    path = tmp_path / "emb.npz"
    np.savez(path, cache=cache, queries=np.concatenate([para, novel]), labels=np.concatenate([src, -np.ones(80, dtype=int)]))
    cfg = {**quick["c"], "embeddings_npz": str(path), "n_cache": [999]}  # the paraphrase sweep is skipped for user data
    res = c_prompt_cache.run(cfg, seed=0, include_external=False)
    bf = next(r for r in res.tables["systems"] if r["system"] == "bruteforce_cosine")
    assert bf["n_cache"] == 200 and bf["n_pos"] == 60 and bf["n_neg"] == 40 and bf["recall_at_1"] > 0.9
    with pytest.raises(ValueError):
        np.savez(tmp_path / "bad.npz", cache=cache, queries=para, labels=np.array([0]))
        c_prompt_cache.load_embeddings(tmp_path / "bad.npz")


# ----------------------------------------------------------------------------- D
def test_workload_d_capacity_physics(quick_suite):
    res = quick_suite.get("d")
    cap = res.tables["capacity"]
    for alg in ("map", "hrr"):
        for dim in (512, 1024):
            lo, hi = (next(r for r in cap if r["algebra"] == alg and r["dim"] == dim and r["triples"] == t) for t in (30, 60))
            assert lo["single_hop_acc"] >= hi["single_hop_acc"] and lo["ghost_rate"] <= hi["ghost_rate"]
            assert lo["path_acc_h3"] <= lo["path_acc_h1"] + 1e-9  # errors compound along a walk
        assert next(r for r in cap if r["algebra"] == alg and r["dim"] == 1024 and r["triples"] == 30)["single_hop_acc"] > 0.95
    sp = res.tables["speed"]
    holo = max(r["hops_per_ms_single"] for r in sp if r["family"] == "neurostore")
    dct = max(r["hops_per_ms_single"] for r in sp if r["system"] == "adjacency_dict")
    assert dct > 10 * holo  # honest result: pointer chasing is far faster on a CPU
    assert all(r["reverse_hit_rate"] <= 0.1 for r in res.tables["direction_control"])


# ----------------------------------------------------------------------------- capacity and CIM
def test_capacity_workload_reproduces_the_closed_forms(quick_suite):
    summ = quick_suite.get("capacity").tables["capacity_summary"]
    for s in (x for x in summ if x["memory"] == "linear_matrix"):
        assert s["c_max_interpolated"] == pytest.approx(s["c_max_theory"], rel=0.25)
        assert 0 < s["eta"] < 0.05
    raw = [s for s in summ if s["memory"] == "vsa_bundle_raw"]
    assert all(s["raw_ber_at_smallest_n"] > 0.2 for s in raw)  # reading a bundle bit-by-bit is hopeless at any D
    hop = [s for s in summ if s["memory"] == "modern_hopfield"]
    assert all(s["censored"] and s["eta"] == pytest.approx(1 / 32, rel=0.05) for s in hop)  # float32 storage of +-1 bits


def test_cim_workload_separates_emulated_accuracy_from_modelled_latency(quick_suite):
    res = quick_suite.get("cim")
    tol = {(r["query_noise"], r["param"], r["value"], r["variant"]): r for r in res.tables["tolerance"] if r["dim"] == 256}
    assert tol[(0.0, "ideal", 0, "")]["ident_acc"] == 1.0
    noisy, mild = tol[(0.3, "read_noise", 1.0, "ideal ADC")]["ident_acc"], tol[(0.3, "read_noise", 0.1, "ideal ADC")]["ident_acc"]
    assert noisy < 0.5 and mild > noisy  # read noise equal to the tile full scale destroys search; 10% degrades it
    assert tol[(0.3, "combined_realistic", 1, "see params.combined")]["ident_acc"] >= 0.95
    for r in res.tables["latency_model"]:
        assert r["kind"] == "MODELED (not measured)" and r["modeled_ns"] > 0 and r["cpu_best_us"] > 0
    assert res.headlines == []  # a model must never enter the measured scorecard
    assert any("NOT measurements" in n for n in res.notes)


def test_individual_workload_runs_standalone(quick):
    assert capacity.run(quick["capacity"], seed=2).tables["capacity_curves"]
    assert cim_emulation.run(quick["cim"], seed=2).tables["tolerance_limits"]
    assert d_graph_walk.run(quick["d"], seed=2).tables["capacity_summary"]
