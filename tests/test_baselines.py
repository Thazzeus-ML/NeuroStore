"""Every baseline must be *correct* before it can be a fair comparison point."""
import itertools
import math
import subprocess

import numpy as np
import pytest

from conftest import requires_datasketch, requires_faiss, requires_qdrant, requires_redis, requires_rocksdict
from neurostore.baselines import (
    AdjacencyDictGraph,
    BitSamplingLSHSystem,
    BruteForceCosineCache,
    BruteForceHammingSystem,
    CompositeIndexSet,
    DatasketchLSHSystem,
    DenseSuccessorTable,
    DictKVSystem,
    ExactHashCache,
    FaissBinaryFlatSystem,
    FaissBinaryHNSWSystem,
    FaissFlatCache,
    FaissHNSWCache,
    FlyHashBinaryHNSWCache,
    FlyHashInvertedCache,
    FlyHashScanCache,
    FuzzyScan,
    InvertedIndexIntersect,
    LinearScan,
    QdrantCache,
    QdrantSystem,
    RedisKVSystem,
    RocksKVSystem,
    SimHashCache,
    availability,
)
from neurostore.common import flip_exact, make_rng, random_bipolar
from neurostore.workloads.c_prompt_cache import make_paraphrases, make_prompts
from neurostore.workloads.d_graph_walk import make_graph

N, D = 120, 256


@pytest.fixture
def data(rng):
    return random_bipolar(rng, N, D), random_bipolar(rng, N, D)


ALL_ASSOC = [
    pytest.param(DictKVSystem, id="dict"),
    pytest.param(RocksKVSystem, id="rocksdb", marks=requires_rocksdict),
    pytest.param(RedisKVSystem, id="redis", marks=requires_redis),
    pytest.param(BruteForceHammingSystem, id="bruteforce"),
    pytest.param(FaissBinaryFlatSystem, id="faiss-flat", marks=requires_faiss),
    pytest.param(FaissBinaryHNSWSystem, id="faiss-hnsw", marks=requires_faiss),
    pytest.param(QdrantSystem, id="qdrant", marks=requires_qdrant),
    pytest.param(lambda: BitSamplingLSHSystem(8, 32), id="lsh"),
    pytest.param(DatasketchLSHSystem, id="datasketch", marks=requires_datasketch),
]


# ----------------------------------------------------------------------------- the common contract
@pytest.mark.parametrize("make", ALL_ASSOC)
def test_every_system_returns_the_stored_value_for_a_clean_key_and_nothing_for_an_unknown_one(make, data, rng):
    keys, values = data
    s = make()
    try:
        s.build(keys, values)
        assert (s.retrieve_values(keys) == values).all()
        unknown = random_bipolar(rng, 5, D)
        out = s.retrieve_values(unknown)
        exact_only = s.family == "kv"
        if exact_only:  # an exact store must answer "nothing", never a wrong value
            assert (out == 0).all()
        assert out.shape == (5, D)
    finally:
        s.close()


@pytest.mark.parametrize("make", ALL_ASSOC)
def test_single_operation_paths_agree_with_the_batch_path_and_writes_are_visible(make, data, rng):
    keys, values = data
    s = make()
    try:
        s.build(keys, values)
        q = flip_exact(rng, keys[:6], 8)
        prepared = [s.prepare_read(x) for x in q]
        for p in prepared:
            s.read_one(p)  # must not raise
        new_k, new_v = random_bipolar(rng, 2, D), random_bipolar(rng, 2, D)
        for i in range(2):
            s.write_one(s.prepare_write(new_k[i], new_v[i]))
        assert (s.retrieve_values(new_k) == new_v).all()
        assert (s.retrieve_values(keys[:10]) == values[:10]).all()  # old content intact
        assert s.memory_bytes() > N * D / 8  # at least the packed payload
    finally:
        s.close()


@pytest.mark.parametrize("make", [p for p in ALL_ASSOC if p.id in ("dict", "rocksdb", "redis")])
def test_exact_match_stores_fail_on_any_bit_flip(make, data, rng):
    keys, values = data
    s = make()
    try:
        s.build(keys, values)
        assert (s.retrieve_values(flip_exact(rng, keys, 1)) == 0).all()
    finally:
        s.close()


# ----------------------------------------------------------------------------- nearest-neighbour systems
def test_bruteforce_hamming_is_exact_nearest_neighbour(data, rng):
    keys, values = data
    s = BruteForceHammingSystem()
    s.build(keys, values)
    q = flip_exact(rng, keys, 90)
    ref = (q[:, None, :] != keys[None, :, :]).sum(axis=2).argmin(axis=1)
    assert np.array_equal(s.retrieve_ids(q), ref)


@requires_faiss
def test_faiss_flat_equals_bruteforce_and_hnsw_is_near_exact(data, rng):
    keys, values = data
    bf, flat, hnsw = BruteForceHammingSystem(), FaissBinaryFlatSystem(), FaissBinaryHNSWSystem()
    for s in (bf, flat, hnsw):
        s.build(keys, values)
    q = flip_exact(rng, keys, 80)
    assert np.array_equal(flat.retrieve_ids(q), bf.retrieve_ids(q))
    assert (hnsw.retrieve_ids(q) == bf.retrieve_ids(q)).mean() >= 0.97


@requires_qdrant
def test_qdrant_cosine_agrees_with_hamming_ranking(data, rng):
    keys, values = data
    bf, qd = BruteForceHammingSystem(), QdrantSystem()
    bf.build(keys, values)
    qd.build(keys, values)
    try:
        q = flip_exact(rng, keys[:30], 60)
        assert np.array_equal(qd.retrieve_ids(q), bf.retrieve_ids(q))  # for bipolar vectors cosine and Hamming rank identically
    finally:
        qd.close()


# ----------------------------------------------------------------------------- LSH
def test_bitsampling_lsh_collision_probability_matches_the_analytic_formula():
    rng = make_rng(11)
    keys, values = random_bipolar(rng, 300, 512), random_bipolar(rng, 300, 512)
    s = BitSamplingLSHSystem(k_bits=8, n_tables=16, seed=1)
    s.build(keys, values)
    for noise in (0.1, 0.2, 0.3):
        q = flip_exact(rng, np.tile(keys, (4, 1)), int(512 * noise))
        found = (s.retrieve_ids(q) == np.tile(np.arange(300), 4)).mean()
        th = s.p_found(noise)
        se = math.sqrt(th * (1 - th) / len(q))
        assert abs(found - th) < 4 * se + 0.02, (noise, found, th)


def test_lsh_tuning_trades_noise_tolerance_for_candidates(data, rng):
    keys, values = data
    robust, strict = BitSamplingLSHSystem(6, 64), BitSamplingLSHSystem(14, 8)
    for s in (robust, strict):
        s.build(keys, values)
    q = flip_exact(rng, keys, 70)
    assert (robust.retrieve_ids(q) == np.arange(N)).mean() > (strict.retrieve_ids(q) == np.arange(N)).mean() + 0.3
    with pytest.raises(ValueError):
        BitSamplingLSHSystem(0, 4)


@requires_datasketch
def test_datasketch_lsh_recovers_lightly_corrupted_keys(rng):
    keys, values = random_bipolar(rng, 80, 512), random_bipolar(rng, 80, 512)
    s = DatasketchLSHSystem()
    s.build(keys, values)
    q = flip_exact(rng, keys, 25)  # 5%
    assert (s.retrieve_ids(q) == np.arange(80)).mean() > 0.95


# ----------------------------------------------------------------------------- real external services
@requires_redis
def test_redis_server_is_started_and_stopped(data):
    keys, values = data
    s = RedisKVSystem()
    proc = s._proc
    s.build(keys, values)
    assert proc.poll() is None and s.memory_bytes() > N * D / 8
    s.close()
    assert proc.poll() is not None  # no stray server left behind
    s.close()  # idempotent


@requires_rocksdict
def test_rocksdb_directory_is_cleaned_up(data):
    import os

    keys, values = data
    s = RocksKVSystem()
    path = s._dir
    s.build(keys, values)
    assert s.memory_bytes() > 0 and os.path.isdir(path)
    s.close()
    assert not os.path.exists(path)


# ----------------------------------------------------------------------------- relational baselines
def _table(rng, m=400, f=6, v=120):
    return rng.integers(0, v, (m, f)).astype(np.uint16), [v] * f


def _unique_cues(rng, table, index, k, q):
    src, flds, vals = [], [], []
    while len(src) < q:
        i = int(rng.integers(0, table.shape[0]))
        fl = rng.permutation(table.shape[1])[:k]
        v = table[i, fl]
        if index.match_all(fl, v).size == 1:
            src.append(i)
            flds.append(fl)
            vals.append(v)
    return np.array(src), np.stack(flds), np.stack(vals).astype(np.int64)


def test_all_exact_relational_methods_agree_on_clean_unique_cues(rng):
    table, vocab = _table(rng)
    inv = InvertedIndexIntersect(table, vocab)
    src, flds, vals = _unique_cues(rng, table, inv, 3, 60)
    methods = [inv, LinearScan(table), FuzzyScan(table), CompositeIndexSet(table, 3)]
    for m in methods:
        assert np.array_equal(m.identify(flds, vals), src), m.name
    tgt = np.array([rng.choice(np.setdiff1d(np.arange(6), f)) for f in flds])
    assert np.array_equal(methods[0].reconstruct(flds, vals, tgt)["values"], table[src, tgt])


def test_only_the_fuzzy_scan_survives_a_corrupted_cue_field(rng):
    table, vocab = _table(rng, v=400)
    inv = InvertedIndexIntersect(table, vocab)
    src, flds, vals = _unique_cues(rng, table, inv, 4, 80)
    vals = vals.copy()
    vals[:, 0] = (vals[:, 0] + 1) % 400
    assert (LinearScan(table).identify(flds, vals) == src).mean() < 0.05
    assert (inv.identify(flds, vals) == src).mean() < 0.05
    assert (CompositeIndexSet(table, 4).identify(flds, vals) == src).mean() < 0.05
    assert (FuzzyScan(table).identify(flds, vals) == src).mean() > 0.9


def test_a_missing_record_is_reported_as_minus_one(rng):
    table, vocab = _table(rng, v=300)
    fields, values = np.array([[0, 1, 2]]), np.array([[299, 299, 299]])
    assert not (table[:, [0, 1, 2]] == 299).all(axis=1).any()  # the cue really matches no record
    for m in (InvertedIndexIntersect(table, vocab), LinearScan(table), CompositeIndexSet(table, 3)):
        assert m.identify(fields, values)[0] == -1, m.name
        assert m.reconstruct(fields, values, np.array([4]))["values"][0] == -1
    assert FuzzyScan(table).identify(fields, values)[0] >= 0  # fuzzy always returns its best partial match


def test_composite_indexes_blow_up_combinatorially_while_the_inverted_index_does_not(rng):
    table, vocab = _table(rng, m=300, f=8, v=50)
    comp = {k: CompositeIndexSet(table, k) for k in (2, 4)}
    assert comp[2].n_indexes == math.comb(8, 2) and comp[4].n_indexes == math.comb(8, 4)
    assert comp[4].memory_bytes() > 2 * comp[2].memory_bytes()
    inv = InvertedIndexIntersect(table, vocab)
    assert inv.memory_bytes() < comp[2].memory_bytes() / 5
    assert 2**8 - 1 == 255  # indexes needed to serve ANY field subset


# ----------------------------------------------------------------------------- graph baselines
def test_pointer_chasing_baselines_are_exact_and_agree(rng):
    e, r = 50, 8
    tri = make_graph(rng, e, r, 120)
    dct, tab = AdjacencyDictGraph(tri), DenseSuccessorTable(tri, e, r)
    assert all(dct.hop(int(s), int(rr)) == o for s, rr, o in tri)
    assert np.array_equal(tab.hop_batch(tri[:, 0], tri[:, 1]), tri[:, 2])
    assert dct.hop(0, 99) == -1
    start = tri[:5, 0]
    rels = np.stack([tri[:5, 1]] * 2, axis=1)
    assert np.array_equal(tab.walk_batch(start, rels)[:, 0], tri[:5, 2])
    assert dct.walk(int(tri[0, 0]), [int(tri[0, 1])]) == [int(tri[0, 2])]
    assert tab.memory_bytes() == e * r * 4 and dct.memory_bytes() > tab.memory_bytes()


# ----------------------------------------------------------------------------- cache systems
@pytest.fixture(scope="module")
def prompts():
    rng = make_rng(21)
    centers = rng.standard_normal((10, 64)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    cache = make_prompts(rng, 600, centers, 0.5)
    src = rng.integers(0, 600, 200)
    return {"cache": cache, "src": src, "para": make_paraphrases(rng, cache[src], 0.9), "novel": make_prompts(rng, 200, centers, 0.5)}


def test_prompt_generators_hit_their_target_cosines(prompts):
    cache, src, para = prompts["cache"], prompts["src"], prompts["para"]
    assert np.allclose(np.linalg.norm(cache, axis=1), 1, atol=1e-5)
    assert np.allclose((cache[src] * para).sum(axis=1), 0.9, atol=1e-4)
    sims = (cache @ cache.T)[~np.eye(600, dtype=bool)]
    same_topic = sims[sims > 0.25]  # 10 topics: ~1/10 of unrelated pairs share one and sit at cosine ~topic_rho = 0.5
    assert 0.07 < same_topic.size / sims.size < 0.13
    assert 0.45 < same_topic.mean() < 0.62
    assert abs(sims[sims <= 0.25].mean()) < 0.05  # different topics: uncorrelated


def test_exact_hash_misses_every_paraphrase_but_finds_identical_vectors(prompts):
    s = ExactHashCache()
    s.build(prompts["cache"])
    assert (s.search(prompts["para"])[0] == -1).all()
    assert (s.search(prompts["cache"][:50])[0] == np.arange(50)).all()


def test_exact_cosine_and_simhash_find_paraphrases(prompts):
    for s in (BruteForceCosineCache(), SimHashCache(2048)):
        s.build(prompts["cache"])
        ids, sc = s.search(prompts["para"])
        assert (ids == prompts["src"]).mean() > 0.98, s.name
    bc = BruteForceCosineCache()
    bc.build(prompts["cache"])
    _, sp = bc.search(prompts["para"])
    _, sn = bc.search(prompts["novel"])
    assert np.median(sp) > np.percentile(sn, 95)  # paraphrases score above novel prompts


def test_flyhash_scan_and_inverted_index_are_two_implementations_of_the_same_function(prompts):
    scan, inv = FlyHashScanCache(1024, 32, seed=3), FlyHashInvertedCache(1024, 32, seed=3)
    scan.build(prompts["cache"])
    inv.build(prompts["cache"])
    i1, s1 = scan.search(prompts["para"])
    i2, s2 = inv.search(prompts["para"])
    assert np.array_equal(i1, i2) and np.allclose(s1, s2)
    assert (i1 == prompts["src"]).mean() > 0.9
    a, b = prompts["para"][0], prompts["novel"][0]
    assert scan.search_one(scan.prepare_one(a))[0] == inv.search_one(inv.prepare_one(a))[0] == i1[0]
    assert scan.search_one(scan.prepare_one(b)) == pytest.approx(inv.search_one(inv.prepare_one(b)))


def test_flyhash_memory_is_the_packed_code_plus_fixed_projection(prompts):
    scan = FlyHashScanCache(2048, 64)
    scan.build(prompts["cache"])
    assert scan.memory_bytes() == 600 * 2048 // 8 and scan.fixed_bytes() == 64 * 2048 * 4
    sim512 = SimHashCache(512)
    sim512.build(prompts["cache"])
    assert sim512.memory_bytes() == 600 * 64 and sim512.memory_bytes() < scan.memory_bytes()  # 512-bit vs 2048-bit codes
    assert scan.memory_bytes() == 600 * 64 * 4  # a 2048-bit code costs the same as a 64-d float32 vector (256 B)


@requires_faiss
def test_faiss_caches(prompts):
    bc, flat, hnsw = BruteForceCosineCache(), FaissFlatCache(), FaissHNSWCache()
    for s in (bc, flat, hnsw):
        s.build(prompts["cache"])
    ref, _ = bc.search(prompts["para"])
    assert np.array_equal(flat.search(prompts["para"])[0], ref)
    assert (hnsw.search(prompts["para"])[0] == ref).mean() >= 0.97
    assert hnsw.memory_bytes() > flat.memory_bytes()  # graph links on top of the vectors
    fb = FlyHashBinaryHNSWCache(1024, 32, seed=3)
    scan = FlyHashScanCache(1024, 32, seed=3)
    fb.build(prompts["cache"])
    scan.build(prompts["cache"])
    assert (fb.search(prompts["para"])[0] == scan.search(prompts["para"])[0]).mean() >= 0.95
    q = fb.prepare_one(prompts["para"][0])
    assert fb.search_one(q)[0] == fb.search(prompts["para"][:1])[0][0]


@requires_qdrant
def test_qdrant_cache_matches_exact_cosine_and_respects_its_size_cap(prompts):
    bc, qd = BruteForceCosineCache(), QdrantCache()
    bc.build(prompts["cache"])
    qd.build(prompts["cache"])
    try:
        assert np.array_equal(qd.search(prompts["para"][:40])[0], bc.search(prompts["para"][:40])[0])
        assert qd.max_items == 20_000
    finally:
        qd.close()


# ----------------------------------------------------------------------------- catalogue
def test_availability_reports_every_catalogued_baseline():
    info = availability()
    assert {"python_dict", "bruteforce_hamming", "bitsampling_lsh", "simhash"} <= set(info)
    assert all(isinstance(v["available"], bool) and v["family"] in ("kv", "ann", "lsh") for v in info.values())
    assert info["python_dict"]["available"]
