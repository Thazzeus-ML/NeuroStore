"""Utilities: noise injection, bit packing, statistics, timing, memory accounting."""
import json
import tracemalloc

import numpy as np
import pytest

from neurostore import common as C
from neurostore.results import to_jsonable


# ----------------------------------------------------------------------------- RNG / data
def test_make_rng_is_deterministic_and_passthrough():
    a, b = C.make_rng(7).random(5), C.make_rng(7).random(5)
    assert np.array_equal(a, b)
    g = np.random.default_rng(1)
    assert C.make_rng(g) is g


def test_spawn_seeds_are_distinct_and_reproducible():
    s = C.spawn_seeds(42, 6)
    assert len(set(s)) == 6 and s == C.spawn_seeds(42, 6) and s != C.spawn_seeds(43, 6)


def test_random_bipolar_values_dtype_and_balance(rng):
    x = C.random_bipolar(rng, 200, 512)
    assert x.dtype == np.int8 and set(np.unique(x)) == {-1, 1}
    assert abs(x.mean()) < 0.02  # 100k samples


# ----------------------------------------------------------------------------- noise injection
@pytest.mark.parametrize("n_flips", [0, 1, 37, 255, 512])
def test_flip_exact_flips_exactly_n_per_row(rng, n_flips):
    x = C.random_bipolar(rng, 40, 512)
    y = C.flip_exact(rng, x, n_flips)
    assert ((x != y).sum(axis=1) == n_flips).all()
    assert set(np.unique(y)) <= {-1, 1}


def test_flip_exact_does_not_mutate_input_and_handles_1d(rng):
    x = C.random_bipolar(rng, 1, 128)[0]
    keep = x.copy()
    y = C.flip_exact(rng, x, 10)
    assert np.array_equal(x, keep) and y.shape == (128,) and (x != y).sum() == 10


def test_flip_exact_positions_are_uniform(rng):
    x = np.ones((4000, 64), dtype=np.int8)
    freq = (C.flip_exact(rng, x, 16) == -1).mean(axis=0)  # expected 16/64 = 0.25 at every position
    assert np.abs(freq - 0.25).max() < 0.04


def test_noise_to_flips_rounds_and_clips():
    assert C.noise_to_flips(512, 0.05) == 26 and C.noise_to_flips(512, 0) == 0
    assert C.noise_to_flips(100, 2.0) == 100 and C.noise_to_flips(100, -1) == 0


def test_inject_bit_flip_noise_matches_reference_semantics(rng):
    v = C.random_bipolar(rng, 1, 512)[0]
    noisy = C.inject_bit_flip_noise(rng, v, 0.1)
    assert (v != noisy).sum() == round(512 * 0.1)


# ----------------------------------------------------------------------------- bit packing
@pytest.mark.parametrize("dim", [8, 63, 64, 65, 512, 1000])
def test_pack_unpack_roundtrip_and_word_width(rng, dim):
    bits = (C.random_bipolar(rng, 7, dim) > 0).astype(np.uint8)
    w = C.pack_bits(bits)
    assert w.dtype == np.uint64 and w.shape == (7, -(-dim // 64))
    assert np.array_equal(C.unpack_bits(w, dim), bits)


def test_packed_hamming_equals_unpacked_hamming(rng):
    a, b = C.random_bipolar(rng, 30, 777), C.random_bipolar(rng, 50, 777)
    d_ref = (a[:, None, :] != b[None, :, :]).sum(axis=2)
    d = C.packed_distance_matrix(C.pack_bipolar(a), C.pack_bipolar(b))
    assert np.array_equal(d, d_ref)
    assert np.array_equal(C.packed_distance_to_all(C.pack_bipolar(a[3]), C.pack_bipolar(b)), d_ref[3])


def test_packed_and_overlap_counts_shared_ones(rng):
    a = (rng.random((5, 300)) < 0.3).astype(np.uint8)
    b = (rng.random((9, 300)) < 0.3).astype(np.uint8)
    ref = a.astype(int) @ b.astype(int).T
    got = C.packed_distance_matrix(C.pack_bits(a), C.pack_bits(b), op="and")
    assert np.array_equal(got, ref)


def test_popcount_lut_fallback_matches_native(rng):
    x = rng.integers(0, 2**63, size=(10, 4), dtype=np.uint64) ^ (rng.integers(0, 2, size=(10, 4), dtype=np.uint64) << np.uint64(63))
    ref = np.array([[bin(int(v)).count("1") for v in row] for row in x])
    assert np.array_equal(C.popcount_u64_lut(x), ref)
    assert np.array_equal(C.popcount_u64(x), ref)
    assert np.array_equal(C.popcount_rows(x), ref.sum(axis=1))


def test_packed_distance_matrix_chunking_is_invisible(rng):
    a, b = C.random_bipolar(rng, 33, 256), C.random_bipolar(rng, 40, 256)
    full = C.packed_distance_matrix(C.pack_bipolar(a), C.pack_bipolar(b))
    tiny = C.packed_distance_matrix(C.pack_bipolar(a), C.pack_bipolar(b), max_temp_bytes=1)
    assert np.array_equal(full, tiny)


# ----------------------------------------------------------------------------- statistics
def test_wilson_interval_known_values():
    lo, hi = C.wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = C.wilson_interval(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.33
    lo, hi = C.wilson_interval(10, 10)
    assert hi == 1.0 and 0.67 < lo < 0.75
    assert C.wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_narrows_with_n_and_contains_rate():
    w = [C.wilson_interval(int(0.9 * n), n) for n in (20, 200, 2000)]
    assert w[0][1] - w[0][0] > w[1][1] - w[1][0] > w[2][1] - w[2][0]
    assert all(lo <= 0.9 <= hi for lo, hi in w)


def test_proportion_dict():
    p = C.proportion(90, 100)
    assert p["rate"] == 0.9 and p["k"] == 90 and p["n"] == 100 and p["lo"] < 0.9 < p["hi"]


def test_normal_cdf_and_sf():
    assert float(C.norm_cdf(0.0)) == pytest.approx(0.5)
    assert float(C.norm_cdf(1.96)) == pytest.approx(0.975, abs=1e-4)
    assert float(C.norm_sf(3.0902)) == pytest.approx(1e-3, rel=2e-3)
    assert float(C.norm_sf(8.0)) > 0  # no underflow to a flat zero in the tail


def test_threshold_crossing_interpolates_and_uses_running_minimum():
    tc = C.threshold_crossing([0, 0.1, 0.2, 0.3], [1.0, 0.995, 0.9, 0.5], 0.99)
    assert tc["last_ok"] == 0.1 and tc["first_fail"] == 0.2
    assert 0.1 < tc["crossing"] < 0.2
    # a lucky recovery after the first failure must not move the threshold
    tc2 = C.threshold_crossing([0, 0.1, 0.2, 0.3], [1.0, 0.98, 0.995, 0.5], 0.99)
    assert tc2["last_ok"] == 0 and tc2["first_fail"] == 0.1


def test_threshold_crossing_edge_cases():
    assert C.threshold_crossing([0, 1], [1.0, 1.0], 0.99) == {"last_ok": 1, "first_fail": None, "crossing": None}
    tc = C.threshold_crossing([0, 1], [0.5, 0.4], 0.99)
    assert tc["last_ok"] is None and tc["first_fail"] == 0


# ----------------------------------------------------------------------------- timing
def test_time_calls_measures_each_call():
    ns = C.time_calls(lambda x: sum(range(x)), [10, 1000, 100000], warmup=0)
    assert ns.dtype == np.int64 and (ns > 0).all() and ns[2] > ns[0]


def test_time_calls_warmup_runs_but_is_not_recorded():
    seen = []
    ns = C.time_calls(seen.append, list(range(10)), warmup=3)
    assert len(ns) == 10 and len(seen) == 13


def test_time_calls_restores_gc_state():
    import gc

    gc.enable()
    C.time_calls(lambda x: x, [1, 2, 3])
    assert gc.isenabled()


def test_summarize_latency_percentiles_in_microseconds():
    s = C.summarize_latency(np.array([1000, 2000, 3000, 4000, 100000]))
    assert s["n"] == 5 and s["p50_us"] == 3.0 and s["min_us"] == 1.0 and s["max_us"] == 100.0 and s["p99_us"] > s["p50_us"]
    assert np.isnan(C.summarize_latency(np.array([]))["p50_us"])


def test_timer_overhead_is_small_and_positive():
    assert 0 < C.timer_overhead_ns(2000) < 20000


# ----------------------------------------------------------------------------- growable buffer
def test_growable_rows_amortised_growth_and_content():
    g = C.GrowableRows(4, np.float32)
    for i in range(100):
        assert g.append(np.full(4, i)) == i
    assert len(g) == 100 and np.array_equal(g.view[:, 0], np.arange(100))
    g.extend(np.ones((7, 4)))
    assert len(g) == 107 and g.nbytes >= 107 * 16
    g.shrink_to_fit()
    assert g.nbytes == 107 * 16


def test_growable_rows_does_not_copy_on_every_append():
    g = C.GrowableRows(8, np.uint64)
    buffers = set()
    for i in range(200):
        g.append(np.zeros(8, dtype=np.uint64))
        buffers.add(id(g._buf))
    assert len(buffers) <= 12  # doubling: ~log2(200) reallocations, not 200


# ----------------------------------------------------------------------------- memory accounting
def test_deep_sizeof_counts_contents_and_shared_objects_once():
    blob = b"x" * 1000
    d = {1: blob, 2: blob}  # the same bytes object twice
    assert 1000 < C.deep_sizeof(d) < 2000
    arr = np.zeros(10_000, dtype=np.float32)
    assert C.deep_sizeof(arr) >= arr.nbytes
    assert C.deep_sizeof({"a": arr}) >= arr.nbytes
    view = arr[:100]
    assert C.deep_sizeof([arr, view]) < arr.nbytes + 1000  # a view must not double count its base


def test_deep_sizeof_agrees_with_tracemalloc_within_a_factor_of_two():
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    d = {bytes([i % 256]) * 8 + i.to_bytes(4, "little"): bytes(64) for i in range(20000)}
    allocated = tracemalloc.get_traced_memory()[0] - base
    tracemalloc.stop()
    est = C.deep_sizeof(d)
    assert 0.5 < est / allocated < 2.0


def test_fmt_bytes():
    assert C.fmt_bytes(512) == "512 B" and C.fmt_bytes(2048) == "2.0 KiB" and C.fmt_bytes(3 * 1024**2) == "3.0 MiB"
    assert C.fmt_bytes(float("nan")) == "n/a"


# ----------------------------------------------------------------------------- threads / environment
def test_limit_threads_is_a_noop_for_none_and_restores():
    with C.limit_threads(None):
        pass
    with C.limit_threads(1):
        x = np.ones((50, 50)) @ np.ones((50, 50))
    assert x.shape == (50, 50)


def test_environment_info_is_complete_and_json_serialisable():
    env = C.environment_info()
    for k in ("python", "numpy", "cpu_model", "cpu_count", "timer_overhead_ns", "optional_libs"):
        assert k in env
    json.dumps(to_jsonable(env), allow_nan=False)
