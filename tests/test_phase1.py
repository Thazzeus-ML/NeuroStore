import numpy as np
from hypothesis import given, strategies as st
from neurostore.encoder import SparseEncoder, normalize_features, q8
from neurostore.spikelog import SpikeLog
from neurostore.rates import RateCounters
from neurostore.layouts import SPIKE_LAYOUTS
from sim.generator import World, WorldConfig


@given(feats=st.lists(st.integers(-5, 900), max_size=120), seed=st.integers(0, 20),
       k=st.sampled_from([4, 16, 32, 64]), thseed=st.integers(0, 5))
def test_exactly_k_unique_units(feats, seed, k, thseed):
    enc = SparseEncoder(800, 1024, k, 2, seed)
    th = np.random.default_rng(thseed).normal(0, 1, 1024).astype(np.float32)
    f = normalize_features(feats, 800)
    u, s = enc.encode(f, th)
    assert len(u) == k and len(np.unique(u)) == k and np.all(np.diff(u) > 0)
    assert u.min() >= 0 and u.max() < 1024
    assert np.all((s >= 0) & (s <= 1))


@given(feats=st.lists(st.integers(0, 799), max_size=60), seed=st.integers(0, 20))
def test_same_event_same_state_same_code_and_no_mutation(feats, seed):
    enc = SparseEncoder(800, 1024, 32, 2, seed)
    enc.theta = np.random.default_rng(seed).normal(0, .5, 1024).astype(np.float32)
    before = enc.theta.copy()
    f = normalize_features(feats, 800)
    a, sa = enc.encode(f); b, sb = enc.encode(f.copy())
    assert np.array_equal(a, b) and np.array_equal(sa, sb) and np.array_equal(before, enc.theta)
    enc2 = SparseEncoder(800, 1024, 32, 2, seed); enc2.theta = before.copy()
    assert np.array_equal(enc2.encode(f)[0], a)               # same seed => same projection


def test_same_concept_overlaps_more():
    w = World(WorldConfig(), 5); s = w.stream("test", 600)
    enc = SparseEncoder(800, 1024, 32, 2, 5)
    codes = [enc.encode(s.event(i))[0] for i in range(len(s))]
    single = [i for i in range(len(s)) if len(s.concepts[i]) == 1]
    rng = np.random.default_rng(0); same, diff = [], []
    for _ in range(1500):
        i, j = rng.choice(single, 2, replace=False)
        o = len(np.intersect1d(codes[i], codes[j])) / 32
        (same if s.concepts[i] == s.concepts[j] else diff).append(o)
    assert len(same) > 5 and np.mean(same) > np.mean(diff) + 0.1


@given(F=st.sampled_from([100, 800, 3000]), k=st.sampled_from([8, 32]), nf=st.integers(0, 90))
def test_bytes_per_event_depend_on_k_not_features(F, k, nf):
    enc = SparseEncoder(F, 1024, k, 2, 0)
    log = SpikeLog(k)
    f = np.sort(np.random.default_rng(nf).choice(F, min(nf, F), replace=False)).astype(np.int32)
    u, s = enc.encode(f)
    b = log.append(0, 0.0, u, q8(s), np.zeros(k, np.uint8))
    assert b == 8 + 4 * k


@given(times=st.lists(st.floats(0, 40), min_size=1, max_size=80), seed=st.integers(0, 10))
def test_rate_counters_match_bruteforce(times, seed):
    times = sorted(times); rng = np.random.default_rng(seed)
    N = 50; rc = RateCounters(N, window=10.0, n_sub=5)
    evs = []
    for t in times:
        u = np.sort(rng.choice(N, 4, replace=False)); rc.update(t, u); evs.append((t, u))
    bl = 2.0; head = int(times[-1] // bl)
    sl = np.zeros(N, int); nev = 0
    for t, u in evs:
        if head - 5 < int(t // bl) <= head:
            sl[u] += 1; nev += 1
    cnt, n = rc.sliding()
    assert np.array_equal(cnt, sl) and n == nev
    tot = np.zeros(N, int)
    for t, u in evs: tot[u] += 1
    assert np.array_equal(rc.total, tot)
    # tumbling: closed windows + current window partition the stream
    s = sum((c.sum() for _, c, _ in rc.closed), 0) + rc.cur.sum()
    assert s == tot.sum()


def test_layouts_store_identical_content():
    k = 8; rng = np.random.default_rng(1)
    logs = {n: SpikeLog(k, n) for n in SPIKE_LAYOUTS}
    rows = []
    for e in range(200):
        u = np.sort(rng.choice(300, k, replace=False)).astype(np.int32)
        r = (e * 3, float(e) * 0.5, u, rng.integers(0, 256, k).astype(np.uint8), rng.integers(0, 256, k).astype(np.uint8))
        rows.append(r)
        for lg in logs.values(): lg.append(*r)
    idx = rng.integers(0, 200, 50)
    ref = logs["aos"]
    for n, lg in logs.items():
        for a, b in zip(lg.gather(idx), ref.gather(idx)): assert np.array_equal(a, b), n
        for i in (0, 77, 199):
            got = lg.pattern(i)
            assert got[0] == rows[i][0] and got[1] == rows[i][1] and np.array_equal(got[2], rows[i][2])
        assert np.array_equal(lg.event_ids(), ref.event_ids()) and np.array_equal(lg.times(), ref.times())
        assert lg.store.field_sum("strength") == ref.store.field_sum("strength")
        for u in (3, 10, 150):
            a, b = lg.store.unit_history(u), ref.store.unit_history(u)
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_volley_crosses_slab_boundaries():
    from neurostore.layouts.volley import VolleyStore
    v = VolleyStore(4, V=4, tiles_per_slab=2)   # 8 events per slab
    for e in range(30):
        v.append(e, float(e), np.arange(4) + e, np.full(4, e % 256), np.zeros(4))
    assert v.n == 30 and len(v.slabs) == 4
    assert v.event(29)[0] == 29 and np.array_equal(v.event(17)[2], np.arange(4) + 17)
    assert list(v.event_ids()) == list(range(30))
