import numpy as np
from hypothesis import given, strategies as st
from neurostore.layouts import SPIKE_LAYOUTS, make_store, CSRGraph, DictGraph
from neurostore.layouts.volley import VolleyStore


@given(n=st.integers(1, 150), k=st.sampled_from([1, 4, 8]), seed=st.integers(0, 1000))
def test_spike_layouts_are_equivalent(n, k, seed):
    rng = np.random.default_rng(seed)
    stores = {nm: (VolleyStore(k, V=4, tiles_per_slab=3) if nm == "volley" else make_store(nm, k, cap_events=4)) for nm in SPIKE_LAYOUTS}
    stores["volley"].name = "volley"
    rows = []
    for e in range(n):
        u = np.sort(rng.choice(60, k, replace=False)).astype(np.int32)
        r = (int(rng.integers(0, 2 ** 31)), float(e) * 0.37, u, rng.integers(0, 256, k).astype(np.uint8), rng.integers(0, 256, k).astype(np.uint8))
        rows.append(r)
        for s in stores.values():
            s.append(*r)
    for s in stores.values():
        s.build_postings()
    idx = rng.integers(0, n, 20)
    ref = stores["aos"]
    for nm, s in stores.items():
        assert s.n == n
        for a, b in zip(s.gather(idx), ref.gather(idx)):
            assert np.array_equal(a, b), nm
        for i in (0, n // 2, n - 1):
            e = s.event(i)
            assert e[0] == rows[i][0] and np.isclose(e[1], np.float32(rows[i][1])) and np.array_equal(e[2], rows[i][2])
            assert np.array_equal(e[3], rows[i][3]) and np.array_equal(e[4], rows[i][4])
        assert np.array_equal(s.event_ids(), ref.event_ids()) and np.array_equal(s.times(), ref.times())
        for f in ("strength", "surprise", "unit"):
            assert s.field_sum(f) == ref.field_sum(f), (nm, f)
        for u in (0, 7, 33, 59):
            h0, h1, h2 = ref.unit_history(u), s.unit_history(u), s.unit_history_indexed(u)
            for h in (h1, h2):
                assert np.array_equal(h[0], h0[0]) and np.array_equal(h[1], h0[1]), (nm, u)
        assert np.array_equal(s.eids_at(idx), ref.eids_at(idx)) and np.array_equal(s.times_at(idx), ref.times_at(idx))


@given(seed=st.integers(0, 10_000), ne=st.integers(1, 80), lm=st.sampled_from([None, 0.2]))
def test_graph_layouts_are_equivalent(seed, ne, lm):
    rng = np.random.default_rng(seed); N = 16
    keys = rng.choice(N * N, min(ne, N * N), replace=False)
    pre, post = keys // N, keys % N
    w = rng.random(len(keys)); lag = rng.random(len(keys)) * 2; ep = rng.integers(0, 6, len(keys))
    csr = CSRGraph(N, pre, post, w, lag, ep); dg = DictGraph.from_csr(csr)
    dt = 0.9 ** np.arange(20); E = 7
    units = rng.choice(N, 5, replace=False); x = rng.random(5)
    def acc(g):
        wh, po, we = g.gather(units, dt, E, lm)
        return np.bincount(po, weights=we * x[wh], minlength=N)
    assert np.allclose(acc(csr), acc(dg), atol=1e-5)
    for a, b in zip(csr.edges(dt, E), dg.edges(dt, E)):
        assert np.allclose(a, b, atol=1e-6)
    for u in range(N):
        for a, b in zip(csr.out_edges(u), dg.out_edges(u)):
            assert np.array_equal(np.sort(a.astype(float)), np.sort(b.astype(float))) or np.allclose(a, b)
        ci, di = csr.in_edges(u), dg.in_edges(u)
        oc, od = np.argsort(ci[0], kind="stable"), np.argsort(di[0], kind="stable")
        assert np.array_equal(ci[0][oc], di[0][od]) and np.allclose(ci[1][oc], di[1][od])
