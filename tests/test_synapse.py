import numpy as np
from hypothesis import given, strategies as st
from neurostore.config import Config
from neurostore.store import NeuroStore
from neurostore.synapse import SynapseLattice

N = 12
op = st.one_of(st.just(("tick",)), st.just(("merge",)),
               st.tuples(st.just("add"), st.integers(0, 2 ** 31 - 1)))


def apply(ops, lam, R=None, prune=False):
    lat = SynapseLattice(N, lam=lam, prune_R=R, eps=0.0 if not prune else 1e-6)
    eager = np.zeros((N, N)); d = 1.0 - lam; first = True
    for o in ops:
        if o[0] == "tick":
            lat.tick(); eager *= d
        elif o[0] == "merge":
            lat.merge(prune=prune)
        else:
            rng = np.random.default_rng(o[1]); n = int(rng.integers(1, 20))
            pre, post = rng.integers(0, N, n), rng.integers(0, N, n); inc = rng.random(n).astype(np.float32)
            lat.add_delta(pre, post, inc, inc * 0.5)
            np.add.at(eager, (pre, post), inc.astype(np.float64))
    return lat, eager


@given(ops=st.lists(op, max_size=40), lam=st.sampled_from([0.01, 0.05, 0.2]))
def test_lazy_decay_equals_eager_decay(ops, lam):
    lat, eager = apply(ops, lam)
    assert np.allclose(lat.effective_matrix(), eager, rtol=1e-4, atol=1e-5)
    lat.merge(prune=False)
    assert np.allclose(lat.effective_matrix(), eager, rtol=1e-4, atol=1e-5)


@given(ops=st.lists(op, min_size=1, max_size=30), q=st.lists(st.integers(0, N - 1), min_size=1, max_size=6, unique=True),
       norm=st.booleans())
def test_reads_from_lattice_plus_delta_equal_merged_reads(ops, q, norm):
    lat, _ = apply(ops, 0.05)
    q = np.array(q); x = np.linspace(0.5, 1.0, len(q))
    a = lat.accumulate(q, x, normalize=norm); s1 = lat.spread(q, x, hops=2, keep=0, normalize=norm)
    lat.merge(prune=False)
    b = lat.accumulate(q, x, normalize=norm); s2 = lat.spread(q, x, hops=2, keep=0, normalize=norm)
    assert np.allclose(a, b, rtol=1e-4, atol=1e-6) and np.allclose(s1, s2, rtol=1e-3, atol=1e-6)


def test_prune_keeps_strongest_R_per_row():
    lat = SynapseLattice(N, lam=0.0, prune_R=3, eps=0.0)
    rng = np.random.default_rng(0)
    W = rng.random((N, N)); pre, post = np.nonzero(W > -1)
    lat.add_delta(pre, post, W[pre, post]); lat.merge()
    for u in range(N):
        p, w, _, _ = lat.csr.out_edges(u)
        assert len(p) == 3 and np.allclose(np.sort(w), np.sort(W[u])[-3:], atol=1e-6)


def test_transpose_is_consistent():
    lat = SynapseLattice(N, lam=0.0, prune_R=None)
    rng = np.random.default_rng(1); pre, post = rng.integers(0, N, 60), rng.integers(0, N, 60)
    lat.add_delta(pre, post, rng.random(60)); lat.merge(prune=False)
    out = {(int(u), int(v)): float(w) for u in range(N) for v, w in zip(*lat.csr.out_edges(u)[:2])}
    inn = {(int(u), int(v)): float(w) for v in range(N) for u, w in zip(*lat.in_edges(v)[:2])}
    assert out == inn


def test_spread_is_hop_bounded_and_decayed():
    lat = SynapseLattice(5, lam=0.0, prune_R=None)
    lat.add_delta([0, 1, 2], [1, 2, 3], [1.0, 1.0, 1.0]); lat.merge()
    s = lat.spread([0], hops=2, hop_decay=0.5, keep=0, normalize=False)
    assert np.allclose(s, [0, 0.5, 0.25, 0, 0])
    s3 = lat.spread([0], hops=3, hop_decay=0.5, keep=0, normalize=False)
    assert np.isclose(s3[3], 0.125)


def test_hebbian_kernel_favours_pre_before_post():
    cfg = Config(F=100, N=64, k=8, d=2, consolidate="off", asm_max=8, merge_every=1, prune_R=None, lam=0.0)
    s = NeuroStore(cfg)
    A, B = np.arange(0, 8), np.arange(20, 28)
    for e in range(40):                 # A then B, repeatedly, 1 time unit apart
        s.log.append(2 * e, 10.0 * e, A, np.full(8, 255, np.uint8), np.zeros(8, np.uint8))      # pairs spaced 10 apart:
        s.log.append(2 * e + 1, 10.0 * e + 1, B, np.full(8, 255, np.uint8), np.zeros(8, np.uint8))  # B->next A is outside the lag window
    s.cons.run_round(positions=np.arange(80))
    fwd = s.lattice.accumulate(A)[B].sum(); bwd = s.lattice.accumulate(B)[A].sum()
    assert fwd > 0 and bwd == 0         # only the pre-before-post direction is learned between A and B
    # typical lag of A->B edges is ~1 (B follows A by 1 time unit)
    p, _, lag, _ = s.lattice.csr.out_edges(0)
    assert np.isclose(float(lag[np.isin(p, B)].mean()), 1.0, atol=0.05)
