"""VSA algebras, the role-filler record store (Workload B) and the holographic graph (Workload D)."""
import numpy as np
import pytest

from neurostore import theory
from neurostore.common import make_rng, random_bipolar
from neurostore.primitives import HolographicGraphMemory, HRRAlgebra, MAPAlgebra, VSARecordStore
from neurostore.workloads.d_graph_walk import make_graph, make_walks


# ----------------------------------------------------------------------------- algebras
@pytest.mark.parametrize("Alg", [MAPAlgebra, HRRAlgebra])
def test_bind_unbind_is_exact_and_binding_decorrelates(Alg):
    a = Alg(2048, seed=1)
    x, y = a.random_vectors(2)
    bound = a.bind(x, y)
    assert float(a.cosine(a.unbind(bound, x), y)[0, 0]) == pytest.approx(1.0, abs=1e-4)
    assert abs(float(a.cosine(bound, x)[0, 0])) < 0.1 and abs(float(a.cosine(bound, y)[0, 0])) < 0.1


@pytest.mark.parametrize("Alg", [MAPAlgebra, HRRAlgebra])
def test_permutation_is_invertible_and_decorrelating(Alg):
    a = Alg(1024, seed=2)
    x = a.random_vectors(1)[0]
    assert np.allclose(a.inverse_permute(a.permute(x, 3), 3), x)
    assert abs(float(a.cosine(a.permute(x), x)[0, 0])) < 0.15


def test_hrr_vectors_are_unitary():
    v = HRRAlgebra(1000, seed=0).random_vectors(5)
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert np.allclose(np.abs(np.fft.rfft(v, axis=1)), 1, atol=1e-4)


def test_cyclic_shift_commutes_with_circular_convolution_but_a_random_permutation_does_not():
    """Why direction needs a random permutation: for HRR a cyclic shift is a convolution with a delta,
    so it cannot make a binding order-sensitive."""
    a = HRRAlgebra(512, seed=3)
    x, y = a.random_vectors(2)
    assert np.allclose(np.roll(a.bind(x, y), 1), a.bind(np.roll(x, 1), y), atol=1e-5)  # commutes
    assert abs(float(a.cosine(a.bind(a.permute(x), y), a.bind(x, a.permute(y)))[0, 0])) < 0.2  # random permutation: does not


def test_map_bundle_resembles_its_parts_and_binarisation_breaks_ties_without_bias():
    a = MAPAlgebra(8192, seed=4)
    parts = a.random_vectors(3)
    b = a.bundle(parts, binarize=False)
    assert all(float(a.cosine(b, p)[0, 0]) > 0.5 for p in parts)
    even = a.bundle(a.random_vectors(2))  # 2 terms -> ~50% of positions tie
    assert set(np.unique(even)) == {-1.0, 1.0}
    ties = a.random_vectors(2)
    summed = ties.sum(axis=0)
    out = a.binarize(summed)
    tie_pos = summed == 0
    assert abs(out[tie_pos].mean()) < 0.1  # the reference's "ties -> +1" would give exactly +1.0


# ----------------------------------------------------------------------------- record store
def _records(rng, m=400, f=6, v=80):
    return rng.integers(0, v, (m, f)), [v] * f


def _cues(rng, recs, k, q):
    src = rng.integers(0, recs.shape[0], q)
    f = recs.shape[1]
    flds = np.stack([rng.permutation(f)[:k] for _ in range(q)])
    vals = np.take_along_axis(recs[src], flds, axis=1)
    tgt = np.array([rng.choice(np.setdiff1d(np.arange(f), fl)) for fl in flds])
    return src, flds, vals, tgt


def test_two_stage_lookup_identifies_and_reconstructs(rng):
    recs, vocab = _records(rng)
    store = VSARecordStore(vocab, dim=2048, seed=1)
    store.add_records(recs)
    src, flds, vals, tgt = _cues(rng, recs, 3, 150)
    out = store.reconstruct(flds, vals, tgt)
    ok_id = (out["ids"] == src) | (recs[out["ids"]][np.arange(150)[:, None], flds] == vals).all(axis=1)  # ties on non-unique cues are fine
    assert ok_id.mean() > 0.98
    assert (out["values"] == recs[src, tgt]).mean() > 0.97
    assert out["unbound"].shape == (150, 2048)


def test_accuracy_improves_with_dimension_and_with_more_known_fields(rng):
    recs, vocab = _records(rng, m=600, v=200)

    def acc(dim, k):
        store = VSARecordStore(vocab, dim=dim, seed=7)
        store.add_records(recs)
        r2 = make_rng(99)
        src, flds, vals, tgt = _cues(r2, recs, k, 200)
        return (store.identify(flds, vals)[0] == src).mean()

    assert acc(128, 3) < acc(2048, 3)
    assert acc(1024, 2) < acc(1024, 5) + 1e-9
    assert acc(2048, 4) > 0.97


def test_a_corrupted_cue_field_degrades_but_does_not_destroy_identification(rng):
    recs, vocab = _records(rng, m=600, v=200)
    store = VSARecordStore(vocab, dim=4096, seed=3)
    store.add_records(recs)
    src, flds, vals, _ = _cues(rng, recs, 4, 200)
    bad = vals.copy()
    bad[:, 0] = (bad[:, 0] + 1) % 200
    ident = (store.identify(flds, bad)[0] == src).mean()
    assert ident > 0.9  # 3 of 4 clues intact


def test_REGRESSION_original_harness_unbound_a_role_that_is_not_in_the_cue(rng):
    """The original Workload B bundled fields {dept, title, loc} and unbound the ID role: no information."""
    dim, trials = 4096, 80
    cos_ref = []
    for t in range(trials):
        vsa = MAPAlgebra(dim, seed=t)
        role_id, role_dept, role_title, role_loc = vsa.random_vectors(4)
        val_id, val_dept, val_title, val_loc = vsa.random_vectors(4)
        partial_cue = vsa.bundle([vsa.bind(role_dept, val_dept), vsa.bind(role_title, val_title), vsa.bind(role_loc, val_loc)])
        cos_ref.append(float(np.dot(vsa.unbind(partial_cue, role_id), val_id)) / dim)
    cos_ref = np.array(cos_ref)
    assert abs(cos_ref.mean()) < 4 * (1 / np.sqrt(dim)) / np.sqrt(trials) + 1e-3  # unbiased noise
    assert cos_ref.std() == pytest.approx(1 / np.sqrt(dim), rel=0.25)
    assert (cos_ref > 0.6).sum() == 0  # the original success criterion can never be met


def test_REGRESSION_two_stage_recovers_the_missing_field_where_the_original_method_cannot(rng):
    dim, f, v = 4096, 5, 100
    recs = rng.integers(0, v, (50, f))
    store = VSARecordStore([v] * f, dim=dim, seed=5)
    store.add_records(recs)
    src = rng.integers(0, 50, 60)
    flds = np.tile([1, 2, 3], (60, 1))  # cue: dept, title, loc -- omit the ID field 0 and salary field 4
    vals = recs[src][:, [1, 2, 3]]
    out = store.reconstruct(flds, vals, np.zeros(60, dtype=int))
    true_fill = store.filler(np.zeros(60, dtype=int), recs[src, 0])
    cos = (out["unbound"] * true_fill).sum(axis=1) / dim
    assert cos.mean() > 0.3  # ~0.375 expected for a 5-term sign bundle
    assert (out["values"] == recs[src, 0]).mean() > 0.99


def test_record_store_memory_accounting(rng):
    recs, vocab = _records(rng, m=100, v=50)
    store = VSARecordStore(vocab, dim=512, seed=1)
    store.add_records(recs)
    assert store.record_bytes() == 100 * 512 * 4
    assert store.memory_bytes() > store.record_bytes()
    assert store.ideal_packed_bytes() < store.memory_bytes() / 20
    store.add_records(recs[:10])
    assert store.records.shape == (110, 512)


# ----------------------------------------------------------------------------- graph generator
def test_make_graph_invariants(rng):
    e, r, t = 40, 6, 120
    tri = make_graph(rng, e, r, t)
    assert tri.shape == (t, 3)
    assert len({(s, rr) for s, rr, _ in tri}) == t  # (subject, relation) unique
    assert set(range(e)) <= set(tri[:, 0])  # every entity has an out-edge
    assert (tri[:, 0] != tri[:, 2]).all()  # no self loops
    assert tri[:, 1].max() < r and tri[:, 2].max() < e
    with pytest.raises(ValueError):
        make_graph(rng, e, r, e - 1)
    with pytest.raises(ValueError):
        make_graph(rng, e, r, e * r + 1)


def test_make_walks_only_follow_existing_edges(rng):
    e, r = 30, 5
    tri = make_graph(rng, e, r, 70)
    edges = {(int(s), int(rr)): int(o) for s, rr, o in tri}
    start, rels, objs = make_walks(rng, tri, e, 200, 4)
    cur = start.copy()
    for h in range(4):
        assert all(edges[(int(c), int(rr))] == int(o) for c, rr, o in zip(cur, rels[:, h], objs[:, h]))
        cur = objs[:, h]


# ----------------------------------------------------------------------------- holographic graph
@pytest.mark.parametrize("alg", ["map", "hrr"])
def test_walks_are_exact_far_below_capacity(alg, rng):
    e, r = 60, 16
    tri = make_graph(rng, e, r, 90)
    g = HolographicGraphMemory(e, r, dim=4096, algebra=alg, seed=3)
    g.add_triples(tri)
    nxt, cos = g.hop(tri[:, 0], tri[:, 1])
    assert (nxt == tri[:, 2]).all()
    assert cos.mean() == pytest.approx(1 / np.sqrt(90), rel=0.2)  # signal ~ 1/sqrt(T)
    start, rels, objs = make_walks(rng, tri, e, 100, 4)
    assert (g.walk(start, rels) == objs).all()


@pytest.mark.parametrize("alg", ["map", "hrr"])
def test_edge_direction_is_encoded(alg, rng):
    e, r = 80, 16
    tri = make_graph(rng, e, r, 100)
    g = HolographicGraphMemory(e, r, dim=4096, algebra=alg, seed=4)
    g.add_triples(tri)
    rev, _ = g.hop(tri[:, 2], tri[:, 1])  # ask "who points to o via r" as if o were a subject
    legit = np.array([any((tri[:, 0] == o) & (tri[:, 1] == rr) & (tri[:, 2] == s)) for s, rr, o in tri])
    assert (rev[~legit] == tri[~legit, 0]).mean() < 0.05


@pytest.mark.slow
@pytest.mark.parametrize("alg,tol", [("map", 0.04), ("hrr", 0.05)])
def test_graph_accuracy_tracks_the_closed_form_capacity_law(alg, tol):
    e, r, dim, t = 100, 32, 2048, 200
    accs = []
    for gi in range(12):
        rng = make_rng([8, gi])
        tri = make_graph(rng, e, r, t)
        g = HolographicGraphMemory(e, r, dim, alg, seed=gi)
        g.add_triples(tri)
        accs.append((g.hop(tri[:, 0], tri[:, 1])[0] == tri[:, 2]).mean())
    assert abs(np.mean(accs) - theory.holographic_p_correct(dim, t, e)) < tol  # HRR sits ~1-2 points below the Gaussian model


def test_accuracy_falls_monotonically_as_triples_are_added(rng):
    e, r = 100, 32
    accs = []
    for t in (100, 300, 900):
        tri = make_graph(make_rng([1, t]), e, r, t)
        g = HolographicGraphMemory(e, r, 1024, "map", seed=1)
        g.add_triples(tri)
        accs.append((g.hop(tri[:, 0], tri[:, 1])[0] == tri[:, 2]).mean())
    # closed form at D=1024, E=100: T=100 -> 0.74, T=300 -> ~0.25, T=900 -> ~0.1
    assert accs[0] > accs[1] > accs[2] and accs[0] > 0.6 and accs[2] < 0.2
    assert accs[0] == pytest.approx(theory.holographic_p_correct(1024, 100, 100), abs=0.12)


def test_incremental_and_bulk_insertion_build_the_same_memory(rng):
    e, r = 40, 8
    tri = make_graph(rng, e, r, 80)
    for alg in ("map", "hrr"):
        a, b = HolographicGraphMemory(e, r, 512, alg, seed=2), HolographicGraphMemory(e, r, 512, alg, seed=2)
        a.add_triples(tri)
        b.add_triples(tri[:30])
        b.add_triples(tri[30:])
        assert np.allclose(a.memory, b.memory, atol=1e-4) and a.n_triples == b.n_triples == 80


def test_graph_memory_accounting_and_validation():
    g = HolographicGraphMemory(50, 4, 256, "map", seed=0)
    assert g.trace_bytes() == 256 * 4 and g.memory_bytes() == 256 * 4 + 54 * 256 * 4
    with pytest.raises(ValueError):
        HolographicGraphMemory(10, 2, 64, "quaternion")
