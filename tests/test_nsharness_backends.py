"""The five backends: property tests against brute force, internal consistency, and memory accounting from real bytes."""
import numpy as np
import pytest

from nsharness.backends import make_backend, BACKENDS
from nsharness.core import Split
from nsharness.pipeline import Pipeline


def clustered(n=500, d=24, clusters=12, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(clusters, d))
    X = C[rng.integers(0, clusters, n)] + noise * rng.normal(size=(n, d))
    X = ((X - X.mean(0)) / X.std(0)).astype(np.float32)
    return X


def cosine(Q, X):
    return (Q / np.linalg.norm(Q, axis=1, keepdims=True)) @ (X / np.linalg.norm(X, axis=1, keepdims=True)).T


# ---------------------------------------------------------------- exact
@pytest.mark.parametrize("k", [1, 5, 25])
def test_exact_equals_bruteforce(k):
    X = clustered(); Q = X[:30] + 0.2
    be = make_backend("exact").fit(X)
    idx, sim = be.search(Q, k)
    S = cosine(Q, X)
    assert np.allclose(sim, -np.sort(-S, axis=1)[:, :k], atol=1e-5)
    assert np.allclose(np.take_along_axis(S, idx, 1), sim, atol=1e-5)


def test_exact_chunking_does_not_change_results():
    X = clustered(); Q = X[:40] * 1.1
    a = make_backend("exact").fit(X)
    b = make_backend("exact").fit(X)
    b._rows = lambda: 7
    assert np.array_equal(a.search(Q, 5)[0], b.search(Q, 5)[0])


def test_exact_masked_cosine_uses_only_observed_features():
    rng = np.random.default_rng(1)
    X = clustered(); Q = X[:20] + 0.3 * rng.normal(size=(20, X.shape[1])).astype(np.float32)
    mask = rng.random(Q.shape) < 0.7
    mask[:, 0] = True
    be = make_backend("exact").fit(X)
    idx, sim = be.search(np.where(mask, Q, 0), 3, mask)
    for i in range(20):
        m = mask[i]
        brute = (X[:, m] @ Q[i, m]) / (np.linalg.norm(X[:, m], axis=1) * np.linalg.norm(Q[i, m]))
        assert sim[i, 0] == pytest.approx(brute.max(), abs=1e-5) and idx[i, 0] == brute.argmax()


def test_exact_l2_equals_bruteforce_including_masked():
    rng = np.random.default_rng(4)
    X = clustered(); Q = X[:25] + 0.3 * rng.normal(size=(25, X.shape[1])).astype(np.float32)
    be = make_backend("exact_l2").fit(X)
    idx, sim = be.search(Q, 6)
    msd = ((Q[:, None, :] - X[None, :, :]) ** 2).mean(-1)
    assert np.allclose(sim, -np.sort(msd, axis=1)[:, :6], atol=1e-4) and (idx[:, 0] == msd.argmin(1)).all()
    mask = rng.random(Q.shape) < 0.6
    mask[:, :2] = True
    idx, sim = be.search(np.where(mask, Q, 0), 3, mask)
    for i in range(25):
        m = mask[i]
        brute = ((X[:, m] - Q[i, m]) ** 2).mean(1)
        assert sim[i, 0] == pytest.approx(-brute.min(), abs=1e-4) and idx[i, 0] == brute.argmin()
    assert np.allclose(be.raw(idx[:, 0]), X[idx[:, 0]]) and be.index_bytes() == X.nbytes + 4 * len(X)
    assert be.reconstruct(Q, 1).shape == Q.shape


def test_weights_and_readout():
    X = clustered(); y = np.arange(len(X), dtype=np.float32)
    be = make_backend("exact").fit(X)
    assert be.readout(X[:5], y, 1)[0] == 0.0                                   # k=1 returns the item's own label
    assert be.readout(X[:3], y, len(X))[0] == pytest.approx(y.mean(), rel=1e-6)  # uniform over everything = the mean
    sharp = make_backend("exact", beta=1e4).fit(X)
    assert np.allclose(sharp.readout(X[:5], y, 10), y[:5])                     # a very sharp softmax is a 1-NN lookup
    w = be.weights(np.array([[0.9, 0.5, -np.inf]]))
    assert w[0, 2] == 0 and w[0, 0] == w[0, 1] == 1                            # uniform weights ignore padding
    w = make_backend("exact", beta=5.0).weights(np.array([[0.9, 0.5, -np.inf]]))
    assert w[0, 0] == 1.0 and 0 < w[0, 1] < 1 and w[0, 2] == 0


# ---------------------------------------------------------------- ANN
def test_ann_probing_every_list_is_exact():
    X = clustered(n=600); Q = X[:40] + 0.3
    ann = make_backend("ann", nlist=20, nprobe=20).fit(X, 0)
    ex = make_backend("exact").fit(X)
    ia, sa = ann.search(Q, 8); ie, se = ex.search(Q, 8)
    assert np.allclose(sa, se, atol=1e-5)
    assert all(set(a) == set(b) for a, b in zip(ia.tolist(), ie.tolist()))


def test_ann_l2_probing_every_list_equals_exact_l2_and_counts_norm_bytes():
    X = clustered(n=600); Q = X[:40] + 0.3
    ann = make_backend("ann", nlist=20, nprobe=20, metric="l2").fit(X, 0)
    ex = make_backend("exact_l2").fit(X)
    ia, sa = ann.search(Q, 8); ie, se = ex.search(Q, 8)
    assert np.allclose(sa, se, atol=1e-4) and all(set(a) == set(b) for a, b in zip(ia.tolist(), ie.tolist()))
    assert ann.index_bytes() == ann.C.nbytes + ann.ptr.nbytes + ann.order.nbytes + ann.Xl.nbytes + ann.xsq.nbytes
    q8 = make_backend("ann", nlist=20, nprobe=20, metric="l2", sq8=True).fit(X, 0)
    i8, s8 = q8.search(Q, 8)
    assert np.mean([len(set(a) & set(b)) / 8 for a, b in zip(i8.tolist(), ie.tolist())]) > 0.85 and q8.index_bytes() < ann.index_bytes()
    short = make_backend("ann", nlist=30, nprobe=1, metric="l2").fit(clustered(n=60), 0)
    idx, sim = short.search(X[:5], 10)
    assert (idx >= 0).all() and np.isfinite(sim).all()


def test_ann_recall_grows_with_nprobe_and_sq8_costs_little():
    X = clustered(n=1500, noise=1.0); Q = X[:60] + 0.5
    ex = make_backend("exact").fit(X)
    ie, _ = ex.search(Q, 10)
    rec = {}
    for npb in (1, 4, 16):
        ia, _ = make_backend("ann", nlist=32, nprobe=npb).fit(X, 0).search(Q, 10)
        rec[npb] = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ia.tolist(), ie.tolist())])
    assert rec[1] <= rec[4] <= rec[16] and rec[16] > 0.9 and rec[1] < rec[16]
    i8, _ = make_backend("ann", nlist=32, nprobe=32, sq8=True).fit(X, 0).search(Q, 10)
    assert np.mean([len(set(a) & set(b)) / 10 for a, b in zip(i8.tolist(), ie.tolist())]) > 0.9


def test_ann_never_returns_padding():
    X = clustered(n=60)
    ann = make_backend("ann", nlist=30, nprobe=1).fit(X, 0)                   # lists hold ~2 items; k=10 forces the fallback path
    idx, sim = ann.search(X[:10], 10)
    assert (idx >= 0).all() and np.isfinite(sim).all() and all(len(set(r)) == 10 for r in idx.tolist())


def test_ann_is_deterministic_per_seed():
    X = clustered()
    a = make_backend("ann", nprobe=2).fit(X, 3).search(X[:20], 5)[0]
    assert np.array_equal(a, make_backend("ann", nprobe=2).fit(X, 3).search(X[:20], 5)[0])


# ---------------------------------------------------------------- FlyHash
def test_flyhash_code_and_overlap_search_match_bruteforce():
    X = clustered(n=400); Q = X[:25] + 0.3
    be = make_backend("flyhash", m=512, frac=0.05).fit(X, 1)
    assert be.k_act == round(0.05 * 512)
    codes = be._encode(X)
    assert codes.shape == (400, be.k_act) and (np.diff(codes, axis=1) > 0).all() and codes.min() >= 0 and codes.max() < 512
    bit = np.zeros((400, 512), np.int32); bit[np.arange(400)[:, None], codes] = 1
    qc = be._encode(Q)
    qb = np.zeros((25, 512), np.int32); qb[np.arange(25)[:, None], qc] = 1
    overlap = qb @ bit.T
    idx, sim = be.search(Q, 6)
    for i in range(25):
        assert np.allclose(sim[i], overlap[i, idx[i]] / be.k_act)
        assert sorted(overlap[i, idx[i]], reverse=True) == list(overlap[i, idx[i]])
        assert overlap[i, idx[i]].min() >= np.sort(overlap[i])[-6]            # the returned items really are the 6 best overlaps


def test_flyhash_retrieves_itself_and_prefers_near_neighbours():
    rng = np.random.default_rng(0)
    X = clustered(n=800, d=32)
    be = make_backend("flyhash", m=1024).fit(X, 0)
    idx, _ = be.search(X[:50], 1)
    assert (idx[:, 0] == np.arange(50)).mean() > 0.95
    near = be.search(X[:50] + 0.2 * rng.normal(size=(50, 32)).astype(np.float32), 1)[0][:, 0]
    assert (near == np.arange(50)).mean() > 0.7


def test_flyhash_is_deterministic_and_entropy_bound_is_below_bitmap():
    X = clustered()
    a, b = make_backend("flyhash").fit(X, 5), make_backend("flyhash").fit(X, 5)
    assert np.array_equal(a.post, b.post) and np.array_equal(a.search(X[:9], 4)[0], b.search(X[:9], 4)[0])
    assert not np.array_equal(a.post, make_backend("flyhash").fit(X, 6).post)
    rep = a.memory_report()
    # at 5 % activity a 16-bit index list (0.8 bit per unit) is smaller than a bitmap (1 bit per unit); the entropy bound is far below both
    assert rep["alt:entropy_bound"] < rep["alt:index_list_u16"] < rep["alt:bitmap_packed"]


# ---------------------------------------------------------------- Hopfield
def test_hopfield_limits_and_full_attention():
    X = clustered(); y = np.arange(len(X), dtype=np.float32)
    hot = make_backend("hopfield", beta=1e4).fit(X)
    assert np.allclose(hot.readout(X[:6], y, 1), y[:6])                         # beta -> infinity: recall of the best pattern
    cold = make_backend("hopfield", beta=0.0).fit(X)
    assert np.allclose(cold.readout(X[:3], y, 1), y.mean(), rtol=1e-5)          # beta = 0: the mean of all stored labels
    topk = make_backend("hopfield", beta=20.0, full_attention=False).fit(X)
    ex = make_backend("exact", beta=20.0).fit(X)
    assert np.allclose(topk.readout(X[:6], y, 5), ex.readout(X[:6], y, 5))


def test_hopfield_l2_attention_is_a_gaussian_kernel_regression():
    X = clustered(); y = np.arange(len(X), dtype=np.float32)
    be = make_backend("hopfield", beta=2.0, metric="l2").fit(X)
    Q = X[:4] + 0.2
    w = np.exp(-2.0 * ((Q[:, None, :] - X[None, :, :]) ** 2).mean(-1))
    assert np.allclose(be.readout(Q, y, 1), (w * y).sum(1) / w.sum(1), rtol=1e-4)
    assert np.allclose(make_backend("hopfield", beta=1e5, metric="l2").fit(X).readout(X[:5], y, 1), y[:5])
    mask = np.ones((4, X.shape[1]), bool); mask[:, ::3] = False
    R = make_backend("hopfield", beta=20.0, metric="l2").fit(X).reconstruct(np.where(mask, Q, 0).astype(np.float32), 5, mask)
    assert np.allclose(R[mask], Q[mask])


def test_hopfield_completion_clamps_observed_and_fills_missing():
    rng = np.random.default_rng(2)
    X = clustered(n=800, d=30)
    T = X[:40]; mask = rng.random(T.shape) > 0.3
    Q = np.where(mask, T, 0).astype(np.float32)
    be = make_backend("hopfield", beta=30.0, steps=2).fit(X)
    R = be.reconstruct(Q, 5, mask)
    assert np.allclose(R[mask], Q[mask])
    err = ((R - T)[~mask] ** 2).mean()
    assert err < 0.5 * (T[~mask] ** 2).mean()                                    # clearly better than mean imputation


def test_hopfield_energy_ranks_stored_above_noise():
    X = clustered(); rng = np.random.default_rng(777)           # not the generator's seed: that would replay the cluster centres
    be = make_backend("hopfield", beta=20.0).fit(X)
    fam_in, fam_out = be.familiarity(X[:50]), be.familiarity(rng.normal(size=(50, X.shape[1])).astype(np.float32))
    assert {"max_sim", "margin", "mean_top10", "neg_energy"} <= set(fam_in)
    assert fam_in["neg_energy"].min() > fam_out["neg_energy"].max()
    assert (fam_in["neg_energy"] >= fam_in["max_sim"] - 1e-6).all()              # lse/beta >= max


# ---------------------------------------------------------------- VSA
def test_vsa_level_codebook_is_locality_preserving():
    be = make_backend("vsa", D=2048, levels=9).fit(clustered(n=50, d=6), 0)
    L = be._lev.astype(int)
    ham = lambda i, j: int((L[i] != L[j]).sum())
    assert ham(0, 8) == 1024 and ham(0, 4) == 512 and ham(2, 6) == 512 and ham(3, 4) == 128
    assert ham(0, 1) < ham(0, 2) < ham(0, 8) + 1


def test_vsa_hamming_search_matches_bruteforce_and_finds_itself():
    X = clustered(n=300, d=20); be = make_backend("vsa", D=2048).fit(X, 0)
    bits = np.unpackbits(be.H.view(np.uint8), axis=1, bitorder="little")
    q = be._encode(be._levels(X[:15] + 0.2))
    qbits = np.unpackbits(q.view(np.uint8), axis=1, bitorder="little")
    ham = (qbits[:, None, :] != bits[None, :, :]).sum(-1)
    idx, sim = be.search(X[:15] + 0.2, 4)
    assert np.allclose(sim, 1 - 2 * np.take_along_axis(ham, idx, 1) / 2048)
    assert np.allclose(sim[:, 0], 1 - 2 * ham.min(1) / 2048)
    assert (be.search(X[:40], 1)[0][:, 0] == np.arange(40)).mean() > 0.95


def test_vsa_masked_encoding_bundles_only_observed_features():
    X = clustered(n=40, d=9); be = make_backend("vsa", D=1024).fit(X, 0)
    lvl = be._levels(X[:6]); rng = np.random.default_rng(0)
    mask = rng.random((6, 9)) > 0.4; mask[:, :3] = True
    got = np.unpackbits(be._encode(lvl, mask).view(np.uint8), axis=1, bitorder="little")
    for i in range(6):
        cnt = sum(be._table[j][lvl[i, j]].astype(int) for j in range(9) if mask[i, j])
        nobs = mask[i].sum()
        want = np.where(2 * cnt > nobs, 1, np.where(2 * cnt < nobs, 0, be._tie_bits))
        assert np.array_equal(got[i], want)


def test_vsa_decode_recovers_stored_levels():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(1, 10)).astype(np.float32).clip(-2.9, 2.9)
    be = make_backend("vsa", D=8192, levels=32).fit(X, 0)
    R = be.reconstruct(X, 1)
    step = 6.0 / 31
    assert np.abs(R - X).max() <= 2 * step                                      # within two quantisation steps for every feature


# ---------------------------------------------------------------- accounting and registry
def test_memory_reports_equal_real_array_bytes():
    X = clustered(n=300, d=20)
    ex = make_backend("exact").fit(X)
    assert ex.index_bytes() == ex.Xn.nbytes + ex.norm.nbytes == 300 * 20 * 4 + 300 * 4
    hf = make_backend("hopfield").fit(X)
    assert hf.index_bytes() == ex.index_bytes()
    a32 = make_backend("ann", nlist=10).fit(X, 0); a8 = make_backend("ann", nlist=10, sq8=True).fit(X, 0)
    assert a32.index_bytes() == a32.C.nbytes + a32.ptr.nbytes + a32.order.nbytes + a32.Xl.nbytes
    assert a8.index_bytes() == a8.C.nbytes + a8.ptr.nbytes + a8.order.nbytes + a8.Xq.nbytes + a8.scale.nbytes
    assert a8.Xq.nbytes == 300 * 20 and a8.index_bytes() < a32.index_bytes()
    fl = make_backend("flyhash", m=256).fit(X, 0)
    assert fl.index_bytes() == fl.post.nbytes + fl.ptr.nbytes + fl.proj_idx.nbytes
    assert fl.post.nbytes == 300 * fl.k_act * 4
    vs = make_backend("vsa", D=1024, levels=16).fit(X, 0)
    assert vs.index_bytes() == vs.H.nbytes + 16 * 1024 // 8 + 20 * 1024 // 8 + 1024 // 8 and vs.H.nbytes == 300 * 1024 // 8
    for be in (ex, hf, a32, a8, fl, vs):
        rep = be.memory_report()
        assert all(isinstance(v, int) and v >= 0 for v in rep.values())
        assert be.index_bytes() == sum(v for k, v in rep.items() if not k.startswith(("alt:", "cache:")))


def test_pipeline_end_to_end_and_report():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(300, 10, 4)).astype(np.float32); y = Z[:, -1, 0] * 10
    mem = Split(Z, y, np.arange(300))
    for name in BACKENDS:
        cfg = {"nprobe": 4} if name == "ann" else ({"D": 1024} if name == "vsa" else {})
        pipe = Pipeline(name, cfg, {"kind": "stats"}, k=5).fit(mem, 0)
        assert pipe.predict(Z[:7]).shape == (7,)
        rep = pipe.memory_report()
        assert rep["labels_f32"] == 300 * 4 and pipe.deployed_bytes() == sum(v for k, v in rep.items() if ".alt:" not in k and ".cache:" not in k)
        fam = pipe.familiarity(Z[:7])
        assert all(len(v) == 7 for v in fam.values()) and {"max_sim", "margin", "mean_top10"} <= set(fam)
    with pytest.raises(KeyError):
        make_backend("lsh")
