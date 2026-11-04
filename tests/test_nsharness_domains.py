"""Domain adapters: the synthetic world validates the harness against theory; the C-MAPSS adapter is tested on fake files with the real format."""
from pathlib import Path
import numpy as np
import pytest

from nsharness.domains import synthetic, cmapss
from nsharness.pipeline import Pipeline
from nsharness.stats import analytic_flat_recall, wilson_ci


# ---------------------------------------------------------------- the harness reproduces theory
@pytest.mark.parametrize("D,N,p", [(256, 512, 0.35), (256, 512, 0.40), (512, 512, 0.40), (1024, 1024, 0.44), (128, 256, 0.30)])
def test_exact_pipeline_matches_closed_form_recall(D, N, p):
    dom = synthetic.pm1_world(D, N, p, n_queries=4000, seed=11)
    pipe = Pipeline("exact", {}, {"kind": "flatten", "standardize": False}, k=1).fit(dom.memory, 0)
    q = dom.queries["queries"]
    top1 = pipe.neighbors(q.Z, k=1)[0][:, 0]
    hits = int((top1 == q.y).sum())
    lo, hi = wilson_ci(hits, len(q), z=3.5)
    assert lo <= analytic_flat_recall(D, N, p) <= hi, (hits / len(q), analytic_flat_recall(D, N, p))


def test_pm1_world_is_deterministic_and_flip_rate_is_right():
    a, b = synthetic.pm1_world(64, 32, 0.25, 500, 3), synthetic.pm1_world(64, 32, 0.25, 500, 3)
    assert np.array_equal(a.queries["queries"].Z, b.queries["queries"].Z)
    q = a.queries["queries"]
    flips = (q.Z[:, 0, :] != a.memory.Z[q.y, 0, :]).mean()
    assert flips == pytest.approx(0.25, abs=0.01)


def test_blob_world_shapes_and_query_locality():
    X, Q = synthetic.blob_world(2000, 16, 10, 0.5, 50, 0)
    assert X.shape == (2000, 16) and Q.shape == (50, 16) and X.dtype == np.float32
    d = ((Q[:, None, :] - X[None, :, :]) ** 2).sum(-1).min(1)
    assert np.median(d) < 16 * 0.5 ** 2 * 0.6                       # queries are within ~noise of a stored point


# ---------------------------------------------------------------- C-MAPSS adapter on fake files
OPS = np.array([[0, 0, 100], [10, 0.25, 100], [20, 0.7, 100], [25, 0.62, 60], [35, 0.84, 100], [42, 0.84, 100]], float)


def fake_table(rng, n_eng, lens, regimes, shift=0.0):
    rows = []
    for u in range(1, n_eng + 1):
        L = lens[u - 1]
        t = np.arange(1, L + 1)
        reg = rng.integers(0, regimes, L)
        ops = OPS[reg] + rng.normal(0, 0.002, (L, 3)) if regimes > 1 else np.column_stack([rng.normal(0, 0.002, L), rng.normal(0, 0.0003, L), np.full(L, 100.0)])
        S = np.zeros((L, 21))
        for j in range(21):
            if j in (0, 4, 9):                         # constant sensors (s1, s5, s10)
                S[:, j] = 100.0 + j
            elif j == 5:                               # s6: two distinct values
                S[:, j] = 21.6 + 0.01 * (rng.random(L) > 0.5)
            elif j == 3:                               # s4: only 5 distinct values
                S[:, j] = rng.integers(0, 5, L) * 0.1 + 1.0
            else:
                off = 50.0 * reg if regimes > 1 else 0.0
                S[:, j] = j + shift + off + (t / L) ** 2 * (1 + 0.3 * j) + 0.05 * rng.normal(size=L)
        rows.append(np.column_stack([np.full(L, u), t, ops, S]))
    return np.concatenate(rows)


def write_fake(root: Path, fd="FD001", n_train=7, n_test=5, regimes=1, seed=0, test_shift=0.0):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    train = fake_table(rng, n_train, rng.integers(40, 70, n_train), regimes)
    tl = rng.integers(10, 50, n_test)                                           # test engines are truncated (some shorter than the window)
    test = fake_table(rng, n_test, tl, regimes, shift=test_shift)
    rul = rng.integers(5, 140, n_test)
    for name, A in (("train", train), ("test", test)):
        np.savetxt(root / f"{name}_{fd}.txt", A, fmt="%.6f")
    np.savetxt(root / f"RUL_{fd}.txt", rul, fmt="%d")
    return train, test, rul


@pytest.fixture
def fake1(tmp_path):
    train, test, rul = write_fake(tmp_path, "FD001")
    return tmp_path, train, test, rul


def test_windows_padding_targets_and_labels(fake1):
    root, train, test, rul = fake1
    dom = cmapss.build_domain(root, "FD001", window=8, cap=20)
    mem, q, prep = dom.memory, dom.queries["test_last"], dom.info["prep"]
    assert mem.Z.shape == (len(train), 8, len(prep.sensors)) and dom.info["n_train_engines"] == 7
    for u in (1, 4):
        rows = np.where(mem.groups == u)[0]
        L = len(rows)
        assert list(mem.meta["t"][rows]) == list(range(1, L + 1))
        assert np.array_equal(mem.y[rows], np.minimum(L - np.arange(1, L + 1), 20))
        zr, _ = cmapss.apply_prep(train[train[:, 0] == u], prep)
        assert np.allclose(mem.Z[rows[0]], zr[0][None, :].repeat(8, 0))                    # first window: first row repeated (left pad)
        assert np.allclose(mem.Z[rows[10], -1], zr[10]) and np.allclose(mem.Z[rows[10], 0], zr[3])   # window ends at cycle 11, starts at 4
    assert len(q) == 5 and np.array_equal(q.y, rul.astype(float)) and np.array_equal(q.meta["y_capped"], np.minimum(rul, 20))
    last_rows = [test[test[:, 0] == u][-1] for u in range(1, 6)]
    zt, _ = cmapss.apply_prep(np.array(last_rows), prep)
    assert np.allclose(q.Z[:, -1], zt)                                                    # the query is the engine's FINAL window
    assert mem.y.max() == 20 and q.meta["t"].tolist() == [len(test[test[:, 0] == u]) for u in range(1, 6)]


def test_constant_and_low_resolution_channels_are_dropped_from_training_data_only(fake1):
    root, *_ = fake1
    dom = cmapss.build_domain(root, "FD001", window=8, cap=20)
    names = dom.info["channels"]
    assert not {"s1", "s5", "s10", "s6", "s4"} & set(names) and "s2" in names and len(names) == 21 - 5
    other = root / "other"
    write_fake(other, "FD001", seed=0, test_shift=5.0)                                  # same train file, a very different test file
    dom2 = cmapss.build_domain(other, "FD001", window=8, cap=20)
    p1, p2 = dom.info["prep"], dom2.info["prep"]
    assert np.array_equal(p1.sensors, p2.sensors) and np.array_equal(p1.mu, p2.mu) and np.array_equal(p1.sd, p2.sd)
    assert np.array_equal(dom.memory.Z, dom2.memory.Z) and not np.array_equal(dom.queries["test_last"].Z, dom2.queries["test_last"].Z)


def test_operating_regimes_are_recovered_and_normalised_separately(tmp_path):
    write_fake(tmp_path, "FD002", n_train=12, n_test=4, regimes=6)
    dom = cmapss.build_domain(tmp_path, "FD002", window=6, cap=30)
    mem, prep = dom.memory, dom.info["prep"]
    assert prep.centers.shape == (6, 3) and sorted(np.unique(mem.meta["regime"])) == list(range(6))
    last = mem.Z[:, -1, :]
    for r in range(6):
        sel = last[mem.meta["regime"] == r]
        assert np.allclose(sel.mean(0), 0, atol=0.25) and np.allclose(sel.std(0), 1, atol=0.3)
    assert np.abs(last).max() < 6                                                          # without per-regime scaling the +50/regime offsets would explode


def test_foreign_split_goes_through_the_reference_preprocessing(tmp_path):
    ref, other = tmp_path / "ref", tmp_path / "other"
    write_fake(ref, "FD001", seed=0)
    write_fake(other, "FD003", seed=9)
    dom = cmapss.build_domain(ref, "FD001", window=8, cap=20)
    prep = dom.info["prep"]
    # place the FD003 file next to the reference so a single root serves both
    for f in ("train_FD003.txt",):
        (ref / f).write_bytes((other / f).read_bytes())
    fs = cmapss.foreign_split(ref, "FD003", prep, 8, 20, stride=5)
    A = cmapss.read_table(ref / "train_FD003.txt")
    assert fs.Z.shape[2] == len(prep.sensors) and fs.Z.shape[1] == 8 and len(fs) < len(A)
    first_engine_row0 = A[A[:, 0] == 1][0]
    z0, _ = cmapss.apply_prep(first_engine_row0[None, :], prep)
    assert fs.meta["t"][0] == 1 and np.allclose(fs.Z[0, -1], z0[0])                         # window 1 ends on the engine's first row


@pytest.mark.parametrize("breakage,match", [("cycles", "cycles"), ("rul", "RUL labels"), ("nan", "finite"), ("cols", "finite")])
def test_loader_rejects_malformed_files(tmp_path, breakage, match):
    train, test, rul = write_fake(tmp_path, "FD001")
    if breakage == "cycles":
        bad = train.copy(); bad[5, 1] = 99
        np.savetxt(tmp_path / "train_FD001.txt", bad, fmt="%.6f")
    elif breakage == "rul":
        np.savetxt(tmp_path / "RUL_FD001.txt", rul[:-1], fmt="%d")
    elif breakage == "nan":
        bad = train.copy(); bad[3, 7] = np.nan
        np.savetxt(tmp_path / "train_FD001.txt", bad, fmt="%.6f")
    else:
        np.savetxt(tmp_path / "train_FD001.txt", train[:, :25], fmt="%.6f")
    with pytest.raises(ValueError, match=match):
        cmapss.build_domain(tmp_path, "FD001", window=8, cap=20)


REAL = Path(__file__).resolve().parent.parent / "data" / "cmapss"


@pytest.mark.skipif(not (REAL / "train_FD001.txt").exists(), reason="NASA files not downloaded (make pdm-fetch)")
def test_real_files_have_the_published_shape():
    for fd in ("FD001", "FD003"):
        dom = cmapss.build_domain(REAL, fd, window=30, cap=125)
        assert dom.info["n_train_rows"] == cmapss.KNOWN_ROWS[fd][0] and len(dom.memory) == cmapss.KNOWN_ROWS[fd][0]
        assert (dom.info["n_train_engines"], dom.info["n_test_engines"]) == cmapss.KNOWN_ENGINES[fd]
        assert dom.memory.y.max() == 125 and dom.queries["test_last"].Z.shape[0] == cmapss.KNOWN_ENGINES[fd][1]
    d1 = cmapss.build_domain(REAL, "FD001", window=30, cap=125)
    assert d1.info["channels"] == ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
