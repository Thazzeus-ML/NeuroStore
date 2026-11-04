"""NASA C-MAPSS turbofan degradation adapter (FD001-FD004).

Every preprocessing parameter is fitted on the TRAINING file only (regime centres, channel selection, per-regime z-scoring).
Windows: W consecutive cycles ending at cycle t, left-padded by repeating the first cycle. Training memory = every cycle of every
training engine (stride 1) with piecewise-linear RUL target min(RUL, cap). Official test queries = the last window of each test
engine, labelled with the official RUL file (raw) and its capped version.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..core import Split, DomainData
from .._util import kmeans, seed_stream

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
# row counts of the published files (train rows, test rows) and engine counts (train, test) as found in the downloaded files
KNOWN_ROWS = {"FD001": (20631, 13096), "FD002": (53759, 33991), "FD003": (24720, 16596), "FD004": (61249, 41214)}
KNOWN_ENGINES = {"FD001": (100, 100), "FD002": (260, 259), "FD003": (100, 100), "FD004": (249, 248)}


def read_table(path) -> np.ndarray:
    A = np.loadtxt(path, dtype=np.float64)
    if A.ndim != 2 or A.shape[1] != 26 or not np.isfinite(A).all():
        raise ValueError(f"{path}: expected a finite (n, 26) table, got {A.shape}")
    return A


def read_rul(path) -> np.ndarray:
    r = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if not np.isfinite(r).all() or (r < 0).any():
        raise ValueError(f"{path}: invalid RUL values")
    return r


def split_engines(A: np.ndarray) -> dict:
    """{unit: rows} after checking that every engine's cycles are exactly 1..L in order."""
    units = A[:, 0].astype(int)
    out = {}
    for u in np.unique(units):
        rows = A[units == u]
        if not np.array_equal(rows[:, 1], np.arange(1, len(rows) + 1)):
            raise ValueError(f"engine {u}: cycles are not 1..{len(rows)} in order")
        out[int(u)] = rows
    return out


@dataclass
class Prep:
    """Preprocessing fitted on a training table."""
    fd: str
    sensors: np.ndarray            # indices (0..20) of the sensor channels kept
    centers: np.ndarray | None     # (R, 3) operating-condition centres, None for a single regime
    mu: np.ndarray                 # (R, C)
    sd: np.ndarray                 # (R, C)
    min_unique: int

    @property
    def names(self):
        return [f"s{i + 1}" for i in self.sensors]


def assign_regime(A: np.ndarray, centers) -> np.ndarray:
    if centers is None:
        return np.zeros(len(A), np.int64)
    ops = A[:, 2:5]
    return np.argmin(((ops[:, None, :] - centers[None, :, :]) ** 2).sum(-1), axis=1)


def fit_prep(train: np.ndarray, fd: str, seed: int = 0, min_unique: int = 10) -> Prep:
    R = N_REGIMES[fd]
    centers = None
    if R > 1:
        ops = train[:, 2:5]
        scale = ops.std(0) + 1e-9                                   # k-means on standardised op settings
        c, _ = kmeans((ops / scale).astype(np.float32), R, seed_stream(seed, 41), iters=25, n_init=3)
        centers = (c * scale).astype(np.float64)
    reg = assign_regime(train, centers)
    S = train[:, 5:]
    keep = []
    for j in range(21):
        ok = all(len(np.unique(S[reg == r, j])) >= min_unique for r in range(R))
        if ok:
            keep.append(j)
    keep = np.array(keep, int)
    mu = np.stack([S[reg == r][:, keep].mean(0) for r in range(R)])
    sd = np.stack([np.maximum(S[reg == r][:, keep].std(0), 1e-9) for r in range(R)])
    return Prep(fd, keep, centers, mu, sd, min_unique)


def apply_prep(A: np.ndarray, prep: Prep):
    reg = assign_regime(A, prep.centers)
    S = A[:, 5:][:, prep.sensors]
    return ((S - prep.mu[reg]) / prep.sd[reg]).astype(np.float32), reg


def windows_for_engine(Zrows: np.ndarray, W: int) -> np.ndarray:
    """(L, C) -> (L, W, C): window t ends at row t, left-padded with the first row."""
    pad = np.concatenate([np.repeat(Zrows[:1], W - 1, axis=0), Zrows], axis=0)
    return np.lib.stride_tricks.sliding_window_view(pad, W, axis=0).transpose(0, 2, 1).copy()


def make_split(A: np.ndarray, prep: Prep, W: int, cap: int, last_only: bool = False, rul_final=None, stride: int = 1) -> Split:
    """Windows of every engine in table A. Training tables: target = min(L - t, cap). last_only=True: only each engine's final window,
    target = the official RUL of that engine (rul_final[i] for the i-th engine in unit order), raw; meta['y_capped'] holds min(raw, cap)."""
    Zrows, reg = apply_prep(A, prep)
    eng = split_engines(A)
    units = sorted(eng)
    Zs, ys, gs, ts, rr, rawl = [], [], [], [], [], []
    start = 0
    for i, u in enumerate(units):
        L = len(eng[u])
        zr, rg = Zrows[start:start + L], reg[start:start + L]
        start += L
        win = windows_for_engine(zr, W)
        if last_only:
            sel = np.array([L - 1])
            rul = np.array([rul_final[i]], np.float64)
        else:
            sel = np.arange(0, L, stride)
            rul = (L - (sel + 1)).astype(np.float64)
        Zs.append(win[sel]); gs.append(np.full(len(sel), u)); ts.append(sel + 1); rr.append(rg[sel]); rawl.append(rul)
    raw = np.concatenate(rawl)
    meta = {"t": np.concatenate(ts), "rul_raw": raw, "regime": np.concatenate(rr)}
    y = raw if last_only else np.minimum(raw, cap)
    if last_only:
        meta["y_capped"] = np.minimum(raw, cap)
    return Split(np.concatenate(Zs), y, np.concatenate(gs), meta)


def build_domain(root, fd: str = "FD001", window: int = 30, cap: int = 125, seed: int = 0, min_unique: int = 10) -> DomainData:
    root = str(root)
    train = read_table(f"{root}/train_{fd}.txt")
    test = read_table(f"{root}/test_{fd}.txt")
    rul = read_rul(f"{root}/RUL_{fd}.txt")
    prep = fit_prep(train, fd, seed, min_unique)
    memory = make_split(train, prep, window, cap)
    n_test = len(np.unique(test[:, 0]))
    if len(rul) != n_test:
        raise ValueError(f"{fd}: {n_test} test engines but {len(rul)} RUL labels")
    q = make_split(test, prep, window, cap, last_only=True, rul_final=rul)
    return DomainData(f"cmapss-{fd}", "regression", memory, {"test_last": q},
                      info={"fd": fd, "window": window, "cap": cap, "channels": prep.names, "prep": prep,
                            "n_train_rows": len(train), "n_train_engines": len(np.unique(train[:, 0])), "n_test_engines": n_test})


def foreign_split(root, fd_other: str, prep_ref: Prep, window: int, cap: int, stride: int = 5) -> Split:
    """Windows from another sub-dataset's TRAINING file, pushed through the REFERENCE preprocessing (same channels, same
    normalisation, same regime logic): the out-of-distribution sets for a memory built on prep_ref's sub-dataset."""
    A = read_table(f"{root}/train_{fd_other}.txt")
    return make_split(A, prep_ref, window, cap, stride=stride)
