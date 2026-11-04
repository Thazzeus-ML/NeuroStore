from __future__ import annotations
import numpy as np


def l2_normalize(X, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)


def iter_chunks(n: int, size: int):
    for s in range(0, n, size):
        yield s, min(n, s + size)


def topk_desc(S: np.ndarray, k: int):
    """Row-wise top-k of S (q, n), descending. Returns (idx, vals). Equal values are ordered arbitrarily but
    deterministically (k == 1 uses argmax: lowest index wins)."""
    q, n = S.shape
    k = max(1, min(int(k), n))
    if k == 1:
        idx = S.argmax(axis=1)[:, None]
    elif k == n:
        idx = np.argsort(-S, axis=1, kind="stable")
    else:
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(S, part, axis=1)
        idx = np.take_along_axis(part, np.argsort(-vals, axis=1, kind="stable"), axis=1)
    return idx.astype(np.int64), np.take_along_axis(S, idx, axis=1)


def csr_gather(ptr: np.ndarray, rows: np.ndarray):
    """Positions of every entry of CSR rows `rows` (vectorised, no python loop)."""
    rows = np.asarray(rows, np.int64)
    starts = ptr[rows].astype(np.int64)
    lens = ptr[rows + 1].astype(np.int64) - starts
    tot = int(lens.sum())
    if tot == 0:
        return np.zeros(0, np.int64)
    cum = np.cumsum(lens)
    return np.repeat(starts - (cum - lens), lens) + np.arange(tot)


def softmax_rows(S: np.ndarray) -> np.ndarray:
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S, dtype=np.float64)
    return (E / E.sum(axis=1, keepdims=True)).astype(np.float32)


def logsumexp_rows(S: np.ndarray) -> np.ndarray:
    m = S.max(axis=1)
    return (m + np.log(np.exp(S - m[:, None], dtype=np.float64).sum(axis=1))).astype(np.float64)


def popcount64(x: np.ndarray) -> np.ndarray:
    """Per-element population count of uint64 (np.bitwise_count on NumPy >= 2, lookup table otherwise)."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    lut = np.array([bin(i).count("1") for i in range(256)], np.uint8)
    return lut[x.view(np.uint8)].reshape(x.shape + (8,)).sum(-1)


def kmeans(X: np.ndarray, k: int, rng: np.random.Generator, iters: int = 15, spherical: bool = False, n_init: int = 1):
    """Lloyd's k-means. spherical=True: unit-norm data and centroids, assignment by dot product. Returns (centers, assign)."""
    X = np.asarray(X, np.float32)
    n = len(X)
    k = max(1, min(k, n))
    best = None
    for _ in range(n_init):
        if spherical:
            C = X[rng.choice(n, k, replace=False)].copy()
        else:                                            # k-means++ seeding
            C = np.empty((k, X.shape[1]), np.float32)
            C[0] = X[rng.integers(n)]
            d2 = ((X - C[0]) ** 2).sum(1)
            for j in range(1, k):
                p = d2 / d2.sum() if d2.sum() > 0 else None
                C[j] = X[rng.choice(n, p=p)]
                d2 = np.minimum(d2, ((X - C[j]) ** 2).sum(1))
        for _ in range(iters):
            if spherical:
                a = np.argmax(X @ C.T, axis=1)
            else:
                a = np.argmin(((X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]), axis=1)
            for j in range(k):
                m = a == j
                if m.any():
                    C[j] = X[m].mean(0)
                else:                                     # re-seed an empty cluster
                    C[j] = X[rng.integers(n)]
            if spherical:
                C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        a = np.argmax(X @ C.T, axis=1) if spherical else np.argmin(((X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]), axis=1)
        inertia = float(((X - C[a]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, C.copy(), a)
    return best[1], best[2]


def seed_stream(*tags) -> np.random.Generator:
    """Independent reproducible generator per tuple of integer tags."""
    return np.random.default_rng(np.random.SeedSequence([int(t) for t in tags]))
