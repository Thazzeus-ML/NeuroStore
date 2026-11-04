"""Synthetic worlds used to validate the harness against theory and to benchmark scaling."""
from __future__ import annotations
import numpy as np
from ..core import Split, DomainData


def pm1_world(D: int, N: int, p_flip: float, n_queries: int, seed: int) -> DomainData:
    """N random +-1 patterns of D bits; each query is a random stored pattern with every bit flipped w.p. p_flip.
    y = index of the true pattern (so an exact top-1 hit is recall)."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), D, N, int(round(p_flip * 1e6))]))
    X = rng.choice(np.array([-1.0, 1.0], np.float32), size=(N, D))
    tgt = rng.integers(0, N, n_queries)
    flip = rng.random((n_queries, D)) < p_flip
    Q = np.where(flip, -X[tgt], X[tgt]).astype(np.float32)
    mem = Split(X[:, None, :], np.arange(N), np.arange(N))
    qs = Split(Q[:, None, :], tgt, np.arange(n_queries))
    return DomainData(f"pm1(D={D},N={N},p={p_flip})", "classification", mem, {"queries": qs}, info={"D": D, "N": N, "p_flip": p_flip})


def blob_world(n: int, d: int, n_clusters: int, noise: float, n_queries: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Clustered Gaussian vectors for ANN scaling benchmarks: returns (X (n, d), Q (n_queries, d)); queries are noisy copies of stored points."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), n, d, n_clusters]))
    C = rng.normal(size=(n_clusters, d)).astype(np.float32)
    X = (C[rng.integers(0, n_clusters, n)] + noise * rng.normal(size=(n, d)).astype(np.float32)).astype(np.float32)
    src = rng.integers(0, n, n_queries)
    Q = (X[src] + 0.5 * noise * rng.normal(size=(n_queries, d)).astype(np.float32)).astype(np.float32)
    return X, Q
