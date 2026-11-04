"""Latency, throughput, build-time and memory accounting. All timings are wall-clock on the current machine; callers should
pin BLAS threads (see studies) and record `environment()` next to every number."""
from __future__ import annotations
import os
import platform
import time
import tracemalloc
import numpy as np


def environment() -> dict:
    cfg = {}
    try:
        info = np.show_config(mode="dicts")
        cfg = {"blas": info.get("Build Dependencies", {}).get("blas", {}).get("name", "?")}
    except Exception:
        pass
    return {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(), "cpus": os.cpu_count(),
            "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}, **cfg}


def _pct(a, q, min_n):
    return float(np.percentile(a, q)) if len(a) >= min_n else None


def latency_profile(fn, Q, n_samples: int = 1000, warmup: int = 50, seed: int = 0) -> dict:
    """Single-query latency of fn(Q[i:i+1]) in milliseconds over n_samples queries (sampled with replacement if Q is smaller).
    A percentile is reported only when there are enough samples to estimate it: p50 >= 20, p95 >= 100, p99 >= 1000."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(Q), n_samples + warmup)
    for i in idx[:warmup]:
        fn(Q[i:i + 1])
    t = np.empty(n_samples)
    for j, i in enumerate(idx[warmup:]):
        a = time.perf_counter_ns()
        fn(Q[i:i + 1])
        t[j] = (time.perf_counter_ns() - a) / 1e6
    return {"n": int(n_samples), "p50_ms": _pct(t, 50, 20), "p95_ms": _pct(t, 95, 100), "p99_ms": _pct(t, 99, 1000),
            "mean_ms": float(t.mean()), "min_ms": float(t.min())}


def throughput(fn, Q, batch: int = 256, repeats: int = 3, min_seconds: float = 0.5) -> dict:
    """Queries per second when fn is called on batches of `batch` queries; median over repeats of a >= min_seconds timed loop."""
    batch = int(min(batch, len(Q)))
    rates = []
    for _ in range(repeats):
        done, t0 = 0, time.perf_counter()
        while True:
            for a in range(0, len(Q) - batch + 1, batch):
                fn(Q[a:a + batch])
                done += batch
                if time.perf_counter() - t0 >= min_seconds:
                    break
            if time.perf_counter() - t0 >= min_seconds:
                break
        rates.append(done / (time.perf_counter() - t0))
    return {"batch": batch, "qps_median": float(np.median(rates)), "qps_all": [float(r) for r in rates]}


def build_profile(build_fn) -> tuple[object, dict]:
    """Run build_fn() once; return (result, {seconds, tracemalloc_peak_bytes}). NumPy allocations are visible to tracemalloc."""
    tracemalloc.start()
    t0 = time.perf_counter()
    out = build_fn()
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, {"seconds": float(dt), "tracemalloc_peak_bytes": int(peak)}


def search_peak_bytes(fn, Q) -> int:
    """Peak transient allocation of one batch call (tracemalloc)."""
    tracemalloc.start()
    fn(Q)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return int(peak)


def memory_table(pipeline, n_items: int, raw_window_bytes: int, raw_feature_bytes: int) -> dict:
    """Deployed bytes of a fitted pipeline broken down by component, with per-item cost and ratios against the raw store."""
    rep = pipeline.memory_report()
    deployed = pipeline.deployed_bytes()
    return {"components": rep, "deployed_bytes": deployed, "bytes_per_item": deployed / max(n_items, 1),
            "ratio_vs_raw_features": deployed / max(raw_feature_bytes, 1), "ratio_vs_raw_windows": deployed / max(raw_window_bytes, 1)}
