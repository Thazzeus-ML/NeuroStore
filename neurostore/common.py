"""Shared utilities: seeded RNG, hypervector data, noise injection, bit packing,
statistics, timing and memory accounting.

Design rules that the rest of the framework relies on:

* No global RNG state. Every random draw comes from an explicit ``numpy.random.Generator``
  so a run is reproducible from its seed and systems can be compared on *identical* data.
* Noise is injected as an **exact** number of bit flips per vector (not i.i.d. Bernoulli),
  which removes the variance of the flip count from the measurement.
* Timing is per call (``perf_counter_ns``) with GC disabled, and the timer/call overhead
  is measured and reported so that sub-microsecond numbers can be read for what they are.
"""
from __future__ import annotations

import contextlib
import gc
import math
import os
import platform
import sys
import time
import warnings
from typing import Any, Callable, Iterator, Sequence

import numpy as np

# --------------------------------------------------------------------------------------
# RNG
# --------------------------------------------------------------------------------------


def make_rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    """Return a ``Generator`` (pass-through if one is given)."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def spawn_seeds(seed: int, n: int) -> list[int]:
    """Derive ``n`` statistically independent integer seeds from one master seed."""
    children = np.random.SeedSequence(seed).spawn(n)
    return [int(c.generate_state(1)[0]) for c in children]


# --------------------------------------------------------------------------------------
# Hypervector data and noise
# --------------------------------------------------------------------------------------


def random_bipolar(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    """``(n, dim)`` int8 array of i.i.d. uniform {-1, +1}."""
    return (rng.integers(0, 2, size=(n, dim), dtype=np.int8) * 2 - 1).astype(np.int8)


def noise_to_flips(dim: int, rate: float) -> int:
    """Number of bit flips for a noise ``rate`` in a ``dim``-bit vector (rounded, clipped)."""
    return int(min(dim, max(0, round(dim * rate))))


def flip_exact(rng: np.random.Generator, x: np.ndarray, n_flips: int) -> np.ndarray:
    """Copy of bipolar rows ``x`` with **exactly** ``n_flips`` random positions negated per row.

    ``x`` may be 1-D (one vector) or 2-D ``(Q, D)``; the output has the same shape.
    """
    squeeze = x.ndim == 1
    x2 = np.atleast_2d(x)
    q, d = x2.shape
    out = x2.copy()
    if n_flips <= 0:
        pass
    elif n_flips >= d:
        out = -out
    else:
        keys = rng.random((q, d), dtype=np.float32)
        idx = np.argpartition(keys, n_flips, axis=1)[:, :n_flips]
        rows = np.arange(q)[:, None]
        out[rows, idx] = -out[rows, idx]
    return out[0] if squeeze else out


def inject_bit_flip_noise(rng: np.random.Generator, vector: np.ndarray, noise_rate: float) -> np.ndarray:
    """Single-vector convenience wrapper around :func:`flip_exact`."""
    return flip_exact(rng, vector, noise_to_flips(vector.shape[-1], noise_rate))


# --------------------------------------------------------------------------------------
# Bit packing and popcount distances
# --------------------------------------------------------------------------------------

_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """Pack ``(..., D)`` {0,1} arrays into ``(..., ceil(D/64))`` uint64 words (zero padded)."""
    bits = np.asarray(bits, dtype=np.uint8)
    pad = (-bits.shape[-1]) % 64
    if pad:
        bits = np.pad(bits, [(0, 0)] * (bits.ndim - 1) + [(0, pad)])
    packed = np.packbits(bits, axis=-1)
    return np.ascontiguousarray(packed).view(np.uint64)


def pack_bipolar(x: np.ndarray) -> np.ndarray:
    """Pack bipolar (+1 -> 1, anything else -> 0) vectors into uint64 words."""
    return pack_bits(np.asarray(x) > 0)


def unpack_bits(words: np.ndarray, nbits: int) -> np.ndarray:
    """Inverse of :func:`pack_bits` -> ``(..., nbits)`` uint8 array of {0,1}."""
    as_bytes = np.ascontiguousarray(words).view(np.uint8)
    return np.unpackbits(as_bytes, axis=-1)[..., :nbits]


def popcount_u64(x: np.ndarray) -> np.ndarray:
    """Population count of every uint64 element (uses ``np.bitwise_count`` on NumPy >= 2)."""
    if _HAS_BITWISE_COUNT:
        return np.bitwise_count(x)
    return popcount_u64_lut(x)


def popcount_u64_lut(x: np.ndarray) -> np.ndarray:
    """Portable popcount through an 8-bit lookup table (fallback for NumPy 1.x)."""
    x = np.ascontiguousarray(x)
    return _POP8[x.view(np.uint8)].reshape(*x.shape, 8).sum(axis=-1, dtype=np.uint8)


def popcount_rows(x: np.ndarray) -> np.ndarray:
    """Per-row total popcount of a ``(..., W)`` uint64 array -> int32 ``(...)``."""
    return popcount_u64(x).sum(axis=-1, dtype=np.int32)


def packed_distance_to_all(q: np.ndarray, db: np.ndarray, op: str = "xor") -> np.ndarray:
    """Distance/overlap of one packed query ``(W,)`` to every packed row of ``db`` ``(N, W)``.

    ``op='xor'`` -> Hamming distance; ``op='and'`` -> number of shared set bits.
    """
    combined = (db ^ q) if op == "xor" else (db & q)
    return popcount_rows(combined)


def packed_distance_matrix(
    qs: np.ndarray, db: np.ndarray, op: str = "xor", max_temp_bytes: int = 64 << 20
) -> np.ndarray:
    """``(Q, N)`` int32 matrix of packed Hamming distances / overlaps, chunked over queries."""
    qs = np.atleast_2d(qs)
    n, w = db.shape
    per_query = max(1, n * w * 8)
    chunk = max(1, max_temp_bytes // per_query)
    out = np.empty((qs.shape[0], n), dtype=np.int32)
    for s in range(0, qs.shape[0], chunk):
        blk = qs[s : s + chunk, None, :]
        combined = (db[None, :, :] ^ blk) if op == "xor" else (db[None, :, :] & blk)
        out[s : s + chunk] = popcount_rows(combined)
    return out


class GrowableRows:
    """Row buffer with amortised O(1) appends (capacity doubling).

    All incremental-write paths (Hopfield, brute force, LSH, ...) use this so that the
    measured write latency reflects the algorithm and not an O(N) ``np.concatenate``.
    """

    def __init__(self, ncols: int, dtype: Any, capacity: int = 0) -> None:
        self._buf = np.empty((capacity, ncols), dtype=dtype)
        self._n = 0

    def __len__(self) -> int:
        return self._n

    @property
    def view(self) -> np.ndarray:
        """The stored rows (a view, not a copy)."""
        return self._buf[: self._n]

    @property
    def nbytes(self) -> int:
        """Allocated bytes, including any unused doubling slack."""
        return int(self._buf.nbytes)

    def _reserve(self, extra: int) -> None:
        need = self._n + extra
        if need > self._buf.shape[0]:
            new_cap = max(need, 2 * self._buf.shape[0], 16)
            nb = np.empty((new_cap, self._buf.shape[1]), dtype=self._buf.dtype)
            nb[: self._n] = self._buf[: self._n]
            self._buf = nb

    def extend(self, rows: np.ndarray) -> None:
        rows = np.atleast_2d(rows)
        self._reserve(rows.shape[0])
        self._buf[self._n : self._n + rows.shape[0]] = rows
        self._n += rows.shape[0]

    def append(self, row: np.ndarray) -> int:
        """Append one row; returns its index."""
        self._reserve(1)
        self._buf[self._n] = row
        self._n += 1
        return self._n - 1

    def shrink_to_fit(self) -> None:
        self._buf = self._buf[: self._n].copy()


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


def wilson_interval(successes: int, trials: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    lo = 0.0 if successes == 0 else max(0.0, centre - half)  # endpoints are exact, not 1 - 1e-16
    hi = 1.0 if successes == trials else min(1.0, centre + half)
    return (lo, hi)


def proportion(successes: int, trials: int) -> dict[str, float]:
    """``{rate, lo, hi, k, n}`` for a binomial proportion with its Wilson interval."""
    lo, hi = wilson_interval(int(successes), int(trials))
    return {
        "rate": successes / trials if trials else float("nan"),
        "lo": lo,
        "hi": hi,
        "k": int(successes),
        "n": int(trials),
    }


def norm_cdf(x: np.ndarray | float) -> np.ndarray:
    """Standard normal CDF (vectorised ``erfc``; accurate in both tails)."""
    arr = np.asarray(x, dtype=np.float64)
    return 0.5 * np.vectorize(math.erfc)(-arr / math.sqrt(2.0))


def norm_sf(x: np.ndarray | float) -> np.ndarray:
    """Standard normal survival function ``Q(x) = 1 - Phi(x)``."""
    arr = np.asarray(x, dtype=np.float64)
    return 0.5 * np.vectorize(math.erfc)(arr / math.sqrt(2.0))


def threshold_crossing(xs: Sequence[float], ys: Sequence[float], level: float) -> dict[str, float | None]:
    """Where a (noisy) decreasing curve ``ys(xs)`` first drops below ``level``.

    The curve is first made monotone with a running minimum (so one lucky point after a
    failure cannot move the threshold). Returns ``last_ok`` (largest x with y >= level
    and all smaller x also ok), ``first_fail`` (next x) and a linearly interpolated
    ``crossing``. ``last_ok`` is ``None`` if even the first point fails; ``first_fail``
    is ``None`` if the curve never drops below ``level``.
    """
    xs = list(xs)
    ys = list(np.minimum.accumulate(np.asarray(ys, dtype=np.float64)))
    last_ok: float | None = None
    first_fail: float | None = None
    crossing: float | None = None
    for i, (x, y) in enumerate(zip(xs, ys)):
        if y >= level:
            last_ok = x
        else:
            first_fail = x
            if i > 0:
                x0, y0 = xs[i - 1], ys[i - 1]
                crossing = x0 + (y0 - level) / (y0 - y) * (x - x0) if y0 != y else x0
            else:
                crossing = x
            break
    return {"last_ok": last_ok, "first_fail": first_fail, "crossing": crossing}


# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------


def time_calls(
    fn: Callable[[Any], Any], inputs: Sequence[Any], warmup: int = 5, disable_gc: bool = True
) -> np.ndarray:
    """Time ``fn(x)`` individually for every ``x`` in ``inputs``; returns int64 nanoseconds.

    Inputs must be prepared by the caller (conversion cost is not part of the timing).
    """
    inputs = list(inputs)
    for x in inputs[: max(0, min(warmup, len(inputs)))]:
        fn(x)
    out = np.empty(len(inputs), dtype=np.int64)
    pc = time.perf_counter_ns
    was_enabled = gc.isenabled()
    if disable_gc:
        gc.disable()
    try:
        for i, x in enumerate(inputs):
            t0 = pc()
            fn(x)
            out[i] = pc() - t0
    finally:
        if disable_gc and was_enabled:
            gc.enable()
    return out


def summarize_latency(ns: np.ndarray) -> dict[str, float]:
    """Latency summary in microseconds from per-call nanosecond samples."""
    ns = np.asarray(ns, dtype=np.float64)
    if ns.size == 0:
        return {k: float("nan") for k in ("n", "mean_us", "p50_us", "p95_us", "p99_us", "min_us", "max_us")}
    us = ns / 1e3
    return {
        "n": int(us.size),
        "mean_us": float(us.mean()),
        "p50_us": float(np.percentile(us, 50)),
        "p95_us": float(np.percentile(us, 95)),
        "p99_us": float(np.percentile(us, 99)),
        "min_us": float(us.min()),
        "max_us": float(us.max()),
    }


def timer_overhead_ns(reps: int = 20000) -> float:
    """Median cost of an empty timed call (timer reads + Python call), in ns."""
    samples = time_calls(lambda _x: None, [None] * reps, warmup=100)
    return float(np.median(samples))


def time_batches(fn: Callable[[Any], Any], batches: Sequence[Any], warmup: int = 1) -> float:
    """Median wall-clock seconds per batch for ``fn(batch)``."""
    ns = time_calls(fn, batches, warmup=warmup)
    return float(np.median(ns)) / 1e9


# --------------------------------------------------------------------------------------
# Threads
# --------------------------------------------------------------------------------------

_WARNED_THREADS = False


@contextlib.contextmanager
def limit_threads(n: int | None) -> Iterator[None]:
    """Cap BLAS/OpenMP threads (``threadpoolctl``) and FAISS threads for fair per-core numbers.

    ``None``/``0`` leaves the defaults untouched.
    """
    global _WARNED_THREADS
    if not n:
        yield
        return
    restore_faiss = None
    try:
        import faiss  # type: ignore

        restore_faiss = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(int(n))
    except Exception:
        faiss = None  # noqa: F841
    try:
        from threadpoolctl import threadpool_limits

        ctx = threadpool_limits(limits=int(n))
    except ImportError:
        if not _WARNED_THREADS:
            warnings.warn("threadpoolctl not installed: BLAS thread count is not limited")
            _WARNED_THREADS = True
        ctx = contextlib.nullcontext()
    try:
        with ctx:
            yield
    finally:
        if restore_faiss is not None:
            import faiss  # type: ignore

            faiss.omp_set_num_threads(restore_faiss)


# --------------------------------------------------------------------------------------
# Memory accounting
# --------------------------------------------------------------------------------------


def deep_sizeof(obj: Any, _seen: set[int] | None = None) -> int:
    """Recursive ``sys.getsizeof`` that counts shared objects once.

    This is the footprint of the Python data structure *as allocated* (object headers,
    hash-table slack and all), which is what a Python dict/list based baseline really costs.
    """
    seen = _seen if _seen is not None else set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    size = sys.getsizeof(obj)
    if isinstance(obj, np.ndarray):
        if isinstance(obj.base, np.ndarray):
            size += deep_sizeof(obj.base, seen)
        return size
    if isinstance(obj, (bytes, bytearray, str, int, float, bool, complex, type(None))):
        return size
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += deep_sizeof(k, seen) + deep_sizeof(v, seen)
        return size
    if isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += deep_sizeof(item, seen)
        return size
    if hasattr(obj, "__dict__"):
        size += deep_sizeof(vars(obj), seen)
    for slot in getattr(type(obj), "__slots__", ()):
        if hasattr(obj, slot):
            size += deep_sizeof(getattr(obj, slot), seen)
    return size


def fmt_bytes(n: float) -> str:
    """Human readable byte count."""
    if n != n:  # NaN
        return "n/a"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


# --------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _blas_name() -> str:
    try:
        cfg = np.__config__.CONFIG  # type: ignore[attr-defined]
        return str(cfg["Build Dependencies"]["blas"]["name"])
    except Exception:
        return "unknown"


def optional_versions() -> dict[str, str | None]:
    """Versions of the optional baseline libraries (``None`` if not installed)."""
    import importlib.metadata as md

    out: dict[str, str | None] = {}
    for dist in ("faiss-cpu", "qdrant-client", "datasketch", "rocksdict", "redis", "threadpoolctl"):
        try:
            out[dist] = md.version(dist)
        except md.PackageNotFoundError:
            out[dist] = None
    return out


def environment_info() -> dict[str, Any]:
    """Machine/software description stored with every result file."""
    try:
        total_mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        total_mem = None
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas": _blas_name(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": total_mem,
        "timer_overhead_ns": timer_overhead_ns(5000),
        "has_np_bitwise_count": _HAS_BITWISE_COUNT,
        "optional_libs": optional_versions(),
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }
