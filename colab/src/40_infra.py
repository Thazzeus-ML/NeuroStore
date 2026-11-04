# %% id=infra
#@title Infrastructure: RNG, data, timing, recording, self-checks

# ---- reproducible randomness: one independent stream per (SEED, tags...); tags are integers
def rng_for(*tags):
    return np.random.default_rng(np.random.SeedSequence([int(SEED)] + [int(t) for t in tags]))


def bipolar_i8(n, d, rng):
    """n x d random +-1 patterns as int8 (1 byte/element on the host; v1's int64 temporary needed 8)."""
    a = rng.integers(0, 2, size=(n, d), dtype=np.int8)
    a *= 2
    a -= 1
    return a


def flip_i8(x, p, rng):
    """Flip each +-1 entry independently with probability p."""
    out = x.copy()
    if p > 0:
        out[rng.random(x.shape, dtype=np.float32) < p] *= -1
    return out


def pack_bits(x_i8):
    """+-1 int8 (n, d) -> uint32 (n, d/32); popcount(a xor b) is the Hamming distance."""
    packed = np.packbits((x_i8 > 0).astype(np.uint8), axis=1)
    return np.ascontiguousarray(packed).view(np.uint32)


def put(x):
    return jax.device_put(np.asarray(x), DEVICE)


@jax.jit
def to_f32(x):
    return x.astype(jnp.float32)


@jax.jit
def to_bf16(x):
    return x.astype(jnp.bfloat16)


def nbytes(x):
    return int(x.size) * int(x.dtype.itemsize)


def q_cap(n, frac=8):
    """Largest query batch whose Q x n float32 score matrix stays below ~1/frac of memory (and 2 GiB)."""
    lim = min((MEM_LIMIT // frac) if MEM_LIMIT else 2 ** 30, 2 * 2 ** 30)
    return max(1, int(lim // (4 * max(n, 1))))


def fits(n, bytes_per_row, frac=0.4):
    return MEM_LIMIT is None or n * bytes_per_row <= frac * MEM_LIMIT


class Budget:
    """Wall-clock budget for one sweep. A sweep that hits it stops and says so (never silently)."""
    def __init__(self, seconds):
        self.t0, self.s = time.perf_counter(), float(seconds)

    def expired(self):
        return time.perf_counter() - self.t0 > self.s

    def used(self):
        return time.perf_counter() - self.t0


# ---- timing -------------------------------------------------------------------------------------------------------
def synchronize(x):
    for leaf in jax.tree_util.tree_leaves(x):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return x


def _pct(a, q, min_n):
    return float(np.percentile(a, q)) if len(a) >= min_n else NAN


def bench(fn, *args, batch=1, samples=None, warmup=None, modes=("latency", "pipelined"), min_sample_ms=4.0, max_inner=500, keep=None):
    """Time fn(*args) on the device.

    first_call_ms     first invocation, includes tracing + compilation when the function is cold
    lat_*             one call per sample, blocked until the result is ready (what a synchronous caller sees)
    pipe_*            `inner` calls dispatched back to back, blocked once, divided by `inner` (device throughput:
                      hides per-call dispatch cost; `inner` is chosen so a sample lasts >= min_sample_ms)
    Pass a list as `keep` to receive the raw single-call latency samples (ms).
    Tail percentiles are only reported when there are enough samples to estimate them: p90 n>=10, p95 n>=20,
    p99 n>=100, p99.9 n>=1000 (otherwise NaN). With 8 samples v1's "p99" was just the maximum.
    """
    samples = P["repeats"] if samples is None else int(samples)
    warmup = P["warmup"] if warmup is None else int(warmup)
    t = time.perf_counter(); synchronize(fn(*args)); first_ms = (time.perf_counter() - t) * 1e3
    for _ in range(warmup):
        synchronize(fn(*args))
    t = time.perf_counter(); synchronize(fn(*args)); one = max(time.perf_counter() - t, 1e-7)
    inner = int(min(max_inner, max(1, math.ceil(min_sample_ms * 1e-3 / one))))
    out = {"first_call_ms": first_ms, "n_samples": samples, "inner": inner}
    gc_on = gc.isenabled()
    gc.disable()
    try:
        if "latency" in modes:
            lat = np.empty(samples)
            for i in range(samples):
                t = time.perf_counter(); synchronize(fn(*args)); lat[i] = (time.perf_counter() - t) * 1e3
            if keep is not None:
                keep.extend(lat.tolist())
            out.update(lat_min_ms=lat.min(), lat_p50_ms=np.percentile(lat, 50), lat_mean_ms=lat.mean(),
                       lat_std_ms=lat.std(ddof=1) if samples > 1 else NAN, lat_p90_ms=_pct(lat, 90, 10), lat_p95_ms=_pct(lat, 95, 20),
                       lat_p99_ms=_pct(lat, 99, 100), lat_p999_ms=_pct(lat, 99.9, 1000), lat_max_ms=lat.max())
        if "pipelined" in modes:
            pipe = np.empty(samples)
            for i in range(samples):
                t = time.perf_counter()
                outs = [fn(*args) for _ in range(inner)]
                synchronize(outs)
                pipe[i] = (time.perf_counter() - t) * 1e3 / inner
            out.update(pipe_min_ms=pipe.min(), pipe_p50_ms=np.percentile(pipe, 50), pipe_std_ms=pipe.std(ddof=1) if samples > 1 else NAN)
    finally:
        if gc_on:
            gc.enable()
    ref = out.get("pipe_p50_ms", out.get("lat_p50_ms"))
    out["qps"] = batch * 1e3 / ref
    out["us_per_query"] = ref * 1e3 / batch
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


# ---- recording (idempotent: re-running a cell replaces its rows instead of duplicating them) ----------------------
RESULTS = {}
_KEYS = ("variant", "n", "d", "query_batch", "noise", "seed", "fault_mode", "severity", "scenario", "path", "metric")


def _norm(v):
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def record(experiment, method, **fields):
    row = dict(META_ROW)
    row.update(experiment=experiment, method=method)
    row.update({k: _norm(v) if not isinstance(v, (list, tuple, dict)) else v for k, v in fields.items()})
    RESULTS[(experiment, method) + tuple(_norm(fields.get(k)) for k in _KEYS)] = row
    return row


def clear(experiment):
    for k in [k for k in RESULTS if k[0] == experiment]:
        del RESULTS[k]


def results_df(experiment=None, **flt):
    df = pd.DataFrame(list(RESULTS.values()))
    if df.empty:
        return df
    if experiment is not None:
        df = df[df.experiment == experiment]
    for k, v in flt.items():
        df = df[df[k] == v]
    return df.reset_index(drop=True)


def show(experiment, cols, sort=None, max_rows=80, **flt):
    df = results_df(experiment, **flt)
    if df.empty:
        print(f"(no rows for {experiment})")
        return df
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    if sort:
        df = df.sort_values([c for c in sort if c in df.columns])
    display(df.head(max_rows).round(4))
    if len(df) > max_rows:
        print(f"... {len(df) - max_rows} more rows in results.csv")
    return df


# ---- self-checks: invariants that must hold on EVERY backend, collected into one table at the end -----------------
CHECKS = []


def check(name, ok, detail=""):
    ok = bool(ok)
    CHECKS[:] = [c for c in CHECKS if c["name"] != name]
    CHECKS.append({"name": name, "ok": ok, "detail": str(detail)})
    print(("PASS  " if ok else "FAIL  ") + name + (f"  | {detail}" if detail else ""))
    return ok


def savefig(fig, name):
    fig.savefig(OUTPUT_DIR / "plots" / f"{name}.png", dpi=140, bbox_inches="tight")


def yerr(r, lo, hi):
    """Error-bar lengths from a point estimate and interval bounds, clipped at 0 (fp edge cases at recall 1.0)."""
    r, lo, hi = (np.asarray(v, float) for v in (r, lo, hi))
    return np.vstack([np.clip(r - lo, 0, None), np.clip(hi - r, 0, None)])


def pooled_recall(correct_list):
    """Pool boolean arrays over seeds -> (recall, lo, hi, trials)."""
    c = np.concatenate(correct_list) if len(correct_list) else np.zeros(0, bool)
    lo, hi = wilson_ci(int(c.sum()), len(c))
    return (float(c.mean()) if len(c) else NAN), lo, hi, int(len(c))


SEEDS = list(range(P["seeds"]))
print(f"infrastructure ready: {len(SEEDS)} seeds, device-memory limit = {None if MEM_LIMIT is None else round(MEM_LIMIT / 2**30, 2)} GiB")
