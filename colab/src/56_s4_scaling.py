# %% [markdown] id=s4-md
# ## 4. Systems scaling: N sweep, batch sweep, single-request latency, compile cost, memory
#
# v1's largest workload (N = 2048, 256 queries, D = 512) is about 0.5 GFLOP, which a modern accelerator finishes in
# microseconds; the measured milliseconds were launch and synchronisation overhead, so v1 could not distinguish a CPU from
# a GPU or TPU. This section therefore:
#
# * measures the **dispatch floor** (an empty kernel plus a blocking read) and flags every row whose latency is within
#   5× of it as *launch-bound*;
# * sweeps **N** until memory or the profile budget stops it and reports achieved GFLOP/s and GB/s;
# * sweeps the **batch size** (Q = 1 … thousands) to separate latency from throughput;
# * measures the **single-request latency distribution** (Q = 1) with hundreds to thousands of samples, both
#   device-resident and including host → device → host transfer, and only then reports p99 / p99.9;
# * reports **first-call (compile) cost** and **device memory in use**.
#
# On a CPU runtime these sweeps are only a functional check; they say nothing about accelerator behaviour.

# %% id=s4-code
#@title 4. Scaling sweeps, batch sweep, latency distribution
clear("scaling_N"); clear("scaling_Q"); clear("latency_dist"); clear("scaling_summary"); clear("dispatch_floor")
SWEEP_KERNELS = ["f32_highest", "f32_default", "bf16_acc32", "int8_acc32", "packed_popcount"]
BATCH_KERNELS = ["f32_highest", "bf16_acc32", "int8_acc32", "packed_popcount"]
LAT_SAMPLES = {}
HOST_OPERAND = {"f32": lambda a: to_f32(put(a)), "bf16": lambda a: to_bf16(put(a)), "i8": lambda a: put(a), "pack": lambda a: put(pack_bits(a))}


def n_sweep_list():
    top = int(math.log2(NMAX))
    es = list(range(8 if PROFILE == "smoke" else 10, top + 1, 1 if PROFILE == "smoke" else 2))
    if es[-1] != top:
        es.append(top)
    return [2 ** e for e in es if fits(2 ** e, D * 7.2)]


# dispatch floor: the cost of launching an (almost) empty kernel and reading its result back
_tiny = jax.jit(lambda x: x + 1.0)
_x = put(np.ones(8, np.float32))
_fl = bench(_tiny, _x, batch=1, samples=max(200, P["lat_samples"] // 2), modes=("latency",))
DISPATCH_FLOOR_MS = _fl["lat_p50_ms"]
record("dispatch_floor", "empty_kernel", d=D, query_batch=1, **_fl)
print(f"dispatch floor (empty kernel + blocking read): p50 {DISPATCH_FLOOR_MS * 1e3:.1f} us, p99 {_fl['lat_p99_ms'] * 1e3:.1f} us "
      f"(from {_fl['n_samples']} samples)")

if RUN_SCALING:
    ns = n_sweep_list()
    big = bipolar_i8(ns[-1], D, rng_for(4, 0))
    bud = Budget(P["budget_s"])
    for n in ns:
        if bud.expired():
            print(f"N sweep stopped before N={n}: budget of {bud.s:.0f}s used")
            break
        Qn = min(P["Q"], q_cap(n))
        tg, q_i8 = make_queries(big[:n], Qn, 0.25, rng_for(4, 1, n))
        ops = operands(big[:n], q_i8)
        nv = min(Qn, 64)
        rmax = ref_max_scores(ops["f32"][0][:nv], ops["f32"][1])
        for name in SWEEP_KERNELS:
            if kernel_allowed(name, n):
                row = run_kernel("scaling_N", name, ops, q_i8, big[:n], rmax, tg, n, nv, 0.25)
                row["launch_bound"] = bool(row["lat_p50_ms"] < 5 * DISPATCH_FLOOR_MS)
        del ops
        gc.collect()
    sc = results_df("scaling_N")
    for name in SWEEP_KERNELS:
        g = sc[sc.method == name].sort_values("n")
        if len(g) >= 2:
            tail = g.tail(3)
            record("scaling_summary", name, n=int(g.n.max()), d=D, scenario="N_sweep", scaling_exponent=loglog_slope(tail.n, tail.pipe_p50_ms),
                   best_gflops=float(g.gflops_effective.max()), best_gbps=float(g.gbps_memory.max()), largest_n=int(g.n.max()),
                   pipe_ms_at_largest=float(g.pipe_p50_ms.iloc[-1]), launch_bound_rows=int((g.lat_p50_ms < 5 * DISPATCH_FLOOR_MS).sum()), rows=len(g))
    show("scaling_N", ["method", "n", "query_batch", "optimal_rate", "first_call_ms", "lat_p50_ms", "pipe_p50_ms", "gflops_effective", "gbps_memory", "device_bytes_in_use"],
         sort=["method", "n"], max_rows=60)
    show("scaling_summary", ["method", "largest_n", "scaling_exponent", "best_gflops", "best_gbps", "pipe_ms_at_largest", "launch_bound_rows", "rows"], sort=["method"])
    nbad = sc[sc.method.isin(["f32_highest", "bf16_acc32", "int8_acc32", "packed_popcount", "f32_default"])]
    check("S4 every kernel in the N sweep selects a maximal-score pattern", bool((nbad.optimal_rate == 1.0).all()), f"{len(nbad)} (kernel, N) cells")

    # ---------------- batch-size sweep at a fixed N
    nb = min(ns[-1], 2 ** 16)
    qs = [q for q in P["q_sweep"] if q <= q_cap(nb)]
    mem_i8 = big[:nb]
    tg_all, q_all = make_queries(mem_i8, max(qs), 0.25, rng_for(4, 2))
    ops_all = operands(mem_i8, q_all)
    rmax32 = ref_max_scores(ops_all["f32"][0][:32], ops_all["f32"][1])
    bud = Budget(P["budget_s"])
    for qi in qs:
        if bud.expired():
            print(f"batch sweep stopped before Q={qi}: budget used")
            break
        sl = {fam: (qop[:qi], mop) for fam, (qop, mop) in ops_all.items()}
        for name in BATCH_KERNELS:
            if kernel_allowed(name, nb):
                run_kernel("scaling_Q", name, sl, q_all[:qi], mem_i8, rmax32, tg_all[:qi], nb, min(qi, 32), 0.25)
    show("scaling_Q", ["method", "n", "query_batch", "lat_p50_ms", "pipe_p50_ms", "qps", "us_per_query", "gflops_effective"], sort=["method", "query_batch"], max_rows=40)
    del ops_all
    gc.collect()

    # ---------------- single-request latency distribution (Q = 1)
    nl = min(ns[-1], 2 ** 14)
    mem_i8 = big[:nl]
    _, q1 = make_queries(mem_i8, 1, 0.25, rng_for(4, 3))
    ops1 = operands(mem_i8, q1)
    for name in ("f32_highest", "bf16_acc32", "packed_popcount"):
        if not kernel_allowed(name, nl):
            continue
        fn, fam, eb = KERNELS[name]
        qop, mop = ops1[fam]
        keep = []
        st = bench(fn, qop, mop, batch=1, samples=P["lat_samples"], modes=("latency",), keep=keep)
        LAT_SAMPLES[(name, "device_resident")] = np.array(keep)
        record("latency_dist", name, n=nl, d=D, query_batch=1, path="device_resident", **st)
        keep = []
        st = bench(lambda: np.asarray(fn(HOST_OPERAND[fam](q1.copy()), mop)), batch=1, samples=P["lat_samples"], modes=("latency",), keep=keep)
        LAT_SAMPLES[(name, "host_roundtrip")] = np.array(keep)
        record("latency_dist", name, n=nl, d=D, query_batch=1, path="host_roundtrip", **st)
    show("latency_dist", ["method", "path", "n", "n_samples", "first_call_ms", "lat_min_ms", "lat_p50_ms", "lat_p90_ms", "lat_p95_ms", "lat_p99_ms", "lat_p999_ms", "lat_max_ms"],
         sort=["method", "path"])
    if P["lat_samples"] < 100:
        print("Fewer than 100 latency samples in this profile: p99 / p99.9 are intentionally reported as NaN.")

    # ---------------- plots
    fig, axes = plt.subplots(1, 4, figsize=(19, 3.9))
    for name in SWEEP_KERNELS:
        g = sc[sc.method == name].sort_values("n")
        if len(g):
            axes[0].plot(g.n, g.pipe_p50_ms, "o-", label=name); axes[1].plot(g.n, g.gbps_memory, "o-", label=name)
    axes[0].axhline(DISPATCH_FLOOR_MS, color="gray", ls=":", label="dispatch floor")
    axes[0].set(xscale="log", yscale="log", xlabel="stored patterns N", ylabel="pipelined ms / call", title="time vs N"); axes[0].legend(fontsize=7)
    axes[1].set(xscale="log", yscale="log", xlabel="stored patterns N", ylabel="GB/s of memory read", title="achieved bandwidth")
    sq = results_df("scaling_Q")
    for name in BATCH_KERNELS:
        g = sq[sq.method == name].sort_values("query_batch")
        if len(g):
            axes[2].plot(g.query_batch, g.qps, "o-", label=name)
    axes[2].set(xscale="log", yscale="log", xlabel="batch size Q", ylabel="queries / s", title=f"throughput vs batch (N={nb})"); axes[2].legend(fontsize=7)
    for (name, path), arr in LAT_SAMPLES.items():
        if name == "f32_highest":
            axes[3].hist(arr * 1e3, bins=60, alpha=.55, label=path, log=True)
    axes[3].set(xscale="log", xlabel="latency (us)", ylabel="count", title=f"single-request latency (f32, N={nl})"); axes[3].legend(fontsize=7)
    for a in axes:
        a.grid(alpha=.25, which="both")
    plt.tight_layout(); savefig(fig, "s4_scaling"); plt.show()
else:
    print("Scaling study disabled.")
