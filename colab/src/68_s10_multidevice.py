# %% [markdown] id=s10-md
# ## 10. Multi-device scaling
#
# Everything above benchmarks **one** device. When the runtime exposes several (a Colab TPU has 8 cores), this section runs
# the same exact top-1 kernel two ways with `jax.sharding`:
#
# * **data parallel**: memory replicated, queries sharded across devices. Throughput should scale with the device count
#   until the batch is too small to keep every device busy.
# * **memory sharded**: memory split across devices, queries replicated, the argmax reduced across devices. This is how a
#   store larger than one device's memory would be served; it adds a collective to every query.
#
# Both are checked for exactness against a single-device reference (picked pattern attains the maximal score). With one device the
# section is skipped. To exercise the logic on a CPU runtime, set `SIMULATE_CPU_DEVICES = 4` in the configuration cell and
# restart the runtime; the timings are then meaningless (virtual devices share the same cores), but the correctness checks are real.

# %% id=s10-code
#@title 10. Multi-device: data-parallel vs memory-sharded
clear("multidevice")
if RUN_MULTIDEVICE and len(DEVICES) > 1:
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS
    nd = len(DEVICES)
    mesh = Mesh(np.array(DEVICES), ("d",))
    n_md = 2 ** int(math.log2(min(NMAX, 2 ** 16)))
    while n_md % nd:
        n_md //= 2
    q_md = max(nd, (min(max(P["Q"], 8 * nd), q_cap(n_md)) // nd) * nd)
    mem_i8 = bipolar_i8(n_md, D, rng_for(10, 0))
    tg, q_i8 = make_queries(mem_i8, q_md, 0.25, rng_for(10, 1))
    m32, q32 = to_f32(put(mem_i8)), to_f32(put(q_i8))
    rmax = ref_max_scores(q32[: min(q_md, 64)], m32)
    nvv = min(q_md, 64)
    rep, shard_rows = NamedSharding(mesh, PS()), NamedSharding(mesh, PS("d", None))

    def timing_row(name, fn, args, **kw):
        pred = np.asarray(fn(*args))
        st = bench(fn, *args, batch=q_md)
        return record("multidevice", name, n=n_md, d=D, query_batch=q_md, devices=nd, optimal_rate=optimal_rate(pred[:nvv], q_i8[:nvv], mem_i8, rmax[:nvv]),
                      recall_at_1=float(np.mean(pred == tg)), **st, **kw)

    r1 = timing_row("single_device", K_F32_HI, (q32, m32), speedup_vs_single=1.0, efficiency=1.0)
    dp_args = (jax.device_put(np.asarray(q32), shard_rows), jax.device_put(np.asarray(m32), rep))
    r2 = timing_row("data_parallel", K_F32_HI, dp_args)
    ms_args = (jax.device_put(np.asarray(q32), rep), jax.device_put(np.asarray(m32), shard_rows))
    r3 = timing_row("memory_sharded", K_F32_HI, ms_args)
    for row in (r2, r3):
        row["speedup_vs_single"] = r1["pipe_p50_ms"] / row["pipe_p50_ms"]
        row["efficiency"] = row["speedup_vs_single"] / nd
    show("multidevice", ["method", "devices", "n", "query_batch", "optimal_rate", "lat_p50_ms", "pipe_p50_ms", "qps", "speedup_vs_single", "efficiency"], sort=["method"])
    check("S10 data-parallel and memory-sharded results attain the maximal score (match the single-device reference)",
          bool(r2["optimal_rate"] == 1.0 and r3["optimal_rate"] == 1.0 and r1["optimal_rate"] == 1.0), f"{nd} devices, N={n_md}, Q={q_md}")
    if BACKEND == "cpu":
        print("NOTE: the devices are virtual CPU devices on the same cores; speedups above are not hardware scaling.")
else:
    print(f"Multi-device section skipped ({len(DEVICES)} local device(s); needs more than one and RUN_MULTIDEVICE).")
