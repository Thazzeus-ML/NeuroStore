# %% [markdown] id=s7-md
# ## 7. Sparse distributed memory (SDM)
#
# `M` random ±1 hard-location addresses; a pattern activates its `K` nearest locations (fixed-K variant, **not** the
# report's exact address-activation rule); writes add the value to the activated counters; a read sums the activated
# counters and takes the sign. Autoassociative here: key = value.
#
# What this section adds to v1:
#
# * **Two implementations**, verified bit-for-bit equal: v1's dense `mask.T @ values` (cost O(n·M·D), materialises an
#   n×M mask) and a scatter-add write / gather read (cost O(n·K·D)). Which is faster depends on the hardware.
# * **Capacity sweeps** over `M`, `K`, load and noise, with Wilson intervals and optional iterated reads.
# * **Output BER vs input noise.** In v1 at 15 % input noise the SDM's output had *more* bit errors than its input for n ≥ 100;
#   `improves_over_input` makes that visible.
# * **Memory efficiency.** Bytes per *reliably stored* pattern (90 % exact recall), against a flat bit-packed memory,
#   whose retrieval success at the same noise is given by the analytic column.

# %% id=s7-code
#@title 7. SDM: implementations, capacity, noise tolerance, bytes per pattern
clear("sdm_impl"); clear("sdm_capacity"); clear("sdm_summary")


@partial(jax.jit, static_argnames=("k",))
def sdm_locations(keys, addresses, k):
    return lax.top_k(jnp.matmul(keys, addresses.T, precision=HI), k)[1]


@partial(jax.jit, static_argnames=("k",))
def sdm_write_dense(keys, values, addresses, k):          # v1's formulation
    loc = sdm_locations(keys, addresses, k)
    mask = jnp.zeros((keys.shape[0], addresses.shape[0]), jnp.float32).at[jnp.arange(keys.shape[0])[:, None], loc].set(1.0)
    return jnp.matmul(mask.T, values, precision=HI)


@partial(jax.jit, static_argnames=("k",))
def sdm_write_scatter(keys, values, addresses, k):
    loc = sdm_locations(keys, addresses, k)
    return jnp.zeros((addresses.shape[0], values.shape[1]), jnp.float32).at[loc.reshape(-1)].add(jnp.repeat(values, k, axis=0))


@partial(jax.jit, static_argnames=("k",))
def sdm_read_dense(queries, addresses, counters, k):
    loc = sdm_locations(queries, addresses, k)
    mask = jnp.zeros((queries.shape[0], addresses.shape[0]), jnp.float32).at[jnp.arange(queries.shape[0])[:, None], loc].set(1.0)
    return jnp.where(jnp.matmul(mask, counters, precision=HI) >= 0, 1.0, -1.0)


@partial(jax.jit, static_argnames=("k",))
def sdm_read_gather(queries, addresses, counters, k):
    loc = sdm_locations(queries, addresses, k)
    return jnp.where(counters[loc].sum(axis=1) >= 0, 1.0, -1.0)


def sdm_read_iter(x, addresses, counters, k, iters):
    for _ in range(iters):
        x = sdm_read_gather(x, addresses, counters, k)
    return x


if RUN_SDM:
    Ds = P["sdm_dim"]
    # ---------------- (a) dense mask vs scatter/gather: equality and speed
    n_impl = P["sdm_loads"][len(P["sdm_loads"]) // 2]
    k_impl = P["sdm_K"][len(P["sdm_K"]) // 2]
    all_equal = True
    for M in sorted({P["sdm_M"][0], P["sdm_M"][-1]}):
        addr = to_f32(put(bipolar_i8(M, Ds, rng_for(7, 0, M)))); pat = to_f32(put(bipolar_i8(n_impl, Ds, rng_for(7, 1, M))))
        c_d, c_s = sdm_write_dense(pat, pat, addr, k_impl), sdm_write_scatter(pat, pat, addr, k_impl)
        r_d, r_s = sdm_read_dense(pat, addr, c_d, k_impl), sdm_read_gather(pat, addr, c_s, k_impl)
        eq = bool(jnp.array_equal(c_d, c_s)) and bool(jnp.array_equal(r_d, r_s))
        all_equal &= eq
        for name, (wf, wargs, rf, rargs, ctr) in {"dense_mask": (sdm_write_dense, (pat, pat, addr, k_impl), sdm_read_dense, (pat, addr, c_d, k_impl), c_d),
                                                  "scatter_gather": (sdm_write_scatter, (pat, pat, addr, k_impl), sdm_read_gather, (pat, addr, c_s, k_impl), c_s)}.items():
            w = bench(wf, *wargs, batch=n_impl); rd = bench(rf, *rargs, batch=n_impl)
            record("sdm_impl", name, n=n_impl, d=Ds, variant=M, query_batch=n_impl, hard_locations=M, k_active=k_impl, identical_to_dense=eq,
                   write_pipe_ms=w["pipe_p50_ms"], write_lat_ms=w["lat_p50_ms"], read_pipe_ms=rd["pipe_p50_ms"], read_lat_ms=rd["lat_p50_ms"],
                   write_first_call_ms=w["first_call_ms"], temp_mask_bytes=n_impl * M * 4 if name == "dense_mask" else 0, counters_bytes=nbytes(ctr))
    check("S7 scatter/gather SDM is bit-identical to the dense-mask SDM", all_equal)
    show("sdm_impl", ["method", "hard_locations", "n", "k_active", "write_pipe_ms", "read_pipe_ms", "write_first_call_ms", "temp_mask_bytes", "identical_to_dense"], sort=["hard_locations", "method"])

    # ---------------- (b) capacity / noise sweep
    acc, maxabs = {}, {}
    bud = Budget(P["budget_s"])
    for M in P["sdm_M"]:
        for si in SEEDS:
            addr = to_f32(put(bipolar_i8(M, Ds, rng_for(7, 2, M, si))))
            for n in P["sdm_loads"]:
                if bud.expired():
                    break
                pats = bipolar_i8(n, Ds, rng_for(7, 3, M, n, si)); p32 = to_f32(put(pats))
                for K in P["sdm_K"]:
                    ctr = sdm_write_scatter(p32, p32, addr, K)
                    maxabs[(M, K, n)] = max(maxabs.get((M, K, n), 0), float(jnp.max(jnp.abs(ctr))))
                    for noise in P["sdm_noise"]:
                        q = to_f32(put(flip_i8(pats, noise, rng_for(7, 4, M, K, n, int(noise * 1000), si))))
                        for iters in ((1,) if noise == 0 else (1, 3)):
                            out = np.asarray(sdm_read_iter(q, addr, ctr, K, iters))
                            d = acc.setdefault((M, K, n, noise, iters), {"exact": [], "ber": []})
                            d["exact"].append((out == pats).all(axis=1)); d["ber"].append((out != pats).mean(axis=1))
    if bud.expired():
        print(f"SDM capacity sweep cut short after {bud.s:.0f}s: later (M, load) cells are missing")
    for (M, K, n, noise, iters), d in sorted(acc.items()):
        r1, lo, hi, trials = pooled_recall(d["exact"])
        ber = float(np.concatenate(d["ber"]).mean())
        record("sdm_capacity", "sdm_fixed_k", n=n, d=Ds, noise=noise, variant=f"M{M}_K{K}_it{iters}", hard_locations=M, k_active=K, read_iters=iters, trials=trials,
               exact_pattern_recall=r1, exact_pattern_recall_lo=lo, exact_pattern_recall_hi=hi, bit_error_rate_out=ber, ber_in=noise,
               improves_over_input=bool(ber < noise) if noise > 0 else NAN, max_abs_counter=maxabs.get((M, K, n), NAN),
               flat_search_recall_analytic=analytic_flat_recall(Ds, n, noise), load_per_location=n * K / M)
    cap = results_df("sdm_capacity")
    for (M, K), g in cap[(cap.noise == 0) & (cap.read_iters == 1)].groupby(["hard_locations", "k_active"]):
        ok = g[g.exact_pattern_recall >= 0.9]
        n90 = int(ok.n.max()) if len(ok) else 0
        mx = float(g[g.n <= max(n90, 1)].max_abs_counter.max()) if n90 else NAN
        counter_bytes = M * Ds * (1 if mx < 127 else 2 if mx < 32767 else 4)
        sdm_bytes = counter_bytes + M * Ds / 8                                # counters + bit-packed addresses
        record("sdm_summary", "sdm_fixed_k", d=Ds, variant=f"M{M}_K{K}", hard_locations=M, k_active=K, reliable_patterns_n90=n90, max_abs_counter_at_n90=mx,
               sdm_bytes_int_counters=sdm_bytes, sdm_bytes_f32=2 * M * Ds * 4, bytes_per_reliable_pattern=(sdm_bytes / n90) if n90 else NAN,
               flat_bitpacked_bytes_per_pattern=Ds / 8, flat_f32_bytes_per_pattern=Ds * 4, memory_overhead_vs_flat_bitpacked=(sdm_bytes / n90) / (Ds / 8) if n90 else NAN)
    show("sdm_summary", ["hard_locations", "k_active", "reliable_patterns_n90", "max_abs_counter_at_n90", "sdm_bytes_int_counters", "bytes_per_reliable_pattern",
                         "flat_bitpacked_bytes_per_pattern", "memory_overhead_vs_flat_bitpacked"], sort=["hard_locations", "k_active"])
    noisy = cap[(cap.noise > 0) & (cap.read_iters == 1)]
    worse = noisy[noisy.improves_over_input == False]  # noqa: E712
    print(f"{len(worse)}/{len(noisy)} noisy-cue cells return a pattern with MORE bit errors than the cue it was given (single read).")
    show("sdm_capacity", ["variant", "n", "noise", "read_iters", "load_per_location", "exact_pattern_recall", "bit_error_rate_out", "ber_in", "improves_over_input",
                          "flat_search_recall_analytic"], sort=["variant", "noise", "read_iters", "n"], max_rows=40)

    fig, axes = plt.subplots(1, 3, figsize=(16, 3.9))
    for (M, K), g in cap[(cap.noise == 0) & (cap.read_iters == 1)].groupby(["hard_locations", "k_active"]):
        g = g.sort_values("n")
        axes[0].errorbar(g.n, g.exact_pattern_recall, yerr=yerr(g.exact_pattern_recall, g.exact_pattern_recall_lo, g.exact_pattern_recall_hi), marker="o", ms=3, capsize=1.5, label=f"M={M} K={K}")
    fl = cap[(cap.noise == 0)].sort_values("n").drop_duplicates("n")
    axes[0].plot(fl.n, fl.flat_search_recall_analytic, "k--", lw=1, label="flat search (analytic)")
    axes[0].set(xscale="log", xlabel="stored patterns", ylabel="exact whole-pattern recall", title="SDM capacity (noise-free cue)"); axes[0].legend(fontsize=6); axes[0].grid(alpha=.25)
    nz = [x for x in P["sdm_noise"] if x > 0][-1]
    for (M, K), g in cap[(cap.noise == nz) & (cap.read_iters == 1)].groupby(["hard_locations", "k_active"]):
        g = g.sort_values("n"); axes[1].plot(g.n, g.bit_error_rate_out, "o-", ms=3, label=f"M={M} K={K}")
    axes[1].axhline(nz, color="k", ls="--", lw=1, label="input BER"); axes[1].set(xscale="log", xlabel="stored patterns", ylabel="output bit-error rate", title=f"denoising at {int(nz * 100)}% input noise (single read)")
    axes[1].legend(fontsize=6); axes[1].grid(alpha=.25)
    im = results_df("sdm_impl")
    for name in ("dense_mask", "scatter_gather"):
        g = im[im.method == name].sort_values("hard_locations")
        axes[2].plot(g.hard_locations, g.read_pipe_ms, "o-", label=f"read {name}"); axes[2].plot(g.hard_locations, g.write_pipe_ms, "s--", label=f"write {name}")
    axes[2].set(xscale="log", yscale="log", xlabel="hard locations M", ylabel="pipelined ms", title="implementation cost"); axes[2].legend(fontsize=7); axes[2].grid(alpha=.25, which="both")
    plt.tight_layout(); savefig(fig, "s7_sdm"); plt.show()
else:
    print("SDM study disabled.")
