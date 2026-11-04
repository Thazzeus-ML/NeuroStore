# %% [markdown] id=s3-md
# ## 3. Kernel and precision variants for exact flat search
#
# v1 set a global `highest` matmul precision. For ±1 data that is unnecessary: every product is ±1 and every partial
# sum is an integer far below 2²⁴, so bf16 or int8 inputs with a wide accumulator give *bit-identical* scores. On
# hardware where `highest` costs several passes (TPU) or disables TF32 (GPU) it may change throughput materially. This
# section measures that on whatever backend runs it, and checks exactness, instead of assuming either.
#
# | variant | inputs | accumulate | notes |
# |---|---|---|---|
# | `f32_highest` | float32 | fp32 | v1's choice |
# | `f32_default` | float32 | backend default | may use TF32 / bf16 passes depending on hardware |
# | `bf16_acc32` | bfloat16 | fp32 | `preferred_element_type=float32` |
# | `bf16_bf16out` | bfloat16 | rounded to bf16 | what a naive `a @ b.T` on bf16 arrays does; exact only while scores fit in 8 significant bits |
# | `int8_acc32` | int8 | int32 | integer MXU / tensor-core path |
# | `packed_popcount` | 1 bit/element in uint32 words | integer | `popcount(a xor b)` = Hamming distance; 32× smaller than fp32 |
#
# **Correctness criterion.** Exact ties are common for integer scores, so "same index as the reference" is the wrong
# test. A kernel is *exact* iff the pattern it picks attains the true maximum score (`optimal_rate == 1`).

# %% id=s3-code
#@title 3. Kernel / precision variants (accuracy, throughput, footprint)
clear("kernel_variants"); clear("topk_variants"); clear("precision_vs_D")

K_F32_HI = jax.jit(lambda q, m: jnp.argmax(jnp.matmul(q, m.T, precision=HI), axis=1))
K_F32_DEF = jax.jit(lambda q, m: jnp.argmax(jnp.matmul(q, m.T, precision=DEFAULT_PREC), axis=1))
K_BF16_ACC32 = jax.jit(lambda q, m: jnp.argmax(jnp.matmul(q, m.T, preferred_element_type=jnp.float32), axis=1))
K_BF16_OUT = jax.jit(lambda q, m: jnp.argmax(jnp.matmul(q, m.T), axis=1))
K_I8 = jax.jit(lambda q, m: jnp.argmax(lax.dot_general(q, m, (((1,), (1,)), ((), ())), preferred_element_type=jnp.int32), axis=1))


@partial(jax.jit, static_argnames=("block",))
def k_popcount(qw, mw, block):
    """Top-1 by minimum Hamming distance, scanning the memory in blocks so the (Q, block, words) temporary stays small."""
    n, w = mw.shape
    nb = n // block
    mb = mw.reshape(nb, block, w)
    base = jnp.arange(nb, dtype=jnp.int32) * block

    def body(carry, xs):
        best_d, best_i = carry
        blk, b0 = xs
        dist = jnp.sum(lax.population_count(jnp.bitwise_xor(qw[:, None, :], blk[None, :, :])).astype(jnp.int32), axis=-1)
        loc = jnp.argmin(dist, axis=1)
        dm = jnp.take_along_axis(dist, loc[:, None], axis=1)[:, 0]
        better = dm < best_d                                     # strict: keeps the lowest index on ties, like argmax
        return (jnp.where(better, dm, best_d), jnp.where(better, b0 + loc, best_i)), None

    init = (jnp.full((qw.shape[0],), 2 ** 30, jnp.int32), jnp.zeros((qw.shape[0],), jnp.int32))
    (_, best_i), _ = lax.scan(body, init, (mb, base))
    return best_i


def pick_block(n, q, w, max_elems=2 ** 25):
    cap = max(64, max_elems // max(1, q * w))
    b = 1
    while b * 2 <= min(cap, n) and n % (b * 2) == 0:
        b *= 2
    return b


K_POP = lambda q, m: k_popcount(q, m, block=pick_block(m.shape[0], q.shape[0], m.shape[1]))
KERNELS = {          # name -> (callable, operand family, bytes per stored element)
    "f32_highest": (K_F32_HI, "f32", 4.0), "f32_default": (K_F32_DEF, "f32", 4.0), "bf16_acc32": (K_BF16_ACC32, "bf16", 2.0),
    "bf16_bf16out": (K_BF16_OUT, "bf16", 2.0), "int8_acc32": (K_I8, "i8", 1.0), "packed_popcount": (K_POP, "pack", 1 / 8),
}
# CPUs have no fast bf16/int8 matmul in XLA; cap their size so a CPU session does not stall (accelerators are uncapped)
CPU_NMAX = {"bf16_acc32": 2 ** 13, "bf16_bf16out": 2 ** 13, "int8_acc32": 2 ** 12}


def kernel_allowed(name, n):
    return not (BACKEND == "cpu" and n > CPU_NMAX.get(name, 10 ** 12))


def operands(mem_i8, q_i8):
    m8, q8 = put(mem_i8), put(q_i8)
    return {"f32": (to_f32(q8), to_f32(m8)), "bf16": (to_bf16(q8), to_bf16(m8)), "i8": (q8, m8),
            "pack": (put(pack_bits(q_i8)), put(pack_bits(mem_i8)))}


def ref_max_scores(q32, m32, chunk=1 << 16):
    best = None
    for s in range(0, m32.shape[0], chunk):
        blk = jnp.max(jnp.matmul(q32, m32[s:s + chunk].T, precision=HI), axis=1)
        best = blk if best is None else jnp.maximum(best, blk)
    return np.asarray(best)


def optimal_rate(pred, q_i8, mem_i8, ref_max):
    got = (q_i8.astype(np.int32) * mem_i8[np.asarray(pred)].astype(np.int32)).sum(axis=1)
    return float(np.mean(got == ref_max))


def run_kernel(experiment, name, ops, q_i8, mem_i8, rmax, tg, n, nv, noise, **bench_kw):
    """Accuracy (does the pick attain the maximal score?), timing and footprint of one kernel on prepared operands."""
    fn, fam, eb = KERNELS[name]
    qop, mop = ops[fam]
    qn = qop.shape[0]
    pred = np.asarray(fn(qop, mop))
    st = bench(fn, qop, mop, batch=qn, **bench_kw)
    t = st["pipe_p50_ms"] * 1e-3
    used, _, _ = device_memory()
    return record(experiment, name, n=n, d=D, query_batch=qn, noise=noise, optimal_rate=optimal_rate(pred[:nv], q_i8[:nv], mem_i8, rmax[:nv]),
                  recall_at_1=float(np.mean(pred == tg)), gflops_effective=2.0 * qn * n * D / t / 1e9,
                  gbps_memory=(n * D * eb + qn * D * eb) / t / 1e9, operand_bytes=nbytes(mop), device_bytes_in_use=used, **st)


def sizes_for_kernels():
    cand = [256, 1024] if PROFILE == "smoke" else [1024, 8192, 65536, 262144, 1048576]
    return [n for n in cand if n <= NMAX and fits(n, D * 7.2)]


if RUN_KERNELS:
    KN, NV, NOISE_K = sizes_for_kernels(), 128, 0.40
    for n in KN:
        Qn = min(P["Q"], q_cap(n))
        mem_i8 = bipolar_i8(n, D, rng_for(3, 0, n))
        r = rng_for(3, 1, n)
        tg, q_i8 = make_queries(mem_i8, Qn, NOISE_K, r)
        ops = operands(mem_i8, q_i8)
        nv = min(Qn, NV)
        rmax = ref_max_scores(ops["f32"][0][:nv], ops["f32"][1])
        for name in KERNELS:
            if not kernel_allowed(name, n):
                continue
            run_kernel("kernel_variants", name, ops, q_i8, mem_i8, rmax, tg, n, nv, NOISE_K)
        # exact vs approximate top-10 (matters on TPU where lax.top_k over large N is slow)
        q32, m32 = ops["f32"]
        f_ex = jax.jit(lambda q, m: lax.top_k(jnp.matmul(q, m.T, precision=HI), 10)[1])
        f_ap = jax.jit(lambda q, m: lax.approx_max_k(jnp.matmul(q, m.T, precision=HI), 10, recall_target=0.95)[1])
        ex10 = np.asarray(f_ex(q32, m32))
        for name, fn in (("top10_exact", f_ex), ("top10_approx_r0.95", f_ap)):
            try:
                got = np.asarray(fn(q32, m32))
                ov = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(got, ex10)]))
                st = bench(fn, q32, m32, batch=Qn)
                record("topk_variants", name, n=n, d=D, query_batch=Qn, set_overlap_with_exact=ov, hit_at_10=float(np.mean((got == tg[:, None]).any(1))), **st)
            except Exception as exc:                                    # approx_max_k is backend dependent
                print(f"{name} unavailable at N={n}: {type(exc).__name__}: {str(exc)[:100]}")
        del ops, q32, m32
        gc.collect()

    kv = results_df("kernel_variants")
    show("kernel_variants", ["method", "n", "query_batch", "optimal_rate", "first_call_ms", "lat_p50_ms", "pipe_p50_ms", "qps", "gflops_effective",
                             "gbps_memory", "operand_bytes"], sort=["n", "method"])
    show("topk_variants", ["method", "n", "query_batch", "set_overlap_with_exact", "hit_at_10", "lat_p50_ms", "pipe_p50_ms"], sort=["n", "method"])

    exact_names = ["f32_highest", "f32_default", "bf16_acc32", "int8_acc32", "packed_popcount"]
    sub = kv[kv.method.isin(exact_names)]
    check("S3 f32/bf16(acc32)/int8/popcount kernels all select a maximal-score pattern", bool((sub.optimal_rate == 1.0).all()),
          f"{len(sub)} (kernel, N) cells; min optimal_rate={sub.optimal_rate.min():.4f}" if len(sub) else "no cells")
    nb = kv[kv.method == "bf16_bf16out"]
    if len(nb):
        print(f"naive bf16-output variant: optimal_rate min={nb.optimal_rate.min():.4f} at D={D} "
              f"({'exact here' if (nb.optimal_rate == 1).all() else 'NOT exact: scores rounded to bf16 create false ties'}); see precision_vs_D below.")

    # when does rounding the *output* to bf16 matter? scores are even integers up to D: exact while |score| <= 512, spaced by 2^k beyond
    for dd in sorted({512, 2048, 8192, D}):
        nn_, qq_ = 1024, 128
        mem_i8 = bipolar_i8(nn_, dd, rng_for(3, 5, dd)); r = rng_for(3, 6, dd)
        tg, q_i8 = make_queries(mem_i8, qq_, 0.30, r)
        m32, q32 = to_f32(put(mem_i8)), to_f32(put(q_i8))
        s_ref = np.asarray(jnp.matmul(q32, m32.T, precision=HI))
        s_bf = np.asarray(jnp.matmul(to_bf16(q32), to_bf16(m32).T)).astype(np.float32)
        picked = s_ref[np.arange(qq_), s_bf.argmax(1)]
        record("precision_vs_D", "bf16_bf16out", n=nn_, d=dd, query_batch=qq_, noise=0.30, optimal_rate=float(np.mean(picked == s_ref.max(1))),
               frac_scores_changed=float(np.mean(s_bf != s_ref)))
    show("precision_vs_D", ["method", "d", "n", "optimal_rate", "frac_scores_changed"], sort=["d"])

    if len(kv):
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        for name in KERNELS:
            g = kv[kv.method == name].sort_values("n")
            if len(g):
                ax.plot(g.n, g.pipe_p50_ms, marker="o", label=name)
        ax.set(xscale="log", yscale="log", xlabel="stored patterns N", ylabel="pipelined time per call (ms)", title=f"exact top-1, D={D}, {BACKEND}:{DEVICE_KIND}")
        ax.grid(alpha=.25, which="both"); ax.legend(fontsize=8); plt.tight_layout(); savefig(fig, "s3_kernels"); plt.show()
else:
    print("Kernel study disabled.")
