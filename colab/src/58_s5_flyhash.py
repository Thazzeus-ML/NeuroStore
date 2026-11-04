# %% [markdown] id=s5-md
# ## 5. FlyHash-style sparse projection vs SimHash vs exact search
#
# A sparse random projection maps a ±1 input to a larger space; winner-take-all (WTA) keeps the `k` largest of `m`
# activations; similarity is the overlap of the binary codes. **This is a FlyHash-style proxy, not a recreation of the
# report's `FlyHashCoder`.**
#
# What v1 left open and this cell measures:
#
# * **A control.** SimHash (sign of a dense Gaussian projection, compared by Hamming agreement) at the same bit budget
#   shows how much of the FlyHash behaviour is the sparse-WTA design and how much is simply "a short hash".
# * **Code-size accounting.** v1 reported dense float32 codes and an ideal bit-packed size. A k-of-m code has three
#   honest sizes: dense bitmap (`m` bits), index list (`k·⌈log₂ m⌉` bits) and the information-theoretic bound
#   `log₂ C(m,k)` bits. The raw ±1 input needs only `D` bits, so the table shows when the "expanded" code is *larger*
#   than the data it summarises.
# * **Ties.** Activations of a ±1 input through ±1/√fan-in weights take few distinct values, so many units tie at the
#   WTA cut-off and `top_k` breaks the ties by unit index. `boundary_tie_fraction` reports how often this happens.
#   (Replacing the weights with Gaussians removes most ties; the sweep shows whether recall changes.)
# * **Variants:** larger expansion `m`, larger `k`, larger fan-in.

# %% id=s5-code
#@title 5. FlyHash vs SimHash vs exact
clear("fly_vs_simhash"); clear("fly_timing")


def fly_projection(rng, d, m, fanin, kind="binary"):
    rows = np.argpartition(rng.random((m, d)), fanin - 1, axis=1)[:, :fanin]               # distinct inputs per unit
    vals = (rng.choice([-1.0, 1.0], size=(m, fanin)) if kind == "binary" else rng.normal(size=(m, fanin))) / math.sqrt(fanin)
    out = np.zeros((d, m), np.float32)
    out[rows, np.arange(m)[:, None]] = vals
    return out


@partial(jax.jit, static_argnames=("k",))
def wta_encode(x, proj, k):
    a = jnp.matmul(x, proj, precision=HI)
    idx = lax.top_k(a, k)[1]
    return jnp.zeros(a.shape, jnp.float32).at[jnp.arange(a.shape[0])[:, None], idx].set(1.0)


@partial(jax.jit, static_argnames=("k",))
def wta_boundary_ties(x, proj, k):
    top = lax.top_k(jnp.matmul(x, proj, precision=HI), k + 1)[0]
    return jnp.mean((top[:, k - 1] == top[:, k]).astype(jnp.float32))


@jax.jit
def sign_encode(x, proj):
    return jnp.where(jnp.matmul(x, proj, precision=HI) >= 0, 1.0, -1.0)


@jax.jit
def code_search(qc, mc):
    s = jnp.matmul(qc, mc.T, precision=HI)
    return jnp.argmax(s, axis=1), lax.top_k(s, 10)[1]


FLY_METHODS = {
    "exact_flat": dict(kind="exact"),
    "simhash_512b": dict(kind="sim", bits=512),
    "simhash_2048b": dict(kind="sim", bits=2048),
    "fly_m2048_k64_f16 (v1)": dict(kind="fly", m=2048, k=64, fanin=16, w="binary"),
    "fly_gauss_m2048_k64_f16": dict(kind="fly", m=2048, k=64, fanin=16, w="gauss"),
    "fly_m2048_k128_f16": dict(kind="fly", m=2048, k=128, fanin=16, w="binary"),
    "fly_m8192_k256_f16": dict(kind="fly", m=8192, k=256, fanin=16, w="binary"),
    "fly_m2048_k64_f64": dict(kind="fly", m=2048, k=64, fanin=64, w="binary"),
}
if PROFILE == "smoke":
    FLY_METHODS = {k: v for k, v in FLY_METHODS.items() if k in ("exact_flat", "simhash_512b", "fly_m2048_k64_f16 (v1)", "fly_gauss_m2048_k64_f16")}


def code_sizes(spec):
    if spec["kind"] == "exact":
        return dict(code_bits=D, active_units=D, bitmap_bytes=D / 8, index_list_bytes=D / 8, ideal_bytes=D / 8)
    if spec["kind"] == "sim":
        return dict(code_bits=spec["bits"], active_units=spec["bits"], bitmap_bytes=spec["bits"] / 8, index_list_bytes=spec["bits"] / 8, ideal_bytes=spec["bits"] / 8)
    m, k = spec["m"], spec["k"]
    return dict(code_bits=m, active_units=k, bitmap_bytes=m / 8, index_list_bytes=k * math.ceil(math.log2(m)) / 8, ideal_bytes=ideal_sparse_code_bytes(m, k))


def build_encoder(spec, rng):
    if spec["kind"] == "exact":
        return lambda x: x
    if spec["kind"] == "sim":
        R = put(rng.normal(size=(D, spec["bits"])).astype(np.float32))
        return lambda x: sign_encode(x, R)
    pr = put(fly_projection(rng, D, spec["m"], spec["fanin"], spec["w"]))
    return lambda x: wta_encode(x, pr, spec["k"]), pr


if RUN_FLY:
    fly_ns = [n for n in N_VALUES if n >= 256][:2] or N_VALUES[:1]
    acc, ties = {}, {}
    for n in fly_ns:
        for si in SEEDS:
            mem_np = bipolar_i8(n, D, rng_for(5, 0, n, si)); mem32 = to_f32(put(mem_np))
            qsets = {}
            for noise in P["fly_noise"]:
                tg, q_np = make_queries(mem_np, P["Q"], noise, rng_for(5, 1, n, int(round(noise * 1000)), si))
                qsets[noise] = (tg, to_f32(put(q_np)))
            for mi, (name, spec) in enumerate(FLY_METHODS.items()):
                enc = build_encoder(spec, rng_for(5, 2, mi, si))
                pr = None
                if isinstance(enc, tuple):
                    enc, pr = enc
                mc = enc(mem32)
                for noise, (tg, q32) in qsets.items():
                    i1, i10 = (np.asarray(a) for a in code_search(enc(q32), mc))
                    d = acc.setdefault((n, name, noise), {"c1": [], "c10": []})
                    d["c1"].append(i1 == tg); d["c10"].append((i10 == tg[:, None]).any(1))
                    if pr is not None and noise == 0.25:
                        ties.setdefault((n, name), []).append(float(wta_boundary_ties(q32, pr, spec["k"])))
    for (n, name, noise), d in sorted(acc.items()):
        r1, lo, hi, trials = pooled_recall(d["c1"]); r10, lo10, hi10, _ = pooled_recall(d["c10"])
        extra = dict(analytic_recall=analytic_flat_recall(D, n, noise)) if name == "exact_flat" else {}
        spec = FLY_METHODS[name]
        record("fly_vs_simhash", name, n=n, d=D, noise=noise, query_batch=P["Q"], trials=trials, recall_at_1=r1, recall_at_1_lo=lo, recall_at_1_hi=hi,
               recall_at_10=r10, recall_at_10_lo=lo10, recall_at_10_hi=hi10, boundary_tie_fraction=float(np.mean(ties[(n, name)])) if (n, name) in ties else NAN,
               **code_sizes(spec), **extra)
    fv = results_df("fly_vs_simhash")
    n0 = fly_ns[0]
    noise_ref = min(P["fly_noise"], key=lambda x: abs(x - 0.25))
    pivot = fv[fv.n == n0].pivot_table(index="method", columns="noise", values="recall_at_1").round(3)
    print(f"recall@1 at N={n0} (rows: method, columns: fraction of query bits flipped)")
    display(pivot)
    show("fly_vs_simhash", ["method", "code_bits", "active_units", "bitmap_bytes", "index_list_bytes", "ideal_bytes", "boundary_tie_fraction"], sort=["method"], n=n0, noise=noise_ref)
    print(f"raw +-1 input: {D / 8:.0f} bytes/item bit-packed. A code whose bitmap / index-list / entropy-bound size exceeds that is larger than the data it encodes "
          "(only the entropy bound can be smaller, and it needs combinatorial coding to reach).")
    base = fv[(fv.method == "exact_flat") & (fv.n == n0)]
    zz = [abs(r.recall_at_1 - r.analytic_recall) <= 4.5 * math.sqrt(max(r.analytic_recall * (1 - r.analytic_recall), 1e-12) / r.trials) + 1.0 / r.trials for r in base.itertuples()]
    check("S5 exact-search recall in the FlyHash comparison matches the analytic model", all(zz), f"{len(zz)} noise levels at N={n0}")
    ones = np.asarray(wta_encode(to_f32(put(bipolar_i8(32, D, rng_for(5, 9)))), put(fly_projection(rng_for(5, 8), D, 2048, 16)), 64)).sum(1)
    check("S5 every WTA code has exactly k active units", bool((ones == 64).all()), f"counts {sorted(set(ones.astype(int).tolist()))}")

    # timing of encode + search at the largest N of this section (device resident)
    n_t = fly_ns[-1]
    mem_np = bipolar_i8(n_t, D, rng_for(5, 3)); _, q_np = make_queries(mem_np, P["Q"], 0.25, rng_for(5, 4))
    m32, q32 = to_f32(put(mem_np)), to_f32(put(q_np))
    for mi, (name, spec) in enumerate(FLY_METHODS.items()):
        enc = build_encoder(spec, rng_for(5, 2, mi, 0)); enc = enc[0] if isinstance(enc, tuple) else enc
        mc = enc(m32)
        fn = (lambda q, enc=enc, mc=mc: code_search(enc(q), mc))
        record("fly_timing", name, n=n_t, d=D, query_batch=P["Q"], **bench(fn, q32, batch=P["Q"]), **code_sizes(spec))
    show("fly_timing", ["method", "n", "pipe_p50_ms", "qps", "code_bits", "bitmap_bytes"], sort=["pipe_p50_ms"])

    fig, axes = plt.subplots(1, len(fly_ns), figsize=(6.2 * len(fly_ns), 4), sharey=True, squeeze=False)
    for ax, n in zip(axes[0], fly_ns):
        for name in FLY_METHODS:
            g = fv[(fv.n == n) & (fv.method == name)].sort_values("noise")
            ax.errorbar(g.noise, g.recall_at_1, yerr=yerr(g.recall_at_1, g.recall_at_1_lo, g.recall_at_1_hi), marker="o", ms=3, capsize=1.5, label=name, lw=1.2)
        ax.set(title=f"N={n}", xlabel="fraction of query bits flipped"); ax.grid(alpha=.25)
    axes[0][0].set_ylabel("recall@1"); axes[0][-1].legend(fontsize=6.5)
    plt.tight_layout(); savefig(fig, "s5_flyhash"); plt.show()
else:
    print("FlyHash study disabled.")
