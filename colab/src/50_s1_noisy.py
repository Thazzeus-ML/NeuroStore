# %% [markdown] id=s1-md
# ## 1. Noisy-key associative retrieval: exact flat search vs Modern-Hopfield cleanup
#
# Equal-norm ±1 patterns, queries are stored patterns with each bit flipped independently, **the same queries for both
# methods** (paired). Results are pooled over several seeds (each seed = a new memory and new queries) and reported with
# Wilson 95 % intervals. The `analytic` column is the exact expectation for flat search; measured recall must agree
# with it (self-check). Rows where the analytic recall is 1 − 1e-9 or better carry no information about either method.
#
# **How to read it.** Hopfield softmax preserves the ranking of its input similarities, so the *first* stage cannot
# differ from exact search. Any difference comes from the reconstruction-and-cleanup stage (cross-talk between stored
# patterns). The McNemar column tests whether the paired difference is distinguishable from zero.

# %% id=s1-code
#@title 1. Noisy-key retrieval (multi-seed, paired, analytically validated)
clear("noisy_recall"); clear("noisy_recall_timing")


@jax.jit
def exact_eval(mem, q):
    s = jnp.matmul(q, mem.T, precision=HI)
    mx = jnp.max(s, axis=1)
    return jnp.argmax(s, axis=1), lax.top_k(s, 10)[1], jnp.sum(s == mx[:, None], axis=1)


@jax.jit
def hopfield_eval(mem, q, beta):
    s = jnp.matmul(q, mem.T, precision=HI) / mem.shape[1]
    w = jax.nn.softmax(beta * s, axis=-1)
    rec = jnp.matmul(w, mem, precision=HI)                       # attention read-out
    c = jnp.matmul(rec, mem.T, precision=HI)                     # cleanup against the stored patterns
    return jnp.argmax(c, axis=1), lax.top_k(c, 10)[1], rec


def make_queries(mem_np, n_q, noise, rng):
    tg = rng.integers(0, len(mem_np), size=n_q)
    return tg, flip_i8(mem_np[tg], noise, rng)


N_VALUES, NOISE = P["n_values"], P["noise_levels"]
S1 = {}
for n in N_VALUES:
    for si in SEEDS:
        mem_np = bipolar_i8(n, D, rng_for(1, 0, n, si))
        mem = to_f32(put(mem_np))
        for noise in NOISE:
            r = rng_for(1, 1, n, int(round(noise * 1000)), si)
            tg, q_np = make_queries(mem_np, P["Q"], noise, r)
            q = to_f32(put(q_np))
            e1, e10, ties = (np.asarray(a) for a in exact_eval(mem, q))
            h1, h10, rec = hopfield_eval(mem, q, BETA)
            h1, h10, rec = np.asarray(h1), np.asarray(h10), np.asarray(rec)
            d = S1.setdefault((n, noise), {k: [] for k in ("e1", "e10", "h1", "h10", "ties", "ber", "agree")})
            d["e1"].append(e1 == tg); d["h1"].append(h1 == tg)
            d["e10"].append((e10 == tg[:, None]).any(1)); d["h10"].append((h10 == tg[:, None]).any(1))
            d["ties"].append(ties > 1); d["agree"].append(e1 == h1)
            d["ber"].append((np.where(rec >= 0, 1, -1) != mem_np[tg]).mean(axis=1))
        if si == 0:                                   # device timing once per N (does not depend on the noise level)
            r = rng_for(1, 2, n)
            _, q_np = make_queries(mem_np, P["Q"], 0.25, r)
            q = to_f32(put(q_np))
            for name, fn, args in (("exact_flat", exact_eval, (mem, q)), ("hopfield_cleanup", hopfield_eval, (mem, q, BETA))):
                record("noisy_recall_timing", name, n=n, d=D, query_batch=P["Q"], **bench(fn, *args, batch=P["Q"]), index_bytes=nbytes(mem))

rows = []
for (n, noise), d in sorted(S1.items()):
    ana = analytic_flat_recall(D, n, noise)
    e1, h1 = np.concatenate(d["e1"]), np.concatenate(d["h1"])
    b, c = int((e1 & ~h1).sum()), int((~e1 & h1).sum())
    pd_m, pd_lo, pd_hi = paired_diff_ci(h1, e1)
    for tag, name in (("e", "exact_flat"), ("h", "hopfield_cleanup")):
        r1, lo, hi, trials = pooled_recall(d[tag + "1"])
        r10, lo10, hi10, _ = pooled_recall(d[tag + "10"])
        se = math.sqrt(max(ana * (1 - ana), 1e-12) / trials)
        extra = dict(analytic_recall=ana, z_vs_analytic=(r1 - ana) / se) if tag == "e" else dict(
            paired_diff_vs_exact=pd_m, paired_diff_lo=pd_lo, paired_diff_hi=pd_hi, mcnemar_p=mcnemar_exact(b, c),
            disagreement_with_exact=1 - float(np.concatenate(d["agree"]).mean()), ber_out=float(np.concatenate(d["ber"]).mean()), ber_in=noise)
        rows.append(record("noisy_recall", name, n=n, d=D, noise=noise, query_batch=P["Q"], trials=trials, n_seeds=len(SEEDS),
                           recall_at_1=r1, recall_at_1_lo=lo, recall_at_1_hi=hi, recall_at_10=r10, recall_at_10_lo=lo10, recall_at_10_hi=hi10,
                           tie_fraction=float(np.concatenate(d["ties"]).mean()), index_bytes=n * D * 4, **extra))

view = pd.DataFrame(rows)
tbl = view.pivot_table(index=["n", "noise"], columns="method", values=["recall_at_1", "recall_at_1_lo", "recall_at_1_hi"]).round(4)
out = pd.DataFrame({
    "trials": view[view.method == "exact_flat"].set_index(["n", "noise"]).trials,
    "exact [95% CI]": [f"{r:.3f} [{lo:.3f},{hi:.3f}]" for r, lo, hi in zip(*[view[view.method == 'exact_flat'][c] for c in ("recall_at_1", "recall_at_1_lo", "recall_at_1_hi")])],
    "hopfield [95% CI]": [f"{r:.3f} [{lo:.3f},{hi:.3f}]" for r, lo, hi in zip(*[view[view.method == 'hopfield_cleanup'][c] for c in ("recall_at_1", "recall_at_1_lo", "recall_at_1_hi")])],
    "analytic exact": view[view.method == "exact_flat"].set_index(["n", "noise"]).analytic_recall.round(5),
    "Δ hopfield−exact": view[view.method == "hopfield_cleanup"].set_index(["n", "noise"]).paired_diff_vs_exact.round(4),
    "McNemar p": view[view.method == "hopfield_cleanup"].set_index(["n", "noise"]).mcnemar_p.round(4),
    "disagree": view[view.method == "hopfield_cleanup"].set_index(["n", "noise"]).disagreement_with_exact.round(4),
})
display(out)
show("noisy_recall_timing", ["method", "n", "query_batch", "first_call_ms", "lat_p50_ms", "pipe_p50_ms", "qps", "n_samples"], sort=["n", "method"])

# ---- self-check: measured exact recall must match the analytic expectation
bad = [(n, nz, round(r.z_vs_analytic, 2)) for (n, nz), r in
       ((k, view[(view.n == k[0]) & (view.noise == k[1]) & (view.method == "exact_flat")].iloc[0]) for k in S1)
       if abs(r.recall_at_1 - r.analytic_recall) > 4.5 * math.sqrt(max(r.analytic_recall * (1 - r.analytic_recall), 1e-12) / r.trials) + 1.0 / r.trials]
check("S1 exact-search recall agrees with the analytic model (|z| <= 4.5)", not bad, f"violations: {bad}" if bad else f"{len(S1)} (N, noise) cells")
sat = view[(view.method == "exact_flat") & (view.analytic_recall > 1 - 1e-9)]
print(f"{len(sat)}/{len(S1)} (N, noise) cells have an analytic error probability below 1e-9: they cannot discriminate any method.")

fig, axes = plt.subplots(1, len(N_VALUES), figsize=(4.4 * len(N_VALUES), 3.8), sharey=True, squeeze=False)
for ax, n in zip(axes[0], N_VALUES):
    for name, mk in (("exact_flat", "o"), ("hopfield_cleanup", "s")):
        g = view[(view.n == n) & (view.method == name)].sort_values("noise")
        ax.errorbar(g.noise, g.recall_at_1, yerr=yerr(g.recall_at_1, g.recall_at_1_lo, g.recall_at_1_hi), marker=mk, capsize=2, label=name, alpha=.85)
    g = view[(view.n == n) & (view.method == "exact_flat")].sort_values("noise")
    ax.plot(g.noise, g.analytic_recall, "k--", lw=1, label="analytic (exact)")
    ax.set(title=f"N={n}, D={D}", xlabel="fraction of query bits flipped"); ax.grid(alpha=.25)
axes[0][0].set_ylabel("recall@1 (95% Wilson CI)"); axes[0][0].legend(fontsize=8)
plt.tight_layout(); savefig(fig, "s1_noisy_recall"); plt.show()
