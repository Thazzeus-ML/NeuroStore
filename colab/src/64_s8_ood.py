# %% [markdown] id=s8-md
# ## 8. Out-of-distribution rejection
#
# In v1 the OOD queries were independent random vectors; they score about 0.12 maximum cosine against the memory while a
# 25 %-noise known query scores about 0.5, a gap of roughly nine standard deviations, so AUROC = 1.0 for *any* sensible
# score and the test could not fail. v2 uses families that are progressively harder, and four scores:
#
# | family | construction | expected behaviour |
# |---|---|---|
# | `random` | independent ±1 vectors | trivial |
# | `chimera` | half the bits from stored pattern A, half from B | max-cosine ≈ 0.5, same as a 25 %-noise known query, so `max_cos` should fail and the *margin* should work |
# | `partial_f` | fraction `f` copied from one stored pattern, rest random | cosine ≈ f; at `f = 0.5` it is statistically identical to a 25 %-noise known query, so AUROC must be 0.5 (a built-in sanity check of the metric) |
#
# Scores (higher = more "known"): `max_cos`, `margin_top1_top2`, `neg_entropy` of the Hopfield attention weights, `logsumexp`
# (negative Hopfield energy). Thresholds are calibrated on a *separate* set of known queries (5th percentile → nominal 95 % TPR)
# and evaluated on independent known and OOD queries. AUROC intervals are bootstrap intervals.

# %% id=s8-code
#@title 8. OOD rejection: families x scores
clear("ood")


@jax.jit
def ood_scores(mem, q, beta):
    s = jnp.matmul(q, mem.T, precision=HI) / mem.shape[1]
    t2 = lax.top_k(s, 2)[0]
    w = jax.nn.softmax(beta * s, axis=-1)
    return jnp.stack([t2[:, 0], t2[:, 0] - t2[:, 1], jnp.sum(w * jnp.log(w + 1e-30), axis=-1), jax.nn.logsumexp(beta * s, axis=-1) / beta], axis=1)


SCORE_NAMES = ["max_cos", "margin_top1_top2", "neg_entropy", "logsumexp"]
PARTIALS = [0.1, 0.2, 0.3, 0.4, 0.5]
OOD_FAMILIES = ["random", "chimera"] + [f"partial_{f}" for f in PARTIALS]


def ood_queries(name, mem_i8, n_q, rng):
    n, d = mem_i8.shape
    if name.startswith("known_p"):
        return flip_i8(mem_i8[rng.integers(0, n, n_q)], int(name[len("known_p"):]) / 100.0, rng)
    if name == "random":
        return bipolar_i8(n_q, d, rng)
    if name == "chimera":
        a = rng.integers(0, n, n_q)
        b = (a + 1 + rng.integers(0, n - 1, n_q)) % n
        return np.where(rng.random((n_q, d)) < 0.5, mem_i8[a], mem_i8[b]).astype(np.int8)
    f = float(name.split("_")[1])
    return np.where(rng.random((n_q, d)) < f, mem_i8[rng.integers(0, n, n_q)], bipolar_i8(n_q, d, rng)).astype(np.int8)


def auroc_ci(pos, neg, n_boot=150, seed=0):
    rng = np.random.default_rng(seed)
    vals = [auroc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]) for _ in range(n_boot)]
    return tuple(np.quantile(vals, [0.025, 0.975]))


if RUN_OOD:
    n_o, q_o = P["ood_n"], P["ood_q"]
    sets = {k: [] for k in ["known_p25_cal", "known_p25", "known_p40"] + OOD_FAMILIES}
    for si in SEEDS:
        mem_i8 = bipolar_i8(n_o, D, rng_for(8, 0, si)); mem = to_f32(put(mem_i8))
        for fam in sets:
            base = "known_p25" if fam.startswith("known_p25") else fam
            q = ood_queries(base, mem_i8, q_o, rng_for(8, 1, si, list(sets).index(fam)))
            sets[fam].append(np.asarray(ood_scores(mem, to_f32(put(q)), BETA)))
    S = {k: np.concatenate(v) for k, v in sets.items()}
    rows = []
    for si_, sname in enumerate(SCORE_NAMES):
        thr = float(np.quantile(S["known_p25_cal"][:, si_], 0.05))
        tpr = float(np.mean(S["known_p25"][:, si_] >= thr))
        for fam in OOD_FAMILIES + ["known_p40"]:
            pos, neg = (S["known_p25"], S[fam]) if fam != "known_p40" else (S["known_p40"], S["random"])
            pos_s, neg_s = pos[:, si_], neg[:, si_]
            a = auroc(pos_s, neg_s)
            lo, hi = auroc_ci(pos_s, neg_s, seed=si_)
            rows.append(record("ood", sname, scenario=("known25_vs_" + fam) if fam != "known_p40" else "known40_vs_random", n=n_o, d=D, query_batch=len(pos_s), auroc=a, auroc_lo=lo, auroc_hi=hi,
                               calibrated_threshold=thr, tpr_known=tpr, tpr_pos=float(np.mean(pos_s >= thr)), fpr_ood=float(np.mean(neg_s >= thr)),
                               n_pos=len(pos_s), n_neg=len(neg_s)))
    od = results_df("ood")
    piv = od.pivot_table(index="scenario", columns="method", values="auroc")[SCORE_NAMES].round(3)
    print(f"AUROC (positives = known queries; {len(S['known_p25'])} per set pooled over {len(SEEDS)} seeds). 0.5 = chance.")
    display(piv)
    pf = od.pivot_table(index="scenario", columns="method", values="fpr_ood")[SCORE_NAMES].round(3)
    print("False-positive rate of the calibrated threshold (OOD queries wrongly accepted; target is small when TPR ~ 0.95):")
    display(pf)
    print("TPR of the calibrated thresholds on independent known queries:", {r.method: round(r.tpr_known, 3) for r in od.drop_duplicates('method').itertuples()})
    half = od[(od.scenario == "known25_vs_partial_0.5") & (od.method == "max_cos")].iloc[0]
    check("S8 partial_0.5 is indistinguishable from known-25%-noise (AUROC of max_cos within [0.45, 0.55] and CI contains 0.5)",
          0.45 <= half.auroc <= 0.55 and half.auroc_lo <= 0.5 <= half.auroc_hi, f"AUROC={half.auroc:.3f} CI=[{half.auroc_lo:.3f},{half.auroc_hi:.3f}]")
    rnd = od[(od.scenario == "known25_vs_random")]
    print(f"known25 vs random is {'trivial' if (rnd.auroc > 0.999).all() else 'not trivial'} here (AUROC {rnd.auroc.min():.4f}-{rnd.auroc.max():.4f}); harder families are the informative ones.")
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    im = ax.imshow(piv.values, vmin=0.4, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(SCORE_NAMES)), SCORE_NAMES, rotation=20); ax.set_yticks(range(len(piv.index)), piv.index, fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=7)
    fig.colorbar(im, label="AUROC"); ax.set_title("OOD AUROC by family and score"); plt.tight_layout(); savefig(fig, "s8_ood"); plt.show()
else:
    print("OOD study disabled.")
