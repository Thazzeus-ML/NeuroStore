# %% [markdown] id=s6-md
# ## 6. Role–filler VSA
#
# **A. Structured record retrieval.** Five-field records (department, role, location, team, unique ID), each field bound to its
# role vector and bundled into one ±1 record. The cue is the bundle of the first three fields. v1 reported ID top-1 and
# compared it with an "oracle"; but all records sharing a cue tuple have *identical* cue vectors, so any deterministic
# system returns the same record for all of them and its top-1 equals `distinct tuples / records` by construction. The
# informative quantities are therefore **tuple-level accuracy** (does the retrieved record share the cue's three fields?)
# and how it degrades with dimension.
#
# **B. Bundle-decode capacity.** Unbind one role from a bundle of `k` role–filler pairs and clean up against a codebook of `F`
# fillers. A majority bundle of `k` terms correlates with each term by `C(k-1,(k-1)/2)/2^(k-1)`, which makes decoding
# equivalent to flat retrieval with an effective flip probability `(1-c_k)/2`; the closed form from Section A of the helpers
# therefore predicts decode accuracy for every `(D, k)`, and the measurements must agree with it. The bundle sizes probed
# for each dimension are *derived from that closed form* (the sizes at which accuracy should be 99.5 %, 97 %, 90 %, 70 %, 50 %
# and 30 %), so every dimension is measured through its own transition region instead of at a fixed k where large D is
# trivially perfect.

# %% id=s6-code
#@title 6. VSA: record retrieval and bundle-decode capacity
clear("vsa_records"); clear("vsa_decode")


@jax.jit
def vsa_top1(records, cues):
    s = jnp.matmul(cues, records.T, precision=HI)
    return jnp.argmax(s, axis=1), lax.top_k(s, 10)[1]


def vsa_make_records(rng, d, n_rec):
    sizes = [6, 8, 6, 10]
    fields = np.stack([rng.integers(0, s, size=n_rec) for s in sizes] + [np.arange(n_rec)], axis=1)
    roles = bipolar_i8(5, d, rng)
    fillers = [bipolar_i8(s, d, rng) for s in sizes + [n_rec]]
    bound = [roles[j][None, :].astype(np.int16) * fillers[j][fields[:, j]].astype(np.int16) for j in range(5)]
    rec = np.where(sum(bound) >= 0, 1, -1).astype(np.float32)           # five terms: never zero
    cue = np.where(sum(bound[:3]) >= 0, 1, -1).astype(np.float32)       # three terms: never zero
    return fields, rec, cue


@jax.jit
def vsa_bundle(roles, codebook, idx):
    """sign(sum_j role_j * filler[idx[:, j]]) for k = roles.shape[0] terms (k odd: never zero). A rolled loop keeps compile time flat in k."""
    def body(j, acc):
        r = lax.dynamic_index_in_dim(roles, j, axis=0, keepdims=False).astype(jnp.int32)
        ids = lax.dynamic_index_in_dim(idx, j, axis=1, keepdims=False)
        return acc + r[None, :] * codebook[ids].astype(jnp.int32)
    acc = lax.fori_loop(0, roles.shape[0], body, jnp.zeros((idx.shape[0], roles.shape[1]), jnp.int32))
    return jnp.where(acc >= 0, 1.0, -1.0)


@jax.jit
def vsa_decode_first(bundle, role0, codebook):
    u = bundle * role0[None, :].astype(jnp.float32)
    return jnp.argmax(jnp.matmul(u, codebook.astype(jnp.float32).T, precision=HI), axis=1)


def vsa_k_for_accuracy(dd, n_fillers, target, k_max=4001):
    """Smallest odd bundle size at which the closed-form decode accuracy falls to `target` (accuracy decreases with k); bisection."""
    lo, hi = 0, (k_max - 1) // 2                       # k = 2m + 1
    if analytic_vsa_decode(dd, n_fillers, 2 * hi + 1) > target:
        return 2 * hi + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if analytic_vsa_decode(dd, n_fillers, 2 * mid + 1) > target:
            lo = mid + 1
        else:
            hi = mid
    return 2 * lo + 1


if RUN_VSA:
    # ---------------- A. record retrieval
    for dd in P["vsa_dims"]:
        accA = {k: [] for k in ("tuple", "id1", "id10", "uniq")}
        distinct, ambiguous = [], []
        for si in SEEDS:
            fields, rec, cue = vsa_make_records(rng_for(6, 0, dd, si), dd, P["vsa_records"])
            pred, top10 = (np.asarray(a) for a in vsa_top1(put(rec), put(cue)))
            truth = np.arange(len(rec))
            tuples = [tuple(r) for r in fields[:, :3]]
            cnt = Counter(tuples); uniq = np.array([cnt[t] == 1 for t in tuples])
            accA["tuple"].append(np.array([tuples[int(p)] == t for p, t in zip(pred, tuples)]))
            accA["id1"].append(pred == truth); accA["id10"].append((top10 == truth[:, None]).any(1)); accA["uniq"].append((pred == truth)[uniq])
            distinct.append(len(cnt) / len(rec)); ambiguous.append(1 - uniq.mean())
        t1, tlo, thi, trials = pooled_recall(accA["tuple"])
        record("vsa_records", "vsa_proxy", n=P["vsa_records"], d=dd, trials=trials, tuple_level_accuracy=t1, tuple_level_lo=tlo, tuple_level_hi=thi,
               id_top1=float(np.concatenate(accA["id1"]).mean()), id_top10=float(np.concatenate(accA["id10"]).mean()),
               unique_cue_id_top1=float(np.concatenate(accA["uniq"]).mean()) if sum(len(a) for a in accA["uniq"]) else NAN,
               oracle_id_top1_distinct_over_n=float(np.mean(distinct)), ambiguous_cue_fraction=float(np.mean(ambiguous)))
    show("vsa_records", ["d", "n", "trials", "tuple_level_accuracy", "tuple_level_lo", "tuple_level_hi", "id_top1", "oracle_id_top1_distinct_over_n", "id_top10",
                         "unique_cue_id_top1", "ambiguous_cue_fraction"], sort=["d"])

    # ---------------- B. decode capacity vs bundle size and dimension
    F = 64
    for dd in P["vsa_dims"]:
        # k grid chosen from the analytic model so every dimension is probed from ~perfect to ~failing decode
        ks = sorted({1, 5} | {vsa_k_for_accuracy(dd, F, t) for t in P["vsa_targets"]})
        print(f"D={dd}: bundle sizes {ks}")
        for k in ks:
            correct = []
            for si in SEEDS:
                r = rng_for(6, 1, dd, k, si)
                roles = bipolar_i8(k, dd, r); codebook = bipolar_i8(F, dd, r)
                idx = r.integers(0, F, size=(P["vsa_trials"], k)).astype(np.int32)
                bundle = vsa_bundle(put(roles), put(codebook), put(idx))
                correct.append(np.asarray(vsa_decode_first(bundle, put(roles[0]), put(codebook))) == idx[:, 0])
            r1, lo, hi, trials = pooled_recall(correct)
            ana = analytic_vsa_decode(dd, F, k)
            record("vsa_decode", "role_unbind_cleanup", n=F, d=dd, scenario=f"k={k}", trials=trials, bundle_terms=k, recall_at_1=r1, recall_at_1_lo=lo, recall_at_1_hi=hi,
                   analytic_recall=ana, z_vs_analytic=(r1 - ana) / math.sqrt(max(ana * (1 - ana), 1e-12) / trials), bundle_corr=vsa_bundle_corr(k))
    vd = results_df("vsa_decode")
    odd = vd[(vd.analytic_recall > 1e-3) & (vd.analytic_recall < 1 - 1e-3)]
    viol = vd[abs(vd.recall_at_1 - vd.analytic_recall) > 4.5 * np.sqrt(np.maximum(vd.analytic_recall * (1 - vd.analytic_recall), 1e-12) / vd.trials) + 1.0 / vd.trials]
    check("S6 VSA decode accuracy matches the closed-form prediction for every (D, k)", len(viol) == 0, f"{len(vd)} cells, {len(odd)} informative; violations: {len(viol)}")
    print(vd.pivot_table(index="bundle_terms", columns="d", values="recall_at_1").round(3).to_string())
    print("(NaN = that bundle size was not probed for that dimension; grids are dimension specific)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    for dd in P["vsa_dims"]:
        g = vd[vd.d == dd].sort_values("bundle_terms")
        axes[0].errorbar(g.bundle_terms, g.recall_at_1, yerr=yerr(g.recall_at_1, g.recall_at_1_lo, g.recall_at_1_hi), marker="o", capsize=2, label=f"D={dd}")
        axes[0].plot(g.bundle_terms, g.analytic_recall, "k:", lw=1)
    axes[0].set(xlabel="role-filler pairs bundled (k)", ylabel="decode accuracy (F=64 fillers)", title="measured (points) vs closed form (dotted)"); axes[0].legend(); axes[0].grid(alpha=.25)
    ra = results_df("vsa_records").sort_values("d")
    axes[1].errorbar(ra.d, ra.tuple_level_accuracy, yerr=yerr(ra.tuple_level_accuracy, ra.tuple_level_lo, ra.tuple_level_hi), marker="o", capsize=2, label="tuple-level accuracy")
    axes[1].plot(ra.d, ra.id_top1, "s--", label="ID top-1"); axes[1].plot(ra.d, ra.oracle_id_top1_distinct_over_n, "k:", label="ID top-1 bound (distinct/N)")
    axes[1].set(xscale="log", xlabel="dimension D", ylabel="accuracy", title="record retrieval from a 3-field cue"); axes[1].legend(fontsize=8); axes[1].grid(alpha=.25)
    plt.tight_layout(); savefig(fig, "s6_vsa"); plt.show()
else:
    print("VSA study disabled.")
