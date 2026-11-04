# %% [markdown] id=s9-md
# ## 9. Hardware fault-injection simulation
#
# In v1 every fault condition (up to 5 % stored bit flips and Gaussian noise with σ = 0.5) left recall at 1.000, because
# the operating point (N = 256, 20 % query noise) has an error probability near 1e-11: the test was saturated.
#
# **v2 first chooses an operating point from the analytic model**: the query-noise level at which fault-free exact search has
# 95 % recall for this `N` and `D`. Degradation in either direction is then visible, and the fault-free row is checked against
# the model. Faults are applied to the *stored* memory (static) or to the *read-out* (dynamic):
#
# | mode | meaning | severity |
# |---|---|---|
# | `bitflip` | stored bits inverted | probability |
# | `stuck_zero` | cell reads 0 (open) | fraction of cells |
# | `stuck_pos` | cell stuck at +1 | fraction of cells |
# | `dead_rows` | a whole stored pattern unreadable | fraction of patterns |
# | `dead_cols` | a whole dimension unreadable for every pattern | fraction of dimensions |
# | `gauss_static` | fabrication-style additive variation on each stored value | σ |
# | `lognormal_gain` | multiplicative per-cell gain `exp(N(0, σ²))` | σ |
# | `read_noise` | fresh Gaussian noise on every similarity score at every query | σ in cosine units |
# | `adc_bits` | similarity scores quantised to 2^b levels over [-1, 1] | bits (lower is worse) |
#
# These are **functional** fault models. They are not calibrated FeNAND, crossbar or compute-in-memory device physics, and
# say nothing about energy, endurance, retention, line resistance or ADC/DAC power.

# %% id=s9-code
#@title 9. Fault injection at an informative operating point
clear("fault_injection")


def _fscores(mem, q, noise, adc_step):
    s = jnp.matmul(q, mem.T, precision=HI) / mem.shape[1] + noise
    return jnp.where(adc_step > 0, jnp.round((s + 1.0) / jnp.maximum(adc_step, 1e-9)) * adc_step - 1.0, s)


@jax.jit
def faulty_exact(mem, q, noise, adc_step):
    return jnp.argmax(_fscores(mem, q, noise, adc_step), axis=1)


@jax.jit
def faulty_hopfield(mem, q, noise, adc_step, beta):
    w = jax.nn.softmax(beta * _fscores(mem, q, noise, adc_step), axis=-1)
    return jnp.argmax(jnp.matmul(jnp.matmul(w, mem, precision=HI), mem.T, precision=HI), axis=1)


def apply_fault(mem_i8, mode, sev, rng):
    m = mem_i8.astype(np.float32)
    if mode == "bitflip":
        m[rng.random(m.shape) < sev] *= -1
    elif mode == "stuck_zero":
        m[rng.random(m.shape) < sev] = 0.0
    elif mode == "stuck_pos":
        m[rng.random(m.shape) < sev] = 1.0
    elif mode == "dead_rows":
        m[rng.random(len(m)) < sev, :] = 0.0
    elif mode == "dead_cols":
        m[:, rng.random(m.shape[1]) < sev] = 0.0
    elif mode == "gauss_static":
        m += rng.normal(0.0, sev, m.shape).astype(np.float32)
    elif mode == "lognormal_gain":
        m *= np.exp(rng.normal(0.0, sev, m.shape)).astype(np.float32)
    return m


FAULTS = [("bitflip", [0.01, 0.05, 0.10, 0.20, 0.30]), ("stuck_zero", [0.05, 0.10, 0.25, 0.50, 0.75]), ("stuck_pos", [0.05, 0.10, 0.25, 0.50]),
          ("dead_rows", [0.01, 0.05, 0.10, 0.25]), ("dead_cols", [0.05, 0.10, 0.25, 0.50]), ("gauss_static", [0.25, 0.5, 1.0, 2.0, 4.0]),
          ("lognormal_gain", [0.1, 0.3, 0.5, 1.0]), ("read_noise", [0.005, 0.01, 0.02, 0.05]), ("adc_bits", [8, 6, 5, 4, 3, 2])]

if RUN_FAULTS:
    n_f, q_f = P["fault_n"], P["fault_q"]
    p_star = solve_noise_for_recall(D, n_f, 0.95)
    print(f"operating point: N={n_f}, D={D}, query flip probability {p_star:.4f} -> analytic fault-free recall {analytic_flat_recall(D, n_f, p_star):.4f}")
    METHODS = {"exact_flat": lambda m, q, nz, st: faulty_exact(m, q, nz, st), "hopfield_cleanup": lambda m, q, nz, st: faulty_hopfield(m, q, nz, st, BETA)}
    acc = {}
    for si in SEEDS:
        mem_i8 = bipolar_i8(n_f, D, rng_for(9, 0, si))
        tg, q_np = make_queries(mem_i8, q_f, p_star, rng_for(9, 1, si))
        q = to_f32(put(q_np))
        zero_noise, no_adc = put(np.zeros((q_f, n_f), np.float32)), jnp.float32(0.0)
        base_mem = put(mem_i8.astype(np.float32))
        for name, fn in METHODS.items():
            acc.setdefault(("baseline", 0.0, name), []).append(np.asarray(fn(base_mem, q, zero_noise, no_adc)) == tg)
        for mi, (mode, sevs) in enumerate(FAULTS):
            for sev in sevs:
                r = rng_for(9, 2, mi, int(round(sev * 1000)), si)
                mem, nz, st = base_mem, zero_noise, no_adc
                if mode == "read_noise":
                    nz = put(r.normal(0.0, sev, (q_f, n_f)).astype(np.float32))
                elif mode == "adc_bits":
                    st = jnp.float32(2.0 / (2 ** int(sev) - 1))
                else:
                    mem = put(apply_fault(mem_i8, mode, sev, r))
                for name, fn in METHODS.items():
                    acc.setdefault((mode, float(sev), name), []).append(np.asarray(fn(mem, q, nz, st)) == tg)
    ana = analytic_flat_recall(D, n_f, p_star)
    for (mode, sev, name), lst in acc.items():
        r1, lo, hi, trials = pooled_recall(lst)
        base = np.concatenate(acc[("baseline", 0.0, name)]); cur = np.concatenate(lst)
        b, c = int((base & ~cur).sum()), int((~base & cur).sum())                       # baseline-right/fault-wrong, and the reverse
        record("fault_injection", name, n=n_f, d=D, noise=p_star, fault_mode=mode, severity=sev, query_batch=q_f, trials=trials, recall_at_1=r1, recall_at_1_lo=lo,
               recall_at_1_hi=hi, mcnemar_p_vs_fault_free=mcnemar_exact(b, c), degraded_queries=b, improved_queries=c, analytic_fault_free=ana)
    fi = results_df("fault_injection")
    bl = fi[(fi.fault_mode == "baseline") & (fi.method == "exact_flat")].iloc[0]
    check("S9 fault-free exact recall matches the analytic model at the chosen operating point",
          abs(bl.recall_at_1 - ana) <= 4.5 * math.sqrt(ana * (1 - ana) / bl.trials) + 1.0 / bl.trials, f"measured {bl.recall_at_1:.4f}, analytic {ana:.4f}, trials {bl.trials}")
    # first severity with a significant degradation (paired McNemar p < 0.01 and more queries lost than gained)
    first = []
    for mode, sevs in FAULTS:
        order = sevs if mode != "adc_bits" else sevs                                  # listed from mild to harsh in both cases
        for name in METHODS:
            hit = None
            for sev in order:
                r = fi[(fi.fault_mode == mode) & (fi.severity == float(sev)) & (fi.method == name)].iloc[0]
                if r.mcnemar_p_vs_fault_free < 0.01 and r.degraded_queries > r.improved_queries:
                    hit = sev; break
            first.append({"fault_mode": mode, "method": name, "first_significant_degradation": hit if hit is not None else "none up to max severity"})
            record("fault_injection", name, n=n_f, d=D, fault_mode=mode, severity=-1.0, scenario="first_significant_degradation", query_batch=q_f,
                   first_significant_severity=hit if hit is not None else NAN)
    print("Fault severity at which recall first drops significantly below the fault-free value (paired McNemar p < 0.01):")
    display(pd.DataFrame(first).pivot(index="fault_mode", columns="method", values="first_significant_degradation").reindex([m for m, _ in FAULTS]))
    tbl = fi[fi.severity >= 0].copy()
    tbl["recall [95% CI]"] = [f"{r:.3f} [{lo:.3f},{hi:.3f}]" for r, lo, hi in zip(tbl.recall_at_1, tbl.recall_at_1_lo, tbl.recall_at_1_hi)]
    display(tbl.pivot_table(index=["fault_mode", "severity"], columns="method", values="recall_at_1", sort=False).round(3))

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharey=True)
    for ax, (mode, sevs) in zip(axes.ravel(), FAULTS):
        for name, mk in (("exact_flat", "o"), ("hopfield_cleanup", "s")):
            g = fi[(fi.fault_mode == mode) & (fi.method == name) & (fi.severity >= 0)].sort_values("severity")
            ax.errorbar(g.severity, g.recall_at_1, yerr=yerr(g.recall_at_1, g.recall_at_1_lo, g.recall_at_1_hi), marker=mk, ms=4, capsize=2, label=name)
        ax.axhline(ana, color="k", ls=":", lw=1); ax.set_title(mode, fontsize=10); ax.grid(alpha=.25)
        if mode == "adc_bits":
            ax.invert_xaxis()
    axes[0][0].legend(fontsize=7); fig.supxlabel("fault severity"); fig.supylabel("recall@1 (dotted: fault-free analytic)")
    plt.tight_layout(); savefig(fig, "s9_faults"); plt.show()
else:
    print("Fault injection disabled.")
