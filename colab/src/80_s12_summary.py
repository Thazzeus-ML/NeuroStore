# %% [markdown] id=s12-md
# ## 12. Self-checks, findings and claims ledger
#
# Everything below is computed from the rows recorded above; nothing is typed in. A **FAIL** in the self-check table means an
# invariant that must hold on every backend was violated on *this* runtime (a kernel not exact, a measured recall disagreeing
# with theory, two implementations disagreeing). Treat such a runtime's timing results with suspicion until it is explained.

# %% id=s12-code
#@title 12. Summary
df = results_df()
FINDINGS = []


def say(msg):
    FINDINGS.append(msg)
    print("-", msg)


print(f"{len(df)} result rows from {df.experiment.nunique()} experiments on {BACKEND}:{DEVICE_KIND} ({len(DEVICES)} device(s)), profile={PROFILE}, config={CONFIG_HASH}")
display(df.groupby("experiment").size().rename("rows").to_frame().T)
chk = pd.DataFrame(CHECKS)
if len(chk):
    display(chk.assign(status=np.where(chk.ok, "PASS", "FAIL"))[["status", "name", "detail"]])
    print(f"self-checks: {int(chk.ok.sum())}/{len(chk)} passed")
else:
    print("no self-checks were recorded")


def find(fn):
    try:
        fn()
    except (AttributeError, KeyError, IndexError, ValueError):       # a disabled or truncated section must not break the summary
        print(f"  (no data for: {fn.__name__[2:]})")


def f_saturation():
    ex = df[(df.experiment == "noisy_recall") & (df.method == "exact_flat")]
    sat = (ex.analytic_recall > 1 - 1e-9).mean()
    say(f"Noisy recall: {sat:.0%} of the (N, noise) cells have an analytic error probability below 1e-9, so only {1 - sat:.0%} can separate methods.")


def f_hopfield():
    ex = df[(df.experiment == "noisy_recall") & (df.method == "exact_flat")]
    hp = df[(df.experiment == "noisy_recall") & (df.method == "hopfield_cleanup")]
    for n in sorted(ex.n.unique()):
        e = ex[ex.n == n]; cell = e.iloc[(e.analytic_recall - 0.9).abs().argsort().iloc[0]]
        h = hp[(hp.n == n) & (hp.noise == cell.noise)].iloc[0]
        verdict = "no detectable difference" if h.mcnemar_p >= 0.05 else ("Hopfield worse" if h.paired_diff_vs_exact < 0 else "Hopfield better")
        say(f"Exact vs Hopfield at N={int(n)}, noise {cell.noise} (exact recall {cell.recall_at_1:.3f}): paired difference {h.paired_diff_vs_exact:+.3f} "
            f"[{h.paired_diff_lo:+.3f}, {h.paired_diff_hi:+.3f}], McNemar p={h.mcnemar_p:.3g}, {h.disagreement_with_exact:.1%} of queries differ -> {verdict}.")


def f_kernels():
    kv = df[df.experiment == "kernel_variants"]
    n = kv.n.max(); g = kv[kv.n == n].set_index("method")
    base = g.loc["f32_highest", "pipe_p50_ms"]
    parts = [f"{m} {base / g.loc[m, 'pipe_p50_ms']:.2f}x" for m in g.index if m != "f32_highest"]
    say(f"Kernels at N={int(n)}, Q={int(g.query_batch.iloc[0])}: speed relative to f32_highest: " + ", ".join(parts) +
        f"; popcount memory is {g.loc['packed_popcount', 'operand_bytes'] / g.loc['f32_highest', 'operand_bytes']:.3f}x of f32.")


def f_launch():
    sc = df[df.experiment == "scaling_N"]
    lb = (sc.lat_p50_ms < 5 * DISPATCH_FLOOR_MS)
    say(f"Scaling: {lb.mean():.0%} of N-sweep rows are within 5x of the dispatch floor ({DISPATCH_FLOOR_MS * 1e3:.0f} us); "
        f"best achieved {sc.gflops_effective.max():.0f} GFLOP/s and {sc.gbps_memory.max():.1f} GB/s at N up to {int(sc.n.max())}.")


def f_latency():
    ld = df[(df.experiment == "latency_dist") & (df.method == "f32_highest")]
    parts = [f"{r.path}: p50 {r.lat_p50_ms * 1e3:.0f} us, p99 {r.lat_p99_ms * 1e3:.0f} us" if r.lat_p99_ms == r.lat_p99_ms else f"{r.path}: p50 {r.lat_p50_ms * 1e3:.0f} us (p99 needs >=100 samples)" for r in ld.itertuples()]
    say("Single-request latency (Q=1, f32): " + "; ".join(parts) + ".")


def f_fly():
    fv = df[df.experiment == "fly_vs_simhash"]; n0 = fv.n.min(); nz = min(fv.noise.unique(), key=lambda x: abs(x - 0.30))
    g = fv[(fv.n == n0) & (fv.noise == nz)].set_index("method")
    say(f"FlyHash vs controls at N={int(n0)}, {nz:.0%} flips: " + ", ".join(f"{m} {g.loc[m, 'recall_at_1']:.3f}" for m in g.index) +
        f". Sizes: raw input {D / 8:.0f} B; v1 FlyHash code ideal {g[g.index.str.contains('v1')].ideal_bytes.iloc[0]:.0f} B.")


def f_vsa():
    vd = df[df.experiment == "vsa_decode"]; dd = vd.d.max()
    g = vd[vd.d == dd].sort_values("bundle_terms"); below = g[g.recall_at_1 < 0.9]
    ra = df[df.experiment == "vsa_records"]
    say(f"VSA at D={int(dd)}: decode accuracy stays >= 0.9 up to k={int(g[g.recall_at_1 >= 0.9].bundle_terms.max())} bundled pairs"
        + (f" and falls below it at k={int(below.bundle_terms.min())}" if len(below) else "") + f"; tuple-level record accuracy {ra.tuple_level_accuracy.min():.3f}-{ra.tuple_level_accuracy.max():.3f}.")


def f_sdm():
    sm = df[df.experiment == "sdm_summary"].dropna(subset=["memory_overhead_vs_flat_bitpacked"])
    say(f"SDM: best configuration reliably (>=90% exact) stores {int(sm.reliable_patterns_n90.max())} patterns; its memory per reliable pattern is "
        f"{sm.memory_overhead_vs_flat_bitpacked.min():.0f}x-{sm.memory_overhead_vs_flat_bitpacked.max():.0f}x that of a flat bit-packed store.")
    cap = df[(df.experiment == "sdm_capacity") & (df.noise > 0) & (df.read_iters == 1)]
    say(f"SDM: {int((cap.improves_over_input == False).sum())}/{len(cap)} noisy-cue cells return output with more bit errors than the cue.")  # noqa: E712


def f_ood():
    od = df[(df.experiment == "ood") & (df.scenario == "known25_vs_chimera")].set_index("method")
    say("OOD chimera family AUROC: " + ", ".join(f"{m} {od.loc[m, 'auroc']:.2f}" for m in od.index) + " (0.5 = chance).")


def f_faults():
    fi = df[(df.experiment == "fault_injection") & (df.scenario == "first_significant_degradation") & (df.method == "exact_flat")]
    say("Fault injection: first significant degradation of exact search (severity): " + ", ".join(f"{r.fault_mode} {r.first_significant_severity}" for r in fi.itertuples()) + ".")


def f_multi():
    md = df[df.experiment == "multidevice"].set_index("method")
    say(f"Multi-device ({int(md.devices.iloc[0])} devices): data-parallel speedup {md.loc['data_parallel', 'speedup_vs_single']:.2f}x, memory-sharded {md.loc['memory_sharded', 'speedup_vs_single']:.2f}x.")


def f_sparse():
    sp = df[df.experiment == "sparse_accumulate"].pivot_table(index="n", columns="method", values="pipe_p50_ms")
    parts = [f"N={int(n)}: sparse {r['scatter_add']:.3g} ms" + (f" vs dense {r['dense_matmul']:.3g} ms" if r.get('dense_matmul') == r.get('dense_matmul') else " (dense does not fit)") for n, r in sp.iterrows()]
    say("Sparse accumulate: " + "; ".join(parts) + ".")


print("\nFindings:")
for fn in (f_saturation, f_hopfield, f_kernels, f_launch, f_latency, f_fly, f_vsa, f_sdm, f_ood, f_faults, f_multi, f_sparse):
    find(fn)

# ---------------- claims ledger: what each result can and cannot support
LEDGER = []


def claim(text, status, evidence):
    LEDGER.append({"claim": text, "status": status, "evidence": evidence})


nr = df[df.experiment == "noisy_recall"]
if len(nr):
    hp = nr[nr.method == "hopfield_cleanup"]
    sig = hp[(hp.mcnemar_p < 0.05)]
    claim("Hopfield cleanup retrieves as well as exact flat search", "supported only where both are saturated; exact search is better in the informative cells" if len(sig) else "no detectable difference at this sample size",
          f"{len(sig)}/{len(hp)} (N, noise) cells have a significant paired difference (McNemar p<0.05); Section 1")
if CHECKS:
    ok = {c["name"]: c["ok"] for c in CHECKS}
    k_ok = [v for n_, v in ok.items() if n_.startswith("S3")]
    if k_ok:
        claim("bf16 / int8 / bit-packed kernels give the same exact-search answers as fp32 on +-1 data", "supported on this runtime" if all(k_ok) else "NOT supported on this runtime", "Section 3 optimal_rate == 1 for all exact kernels")
sc = df[df.experiment == "scaling_N"]
if len(sc):
    lb = float((sc.lat_p50_ms < 5 * DISPATCH_FLOOR_MS).mean())
    claim("Timings at v1's sizes (N <= 2048, Q <= 256) characterise accelerator throughput", "no" if BACKEND != "cpu" else "n/a on CPU", f"{lb:.0%} of N-sweep rows are launch-bound here; Section 4")
if len(df[df.experiment == "fly_vs_simhash"]):
    claim("FlyHash-style codes are more compact than the data they encode", "no for this projection", "the ideal k-of-m code size exceeds the D-bit raw input in the v1 configuration; Section 5")
if len(df[df.experiment == "sdm_summary"]):
    claim("SDM is memory-efficient relative to flat storage", "no in this setting", "bytes per reliable pattern vs a flat bit-packed store; Section 7")
if len(df[df.experiment == "fault_injection"]):
    claim("Fault injection shows robustness of the architecture", "only within the stated functional models and operating point", "Section 9 (no device physics)")
claim("Multi-core / full-pod scaling", "measured" if len(df[df.experiment == "multidevice"]) else "not measured (single device)", "Section 10")
claim("Energy per query, endurance, ADC/DAC overhead, crossbar area, FeNAND speed-ups", "not measured by this notebook", "no power or device model is present")
claim("Results reproduce the original report", "not established", "independent proxy implementation; seeds and sources not reconciled")
display(pd.DataFrame(LEDGER))
