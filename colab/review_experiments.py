#!/usr/bin/env python3
"""Evidence for colab/REVIEW.md.

Every claim in the review about the ORIGINAL notebook is measured here, on that notebook's own code (executed
unchanged from colab/original/), on whatever JAX backend is installed (the committed evidence was produced on CPU).
Pure-NumPy references come from the v2 helper cell. Writes colab/review_evidence.json.

usage: python colab/review_experiments.py        (requires jax, numpy, pandas, ipython)
"""
from __future__ import annotations
import contextlib, io, json, math, sys, time, warnings
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nbtools  # noqa: E402

ORIG = json.loads((HERE / "original" / "NeuroStore_Colab_Benchmarks.ipynb").read_text())
SRC = {c["id"]: "".join(c["source"]) for c in ORIG["cells"] if c["cell_type"] == "code"}
H = nbtools.exec_cells(nbtools.load_pct(HERE / "src"), ["helpers-numpy"])
wilson, mcnemar, analytic = H["wilson_ci"], H["mcnemar_exact"], H["analytic_flat_recall"]
EV: dict = {}


def run_cells(ids, ns=None):
    ns = {"__name__": "__orig__"} if ns is None else ns
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in ids:
            exec(SRC[i], ns)
    return ns


ALL = [f"ns-code-{i}" for i in (1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19)]
t0 = time.time()
base = run_cells(ALL)          # the original notebook, start to finish, in one namespace
import jax, jax.numpy as jnp  # noqa: E402
EV["environment"] = {"jax": jax.__version__, "backend": jax.default_backend(), "numpy": np.__version__}
SEED, D, Q, BETA = base["SEED"], base["D"], base["Q"], base["BETA"]
bipolar, corrupt, to_dev = base["bipolar"], base["corrupt_bits"], base["to_device"]
exact_top1, hop = base["exact_top1"], base["hopfield_cleanup"]
RES = [r for r in base["RESULTS"]]

# ---------------------------------------------------------------- E0 what the original's own tables say
nr = [r for r in RES if r["experiment"] == "noisy_recall" and r["method"] in ("exact_flat", "hopfield_cleanup")]
fi = [r for r in RES if r["experiment"] == "fault_injection"]
ood = [r for r in RES if r["experiment"] == "ood_rejection"][0]
sdm = [r for r in RES if r["experiment"] == "sdm_capacity"]
EV["E0_saturation_in_original_output"] = {
    "noisy_recall_rows": len(nr),
    "noisy_recall_rows_with_recall_1.0": sum(1 for r in nr if r["recall_at_1"] == 1.0),
    "noisy_recall_rows_noise_le_0.25_all_1.0": all(r["recall_at_1"] == 1.0 for r in nr if r["noise"] <= 0.25),
    "fault_rows": len(fi), "fault_rows_recall_1.0": sum(1 for r in fi if r["recall_at_1"] == 1.0),
    "ood_auroc": ood["auroc"], "ood_fpr": ood["fpr"], "ood_tpr": ood["tpr"],
    "sdm_rows": [{"n": r["n"], "noise": r["noise"], "exact_pattern_recall": r["exact_pattern_recall"], "bit_error_rate_out": r["bit_error_rate"]} for r in sdm],
    "sdm_rows_where_output_ber_exceeds_input_noise": [(r["n"], r["noise"], round(r["bit_error_rate"], 4)) for r in sdm if r["noise"] > 0 and r["bit_error_rate"] > r["noise"]],
}

# ---------------------------------------------------------------- E1 the one discriminative point, paired
def point(n, noise, seed_off=0, q=Q):
    mem_np = bipolar(n, D, SEED + n)
    rng = np.random.default_rng(SEED + 10000 + n + round(noise * 1000) + seed_off)
    tg = rng.integers(0, n, size=q)
    qn = corrupt(mem_np[tg], noise, rng)
    mem, qq = to_dev(mem_np), to_dev(qn)
    return tg, np.asarray(exact_top1(mem, qq)), np.asarray(hop(mem, qq, BETA))


pub = {}
for n in (128, 512):
    tg, pe, ph = point(n, 0.40)
    ce, ch = pe == tg, ph == tg
    b, c = int((ce & ~ch).sum()), int((~ce & ch).sum())
    pub[f"n{n}_noise0.40"] = {"exact": float(ce.mean()), "hopfield": float(ch.mean()), "exact_ci95": wilson(int(ce.sum()), len(ce)),
                              "hopfield_ci95": wilson(int(ch.sum()), len(ch)), "discordant_exact_only": b, "discordant_hopfield_only": c,
                              "mcnemar_p": mcnemar(b, c)}
EV["E1_published_point_paired"] = pub

# ---------------------------------------------------------------- E2 run-to-run spread with 128-query batches
spread = {}
for n in (128, 512):
    d_all, re_, rh_ = [], [], []
    for s in range(1, 301):
        tg, pe, ph = point(n, 0.40, seed_off=s)
        re_.append((pe == tg).mean()); rh_.append((ph == tg).mean())
    re_, rh_ = np.array(re_), np.array(rh_)
    diff = rh_ - re_
    spread[f"n{n}_noise0.40"] = {"batches": 300, "queries_per_batch": Q, "exact_mean": float(re_.mean()), "exact_sd": float(re_.std(ddof=1)),
                                 "hopfield_mean": float(rh_.mean()), "hopfield_sd": float(rh_.std(ddof=1)),
                                 "diff_mean": float(diff.mean()), "diff_sd": float(diff.std(ddof=1)),
                                 "batches_hopfield_better": int((diff > 0).sum()), "batches_exact_better": int((diff < 0).sum()), "batches_equal": int((diff == 0).sum())}
EV["E2_run_to_run_spread"] = spread

# ---------------------------------------------------------------- E3 are exact and Hopfield different algorithms here?
agree = {}
for n in (512, 2048):
    for noise in (0.25, 0.40):
        mem_np = bipolar(n, D, SEED + n)
        mem = to_dev(mem_np)
        rng = np.random.default_rng(SEED + 777 + n)
        tot = dis = e_ok = h_ok = 0
        for _ in range(10):
            tg = rng.integers(0, n, size=2000)
            qq = to_dev(corrupt(mem_np[tg], noise, rng))
            pe, ph = np.asarray(exact_top1(mem, qq)), np.asarray(hop(mem, qq, BETA))
            m = pe != ph
            tot += len(tg); dis += int(m.sum()); e_ok += int((pe[m] == tg[m]).sum()); h_ok += int((ph[m] == tg[m]).sum())
        agree[f"n{n}_noise{noise}"] = {"queries": tot, "disagreements": dis, "disagreement_rate": dis / tot,
                                       "of_which_exact_right": e_ok, "of_which_hopfield_right": h_ok}
EV["E3_exact_vs_hopfield_agreement"] = agree

# ---------------------------------------------------------------- E4 analytic expectation vs measurement
ana = []
for n in (128, 512):
    mem_np = bipolar(n, D, SEED + n)
    mem = to_dev(mem_np)
    for noise in (0.0, 0.10, 0.25, 0.40):
        rng = np.random.default_rng(SEED + 4242 + n + round(noise * 1000))
        ok = tot = 0
        for _ in range(10):
            tg = rng.integers(0, n, size=2000)
            qq = to_dev(corrupt(mem_np[tg], noise, rng))
            ok += int((np.asarray(exact_top1(mem, qq)) == tg).sum()); tot += len(tg)
        a = analytic(D, n, noise)
        se = math.sqrt(max(a * (1 - a), 1e-9) / tot)
        pub_row = [r for r in nr if r["method"] == "exact_flat" and r["n"] == n and abs(r["noise"] - noise) < 1e-9][0]
        ana.append({"n": n, "noise": noise, "analytic": a, "measured_20000q": ok / tot, "z": (ok / tot - a) / se if se > 0 else 0.0,
                    "original_notebook_single_run_128q": pub_row["recall_at_1"]})
EV["E4_analytic_vs_measured_exact_recall"] = ana

# ---------------------------------------------------------------- E5 why OOD and fault experiments cannot discriminate
rng = np.random.default_rng(5)
n_ood = 256
mem_np = bipolar(n_ood, D, SEED + 6000)
mem = to_dev(mem_np)
msim = base["max_similarity"]
known = np.concatenate([np.asarray(msim(mem, to_dev(corrupt(mem_np[rng.integers(0, n_ood, 2000)], 0.25, rng)))) for _ in range(5)])
oodq = np.concatenate([np.asarray(msim(mem, to_dev(bipolar(2000, D, 900 + i)))) for i in range(5)])
EV["E5a_ood_separation"] = {"known_noise": 0.25, "known_min": float(known.min()), "known_p01": float(np.quantile(known, 0.01)),
                            "ood_max": float(oodq.max()), "ood_p99": float(np.quantile(oodq, 0.99)), "auroc_10000x10000": H["auroc"](known, oodq),
                            "gap_between_worst_known_and_best_ood": float(known.min() - oodq.max())}
fault = []
for f in (0.0, 0.001, 0.01, 0.05):
    p_eff = 0.20 + f - 2 * 0.20 * f
    fault.append({"fault": "bitflip", "severity": f, "analytic_recall_n256": analytic(D, 256, p_eff)})
mem_np = bipolar(256, D, SEED + 7001)
for sigma in (0.05, 0.2, 0.5):
    rng = np.random.default_rng(int(sigma * 1000))
    ok = tot = 0
    for _ in range(5):
        tg = rng.integers(0, 256, 2000)
        qn = corrupt(mem_np[tg], 0.20, rng)
        faulty = mem_np + rng.normal(0, sigma, mem_np.shape).astype(np.float32)
        ok += int((np.asarray(exact_top1(to_dev(faulty), to_dev(qn))) == tg).sum()); tot += 2000
    fault.append({"fault": "analog_gaussian", "severity": sigma, "measured_recall_n256_10000q": ok / tot})
EV["E5b_fault_experiment_ceiling"] = fault

# ---------------------------------------------------------------- E6 tail percentiles from 8 or 20 samples
mem_np = bipolar(512, D, SEED + 512)
mem = to_dev(mem_np)
qq = to_dev(corrupt(mem_np[np.random.default_rng(1).integers(0, 512, Q)], 0.25, np.random.default_rng(2)))
exact_top1(mem, qq).block_until_ready()
samples = []
for _ in range(2000):
    t = time.perf_counter(); exact_top1(mem, qq).block_until_ready(); samples.append((time.perf_counter() - t) * 1e3)
samples = np.array(samples)
rng = np.random.default_rng(3)
pr = {}
for k in (8, 20):
    for pct in (95, 99):
        true = float(np.percentile(samples, pct))
        est = np.array([np.percentile(rng.choice(samples, k, replace=False), pct) for _ in range(3000)])
        pr[f"p{pct}_from_{k}_samples"] = {"reference_from_2000_samples_ms": true, "estimate_median_ms": float(np.median(est)),
                                          "estimate_p5_ms": float(np.quantile(est, 0.05)), "estimate_p95_ms": float(np.quantile(est, 0.95)),
                                          "ratio_p95_over_p5": float(np.quantile(est, 0.95) / np.quantile(est, 0.05))}
sub8 = rng.choice(samples, 8, replace=False)
pr["example_8_samples"] = {"p50": float(np.percentile(sub8, 50)), "p95": float(np.percentile(sub8, 95)), "p99": float(np.percentile(sub8, 99)), "max": float(sub8.max())}
EV["E6_percentile_reliability_cpu"] = pr

# ---------------------------------------------------------------- E7 launch-bound regime
tiny = jax.jit(lambda x: x + 1.0)
x = jnp.ones(8); tiny(x).block_until_ready()
ts = []
for _ in range(2000):
    t = time.perf_counter(); tiny(x).block_until_ready(); ts.append((time.perf_counter() - t) * 1e6)
fl = []
for prof, nvals, q in (("quick", [128, 512], 128), ("extended", [128, 512, 2048], 256)):
    for n in nvals:
        flops = 2.0 * q * n * D
        fl.append({"profile": prof, "n": n, "Q": q, "D": D, "flops": flops, "time_at_1_TFLOPs_us": flops / 1e12 * 1e6,
                   "time_at_10_TFLOPs_us": flops / 1e13 * 1e6, "bytes_memory_f32": n * D * 4})
EV["E7_launch_bound"] = {"empty_jit_call_p50_us_this_machine": float(np.percentile(ts, 50)), "empty_jit_call_p99_us_this_machine": float(np.percentile(ts, 99)),
                         "workloads": fl,
                         "original_cpu_batch_p50_ms": {f"{r['method']}_n{r['n']}_noise{r['noise']}": round(r["batch_p50_ms"], 3) for r in nr if r["noise"] == 0.25}}

# ---------------------------------------------------------------- E8 seeds
sdm_seeds = [(n, nz, SEED + n + round(nz * 1000)) for n in (50, 100, 200, 400) for nz in (0.0, 0.15)]
from collections import Counter
cnt = Counter(s for *_, s in sdm_seeds)
EV["E8_seed_derivation"] = {"sdm_noise_rng_seeds": sdm_seeds, "colliding_seeds": [s for s, c in cnt.items() if c > 1],
                            "colliding_pairs": [[(n, nz) for n, nz, s2 in sdm_seeds if s2 == s] for s, c in cnt.items() if c > 1]}

# ---------------------------------------------------------------- E9 re-running a cell
ns2 = run_cells(["ns-code-1", "ns-code-2", "ns-code-3", "ns-code-5"])
c1 = sum(1 for r in ns2["RESULTS"] if r["experiment"] == "noisy_recall")
run_cells(["ns-code-5"], ns2)
c2 = sum(1 for r in ns2["RESULTS"] if r["experiment"] == "noisy_recall")
EV["E9_rerun_duplicates"] = {"noisy_recall_rows_after_first_run": c1, "after_re_executing_the_same_cell": c2}

# ---------------------------------------------------------------- E10 VSA metric is determined by construction
ns3 = run_cells(["ns-code-1", "ns-code-2", "ns-code-3", "ns-code-9"])
pred, tuples, N_ = ns3["pred"], ns3["cue_tuples"], ns3["VSA_N"]
groups: dict = {}
for i, t in enumerate(tuples):
    groups.setdefault(t, []).append(i)
EV["E10_vsa_top1_by_construction"] = {"records": N_, "distinct_cue_tuples": len(groups),
                                      "max_distinct_predictions_within_a_tuple_group": max(len(set(pred[g].tolist())) for g in groups.values()),
                                      "top1_accuracy": float((pred == np.arange(N_)).mean()), "oracle_distinct_over_N": len(groups) / N_,
                                      "tuple_level_accuracy": float(np.mean([tuples[int(p)] == t for p, t in zip(pred, tuples)]))}

# ---------------------------------------------------------------- E11 reduced precision is exact on +-1 data, unless outputs are rounded
rng = np.random.default_rng(11)
n, q = 4096, 512
mem_np = bipolar(n, D, 1).astype(np.int8)
tg = rng.integers(0, n, q)
qn = corrupt(mem_np[tg].astype(np.float32), 0.40, rng).astype(np.int8)
m32, q32 = jnp.asarray(mem_np, jnp.float32), jnp.asarray(qn, jnp.float32)
ref = np.asarray(jnp.matmul(q32, m32.T, precision=jax.lax.Precision.HIGHEST))
variants = {
    "bf16_inputs_f32_accumulate": jnp.matmul(q32.astype(jnp.bfloat16), m32.astype(jnp.bfloat16).T, preferred_element_type=jnp.float32),
    "bf16_inputs_bf16_output": jnp.matmul(q32.astype(jnp.bfloat16), m32.astype(jnp.bfloat16).T).astype(jnp.float32),
    "int8_inputs_int32_accumulate": jax.lax.dot_general(jnp.asarray(qn), jnp.asarray(mem_np), (((1,), (1,)), ((), ())), preferred_element_type=jnp.int32).astype(jnp.float32),
}
prec = {}
mx = ref.max(1)
for k, v in variants.items():
    v = np.asarray(v)
    pick = v.argmax(1)
    prec[k] = {"max_abs_score_error": float(np.abs(v - ref).max()), "frac_queries_with_any_score_error": float((np.abs(v - ref).max(1) > 0).mean()),
               "picked_index_attains_true_max_score": float((ref[np.arange(q), pick] == mx).mean()), "top1_recall": float((pick == tg).mean())}
prec["reference_f32_highest_top1_recall"] = float((ref.argmax(1) == tg).mean())
EV["E11_precision_exactness_cpu"] = prec

# ---------------------------------------------------------------- E12 FlyHash proxy: tied activations at the WTA boundary
rng = np.random.default_rng(base["SEED"] + 3000)
proj = np.zeros((D, base["FLY_DIM"]), np.float32)
for col in range(base["FLY_DIM"]):
    ch = rng.choice(D, size=base["FLY_FANIN"], replace=False)
    proj[ch, col] = rng.choice([-1.0, 1.0], size=base["FLY_FANIN"]) / np.sqrt(base["FLY_FANIN"])
xs = bipolar(5000, D, 77)
act = xs @ proj
k = base["FLY_K"]
srt = np.sort(act, axis=1)[:, ::-1]
tie_at_boundary = srt[:, k - 1] == srt[:, k]
level = srt[:, k - 1:k]
n_at_level = (act == level).sum(1)
n_above = (act > level).sum(1)
distinct = len(np.unique(act))


def fly_recall(weights, n=512, noise=0.35, q=4000, seed=5):
    mem_np = bipolar(n, D, SEED + n)
    rng = np.random.default_rng(seed)
    tg = rng.integers(0, n, q)
    qn = corrupt(mem_np[tg], noise, rng)
    def code(x):
        a = x @ weights
        idx = np.argpartition(-a, k - 1, axis=1)[:, :k]
        c = np.zeros_like(a); np.put_along_axis(c, idx, 1.0, axis=1); return c
    mc, qc = code(mem_np), code(qn)
    return float(((qc @ mc.T).argmax(1) == tg).mean())


rng = np.random.default_rng(99)
gproj = np.zeros_like(proj)
for col in range(base["FLY_DIM"]):
    ch = rng.choice(D, size=base["FLY_FANIN"], replace=False)
    gproj[ch, col] = rng.normal(size=base["FLY_FANIN"]) / np.sqrt(base["FLY_FANIN"])
EV["E12_flyhash_ties"] = {"distinct_activation_values_over_5000_inputs": int(distinct),
                          "fraction_inputs_with_tie_across_the_WTA_boundary": float(tie_at_boundary.mean()),
                          "mean_units_tied_at_boundary_level": float(n_at_level.mean()), "mean_units_strictly_above_boundary": float(n_above.mean()),
                          "recall_at_noise": {str(nz): {"binary_pm1_over_sqrt16 (original)": fly_recall(proj, noise=nz), "gaussian_weights (continuous activations)": fly_recall(gproj, noise=nz)} for nz in (0.25, 0.30, 0.35, 0.40)}}

# ---------------------------------------------------------------- E13 host memory of bipolar()
EV["E13_bipolar_host_memory_gb"] = {f"n={n}": 8 * n * D / 1e9 for n in (2 ** 15, 2 ** 18, 2 ** 20)}

# ---------------------------------------------------------------- E14 SDM: dense mask matmul vs scatter / gather
sdm_dense_build, sdm_dense_read = base["sdm_build"], base["sdm_read"]


@jax.jit
def sdm_build_scatter(keys, values, addresses):
    loc = jax.lax.top_k(keys @ addresses.T, 16)[1]
    out = jnp.zeros((addresses.shape[0], values.shape[1]), jnp.float32)
    return out.at[loc.reshape(-1)].add(jnp.repeat(values, 16, axis=0))


@jax.jit
def sdm_read_gather(queries, addresses, counters):
    loc = jax.lax.top_k(queries @ addresses.T, 16)[1]
    dec = counters[loc].sum(axis=1)
    return jnp.where(dec >= 0, 1.0, -1.0)


sd = {}
for M, n in ((1024, 400), (8192, 400), (32768, 400)):
    addr = to_dev(bipolar(M, 256, 5000)); mem = to_dev(bipolar(n, 256, 5001))
    c_d = sdm_dense_build(mem, mem, addr, 16); c_s = sdm_build_scatter(mem, mem, addr)
    c_d.block_until_ready(); c_s.block_until_ready()
    r_d = sdm_dense_read(mem, addr, c_d, 16); r_s = sdm_read_gather(mem, addr, c_s)
    def t(f, *a):
        f(*a).block_until_ready(); xs_ = []
        for _ in range(7):
            t0_ = time.perf_counter(); f(*a).block_until_ready(); xs_.append((time.perf_counter() - t0_) * 1e3)
        return float(np.median(xs_))
    sd[f"M={M}"] = {"counters_identical": bool(jnp.array_equal(c_d, c_s)), "reads_identical": bool(jnp.array_equal(r_d, r_s)),
                    "build_dense_ms": t(sdm_dense_build, mem, mem, addr, 16), "build_scatter_ms": t(sdm_build_scatter, mem, mem, addr),
                    "read_dense_ms": t(sdm_dense_read, mem, addr, c_d, 16), "read_gather_ms": t(sdm_read_gather, mem, addr, c_s),
                    "dense_mask_bytes": n * M * 4}
EV["E14_sdm_dense_vs_scatter"] = sd

EV["runtime_s"] = time.time() - t0
(HERE / "review_evidence.json").write_text(json.dumps(EV, indent=1, default=float) + "\n")
print(json.dumps({k: v for k, v in EV.items() if k in ("environment", "runtime_s")}, indent=1))
print("wrote", HERE / "review_evidence.json")
