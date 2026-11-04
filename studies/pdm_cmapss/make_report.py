#!/usr/bin/env python3
"""Generate studies/pdm_cmapss/REPORT.md from results/pdm_cmapss/*.json.

The tables, gate verdicts and prediction scores are computed here from the result files. The prose in report_narrative.md may contain
only {{placeholders}} for numbers; every placeholder is resolved from the `facts` dictionary built below (a missing key, or a hand-typed
decimal number in the narrative, is an error).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import evaluate_gates as EG  # noqa: E402
import fetch_data  # noqa: E402

ORDER = ["const", "ridge", "exact", "exact_l2", "hopfield", "ann", "ann_sq8", "flyhash", "vsa"]
SWEEP_ORDER = ["exact", "exact_l2", "hopfield", "ann", "ann_sq8", "flyhash", "vsa", "ridge"]
FAMS = ["noise", "dropout", "stuck", "drift", "spikes", "quantize", "combined"]
LEVELS = ["L25", "L50", "L100"]
facts: dict = {}


def fig(name: str, alt: str) -> str:
    return f"![{alt}](figures/{name})\n\n" if (HERE / "figures" / name).exists() else ""


def jl(d: Path, name):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def f1(x):
    return "n/a" if x is None else f"{x:.1f}"


def f2(x):
    return "n/a" if x is None else f"{x:.2f}"


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def ci(p, lo, hi, nd=1):
    return f"{p:.{nd}f} ({lo:.{nd}f}-{hi:.{nd}f})"


def table(headers, rows, align=None):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(("---:" if (align and align[i] == "r") else "---") for i in range(len(headers))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def meaning_str(family, m):
    if family == "noise":
        return f"sigma {m['noise_sigma_z']:.2f} z"
    if family == "dropout":
        return f"{m['channels_lost']} of {m['of']} channels lost"
    if family == "stuck":
        return f"{m['channels_stuck']} of {m['of']} channels flatlined"
    if family == "drift":
        return f"{m['channels_drifting']} channels ramp to {m['final_offset_z']:.2f} z"
    if family == "spikes":
        return f"{100 * m['spike_probability']:.2f} % of samples hit by +-{m['spike_size_z']:.0f} z"
    if family == "quantize":
        return f"{m['adc_bits']}-bit ADC"
    return f"{m['channels_lost']} lost + sigma {m['noise_sigma_z']:.2f} z + drift {m['final_offset_z']:.2f} z"


def cfg_str(sel):
    st = ", ".join(f"{k}={v}" for k, v in sel["struct"].items())
    return f"{st + ', ' if st else ''}k={sel['k']}, beta={sel['beta']}"


# ----------------------------------------------------------------------------------------------- gates
def gate_evidence(g):
    e, i = g["evidence"], g["id"]
    if g["verdict"] in ("NOT RUN",):
        return "not run"
    if i == "G0":
        return "; ".join(e["problems"]) or f"all files from commit {', '.join(h[:7] for h in e['git_heads'])}"
    if i == "G1":
        return f"{e['configs_inside']}/{e['of']} configurations inside the interval"
    if i == "G2":
        return (f"RMSE {f1(e['rmse_reference'])} ({e['reference']}) vs const {f1(e['rmse_const'])} (ratio {f2(e['ratio_vs_const']['ratio'])}) and ridge "
                f"{f1(e['rmse_ridge'])} (ratio {f2(e['ratio_vs_ridge']['ratio'])})")
    if i.startswith("G3"):
        r = e["ratio_vs_ref"]
        return f"RMSE {f1(e['rmse'])}; ratio vs {e['reference']} {f3(r['ratio'])} ({f3(r['lo'])}-{f3(r['hi'])})"
    if i == "G4":
        mem = ", ".join(f"{b} {v:.2f}x" for b, v in e["deployed_bytes_fraction_of_reference"].items())
        lat = ", ".join(f"{b}: {v['fraction']:.3f} of exact p50, recall@10 {v['recall10']:.2f}, nprobe {v['nprobe']}" for b, v in e["latency_at_scale"].items()) or "no n=500k rows"
        return f"deployed bytes / {e['reference']}: {mem}. Memory part ok: {e['memory_part_ok'] or 'none'}. Latency at scale: {lat}"
    if i == "G5":
        s = "; ".join(f"{b}: better in {len(v)}" + (f" ({', '.join(v)})" if v else "") + f", worse in {len(e['families_worse_than_reference'][b])}" for b, v in e["families_better_than_reference"].items())
        un = e["families_where_level_not_reached"]
        return s + (f". Level not reached for {', '.join(un)}" if un else "")
    if i == "G6a":
        a = e["auroc"]
        return f"AUROC {f3(a['auroc'])} ({f3(a['lo'])}-{f3(a['hi'])})"
    if i == "G6b":
        a, b = e["far"], e["same"]
        return f"far {f3(a['auroc'])} ({f3(a['lo'])}-{f3(a['hi'])}), mid {f3(e['mid']['auroc'])}, same {f3(b['auroc'])} ({f3(b['lo'])}-{f3(b['hi'])})"
    if i == "G6c":
        a = e["auroc"]
        return f"AUROC {f3(a['auroc'])} ({f3(a['lo'])}-{f3(a['hi'])})"
    if i == "G7":
        n = e["nmse"]
        return f"{e['variant']} NMSE {f3(n['nmse'])} ({f3(n['lo'])}-{f3(n['hi'])}) with {e['missing_channels']} of {e['channels']} channels lost; mean imputation {f3(e['mean_imputation_nmse']['nmse'])}"
    return ""


def gates_section(rows):
    for g in rows:
        facts[f"verdict_{g['id'].replace('[', '_').replace(']', '')}"] = g["verdict"]
    return table(["gate", "verdict", "claim", "evidence"], [[g["id"], f"**{g['verdict']}**", g["claim"], gate_evidence(g)] for g in rows])


# ----------------------------------------------------------------------------------------------- predictions
def predictions_section(d: Path):
    t, tm, oo, co = jl(d, "test_panel.json"), jl(d, "timing.json"), jl(d, "ood.json"), jl(d, "completion.json")
    tu = jl(d, "tuning_FD001.json")
    rows = []

    def add(pid, text, verdict, evidence):
        rows.append([pid, text, f"**{verdict}**", evidence])
        facts[f"pred_{pid}"] = verdict

    if tm and tu:
        ref = tu["reference"]
        rb = tm["variants"][ref]["memory"]["deployed_bytes"]
        fy, vs = tm["variants"]["flyhash"]["memory"]["deployed_bytes"] / rb, tm["variants"]["vsa"]["memory"]["deployed_bytes"] / rb
        add("P1", "FlyHash and VSA need at least as many deployed bytes as the exact store", "CONFIRMED" if fy >= 1 and vs >= 1 else "REFUTED",
            f"flyhash {fy:.2f}x, vsa {vs:.2f}x of {ref}")
        facts["bytes_flyhash_x"], facts["bytes_vsa_x"] = f"{fy:.2f}", f"{vs:.2f}"
    if t:
        e = t["subsets"]["FD001"]["variants"]
        rel = e["ann_sq8"]["rmse"] / e["ann"]["rmse"] - 1
        add("P2", "SQ8 costs <= 2 % RMSE relative to fp32 IVF (official test)", "CONFIRMED" if rel <= 0.02 else "REFUTED", f"ann_sq8 / ann - 1 = {100 * rel:+.2f} %")
        r = e["hopfield"]["ratio_vs_ref"]["ratio"]
        add("P3", "Hopfield is within +-3 % RMSE of the reference", "CONFIRMED" if abs(r - 1) <= 0.03 else "REFUTED", f"ratio {r:.3f}")
    if co:
        ok, ev = True, []
        for base in ("exact", "exact_l2", "hopfield"):
            for s in sorted(co["variants"][base]):
                a, b = co["variants"][f"{base}_mask"][s], co["variants"][base][s]
                better = a["hi"] < b["lo"]
                ok &= better
                ev.append(f"{base}@{s}: {a['nmse']:.3f} vs {b['nmse']:.3f}{'' if better else ' (CIs overlap or masked is not lower)'}")
        add("P4", "Knowing which sensors are dead lowers completion NMSE for exact, exact_l2, hopfield at all severities (scored with non-overlapping unpaired CIs, see deviations)",
            "CONFIRMED" if ok else "REFUTED", "; ".join(ev))
    if oo and tu:
        s = oo["variants"][oo["reference"]]["chimera_channel_same"]["max_sim"]
        add("P5", "Same-stage chimera control has AUROC > 0.55", "CONFIRMED" if s["auroc"] > 0.55 else "REFUTED", f"AUROC {f3(s['auroc'])} ({f3(s['lo'])}-{f3(s['hi'])})")
    if tu:
        add("P6", "Euclidean exact beats cosine exact in CV (reference = exact_l2)", "CONFIRMED" if tu["reference"] == "exact_l2" else "REFUTED",
            f"CV RMSE exact {f3(tu['backends']['exact']['selected']['cv_rmse'])}, exact_l2 {f3(tu['backends']['exact_l2']['selected']['cv_rmse'])}")
    return table(["id", "prediction (recorded before the run)", "outcome", "evidence"], rows)


# ----------------------------------------------------------------------------------------------- sections
def validate_section(d: Path):
    v = jl(d, "validate.json")
    if not v:
        return "_not run_"
    a = table(["D", "N", "p_flip", "observed", "99.7 % interval", "closed form", "inside"],
              [[r["D"], r["N"], r["p_flip"], f3(r["observed"]), f"{f3(r['lo'])}-{f3(r['hi'])}", f3(r["analytic"]), "yes" if r["inside"] else "**NO**"] for r in v["analytic"]], list("rrrrrrr"))
    ps = sorted({r["p_flip"] for r in v["panel"]})
    bs = ["exact", "hopfield", "ann", "flyhash", "vsa"]
    panel = table(["bit-flip p"] + bs, [[p] + [next(f"{r['recall']:.3f}" for r in v["panel"] if r["backend"] == b and r["p_flip"] == p) for b in bs] for p in ps])
    c = v["panel_cfg"]
    return (f"Exact search on random +-1 patterns against the closed-form tie-aware recall ({v['meta']['config']['VALIDATE_QUERIES']} queries x "
            f"{len(v['meta']['config']['SEEDS'])} seeds per row):\n\n{a}\n\nThe Colab benchmark's noisy-pattern recall (top-1, D={c['D']}, N={c['N']}, {c['queries']} queries, default-ish "
            f"configurations: ann nprobe 8, flyhash m=4096 / 5 % active, vsa D=4096):\n\n{panel}")


def tuning_section(d: Path):
    out = []
    for fd in ("FD001", "FD002", "FD003", "FD004"):
        t = jl(d, f"tuning_{fd}.json")
        if not t:
            continue
        rows = []
        for n in ["exact", "exact_l2", "hopfield", "ann", "ann_sq8", "flyhash", "vsa"]:
            b = t["backends"][n]
            sel = b["selected"]
            edge = b["edge"]
            flag = ", ".join(x for x, y in (("k at grid edge", edge.get("k_at_grid_edge")), ("beta at grid edge", edge.get("beta_at_grid_edge"))) if y)
            rows.append([n, f3(sel["cv_rmse"]), cfg_str(sel), sel.get("rule", ""), flag])
        bl = t["baselines"]
        out.append(f"**{fd}** ({t['cv']['engines']} engines, {t['cv']['n_windows']} windows, {t['cv']['queries']} CV queries). Reference = **{t['reference']}**; "
                   f"const CV RMSE {f2(bl['const_cv_rmse'])}, ridge {f2(bl['ridge_cv_rmse'])} (lambda {bl['ridge_lambda']}).\n\n" +
                   table(["variant", "CV RMSE", "selected", "rule", "flags"], rows))
    return "\n\n".join(out) or "_not run_"


def test_section(d: Path):
    t = jl(d, "test_panel.json")
    if not t:
        return "_not run_"
    out = []
    for fd, e in t["subsets"].items():
        rows = []
        for n in ORDER:
            v = e["variants"][n]
            r = v.get("ratio_vs_ref")
            rows.append([n + (" (ref)" if n == e["reference"] else ""), ci(v["rmse"], v["rmse_lo"], v["rmse_hi"]), f1(v["rmse_capped"]), f1(v["mae"]),
                         f"{v['nasa_sum']:.0f}", "-" if r is None else f"{r['ratio']:.3f} ({r['lo']:.3f}-{r['hi']:.3f})"])
        out.append(f"**{fd}** ({e['n_test']} test engines, {e['n_train_engines']} training engines, {len(e['channels'])} channels)\n\n" +
                   table(["variant", "RMSE vs official RUL (CI)", "RMSE vs capped", "MAE", "NASA score", "RMSE ratio vs reference (CI)"], rows, list("lrrrrr")))
    return "\n\n".join(out)


def sweeps_section(d: Path):
    s = jl(d, "sweeps.json")
    if not s:
        return "_not run_"
    lv = s["levels"]
    FAMS = list(lv.keys())
    LEVELS = list(next(iter(lv.values())).keys())
    cal = table(["family"] + [f"{l} (target x{lv[FAMS[0]][l]['target']:.2f})" for l in LEVELS],
                [[f] + [f"{meaning_str(f, lv[f][l]['meaning'])} -> x{lv[f][l]['ratio']:.2f}" + ("" if lv[f][l]["reached"] else " **(not reached; max severity)**") for l in LEVELS] for f in FAMS])
    parts = [f"Out-of-fold queries: {s['n_queries']} from {s['n_engines']} engines. Reference = **{s['reference']}**. Severities were calibrated once on the reference so that each family hurts it by the stated multiple.\n\n" + cal]
    for l in LEVELS:
        rows = []
        for f in FAMS:
            row = [f]
            for b in SWEEP_ORDER:
                x = s["variants"][b]["families"][f][l]
                mark = ""
                vr = x.get("vs_ref")
                if vr and b != s["reference"]:
                    mark = " (-)" if vr["hi"] < 0 else (" (+)" if vr["lo"] > 0 else "")
                row.append(f"{x['ratio']:.2f}{mark}")
            rows.append(row)
        body = table(["family"] + SWEEP_ORDER, rows, ["l"] + ["r"] * len(SWEEP_ORDER))
        cap = f"RMSE(corrupted) / RMSE(clean) per backend at level {l}. (-) = degrades significantly LESS than the reference, (+) = significantly MORE (paired 95 % CI)."
        parts.append(f"<details{' open' if l == 'L50' else ''}><summary>Level {l}</summary>\n\n{cap}\n\n{body}\n\n</details>")
    arows = []
    lv_abs = "L50" if "L50" in LEVELS else LEVELS[0]
    for f in FAMS:
        row = [f]
        best = min(s["variants"][b]["families"][f][lv_abs]["rmse_corrupt"] for b in SWEEP_ORDER)
        for b in SWEEP_ORDER:
            x = s["variants"][b]["families"][f][lv_abs]["rmse_corrupt"]
            row.append(f"**{x:.1f}**" if x == best else f"{x:.1f}")
        arows.append(row)
    parts.append(f"Absolute RMSE (cycles, capped target) under corruption at level {lv_abs}; point estimates, lowest in bold. A smaller degradation RATIO can come from a worse clean RMSE, so read it together with this table and the clean RMSE below:\n\n"
                 + table(["family"] + SWEEP_ORDER, arows, ["l"] + ["r"] * len(SWEEP_ORDER)))
    mrows = []
    for b in ("exact", "exact_l2", "hopfield", "vsa"):
        for f in [q for q in ("dropout", "combined") if q in FAMS and "L50" in LEVELS]:
            x, y = s["variants"][b]["families"][f]["L50"], s["variants"][f"{b}_mask"]["families"][f]["L50"]
            mrows.append([b, f, f2(x["ratio"]), f2(y["ratio"]), f"{y['vs_unmasked']['diff']:+.2f} ({y['vs_unmasked']['lo']:+.2f} to {y['vs_unmasked']['hi']:+.2f})"])
    parts.append("Knowing which sensors are dead (mask), level L50:\n\n" + table(["backend", "family", "ratio without mask", "ratio with mask", "paired difference (CI)"], mrows))
    clean = table(["variant", "clean out-of-fold RMSE (capped target)"], [[b, f2(s["variants"][b]["clean_rmse"])] for b in SWEEP_ORDER])
    parts.append(clean)
    return "\n\n".join(parts)


def completion_section(d: Path):
    c = jl(d, "completion.json")
    if not c:
        return "_not run_"
    sev = sorted(next(iter(c["variants"].values())).keys())
    names = ["mean_imputation", "exact", "exact_mask", "exact_l2", "exact_l2_mask", "hopfield", "hopfield_mask", "vsa", "vsa_mask"]
    rows = [[n] + [ci(c["variants"][n][s]["nmse"], c["variants"][n][s]["lo"], c["variants"][n][s]["hi"], 3) for s in sev] for n in names]
    hdr = ["variant"] + [f"dropout {s} ({c['missing_channels'][s]} of {c['channels']} channels)" for s in sev]
    return f"NMSE on the missing features (features are z-scored: mean imputation = 1; lower is better). Reference = **{c['reference']}**.\n\n" + table(hdr, rows)


def ood_section(d: Path):
    o = jl(d, "ood.json")
    if not o:
        return "_not run_"
    sets = ["gauss", "shuffle_time", "permute_channels", "foreign_FD002", "foreign_FD003", "foreign_FD004", "foreign_FD003_late",
            "chimera_channel_same", "chimera_channel_mid", "chimera_channel_far", "chimera_time_same", "chimera_time_mid", "chimera_time_far"]
    sets = [s for s in sets if s in next(iter(o["variants"].values()))]
    names = list(o["variants"])
    rows = [[s] + [ci(o["variants"][n][s]["max_sim"]["auroc"], o["variants"][n][s]["max_sim"]["lo"], o["variants"][n][s]["max_sim"]["hi"], 3) for n in names] for s in sets]
    main = table(["OOD set (ID = held-out FD001 windows)"] + names, rows)
    sc = []
    keys = sorted({k for n in o["variants"] for k in o["variants"][n]["gauss"]})
    for n in ("hopfield", o["reference"], "vsa"):
        for s in [q for q in ("foreign_FD003_late", "chimera_channel_far", "chimera_channel_mid", "chimera_channel_same") if q in o["variants"][n]]:
            v = o["variants"][n][s]
            sc.append([n, s] + [f3(v[k]["auroc"]) if k in v else "-" for k in keys])
    flag = table(["variant", "OOD set"] + keys, sc)
    fl = [[n, s, f2(o["variants"][n][s]["max_sim"]["ood_flagged_at_fpr5"])] for n in (o["reference"], "vsa") for s in ("foreign_FD003_late", "chimera_channel_far", "chimera_channel_same") if s in o["variants"][n]]
    return (f"AUROC with ID windows as the positive class, score = max similarity (higher = more familiar), engine-bootstrap 95 % CI. Foreign set sizes: "
            f"{o['foreign_sizes']}. Empty sets skipped: {o['empty_sets_skipped_per_set'] or 'none'}. Reference = **{o['reference']}**.\n\n{main}\n\n"
            f"Other familiarity scores (AUROC point estimates):\n\n{flag}\n\nShare of OOD items flagged at a 5 % ID false-alarm rate (max similarity):\n\n"
            + table(["variant", "OOD set", "flagged"], fl))


def timing_section(d: Path):
    t = jl(d, "timing.json")
    if not t:
        return "_not run_"
    rows = []
    for n in ["exact", "exact_l2", "hopfield", "ann", "ann_sq8", "flyhash", "vsa"]:
        v = t["variants"][n]
        lp = v["latency_predict"]
        rows.append([n, f"{v['build']['seconds']:.2f}", f"{v['memory']['deployed_bytes'] / 1e6:.2f}", f"{v['memory']['bytes_per_item']:.0f}", f"{v['memory']['ratio_vs_raw_features']:.2f}",
                     f"{lp['p50_ms']:.2f} / {f2(lp['p95_ms'])} / {f2(lp['p99_ms'])}", f"{v['throughput_predict']['qps_median']:.0f}", f"{v['search_peak_bytes_batch'] / 1e6:.1f}"])
    a = table(["variant", "build s", "deployed MB", "bytes / item", "x raw features", "predict latency p50 / p95 / p99 ms", "queries / s (batch 256)", "batch peak MB"], rows, list("lrrrrrrr"))
    comp = []
    for n in ("exact_l2", "ann_sq8", "flyhash", "vsa"):
        for k, b in t["variants"][n]["memory"]["components"].items():
            comp.append([n, k, f"{b:,}"])
    b = table(["variant", "component", "bytes"], comp)
    sc = []
    for n, rows_ in t["scale"].items():
        for r in rows_:
            lt = r["latency"]
            sc.append([n, f"{r['n']:,}", f"{lt['p50_ms']:.2f}", f2(lt['p95_ms']), f"{r['recall10']:.3f}", r.get("nprobe", "-"), f"{r['throughput']['qps_median']:.0f}", f"{r['index_bytes'] / 1e6:.1f}"])
    c = table(["backend", "n", "p50 ms", "p95 ms", "recall@10 vs exact", "nprobe", "queries / s", "index MB"], sc, list("lrrrrrrr"))
    env = t["meta"]["env"]
    return (f"FD001 memory: {t['n_windows']:,} windows x {t['d_features']} features. Single thread (BLAS threads: {env['threads_env']}), {env['cpus']} CPUs visible, NumPy {env['numpy']}, "
            f"Python {env['python']}. Latency = one query through the whole pipeline (featurise + search + readout).\n\n{a}\n\nDeployed bytes by component (alternatives and caches are not counted):\n\n{b}\n\n"
            f"Scale-up on clustered synthetic vectors (d = {t['d_features']}), single-query search with k=10; ANN nprobe = smallest reaching recall@10 >= 0.9:\n\n{c}")


def posthoc_section(d: Path):
    ph = jl(d, "posthoc_quantized.json")
    sw, ts, tm = jl(d, "sweeps.json"), jl(d, "test_panel.json"), jl(d, "timing.json")
    if not ph or not sw:
        return ""
    ref = ph["reference"]
    lv = "L50"
    fams = list(sw["levels"].keys())
    names = ["exact_l2_clip3", "exact_l2_q32", "exact_l1", "exact_l1_clip3"]
    cols = [ref] + names + ["vsa"]
    def ratio(n, f):
        return ph["variants"][n]["families"][f][lv] if n in ph["variants"] else sw["variants"][n]["families"][f][lv]
    t1 = []
    e = ts["subsets"]["FD001"]["variants"]
    for n in cols:
        if n in ph["variants"]:
            v = ph["variants"][n]
            t1.append([n, ci(v["test"]["rmse"], v["test"]["rmse_lo"], v["test"]["rmse_hi"]), f2(v["clean_rmse"]), cfg_str(v["selected"]), f"{v['latency_predict_ms_p50']:.2f}", f"{v['deployed_bytes'] / 1e6:.2f}"])
        else:
            x = e[n]
            t1.append([n + (" (ref)" if n == ref else ""), ci(x["rmse"], x["rmse_lo"], x["rmse_hi"]), f2(sw["variants"][n]["clean_rmse"]), cfg_str(ts["subsets"]["FD001"]["variants"][n]["selected"]),
                       f"{tm['variants'][n]['latency_predict']['p50_ms']:.2f}", f"{tm['variants'][n]['memory']['deployed_bytes'] / 1e6:.2f}"])
    a = table(["store", "official-test RMSE (CI)", "clean out-of-fold RMSE", "selected (k, beta)", "p50 ms", "deployed MB"], t1, list("lrrlrr"))
    t2 = [[f] + [f"{ratio(n, f)['ratio']:.2f}" + (" (-)" if ("vs_ref" in ratio(n, f) and n != ref and ratio(n, f)["vs_ref"]["hi"] < 0) else (" (+)" if ("vs_ref" in ratio(n, f) and n != ref and ratio(n, f)["vs_ref"]["lo"] > 0) else "")) for n in cols] for f in fams]
    b = table(["family"] + cols, t2, ["l"] + ["r"] * len(cols))
    t3 = [[f] + [f"{ratio(n, f)['rmse_corrupt']:.1f}" for n in cols] for f in fams]
    c = table(["family"] + cols, t3, ["l"] + ["r"] * len(cols))
    rows4 = []
    for f in fams:
        row = [f]
        for n in names:
            d_ = ph["variants"][n]["families"][f][lv]["abs_rmse_minus_vsa"]
            row.append(f"{d_['diff']:+.1f} ({d_['lo']:+.1f} to {d_['hi']:+.1f})")
        rows4.append(row)
    dd = table(["family"] + names, rows4)
    return (f"**Exploratory and not pre-registered; it cannot change any gate verdict.** The main sweeps found that VSA degrades less than the reference under every injected corruption. "
            f"This control asks whether that comes from what VSA's encoder does to each feature: it clips features to a bounded range and quantises them to 32 levels, and its level codes make "
            f"the Hamming distance grow linearly with the level difference, an L1-like distance. The controls are brute-force exact stores tuned with the same (k, beta) grid on the same cross-validation "
            f"queries, with the same calibrated severities and corruption seeds as the main sweeps (generated by `posthoc_quantized.py` into `results/pdm_cmapss/posthoc_quantized.json`).\n\n"
            f"Accuracy, speed and size (FD001):\n\n{a}\n\nDegradation ratio at level {lv}, as in the main table ((-)/(+) = paired CI against the reference):\n\n{b}\n\n"
            f"Absolute RMSE under corruption at level {lv} (cycles):\n\n{c}\n\nAbsolute RMSE under corruption minus VSA's, paired engine bootstrap (negative = the control is better than VSA):\n\n{dd}")


def collect_facts(d: Path):
    tu, t, s, c, o, tm, v = (jl(d, n) for n in ("tuning_FD001.json", "test_panel.json", "sweeps.json", "completion.json", "ood.json", "timing.json", "validate.json"))
    F = facts
    ALT = EG.ALT
    if tu:
        ref = tu["reference"]
        F["ref"] = ref
        F["cv_exact"], F["cv_l2"] = f2(tu["backends"]["exact"]["selected"]["cv_rmse"]), f2(tu["backends"]["exact_l2"]["selected"]["cv_rmse"])
        F["cv_cos_over_l2"] = f2(tu["backends"]["exact"]["selected"]["cv_rmse"] / tu["backends"]["exact_l2"]["selected"]["cv_rmse"])
        F["cv_ridge"], F["cv_const"] = f2(tu["baselines"]["ridge_cv_rmse"]), f2(tu["baselines"]["const_cv_rmse"])
        for n, b in tu["backends"].items():
            F[f"cv_{n}"] = f2(b["selected"]["cv_rmse"])
            F[f"cfg_{n}"] = cfg_str(b["selected"])
    if t:
        for fd, e in t["subsets"].items():
            ref = e["reference"]
            for n, x in e["variants"].items():
                F[f"rmse_{n}_{fd}"] = f1(x["rmse"])
                F[f"rmse_{n}_{fd}_ci"] = f"{x['rmse_lo']:.1f}-{x['rmse_hi']:.1f}"
                if "ratio_vs_ref" in x:
                    r = x["ratio_vs_ref"]
                    F[f"ratio_{n}_{fd}"] = f"{r['ratio']:.3f}"
                    F[f"ratio_{n}_{fd}_ci"] = f"{r['lo']:.3f}-{r['hi']:.3f}"
                    F[f"pct_{n}_{fd}"] = f"{100 * (r['ratio'] - 1):+.1f}"
            F[f"ref_vs_const_{fd}"], F[f"ref_vs_ridge_{fd}"] = f"{e['ref_vs_const']['ratio']:.2f}", f"{e['ref_vs_ridge']['ratio']:.2f}"
            F[f"ref_vs_ridge_{fd}_ci"] = f"{e['ref_vs_ridge']['lo']:.2f}-{e['ref_vs_ridge']['hi']:.2f}"
            F[f"ref_vs_const_{fd}_ci"] = f"{e['ref_vs_const']['lo']:.2f}-{e['ref_vs_const']['hi']:.2f}"
            F[f"n_test_{fd}"] = e["n_test"]
        for n in ("const", "ridge", "exact", "exact_l2", "hopfield", "ann", "ann_sq8", "flyhash", "vsa"):                  # FD001 shortcuts
            for k in ("rmse", "rmse_ci", "ratio", "ratio_ci", "pct"):
                key = f"{k}_{n}_FD001"
                if key in F:
                    F[f"{k}_{n}"] = F[key]
        F["rmsecap_ref"] = f1(t["subsets"]["FD001"]["variants"][t["subsets"]["FD001"]["reference"]]["rmse_capped"])
        F["mae_ref"] = f1(t["subsets"]["FD001"]["variants"][t["subsets"]["FD001"]["reference"]]["mae"])
        F["rmse_ref"] = F.get(f"rmse_{t['subsets']['FD001']['reference']}_FD001")
        F["rmse_ref_ci"] = F.get(f"rmse_{t['subsets']['FD001']['reference']}_FD001_ci")
        F["cos_over_l2_test"] = f2(t["subsets"]["FD001"]["variants"]["exact"]["rmse"] / t["subsets"]["FD001"]["variants"]["exact_l2"]["rmse"])
    if s:
        ref = s["reference"]
        lv = "L50" if "L50" in next(iter(s["levels"].values())) else next(iter(next(iter(s["levels"].values()))))
        for b in list(ALT) + [ref]:
            if b in s["variants"]:
                for f, d_ in s["variants"][b]["families"].items():
                    F[f"deg_{b}_{f}"] = f"{d_[lv]['ratio']:.2f}"
                    F[f"abs_{b}_{f}"] = f"{d_[lv]['rmse_corrupt']:.1f}"
        F["clean_sweep_ref"] = f1(s["variants"][ref]["clean_rmse"])
        F["n_families"] = len(s["levels"])
        for b in ALT:
            F[f"n_better_{b}"] = sum(1 for f in s["levels"] if s["variants"][b]["families"][f][lv]["vs_ref"]["hi"] < 0)
            F[f"n_worse_{b}"] = sum(1 for f in s["levels"] if s["variants"][b]["families"][f][lv]["vs_ref"]["lo"] > 0)
            F[f"n_abs_better_{b}"] = sum(1 for f in s["levels"] if s["variants"][b]["families"][f][lv]["rmse_corrupt"] < s["variants"][ref]["families"][f][lv]["rmse_corrupt"])
        for f in ("dropout", "combined"):
            if f in s["levels"] and f"{ref}_mask" in s["variants"]:
                F[f"deg_{ref}_mask_{f}"] = f"{s['variants'][f'{ref}_mask']['families'][f][lv]['ratio']:.2f}"
        F["dropout_channels_L50"] = s["levels"]["dropout"][lv]["meaning"]["channels_lost"] if "dropout" in s["levels"] else "n/a"
    if c:
        for n, d_ in c["variants"].items():
            for sev, x in d_.items():
                F[f"nmse_{n}_{sev}"] = f3(x["nmse"])
                F[f"nmse_{n}_{sev}_ci"] = f"{x['lo']:.3f}-{x['hi']:.3f}"
    if o:
        for n, d_ in o["variants"].items():
            for sname, sc in d_.items():
                F[f"auc_{n}_{sname}"] = f3(sc["max_sim"]["auroc"])
                F[f"auc_{n}_{sname}_ci"] = f"{sc['max_sim']['lo']:.3f}-{sc['max_sim']['hi']:.3f}"
        if "neg_energy" in o["variants"].get("hopfield", {}).get("gauss", {}):
            for sname in ("foreign_FD003_late", "chimera_channel_same", "chimera_channel_mid", "chimera_channel_far"):
                F[f"negE_{sname}"] = f3(o["variants"]["hopfield"][sname]["neg_energy"]["auroc"])
        for sname in next(iter(o["variants"].values())):
            F[f"auc_ref_{sname}"] = F[f"auc_{o['reference']}_{sname}"]
            F[f"auc_ref_{sname}_ci"] = F[f"auc_{o['reference']}_{sname}_ci"]
    if tm:
        ref = tu["reference"]
        rb = tm["variants"][ref]["memory"]["deployed_bytes"]
        for n, x in tm["variants"].items():
            F[f"mb_{n}"] = f"{x['memory']['deployed_bytes'] / 1e6:.2f}"
            F[f"bytes_x_{n}"] = f"{x['memory']['deployed_bytes'] / rb:.2f}"
            F[f"bpi_{n}"] = f"{x['memory']['bytes_per_item']:.0f}"
            F[f"p50_{n}"] = f"{x['latency_predict']['p50_ms']:.2f}"
            F[f"qps_{n}"] = f"{x['throughput_predict']['qps_median']:.0f}"
            F[f"build_{n}"] = f"{x['build']['seconds']:.2f}"
        for n, rows in tm["scale"].items():
            for r in rows:
                F[f"scale_p50_{n}_{r['n']}"] = f"{r['latency']['p50_ms']:.2f}"
                F[f"scale_recall_{n}_{r['n']}"] = f"{r['recall10']:.2f}"
                F[f"scale_mb_{n}_{r['n']}"] = f"{r['index_bytes'] / 1e6:.0f}"
                if "nprobe" in r:
                    F[f"scale_nprobe_{n}_{r['n']}"] = r["nprobe"]
        big = max(r["n"] for r in tm["scale"]["exact"])
        F["scale_big_n"] = f"{big:,}"
        for n in ("ann", "ann_sq8"):
            r = next((r for r in tm["scale"].get(n, []) if r["n"] == big), None)
            if r:
                F[f"scale_speedup_{n}"] = f"{next(x for x in tm['scale']['exact'] if x['n'] == big)['latency']['p50_ms'] / r['latency']['p50_ms']:.0f}"
    ph = jl(d, "posthoc_quantized.json")
    if ph and s and tm:
        ref, lv = s["reference"], "L50"
        fams = list(s["levels"].keys())
        v1 = ph["variants"]["exact_l1"]
        F["ph_l1_rmse"], F["ph_l1_rmse_ci"] = f1(v1["test"]["rmse"]), f"{v1['test']['rmse_lo']:.1f}-{v1['test']['rmse_hi']:.1f}"
        F["ph_l1_clean"], F["ph_l1_p50"], F["ph_l1_mb"] = f2(v1["clean_rmse"]), f"{v1['latency_predict_ms_p50']:.1f}", f"{v1['deployed_bytes'] / 1e6:.2f}"
        F["ph_l1_n_deg_less_than_ref"] = sum(1 for f in fams if v1["families"][f][lv]["vs_ref"]["hi"] < 0)
        F["ph_l1_n_deg_same_as_vsa"] = sum(1 for f in fams if not v1["families"][f][lv]["vs_vsa"]["excludes_zero"])
        F["ph_l1_n_abs_sig_lower_than_vsa"] = sum(1 for f in fams if v1["families"][f][lv]["abs_rmse_minus_vsa"]["hi"] < 0)
        F["ph_l1_n_abs_sig_higher_than_vsa"] = sum(1 for f in fams if v1["families"][f][lv]["abs_rmse_minus_vsa"]["lo"] > 0)
        for n in ("exact_l2_clip3", "exact_l2_q32"):
            F[f"ph_{n}_n_deg_less_than_ref"] = sum(1 for f in fams if ph["variants"][n]["families"][f][lv]["vs_ref"]["hi"] < 0)
            F[f"ph_{n}_rmse"] = f1(ph["variants"][n]["test"]["rmse"])
    if v:
        F["analytic_inside"] = f"{sum(1 for r in v['analytic'] if r['inside'])}/{len(v['analytic'])}"
    F["n_gates_held"] = sum(1 for k, x in facts.items() if k.startswith("verdict_") and x == "HELD")
    F["n_gates"] = sum(1 for k in facts if k.startswith("verdict_"))


def render(narrative: str):
    def rep(m):
        k = m.group(1).strip()
        if k not in facts:
            raise KeyError(f"narrative placeholder {{{{{k}}}}} has no fact")
        return str(facts[k])
    body = re.sub(r"\{\{([^}]+)\}\}", rep, narrative)
    stray = re.findall(r"(?<![\w.-])\d+\.\d+(?![\w.])", re.sub(r"\{\{[^}]+\}\}", "", narrative))
    if stray:
        raise ValueError(f"narrative contains hand-typed decimal numbers {stray}; use placeholders")
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "pdm_cmapss"))
    ap.add_argument("--out", default=str(HERE / "REPORT.md"))
    ap.add_argument("--no-narrative", action="store_true")
    a = ap.parse_args()
    d = Path(a.results)
    proto = HERE / "PROTOCOL.md"
    sha = hashlib.sha256(proto.read_bytes()).hexdigest()
    rows = EG.evaluate(d, data_sums=fetch_data.FILES)
    v = jl(d, "validate.json")
    heads = sorted({jl(d, f)["meta"]["git_head"][:7] for f in ("validate.json", "test_panel.json", "timing.json") if jl(d, f)})
    gates = gates_section(rows)
    preds = predictions_section(d)
    collect_facts(d)
    sections = {
        "validate": validate_section(d), "tuning": tuning_section(d), "test": test_section(d), "sweeps": sweeps_section(d), "completion": completion_section(d),
        "ood": ood_section(d), "timing": timing_section(d), "posthoc": posthoc_section(d)}
    dev = HERE / "deviations.md"
    deviations = dev.read_text().strip() if dev.exists() else "None."
    nar = HERE / "report_narrative.md"
    narrative = render(nar.read_text()) if (nar.exists() and not a.no_narrative) else ""
    md = f"""# PdM study on NASA C-MAPSS: NeuroStore-style memories for remaining-useful-life estimation

_Generated by `studies/pdm_cmapss/make_report.py` from `results/pdm_cmapss/*.json` (result commit {', '.join(heads) or 'n/a'}, protocol sha256 `{sha[:12]}`). Do not edit by hand._

{narrative}

## Gate verdicts (pre-registered, `gates.toml`)

{gates}

## Predictions recorded before the run

{preds}

## E6 Harness validity

{sections['validate']}

## E0 Tuning on training engines (engine-grouped CV)

{sections['tuning']}

## E1 Official test

{fig('fig_accuracy.png', 'Official-test RMSE by store, FD001-FD004')}{sections['test']}

## E2 Corruption sweeps (FD001)

{fig('fig_robustness.png', 'Degradation ratio by corruption family and store')}{sections['sweeps']}

## E3 Completion (FD001)

{sections['completion']}

## E4 Familiarity, OOD and chimeras (FD001 memory)

{fig('fig_chimera.png', 'Chimera detection by life-stage gap')}{sections['ood']}

## E5 Latency, throughput and memory

{fig('fig_scale.png', 'Latency and index size versus stored vectors')}{sections['timing']}

{('## Post-hoc control: why is VSA robust? (exploratory)' + chr(10) + chr(10) + sections['posthoc'] + chr(10)) if sections['posthoc'] else ''}
## Deviations from the protocol

{deviations}

## Reproduce

```
make pdm-fetch && make pdm-study && make pdm-gates && make pdm-report
```
"""
    Path(a.out).write_text(md)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
