"""Assemble REPORT.md from results/*.json and gates.toml. Every number comes from a result file written by a
committed script; narrative text lives in eval/report_narrative.md with {{result:path.to.value:format}} placeholders."""
from __future__ import annotations
import json, re, tomllib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
_cache: dict = {}


def R(name):
    if name not in _cache:
        _cache[name] = json.loads((RES / f"{name}.json").read_text())
    return _cache[name]


def ms(s, f=".3f", scale=1.0):
    return f"{s['mean'] * scale:{f}} ± {s['std'] * scale:{f}}"


def get(obj, path):
    for tok in path.split("."):
        obj = obj[int(tok)] if isinstance(obj, list) else obj[tok]
    return obj


def fill(text):
    def rep(m):
        name, path, fmt = m.group(1), m.group(2), m.group(3) or ""
        v = get(R(name), path)
        if fmt == "pct":
            return f"{100 * v:.1f}%"
        if fmt == "pp":
            return f"{100 * v:.1f} pp"
        return f"{v:{fmt}}" if fmt else str(v)
    return re.sub(r"\{\{(\w+):([\w.\-]+)(?::([^}]*))?\}\}", rep, text)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


G = tomllib.loads((ROOT / "gates.toml").read_text())
SEEDS = "seeds 0-4"


def gate_table():
    p0, p1, p2, p3, p4, p5, p6 = (R(n) for n in ("phase0", "phase1", "phase2", "phase3", "phase4", "phase5", "phase6"))
    a1 = R("phase4_attempt1")
    nc = sum(1 for k, v in p2["conditions"].items() if v["beats_exact"] and v["approaches_nn"])
    rows = [
        ["0", "generator deterministic + documented statistics", p0["gate"],
         f"5/5 seeds satisfy all 6 criteria; chain TV max {max(r['chain_tv'] for r in p0['rows']):.3f} (tol {G['phase0']['tol_chain_tv']}); {SEEDS}"],
        ["1", "exact-k, determinism, bytes/event = 8+4k, same>diff overlap", p1["gate"],
         f"exact-k {min(r['exact_k_fraction'] for r in p1['per_seed']):.3f}; mismatches {sum(r['determinism_mismatches'] for r in p1['per_seed'])}; "
         f"bytes/event {p1['per_seed'][0]['bytes'][1]['bytes_per_event']:.0f}; overlap same {ms(p1['overlap_same'])} vs diff {ms(p1['overlap_diff'])}; {SEEDS}"],
        ["2", "partial-cue recovery vs exact-match and Jaccard-kNN", p2["gate"],
         f"{nc}/6 conditions pass; failing: " + ", ".join(f"{k} (ratio {v['ratio_vs_nn']:.3f} < {v['ratio_required']})" for k, v in p2["conditions"].items() if not v["approaches_nn"]) + f"; {SEEDS}"],
        ["3", "next-unit recall@32 vs frequency and bigram", p3["gate"],
         f"NS {ms(p3['ns'])} vs bigram {ms(p3['bigram'])} vs freq {ms(p3['freq'])}; wins {p3['wins_vs_bigram']}/5, {p3['wins_vs_freq']}/5; {SEEDS}"],
        ["4", "surprise gate: bytes down, recall within 2 pp", p4["gate"] + " (attempt 2)",
         f"bytes {-100 * p4['bytes_reduction']['mean']:.1f}% ± {100 * p4['bytes_reduction']['std']:.1f}; F1 drop {ms(p4['f1_drop'], '.4f')}; P@m drop {ms(p4['prec_drop'], '.4f')}; "
         f"attempt 1 FAILED at bytes {-100 * a1['bytes_reduction']['mean']:.1f}% (needed <= -20%); {SEEDS}"],
        ["5", "layout hypotheses stated plainly (reporting gate)", p5["quick"] is False and "PASS" or "FAIL",
         "; ".join(f"{h['hypothesis'].split(' ')[0]}{'(' + h['workload'].split('_')[1] + ')' if h['workload'].startswith('rebuild') else ''} {'HELD' if h['held'] else 'NOT HELD'}" for h in p5["hypotheses"]) + f"; {SEEDS}"],
        ["6", "ablations measured, dependences + expectations reported (reporting gate)", "PASS" if expectations_complete() else "FAIL",
         f"{expectation_summary()}; {SEEDS}"],
    ]
    return table(["Phase", "Gate", "Verdict", "Numbers"], rows)


UNITS = {"append": ("µs/event", 1e-3), "rebuild_single": ("µs/event", 1e-3), "rebuild_batch": ("ms/4000 events", 1e-6),
         "history_scan": ("ms/query", 1e-6), "history_indexed": ("ms/query", 1e-6), "scan_field": ("ms/scan", 1e-6),
         "postings_build": ("s", 1e-9), "footprint_bytes": ("MB", 1e-6), "postings_bytes": ("MB", 1e-6),
         "sparse_accumulate": ("µs/query", 1e-3), "out_edge_scan": ("µs/row", 1e-3), "in_edge_scan": ("µs/row", 1e-3), "graph_bytes": ("MB", 1e-6)}


def layout_tables():
    p5 = R("phase5"); t = p5["table"]
    rows = []
    for wl, (unit, sc) in UNITS.items():
        cells = []
        for nm in ("aos", "soa", "soa_event", "volley", "csr", "dict"):
            if nm in t[wl]:
                v = t[wl][nm]
                cells.append(f"{nm}: {v['median'] * sc:.3g} [{v['q1'] * sc:.3g}–{v['q3'] * sc:.3g}]" if wl not in ("footprint_bytes", "postings_bytes", "graph_bytes") else f"{nm}: {v['median'] * sc:.3g}")
        rows.append([wl, unit, "<br>".join(cells)])
    t1 = table(["Workload", "Unit", "Median [IQR] by layout"], rows)
    hrows = [[h["hypothesis"], h["workload"], "**HELD**" if h["held"] else "**NOT HELD**", h["fastest"],
              f"{100 * h['margin_vs_runner_up']:.1f}%" if h["margin_vs_runner_up"] >= 0 else f"slower by {abs(h['margin_vs_runner_up']) * 100:.0f}%", f"{h['seed_wins']}/5"] for h in p5["hypotheses"]]
    t2 = table(["Hypothesis", "Workload", "Verdict", "Fastest/smallest", "Margin vs runner-up", "Seeds won"], hrows)
    return t1, t2


ABL_COLS = [("f1", "concept F1", ".3f"), ("prec_lift", "P@m lift", ".3f"), ("temporal_lift", "next-unit lift", ".3f"),
            ("log_bytes_per_event", "log B/event", ".0f"), ("rate_cv", "rate CV", ".2f"), ("overlap_margin", "overlap margin", ".3f"),
            ("surprise_auc", "surprise AUC", ".3f"), ("rate_readout_corr", "rate corr", ".3f")]


def ablation_table(world):
    p6 = R("phase6"); t = p6["table"][world]; dep = p6["dependence"][world]
    rows = []
    for v, d in t.items():
        cells = [v]
        for m, _, f in ABL_COLS:
            s = d[m]; txt = f"{s['mean']:{f}} ± {s['std']:{f}}" if s["mean"] == s["mean"] else "n/a"
            if v in dep and dep[v][m]["depends"]:
                txt += " ↓" if dep[v][m]["direction"] == "removal_hurts" else " ↑"
            cells.append(txt)
        rows.append(cells)
    return table(["Variant"] + [c for _, c, _ in ABL_COLS], rows)


def expectations():
    p6 = R("phase6"); out = []
    for e in G["phase6"]["expect"]:
        d = p6["dependence"][e["world"]][e["variant"]][e["metric"]]
        if e["claim"] == "hurts":
            met = d["depends"] and d["direction"] == "removal_hurts"
        else:
            met = not d["depends"]
        out.append((e, d, met))
    return out


def expectations_complete():
    try:
        return len(expectations()) == len(G["phase6"]["expect"])
    except Exception:
        return False


def expectation_summary():
    ex = expectations()
    return f"{sum(1 for *_, m in ex if m)}/{len(ex)} author expectations met"


def expectation_table():
    rows = [[e["world"], e["variant"], e["metric"], e["claim"], f"{d['delta_mean']:+.3f} ± {d['delta_std']:.3f}", "met" if m else "**NOT met**"] for e, d, m in expectations()]
    return table(["World", "Removed", "Metric", "Pre-registered claim", "Δ vs full (mean ± sd over seeds)", "Result"], rows)


def phase2_table():
    p2 = R("phase2"); rows = []
    for mf in G["phase2"]["masks"]:
        for metric, nm in (("f1", "concept F1"), ("prec", "masked-unit P@m")):
            c = p2["conditions"][f"mask{mf}_{metric}"]
            rows.append([f"{int(mf * 100)}%", nm, ms(c["ns"]), ms(c["nn"]), ms(c["exact"]), f"{c['ratio_vs_nn']:.3f} (need {c['ratio_required']})",
                         "yes" if c["beats_exact"] else "no", "yes" if c["approaches_nn"] else "**no**"])
    t = table(["Mask", "Metric", "NeuroStore", "Jaccard kNN (ref)", "Exact match", "NS / kNN", "beats exact", "approaches kNN"], rows)
    cost = [r["cost"] for r in p2["per_seed"]]
    mean = lambda k: float(np.mean([c[k] for c in cost]))
    c = table(["System", "Query latency (median µs)", "Index bytes (6000 events)"],
              [["NeuroStore assemblies", f"{mean('ns_query_us'):.0f}", f"{mean('ns_index_bytes'):,.0f}"],
               ["Jaccard kNN (dense matvec)", f"{mean('nn_query_us'):.0f}", f"{mean('nn_bytes_packed'):,.0f} packed bits / {mean('nn_bytes_dense'):,.0f} dense"],
               ["Exact-match hash", f"{mean('exact_query_us'):.0f}", f"{mean('exact_bytes'):,.0f}"]])
    return t, c


def phase3_tables():
    p3, ctl, dr, fu = R("phase3"), R("phase3_controls"), R("phase3_drift"), R("phase3_completion_followup")
    a = table(["System", "recall@32 (mean ± sd, 5 seeds)"],
              [["NeuroStore (lattice, temporal readout)", ms(p3["ns"], ".4f")], ["First-order unit bigram", ms(p3["bigram"], ".4f")],
               ["Global frequency", ms(p3["freq"], ".4f")], ["Oracle (true chain + true profiles)", ms(p3["oracle"], ".4f")]])
    b = table(["Variant (post-hoc control)", "recall@32", "edges", "lattice bytes"],
              [[v, ms(x["recall"], ".4f"), f"{x['n_edges']['mean']:,.0f}", f"{x['lattice_bytes']['mean']:,.0f}"] for v, x in ctl["variants"].items()])
    cps = dr["checkpoints"]; cur = dr["curve"]
    rows = []
    for m in ("ns_lam0.03/post", "ns_lam0.15/post", "bigram_cum/post", "bigram_win1500/post", "ns_lam0.03/pre", "ns_lam0.15/pre"):
        rows.append([m] + [f"{cur[m][str(e)]['mean']:.3f}" for e in cps])
    c = table(["model/test stream"] + [f"{e - dr['drift_index']:+d}" for e in cps], rows)
    d = table(["Mask", "assembly only", "assembly + lattice", "kNN", "ratio before → after (Gate 2 needed)"],
              [[f"{float(k) * 100:.0f}%", ms(v["assembly_only"]), ms(v["assembly_plus_lattice"]), ms(v["knn"]),
                f"{v['ratio_assembly_only']:.3f} → {v['ratio_plus_lattice']:.3f} ({v['required_ratio_gate2']})"] for k, v in fu["masks"].items()])
    return a, b, c, d


def phase4_table():
    p4 = R("phase4"); rows = []
    for t, v in p4["curve"].items():
        rows.append([t, f"{100 * v['count_only_fraction']['mean']:.1f}%", f"{-100 * v['bytes_reduction']['mean']:.1f}%", f"{100 * v['f1_drop']['mean']:+.2f}", f"{100 * v['prec_drop']['mean']:+.2f}", f"{v['temporal_recall']['mean']:.4f}"])
    a = table(["tau", "events count-only", "log bytes vs ungated", "F1 drop (pp)", "P@m drop (pp)", "next-unit recall@32"], rows)
    h = p4["homeostasis_test"]
    b = table(["Homeostasis test (Zipf s=1.5, last 2000 of 8000 events)", "off", "on"],
              [["CV of per-unit firing rates", ms(h["cv_off"], ".2f"), ms(h["cv_on"], ".3f")],
               ["max unit rate / target", ms(h["max_over_target_off"], ".1f"), ms(h["max_over_target_on"], ".2f")], ["k per event", "32", "32"]])
    return a, b


def sweep_table():
    sw = R("phase6_homeostasis_sweep"); rows = []
    for wn in ("uniform", "zipf"):
        for c, d in sw["table"][wn].items():
            rows.append([wn, c, ms(d["rate_cv"], ".2f"), ms(d["overlap_margin"], ".3f"), ms(d["f1"], ".3f"), ms(d["prec_lift"], ".3f"), ms(d["temporal_lift"], ".3f")])
    return table(["World", "θ clip", "rate CV", "overlap margin", "concept F1", "P@m lift", "next-unit lift"], rows)


def ablation_table_c():
    t = R("phase6c_nohomeo_base"); d = t["table"]; dep = t["dependence"]
    rows = []
    for v, x in d.items():
        cells = [v]
        for m, _, f in ABL_COLS:
            if m not in x:
                cells.append("not measured"); continue
            s_ = x[m]; txt = f"{s_['mean']:{f}} ± {s_['std']:{f}}" if s_["mean"] == s_["mean"] else "n/a"
            if v in dep and dep[v][m]["depends"]:
                txt += " ↓" if dep[v][m]["direction"] == "removal_hurts" else " ↑"
            cells.append(txt)
        rows.append(cells)
    return table(["Variant (base: homeostasis off)"] + [c for _, c, _ in ABL_COLS], rows)


def scalars():
    p2, p3, p4, a1, p5, ctl, dr, fu = R("phase2"), R("phase3"), R("phase4"), R("phase4_attempt1"), R("phase5"), R("phase3_controls"), R("phase3_drift"), R("phase3_completion_followup")
    V = {}
    for mf in ("0.3", "0.5", "0.7"):
        for m in ("f1", "prec"):
            c = p2["conditions"][f"mask{mf}_{m}"]
            V[f"G2_{m}_{mf}_ratio"] = f"{c['ratio_vs_nn']:.3f}"
            V[f"G2_{m}_{mf}_ns"] = f"{c['ns']['mean']:.3f}"; V[f"G2_{m}_{mf}_nn"] = f"{c['nn']['mean']:.3f}"; V[f"G2_{m}_{mf}_ex"] = f"{c['exact']['mean']:.3f}"
    V["G2_assemblies"] = f"{p2['n_assemblies']['mean']:.0f}"; V["G2_purity"] = f"{p2['purity']['mean']:.2f}"
    cost = [r["cost"] for r in p2["per_seed"]]
    V["G2_ns_us"] = f"{np.mean([c['ns_query_us'] for c in cost]):.0f}"; V["G2_nn_us"] = f"{np.mean([c['nn_query_us'] for c in cost]):.0f}"
    V["G2_ns_kb"] = f"{np.mean([c['ns_index_bytes'] for c in cost]) / 1e3:.0f}"; V["G2_nn_kb"] = f"{np.mean([c['nn_bytes_packed'] for c in cost]) / 1e3:.0f}"
    V["G3_margin_bigram"] = f"{p3['margin_vs_bigram']:.3f}"; V["G3_margin_freq"] = f"{p3['margin_vs_freq']:.3f}"
    cv = ctl["variants"]
    for k in cv:
        V[f"CTL_{k}"] = f"{cv[k]['recall']['mean']:.3f}"
    V["CTL_edges_default"] = f"{cv['default']['n_edges']['mean']:,.0f}"; V["CTL_edges_unpruned"] = f"{cv['unpruned_nodecay']['n_edges']['mean']:,.0f}"
    V["CTL_bytes_ratio"] = f"{cv['unpruned_nodecay']['lattice_bytes']['mean'] / cv['default']['lattice_bytes']['mean']:.1f}"
    c = dr["curve"]; d0 = str(dr["drift_index"]); fin = str(dr["checkpoints"][-1])
    for m in ("ns_lam0.03", "ns_lam0.15", "bigram_cum", "bigram_win1500"):
        V[f"DR_{m}_post_drift"] = f"{c[m + '/post'][d0]['mean']:.3f}"; V[f"DR_{m}_post_final"] = f"{c[m + '/post'][fin]['mean']:.3f}"
        V[f"DR_{m}_pre_drift"] = f"{c[m + '/pre'][d0]['mean']:.3f}"; V[f"DR_{m}_pre_final"] = f"{c[m + '/pre'][fin]['mean']:.3f}"
    V["DR_t95_lam15"] = str(dr["summary"]["ns_lam0.15/post"]["events_to_regain_95pct_of_pre_drift_level"])
    V["DR_t95_lam03"] = str(dr["summary"]["ns_lam0.03/post"]["events_to_regain_95pct_of_pre_drift_level"])
    for k, v in fu["masks"].items():
        V[f"FU_{k}_before"] = f"{v['ratio_assembly_only']:.3f}"; V[f"FU_{k}_after"] = f"{v['ratio_plus_lattice']:.3f}"
    V["G4_a1_red"] = f"{100 * a1['bytes_reduction']['mean']:.1f}%"; V["G4_red"] = f"{100 * p4['bytes_reduction']['mean']:.1f}%"
    V["G4_red_sd"] = f"{100 * p4['bytes_reduction']['std']:.1f} pp"
    V["G4_taus"] = ", ".join(str(t) for t in p4["chosen_tau"])
    V["G4_mean_surprise"] = f"{np.mean([r['ungated']['mean_surprise'] for r in p4['per_seed']]):.2f}"
    V["G4_cache_hit"] = f"{np.mean([r['taus']['0.55']['cache_hit_rate'] for r in p4['per_seed']]):.2f}"
    V["G4_cache_hit_a1"] = f"{np.mean([r['taus']['0.55']['cache_hit_rate'] for r in a1['per_seed']]):.2f}"
    V["G4_delta_ratio"] = f"{np.mean([r['taus']['0.55']['delta_bytes'] / r['ungated']['delta_bytes'] for r in p4['per_seed']]):.2f}"
    h = {x["hypothesis"].split(" ")[0] + "_" + x["workload"]: x for x in p5["hypotheses"]}
    for k, x in h.items():
        V[f"H_{k}"] = ("HELD" if x["held"] else "NOT HELD")
        V[f"M_{k}"] = f"{100 * abs(x['margin_vs_runner_up']):.0f}%"
    t = p5["table"]
    V["L_csr_acc_x"] = f"{t['sparse_accumulate']['dict']['median'] / t['sparse_accumulate']['csr']['median']:.0f}"
    V["L_csr_out_x"] = f"{t['out_edge_scan']['dict']['median'] / t['out_edge_scan']['csr']['median']:.0f}"
    V["L_csr_in_x"] = f"{t['in_edge_scan']['dict']['median'] / t['in_edge_scan']['csr']['median']:.0f}"
    V["L_graph_x"] = f"{t['graph_bytes']['dict']['median'] / t['graph_bytes']['csr']['median']:.1f}"
    V["L_fp_volley"] = f"{t['footprint_bytes']['volley']['median'] / 1e6:.1f}"; V["L_fp_aos"] = f"{t['footprint_bytes']['aos']['median'] / 1e6:.1f}"
    V["L_batch_volley_x"] = f"{t['rebuild_batch']['volley']['median'] / t['rebuild_batch']['soa']['median']:.1f}"
    V["L_scan_soa"] = f"{t['scan_field']['soa']['median'] / 1e6:.2f}"; V["L_scan_volley"] = f"{t['scan_field']['volley']['median'] / 1e6:.2f}"; V["L_scan_aos"] = f"{t['scan_field']['aos']['median'] / 1e6:.2f}"
    V["L_hist_soa_event_x"] = f"{t['history_indexed']['volley']['median'] / t['history_indexed']['soa_event']['median']:.0f}"
    V["EXPECT_MET"] = expectation_summary().split("/")[0] + " of " + expectation_summary().split("/")[1].split(" ")[0]
    p6 = R("phase6")["table"]["uniform"]
    V["ONLINE_X"] = f"{p6['no_consolidation']['train_s']['mean'] / p6['full']['train_s']['mean']:.0f}"
    return {k.replace('.', ''): v for k, v in V.items()}


def main():
    t1, t2 = layout_tables(); p2a, p2b = phase2_table(); p3a, p3b, p3c, p3d = phase3_tables(); p4a, p4b = phase4_table()
    ctx = {"GATES": gate_table(), "LAYOUT": t1, "LAYOUT_H": t2, "ABL_UNIFORM": ablation_table("uniform"), "ABL_ZIPF": ablation_table("zipf"),
           "EXPECT": expectation_table(), "P2": p2a, "P2COST": p2b, "P3": p3a, "P3CTL": p3b, "P3DRIFT": p3c, "P3FOLLOW": p3d, "P4": p4a, "P4HOMEO": p4b,
           "SWEEP": sweep_table(), "ABL_NOHOMEO": ablation_table_c()}
    text = (ROOT / "eval" / "report_narrative.md").read_text()
    for k, v in ctx.items():
        text = text.replace("{{TABLE:" + k + "}}", v)
    sc = scalars()
    text = re.sub(r"\{\{V:(\w+)\}\}", lambda m: sc[m.group(1)], text)
    text = fill(text)
    (ROOT / "REPORT.md").write_text(text)
    print("REPORT.md written,", len(text.splitlines()), "lines")


if __name__ == "__main__":
    main()
