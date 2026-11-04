"""Markdown report generation from a :class:`~neurostore.results.SuiteResult`.

Everything here is computed from the stored measurements -- the scorecard verdicts, the
pass/fail table and the pivots -- so a report can be regenerated from a JSON file
(``neurostore report results.json``) and can never drift from the data it describes.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Sequence

from .common import fmt_bytes
from .results import SuiteResult, WorkloadResult

# ----------------------------------------------------------------------------- formatting helpers


def _cell(x: Any) -> str:
    """Stringify a table cell; a literal ``|`` would split the column, so escape it."""
    return str(x).replace("|", "\\|").replace("\n", " ")


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    headers = [_cell(h) for h in headers]
    rows = [[_cell(c) for c in r] for r in rows]
    if not rows:
        return "_(no data)_\n"
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = lambda cells: "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"  # noqa: E731
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([line(headers), sep, *map(line, rows)]) + "\n"


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f}%"


def num(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if abs(x) >= 1e6:
        return f"{x:.2e}"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{digits}f}"


def sig(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.3g}" if abs(x) < 1000 else f"{x:,.0f}"


def _order(names: Iterable[str], first: Sequence[str] = ()) -> list[str]:
    seen = list(dict.fromkeys(names))
    head = [n for f in first for n in seen if n.startswith(f)]
    return list(dict.fromkeys(head + seen))


def _pivot(rows: list[dict], row_key: str, col_key: str, cell: Callable[[dict], str], rows_order: Sequence | None = None, cols_order: Sequence | None = None,
           corner: str = "") -> str:
    rks = list(rows_order) if rows_order is not None else list(dict.fromkeys(r[row_key] for r in rows))
    cks = list(cols_order) if cols_order is not None else list(dict.fromkeys(r[col_key] for r in rows))
    idx = {(r[row_key], r[col_key]): r for r in rows}
    body = [[rk] + [cell(idx[(rk, ck)]) if (rk, ck) in idx else "" for ck in cks] for rk in rks]
    return md_table([corner or row_key] + [str(c) for c in cks], body)


# ----------------------------------------------------------------------------- per-workload renderers


def _render_a(w: WorkloadResult) -> str:
    out = []
    cfg = w.params
    thr, rec = w.tables["thresholds"], w.tables["recall_vs_noise"]
    dims = cfg["dims"]
    out.append(f"**Noise tolerance** -- largest bit-flip rate with >= {cfg.get('recall_target', 0.99) * 100:g}% exact value recall "
               f"(N={cfg['n_items']} stored pairs, {cfg['trials']} queries/level; grid step {cfg['noise_step'] * 100:g}%). "
               "Cell = last passing level (first failing level).\n")
    names = _order((r["system"] for r in thr), first=("hopfield", "bruteforce", "faiss", "lsh", "datasketch", "qdrant", "python_dict", "rocksdb", "redis", "theory"))
    rows = []
    for n in names:
        cells = []
        for d in dims:
            r = next((x for x in thr if x["dim"] == d and x["system"] == n), None)
            cells.append("" if r is None else f"{pct(r['last_ok'], 0)} ({pct(r['first_fail'], 0)})" if r["first_fail"] is not None else f"{pct(r['last_ok'], 1)}")
        rows.append([n, *cells])
    out.append(md_table(["system"] + [f"D={d}" for d in dims], rows))
    d0 = dims[0]
    shown = [x for x in (0, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5) if any(abs(r["noise"] - x) < 1e-9 for r in rec)]
    out.append(f"\n**Exact value recall vs noise** (D={d0}; Wilson 95% CI in the JSON). `theory_bruteforce_ident` is the closed-form exact-NN identification probability.\n")
    sel = [r for r in rec if r["dim"] == d0 and any(abs(r["noise"] - x) < 1e-9 for x in shown)]
    out.append(_pivot(sel, "system", "noise", lambda r: pct(r["rate"], 1), [n for n in names if any(r["system"] == n for r in sel)], shown, corner="system"))
    if "scale_thresholds" in w.tables:
        st = w.tables["scale_thresholds"]
        ns = sorted({r["n_items"] for r in st})
        out.append(f"\n**Tolerance vs store size** (D={cfg['scale_dim']}): more stored items = more impostors = lower tolerance.\n")
        sn = _order((r["system"] for r in st), first=("hopfield", "bruteforce", "faiss", "lsh", "python_dict", "theory"))
        out.append(md_table(["system"] + [f"N={n}" for n in ns], [[s] + [pct(next((x["last_ok"] for x in st if x["system"] == s and x["n_items"] == n), None), 0) for n in ns] for s in sn]))
    if "beta_thresholds" in w.tables:
        bt, bs = w.tables["beta_thresholds"], w.tables["beta_sweep"]
        steps = sorted(int(k.rsplit("_s", 1)[1]) for k in bt[0] if k.startswith("value_last_ok_s"))
        n_show = 0.4 if any(abs(r["noise"] - 0.4) < 1e-9 for r in bs) else (0.3 if any(abs(r["noise"] - 0.3) < 1e-9 for r in bs) else None)
        out.append(f"\n**Hopfield beta and attractor-iteration sweep** (D={cfg['beta_sweep_dim']}, N={cfg['n_items']}): a single softmax step blends neighbouring patterns, so exact *value* "
                   "recovery breaks before *identification* does; extra attractor steps (key-space cleanup, then read-out) pull the state into one pattern's basin.\n")
        head = ["beta", "identification tolerance"] + [f"value-recall tolerance, {s} step{'s' if s > 1 else ''}" for s in steps]
        if n_show is not None:
            head += [f"value recall @ {n_show * 100:g}%, {s} step{'s' if s > 1 else ''}" for s in steps]
        body = []
        for t_ in bt:
            row = [f"{t_['beta']:g}", pct(t_["ident_last_ok"], 0)] + [pct(t_[f"value_last_ok_s{s}"], 0) for s in steps]
            if n_show is not None:
                row += [pct(next(r["value_recall"] for r in bs if r["beta"] == t_["beta"] and r["steps"] == s and abs(r["noise"] - n_show) < 1e-9), 1) for s in steps]
            body.append(row)
        out.append(md_table(head, body))
    lat = w.tables.get("latency_memory", [])
    for n_items in sorted({r["n_items"] for r in lat}):
        sel = [r for r in lat if r["n_items"] == n_items]
        out.append(f"\n**Latency / memory / density** (D={cfg['latency_dim']}, N={n_items}; reads use {cfg['latency_noise'] * 100:g}% noisy keys; 1 thread; "
                   f"timer+call overhead is in the environment block). `eta` = information bits / allocated bits; `R/W` = read p50 / write p50.\n")
        out.append(md_table(["system", "read p50 us", "read p99 us", "write p50 us", "R/W", "batch q/s", "bytes/pair", "eta", "build s"],
                            [[r["system"], num(r["read_p50_us"], 2), num(r["read_p99_us"], 1), num(r["write_p50_us"], 2), num(r["read_write_ratio"], 1),
                              num(r["qps_batch"], 0), num(r["bytes_per_pair"], 0), sig(r["eta"]), num(r["build_s"], 3)] for r in sel]))
    return "\n".join(out)


def _render_b(w: WorkloadResult) -> str:
    out = []
    cfg, rows = w.params, w.tables["accuracy"]
    dmax = max(cfg["dims"])
    combos = sorted({(r["K"], r["corrupt"]) for r in rows})
    sysnames = _order((r["system"] for r in rows if r["dim"] in (dmax, None)), first=("vsa", "fuzzy", "inverted", "linear", "composite"))
    for key, title in (("ident_rate", "Record identification accuracy"), ("recon_rate", "Missing-field reconstruction accuracy")):
        out.append(f"**{title}** (M={cfg['n_records']} records, F={cfg['n_fields']} fields, vocab {cfg['vocab']}, D={dmax} for VSA; "
                   f"{cfg['n_queries']} unique-cue queries per cell). Columns: K known fields / c corrupted.\n")
        body = []
        for s in sysnames:
            cells = []
            for k, c in combos:
                r = next((x for x in rows if x["system"] == s and x["K"] == k and x["corrupt"] == c and x["dim"] in (dmax, None)), None)
                cells.append(pct(r[key], 1) if r else "")
            body.append([s, *cells])
        out.append(md_table(["system"] + [f"K={k}, c={c}" for k, c in combos], body))
        out.append("")
    k3 = 3 if 3 in cfg["k_values"] else cfg["k_values"][-1]
    out.append(f"**VSA vs dimension** (K={k3}): identification / reconstruction accuracy, unbound-filler cosine and bit agreement.\n")
    body = []
    for r in sorted((x for x in rows if x["family"] == "neurostore" and x["K"] == k3), key=lambda x: (x["system"], x["corrupt"], x["dim"])):
        body.append([r["system"], r["dim"], r["corrupt"], pct(r["ident_rate"], 1), pct(r["recon_rate"], 1), num(r["unbind_cos"], 3), pct(r["bit_agreement"], 1)])
    out.append(md_table(["system", "D", "corrupted", "identify", "reconstruct", "unbind cosine", "bit agreement"], body))
    out.append("\n**Control -- the original harness's method** (unbind a role that is *not* in the cue). Cosine to the true filler is pure noise of scale 1/sqrt(D); it can never exceed the 0.6 success bar.\n")
    out.append(md_table(["D", "mean cosine", "std", "1/sqrt(D)", "success (cos>0.6)"],
                        [[c["dim"], num(c["mean_cos"], 4), num(c["std_cos"], 4), num(c["chance_std"], 4), pct(c["frac_cos_gt_0p6"], 0)] for c in w.tables["reference_method_control"]]))
    lat = w.tables["latency_memory"]
    out.append(f"\n**Latency / memory** (K={cfg['latency_k']}, D={cfg['latency_dim']}, M={cfg['n_records']}, 1 thread). Information content = {w.summary['entropy_bits_per_record']:.1f} bits/record.\n")
    out.append(md_table(["system", "p50 us", "p99 us", "queries/s", "bytes/record", "eta"],
                        [[r["system"], num(r["p50_us"], 1), num(r["p99_us"], 1), num(r["qps"], 0), num(r["bytes_per_record"], 0), sig(r["eta"])] for r in lat]))
    out.append("\n**Index blow-up** -- composite (multi-column) indexes needed to serve any K-of-F cue in O(1):\n")
    out.append(md_table(["K", "indexes for this K", "indexes for any subset", "composite bytes", "inverted-index bytes", "table bytes", "composite measured?"],
                        [[b["K"], b["n_indexes_for_K"], b["n_indexes_any_subset"], fmt_bytes(b["composite_bytes_extrapolated"] or 0), fmt_bytes(b["inverted_bytes"]),
                          fmt_bytes(b["table_bytes"]), "yes" if b["composite_measured"] else "extrapolated"] for b in w.tables["index_blowup"]]))
    return "\n".join(out)


def _render_c(w: WorkloadResult) -> str:
    out = []
    cfg, rows = w.params, w.tables["systems"]
    for n in sorted({r["n_cache"] for r in rows}):
        sel = [r for r in rows if r["n_cache"] == n]
        out.append(f"**N = {n:,} cached prompts** (d={sel[0]['dim']}, paraphrase cosine {cfg['para_cos']}, topic rho {cfg['topic_rho']}; hit threshold calibrated at {cfg['target_fpr'] * 100:g}% false hits; 1 thread).\n")
        out.append(md_table(["system", "recall@1", "TPR@FPR", "real FPR", "ANN agree", "p50 us", "q/s (1)", "q/s (batch)", "B/prompt", "build s"],
                            [[r["system"], pct(r["recall_at_1"], 1), pct(r["tpr_at_fpr"], 1), pct(r["realised_fpr"], 1), pct(r.get("ann_agreement"), 1), num(r["p50_us"], 1),
                              num(r["qps_single"], 0), num(r["qps_batch"], 0), num(r["bytes_per_prompt"], 0), num(r["build_s"], 2)] for r in sel]))
        out.append("")
    sw = w.tables.get("hardness_sweep", [])
    if sw:
        n = sw[0]["n_cache"]
        cs = sorted({r["para_cos"] for r in sw}, reverse=True)
        out.append(f"**Recall@1 vs paraphrase strength** (N={n:,}): how fast each system degrades as paraphrases drift from their source.\n")
        out.append(_pivot(sw, "system", "para_cos", lambda r: pct(r["recall_at_1"], 1), cols_order=cs, corner="system"))
        out.append("\n**True-hit rate at the calibrated false-hit rate** (same sweep):\n")
        out.append(_pivot(sw, "system", "para_cos", lambda r: pct(r["tpr_at_fpr"], 1), cols_order=cs, corner="system"))
    return "\n".join(out)


def _render_d(w: WorkloadResult) -> str:
    out = []
    cfg, cap = w.params, w.tables["capacity"]
    e = cfg["n_entities"]
    out.append(f"**Single-hop accuracy** (E={e} entities, R={cfg['n_relations']} relations; all T triples superposed in ONE D-vector; {cfg['n_graph_samples']} graph(s) per cell). "
               "`ghost rate` = 1 - accuracy. Cell = measured (closed-form theory).\n")
    for alg in cfg["algebras"]:
        sel = [r for r in cap if r["algebra"] == alg]
        out.append(f"_{alg.upper()}_\n")
        out.append(_pivot(sel, "dim", "triples", lambda r: f"{pct(r['single_hop_acc'], 1)} ({pct(r['theory_acc'], 0)})", corner="D \\ triples T"))
        out.append("")
    out.append(f"**Capacity** -- largest T on the grid with single-hop accuracy >= {cfg['accuracy_target'] * 100:g}%, vs closed-form bound `D/(2 ln E)`-type law (E={e}):\n")
    out.append(md_table(["algebra", "D", "T_max (grid)", "first failing T", "T_max (interpolated)", "T_max (theory)"],
                        [[s["algebra"], s["dim"], num(s["t_max_grid"], 0), num(s["first_fail_grid"], 0), num(s["t_max_interpolated"], 0), num(s["t_max_theory"], 0)] for s in w.tables["capacity_summary"]]))
    hops = cfg["hops"]
    d_show = cfg["dims"][len(cfg["dims"]) // 2]
    out.append(f"\n**Multi-hop path accuracy** (D={d_show}; success = every hop of the walk correct; errors compound):\n")
    sel = [r for r in cap if r["dim"] == d_show]
    out.append(md_table(["algebra", "T"] + [f"{h}-hop" for h in hops], [[r["algebra"], r["triples"]] + [pct(r[f"path_acc_h{h}"], 1) for h in hops] for r in sorted(sel, key=lambda x: (x["algebra"], x["triples"]))]))
    out.append("\n**Traversal speed and memory** (a graph far inside capacity -- this measures cost, not accuracy):\n")
    out.append(md_table(["system", "D", "E", "triples", "hops/ms (1 query)", "hops/ms (batch)", "bytes/triple", "trace bytes"],
                        [[r["system"], r["dim"] or "-", r["n_entities"], r["triples"], num(r["hops_per_ms_single"], 1), num(r["hops_per_ms_batch"], 1), num(r["bytes_per_triple"], 0), num(r["trace_bytes"], 0)]
                         for r in w.tables["speed"]]))
    out.append("\n**Direction control** -- querying an edge backwards must not return the subject:\n")
    out.append(md_table(["algebra", "D", "T", "forward accuracy", "reverse hit rate", "chance"],
                        [[r["algebra"], r["dim"], r["triples"], pct(r["forward_acc"], 1), pct(r["reverse_hit_rate"], 1), pct(r["chance"], 1)] for r in w.tables["direction_control"]]))
    return "\n".join(out)


def _render_capacity(w: WorkloadResult) -> str:
    cfg = w.params
    rows = []
    for s in w.tables["capacity_summary"]:
        if s["memory"] == "vsa_bundle_raw":
            continue
        cm = s["c_max_interpolated"] if s["c_max_interpolated"] is not None else s["c_max_last_ok"]
        label = (">= " if s["censored"] else "") + num(cm, 0)
        rows.append([s["dim"], s["memory"] + (f" (noise {s['noise'] * 100:g}%)" if s["memory"] == "modern_hopfield" else ""), s["metric"], label,
                     num(s["c_max_theory"], 0) if s["c_max_theory"] is not None else "-", sig(s["eta"]), fmt_bytes(s["allocated_bytes"])])
    out = [f"**Capacity C_max** -- items stored before BER > {cfg['ber_target'] * 100:g}%, and storage density eta at that point.\n",
           md_table(["D", "memory", "BER measured on", "C_max", "C_max theory", "eta", "allocated"], rows)]
    raw = [s for s in w.tables["capacity_summary"] if s["memory"] == "vsa_bundle_raw"]
    if raw:
        out.append("\nA superposed bundle read **bit by bit** (no cleanup) already has "
                   f"BER {pct(raw[0]['raw_ber_at_smallest_n'], 1)} at N={raw[0]['smallest_n']} for every D -- the per-bit SNR is 1/sqrt(N-1), independent of D.\n")
    return "\n".join(out)


def _render_cim(w: WorkloadResult) -> str:
    out = []
    cfg, rows = w.params, w.tables["tolerance"]
    out.append("**Accuracy under analogue non-idealities -- EMULATED** (top-1 identification; value recall in the JSON). One non-ideality at a time, others ideal "
               f"(N={cfg['n_keys']} keys, {cfg['n_queries']} queries). Read noise is a fraction of a 256-row tile's full-scale output.\n")
    for d in cfg["dims"]:
        for qn in cfg["query_noise"]:
            sel = [r for r in rows if r["dim"] == d and r["query_noise"] == qn and r["param"] != "ideal"]
            ideal = next(r for r in rows if r["dim"] == d and r["query_noise"] == qn and r["param"] == "ideal")
            out.append(f"_D={d}, query bit-flip noise {qn * 100:g}% (ideal digital accuracy {pct(ideal['ident_acc'], 1)})_\n")
            body = []
            for param in dict.fromkeys(r["param"] for r in sel):
                items = [r for r in sel if r["param"] == param]
                for variant in dict.fromkeys(r["variant"] for r in items):
                    series = sorted((r for r in items if r["variant"] == variant), key=lambda r: r["value"])
                    body.append([param, variant, ", ".join(f"{sig(r['value'])}: {pct(r['ident_acc'], 0)}" for r in series)])
            out.append(md_table(["non-ideality", "variant", "value: identification accuracy"], body))
            out.append("")
    out.append("**Tolerance limits** (largest value keeping >= 99% identification; '-' = ideal accuracy itself is below 99% here, or never reached):\n")
    out.append(md_table(["D", "query noise", "parameter", "limit", "unit"],
                        [[x["dim"], pct(x["query_noise"], 0), x["param"], sig(x["max_value_at_99"]) if x["max_value_at_99"] is not None else "-", x["unit"]] for x in w.tables["tolerance_limits"]]))
    out.append("\n**Search latency -- MODELED projection vs MEASURED CPU** (not a measured hardware result; see assumptions below). "
               "CPU = best single-thread time of float32 BLAS mat-vec or packed popcount on this machine.\n")
    out.append(md_table(["N keys", "D", "CPU best us (measured)", "preset", "tiles", "waves", "modeled ns", "speed-up if model holds", "regime"],
                        [[r["n_keys"], r["dim"], num(r["cpu_best_us"], 1), r["preset"], r["tiles"], r["waves"], num(r["modeled_ns"], 1), f"{r['modeled_speedup_vs_cpu_best']:.0f}x", r["regime"]] for r in w.tables["latency_model"]]))
    out.append("\nTiming-model parameters (illustrative placeholders, ns):\n")
    from .primitives.cim import CIMTimingModel

    out.append(md_table(["preset"] + list(next(iter(CIMTimingModel.PRESETS.values())).keys()), [[k] + list(v.values()) for k, v in CIMTimingModel.PRESETS.items()]))
    return "\n".join(out)


RENDERERS: dict[str, Callable[[WorkloadResult], str]] = {"a": _render_a, "b": _render_b, "c": _render_c, "d": _render_d, "capacity": _render_capacity, "cim": _render_cim}
ORDER = ["a", "b", "c", "d", "capacity", "cim"]


# ----------------------------------------------------------------------------- scorecard


def _verdict(ns: float, base: float, higher_is_better: bool, tol: float = 0.05) -> str:
    if ns is None or base is None:
        return "n/a"
    if not higher_is_better:
        ns, base = -ns, -base
        # compare on a signed-safe relative scale
        scale = max(abs(ns), abs(base), 1e-12)
        diff = (ns - base) / scale
    else:
        scale = max(abs(ns), abs(base), 1e-12)
        diff = (ns - base) / scale
    return "WIN" if diff > tol else ("LOSS" if diff < -tol else "TIE")


def scorecard(suite: SuiteResult) -> tuple[list[list[str]], dict[str, int]]:
    rows: list[list[str]] = []
    tally = {"WIN": 0, "TIE": 0, "LOSS": 0}
    for w in suite.workloads:
        for h in w.headlines:
            systems = h["systems"]
            ns = {n: s["value"] for n, s in systems.items() if s["family"] == "neurostore" and s["value"] is not None}
            if not ns:
                continue
            hib = h["higher_is_better"]
            pick = (max if hib else min)
            best_ns = pick(ns, key=ns.get)
            cells = []
            for fam in dict.fromkeys(s["family"] for s in systems.values()):
                if fam in ("neurostore", "theory"):
                    continue
                fs = {n: s["value"] for n, s in systems.items() if s["family"] == fam and s["value"] is not None}
                if not fs:
                    continue
                best = pick(fs, key=fs.get)
                v = _verdict(ns[best_ns], fs[best], hib)
                if v in tally:
                    tally[v] += 1
                cells.append(f"{fam}: {sig(fs[best])} ({best}) **{v}**")
            rows.append([w.workload.upper(), f"{h['metric']} [{h['unit']}{'' if hib else ', lower=better'}]", h["context"], f"{sig(ns[best_ns])} ({best_ns})", "<br>".join(cells) or "no baseline"])
    return rows, tally


# ----------------------------------------------------------------------------- full report


def render_report(suite: SuiteResult) -> str:
    env = suite.environment
    libs = ", ".join(f"{k} {v}" for k, v in (env.get("optional_libs") or {}).items() if v) or "none"
    out = [
        "# NeuroStore benchmark report\n",
        f"- framework `{suite.framework_version}` | profile `{suite.profile}` | seed `{suite.seed}` | generated {suite.created_utc}",
        f"- machine: {env.get('cpu_model')} x{env.get('cpu_count')}, {fmt_bytes(env.get('memory_bytes') or 0)} RAM, {env.get('platform')}",
        f"- software: Python {env.get('python')}, NumPy {env.get('numpy')} ({env.get('blas')}); optional libraries: {libs}",
        f"- timing: per-call `perf_counter_ns`, GC disabled, **1 thread unless stated**; empty-call overhead {env.get('timer_overhead_ns', 0):.0f} ns "
        "(latencies below ~1 us are dominated by Python call overhead).",
        f"- total benchmark time: {sum(w.elapsed_s for w in suite.workloads):.0f} s\n",
    ]
    rows, tally = scorecard(suite)
    if rows:
        out.append("## Scorecard (computed from the data)\n")
        out.append("Best NeuroStore system vs the best system of each competing family on the same headline metric. WIN/LOSS means more than 5% better/worse; otherwise TIE. "
                   "Section-5 CIM latency is excluded: it is a model, not a measurement.\n")
        out.append(md_table(["wl", "metric", "context", "best NeuroStore", "best of each competing family"], rows))
        total = sum(tally.values())
        out.append(f"\n**Tally over {total} comparisons: {tally['WIN']} win, {tally['TIE']} tie, {tally['LOSS']} loss.**\n")
    checks = [(w.workload.upper(), c) for w in suite.workloads for c in w.checks]
    if checks:
        out.append("## Checks against the evaluation spec and benchmark self-validation\n")
        mark = {True: "PASS", False: "**FAIL**", None: "info"}
        out.append(md_table(["wl", "check", "target", "measured", "result"], [[wl, c["name"], c["target"], c["measured"], mark[c["passed"]]] for wl, c in checks]))
        notes = [(c["name"], c["note"]) for _, c in checks if c.get("note")]
        if notes:
            out.append("\nNotes on checks:\n")
            out += [f"- *{n}*: {t}" for n, t in notes]
    for key in ORDER:
        w = suite.get(key)
        if w is None:
            continue
        out.append(f"\n## {w.title}\n")
        out.append(f"_{w.elapsed_s:.0f} s; seed {w.seed}; profile {w.profile}_\n")
        if not w.tables:  # e.g. the workload crashed: say so instead of failing the whole report
            out.append("_No results: the workload produced no tables (see notes and checks)._")
        else:
            try:
                out.append(RENDERERS[key](w))
            except Exception as exc:  # noqa: BLE001 - a renderer bug must not lose the measurements
                out.append(f"_Section could not be rendered ({type(exc).__name__}: {exc}); the raw data is in results.json._")
        if w.notes:
            out.append("\nNotes:\n")
            out += [f"- {n}" for n in w.notes]
    skipped = [(w.workload.upper(), s) for w in suite.workloads for s in w.skipped]
    if skipped:
        out.append("\n## Skipped systems\n")
        out.append(md_table(["wl", "system", "reason"], [[wl, s["system"], s["reason"]] for wl, s in skipped]))
    return "\n".join(out) + "\n"
