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
    n_used = {}
    for r in rec:
        if r["family"] != "theory":
            n_used[r["system"]] = max(n_used.get(r["system"], 0), r["n"])
    capped = {s_: n_ for s_, n_ in n_used.items() if n_ != cfg["trials"]}
    if capped:
        out.append(f"\nQueries per noise level: {cfg['trials']} for every system except " + ", ".join(f"{s_} ({n_})" for s_, n_ in capped.items()) +
                   " (slow Python systems are evaluated on a prefix of the same queries). Thresholds come from one random key set per dimension, so read them to +-1 grid step; "
                   "the closed-form rows are the key-set average.\n")
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
    out.append("\n**Index blow-up** -- composite (multi-column) indexes needed to serve any K-of-F cue in O(1) vs one inverted index per column. "
               "'Python dicts' is what is implemented here; 'compact arrays' is the same index as sorted (key, row id) arrays, which is the fair size to set against the "
               "numpy posting lists -- the C(F,K)-vs-F scaling holds in both.\n")
    out.append(md_table(["K", "indexes for this K", "indexes for any subset", "composite, Python dicts", "composite, compact arrays", "inverted index", "base table"],
                        [[b["K"], b["n_indexes_for_K"], b["n_indexes_any_subset"], fmt_bytes(b["composite_bytes_extrapolated"] or 0) + ("" if b["composite_measured"] else " (extrap.)"),
                          fmt_bytes(b["composite_compact_bytes"]), fmt_bytes(b["inverted_bytes"]), fmt_bytes(b["table_bytes"])] for b in w.tables["index_blowup"]]))
    return "\n".join(out)


def _render_c(w: WorkloadResult) -> str:
    out = []
    cfg, rows = w.params, w.tables["systems"]
    for n in sorted({r["n_cache"] for r in rows}):
        sel = [r for r in rows if r["n_cache"] == n]
        out.append(f"**N = {n:,} cached prompts** (d={sel[0]['dim']}, paraphrase cosine {cfg['para_cos']}, topic rho {cfg['topic_rho']}; true-hit rate read at exactly {cfg['target_fpr'] * 100:g}% false hits; 1 thread).\n")
        out.append(md_table(["system", "recall@1", f"TPR@{cfg['target_fpr'] * 100:g}%FPR", "deployed FPR", "ANN agree", "p50 us", "q/s (1)", "q/s (batch)", "B/prompt index", "B/prompt total", "build s"],
                            [[r["system"], pct(r["recall_at_1"], 1), pct(r["tpr_at_fpr"], 1), pct(r["deployed_fpr"], 1), pct(r.get("ann_agreement"), 1), num(r["p50_us"], 1),
                              num(r["qps_single"], 0), num(r["qps_batch"], 0), num(r["bytes_per_prompt"], 0), num(r["bytes_per_prompt_total"], 0), num(r["build_s"], 2)] for r in sel]))
        out.append("")
    sw = w.tables.get("hardness_sweep", [])
    if sw:
        n = sw[0]["n_cache"]
        cs = sorted({r["para_cos"] for r in sw}, reverse=True)
        out.append(f"**Recall@1 vs paraphrase strength** (N={n:,}): how fast each system degrades as paraphrases drift from their source.\n")
        out.append(_pivot(sw, "system", "para_cos", lambda r: pct(r["recall_at_1"], 1), cols_order=cs, corner="system"))
        out.append(f"\n**True-hit rate at exactly {cfg['target_fpr'] * 100:g}% false hits** (same sweep):\n")
        out.append(_pivot(sw, "system", "para_cos", lambda r: pct(r["tpr_at_fpr"], 1), cols_order=cs, corner="system"))
        bf = [r for r in sw if r["system"] == "bruteforce_cosine"]
        below = [r["para_cos"] for r in bf if r["tpr_at_fpr"] < 0.5]
        above = [r["para_cos"] for r in bf if r["tpr_at_fpr"] >= 0.5]
        if below and above:
            out.append(f"\nRead this table with care below paraphrase cosine ~{min(above):g}. Novel prompts from the cache's own topics sit at cosine ~topic_rho = {cfg['topic_rho']:g} to each other, so the "
                       f"nearest cached neighbour of a novel prompt is already quite similar; here the exact-cosine true-hit rate collapses between paraphrase cosine {max(below):g} and {min(above):g}. "
                       "Once a paraphrase is no more similar to its source than a novel prompt is to its nearest neighbour, **no system can separate them at this false-hit rate** (the task is "
                       "ill-posed there; it is not a defect of any one system), and a noisier code can score slightly above an exact one by chance because every positive sits at exactly the stated cosine. "
                       "Recall@1 (previous table) is unaffected: the nearest cached prompt is still the right one.\n")
    fs = w.tables.get("flyhash_sweep", [])
    if fs:
        paras = sorted({r["para_cos"] for r in fs}, reverse=True)
        out.append("\n**FlyHash hyper-parameter sweep vs SimHash at the same number of bits** (expansion m, winners k incl. ~5% of the original paper, fan-in density; "
                   f"N={w.params.get('sweep_n_cache'):,}). `B/prompt` is the inverted-index / packed-code size. FlyHash is given every chance; SimHash is the plain random-hyperplane code.\n")
        keys = list(dict.fromkeys((r["kind"], r["bits"], r["k"], r["density"]) for r in fs))
        body = []
        for kind, bits, k, dens in keys:
            sel = {r["para_cos"]: r for r in fs if (r["kind"], r["bits"], r["k"], r["density"]) == (kind, bits, k, dens)}
            label = f"flyhash m={bits} k={k} ({k / bits * 100:.1f}%) fan-in {dens:g}" if kind == "flyhash" else f"simhash {bits} bits"
            body.append([label, num(next(iter(sel.values()))["bytes_per_prompt"], 0)] + [pct(sel[p]["recall_at_1"], 1) for p in paras] + [pct(sel[p]["tpr_at_fpr"], 1) for p in paras])
        out.append(md_table(["code", "B/prompt"] + [f"recall@1 @{p:g}" for p in paras] + [f"TPR@FPR @{p:g}" for p in paras], body))
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
    out.append(f"**Capacity** -- largest T with single-hop accuracy >= {cfg['accuracy_target'] * 100:g}% (E={e}), found by bisection on fresh random graphs "
               f"({cfg['capacity_graphs']} per evaluation, 3% resolution), against the closed-form bound ~ D / (2 ln E + margin). The coarse grid above only resolves a factor of two.\n")
    out.append(md_table(["algebra", "D", "T_max measured", "T_max theory", "measured / theory", "grid: last pass / first fail"],
                        [[s["algebra"], s["dim"], (">= " if s["censored"] else "") + num(s["t_max_measured"], 0), num(s["t_max_theory"], 0),
                          num(s["ratio_to_theory"], 2) if s["ratio_to_theory"] is not None else "-", f"{num(s['t_max_grid'], 0)} / {num(s['first_fail_grid'], 0)}"] for s in w.tables["capacity_summary"]]))
    hops = cfg["hops"]
    d_show = cfg["dims"][len(cfg["dims"]) // 2]
    out.append(f"\n**Multi-hop path accuracy** (D={d_show}; success = every hop of the walk correct; errors compound):\n")
    sel = [r for r in cap if r["dim"] == d_show]
    out.append(md_table(["algebra", "T"] + [f"{h}-hop" for h in hops], [[r["algebra"], r["triples"]] + [pct(r[f"path_acc_h{h}"], 1) for h in hops] for r in sorted(sel, key=lambda x: (x["algebra"], x["triples"]))]))
    out.append("\n**Traversal speed and memory** (graph sized to 70% of the closed-form capacity at each (D, E) and at most 100 triples, so the timed memory is an accurate one; "
               "hop time itself does not depend on the number of triples):\n")
    out.append(md_table(["system", "D", "E", "triples", "hops/ms (1 query)", "hops/ms (batch)", "bytes/triple (as built)", "bytes/triple at 99% capacity", "trace bytes", "allocated"],
                        [[r["system"], r["dim"] or "-", r["n_entities"], r["triples"], sig(r["hops_per_ms_single"]), sig(r["hops_per_ms_batch"]), num(r["bytes_per_triple"], 0),
                          num(r.get("bytes_per_triple_at_capacity"), 0) if r.get("bytes_per_triple_at_capacity") else "-", num(r["trace_bytes"], 0), fmt_bytes(r["memory_bytes"])]
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
               "CPU = the fastest single-thread implementation available here (FAISS binary flat if installed, else NumPy).\n")
    out.append(md_table(["N keys", "D", "CPU best us (measured)", "CPU implementation", "preset", "tiles", "waves", "modeled ns", "speed-up if model holds", "regime"],
                        [[r["n_keys"], r["dim"], num(r["cpu_best_us"], 1), r.get("cpu_best_impl", "-"), r["preset"], r["tiles"], r["waves"], num(r["modeled_ns"], 1), f"{r['modeled_speedup_vs_cpu_best']:.0f}x", r["regime"]] for r in w.tables["latency_model"]]))
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


def empty_tally() -> dict[str, dict[str, int]]:
    return {"quality": {"WIN": 0, "TIE": 0, "LOSS": 0}, "cost": {"WIN": 0, "TIE": 0, "LOSS": 0}}


def scorecard(suite: SuiteResult) -> tuple[list[list[str]], dict[str, dict[str, int]]]:
    """Rows for the scorecard table and a tally ``{'quality': {WIN, TIE, LOSS}, 'cost': {...}}``.

    * ``quality`` headlines (accuracy / robustness) compare every system.
    * ``cost`` headlines (latency, throughput, memory, build time) carry an accuracy *gate*: only systems that pass it
      compete. A system cannot win on speed by returning nothing, and NeuroStore is represented by its best system **among
      those that pass the gate** -- not by whichever variant is fastest while another one supplies the accuracy.
    """
    rows: list[list[str]] = []
    tally = empty_tally()
    for w in suite.workloads:
        for h in w.headlines:
            systems = h["systems"]
            kind = h.get("kind", "quality")
            eligible = {n: s for n, s in systems.items() if s.get("passes") is not False}
            excluded = [n for n, s in systems.items() if s.get("passes") is False and s["family"] != "theory"]
            ns = {n: s["value"] for n, s in eligible.items() if s["family"] == "neurostore" and s["value"] is not None}
            any_ns = any(s["family"] == "neurostore" for s in systems.values())
            if not any_ns:
                continue
            hib = h["higher_is_better"]
            pick = (max if hib else min)
            note = f"<br>_gate: {h['gate']}_" if h.get("gate") else ""
            metric = f"{'[cost] ' if kind == 'cost' else ''}{h['metric']} [{h['unit']}{'' if hib else ', lower=better'}]"
            if not ns:
                rows.append([w.workload.upper(), metric, h["context"], "**fails the gate**", "not compared" + note])
                continue
            best_ns = pick(ns, key=ns.get)
            cells = []
            for fam in dict.fromkeys(s["family"] for s in eligible.values()):
                if fam in ("neurostore", "theory"):
                    continue
                fs = {n: s["value"] for n, s in eligible.items() if s["family"] == fam and s["value"] is not None}
                if not fs:
                    continue
                best = pick(fs, key=fs.get)
                v = _verdict(ns[best_ns], fs[best], hib)
                if v in tally[kind]:
                    tally[kind][v] += 1
                cells.append(f"{fam}: {sig(fs[best])} ({best}) **{v}**")
            if excluded:
                cells.append("_excluded by the gate: " + ", ".join(excluded) + "_")
            rows.append([w.workload.upper(), metric, h["context"], f"{sig(ns[best_ns])} ({best_ns})",
                         "<br>".join(cells) + (note if cells else "") or "no baseline"])
    return rows, tally


def tally_text(tally: dict[str, dict[str, int]]) -> str:
    q, c = tally["quality"], tally["cost"]
    return (f"quality (accuracy / robustness): {q['WIN']} win, {q['TIE']} tie, {q['LOSS']} loss over {sum(q.values())} comparisons; "
            f"cost (latency / throughput / memory, accuracy-gated): {c['WIN']} win, {c['TIE']} tie, {c['LOSS']} loss over {sum(c.values())} comparisons")


# ----------------------------------------------------------------------------- full report


def render_report(suite: SuiteResult) -> str:
    env = suite.environment
    libs = ", ".join(f"{k} {v}" for k, v in (env.get("optional_libs") or {}).items() if v) or "none"
    out = [
        "# NeuroStore benchmark report\n",
        f"- framework `{suite.framework_version}` | profile `{suite.profile}` | seed `{suite.seed}` | generated {suite.created_utc}"
        + (f" | code revision `{env['git_commit'][:10]}`{' (uncommitted changes)' if env.get('git_dirty') else ''}" if env.get("git_commit") else ""),
        f"- machine: {env.get('cpu_model')} x{env.get('cpu_count')}, {fmt_bytes(env.get('memory_bytes') or 0)} RAM, {env.get('platform')}",
        f"- software: Python {env.get('python')}, NumPy {env.get('numpy')} ({env.get('blas')}); optional libraries: {libs}",
        f"- timing: per-call `perf_counter_ns`, GC disabled, "
        + ("**1 thread unless stated** (enforced with threadpoolctl)" if (env.get("optional_libs") or {}).get("threadpoolctl") else
           "**thread count NOT enforced (threadpoolctl missing): BLAS/OpenMP libraries may have used every core, so throughput and latency are not single-thread numbers**")
        + f"; empty-call overhead {env.get('timer_overhead_ns', 0):.0f} ns (latencies below ~1 us are dominated by Python call overhead).",
        f"- total benchmark time: {sum(w.elapsed_s for w in suite.workloads):.0f} s\n",
    ]
    rows, tally = scorecard(suite)
    if rows:
        out.append("## Scorecard (computed from the data)\n")
        out.append("Best NeuroStore system vs the best system of each competing family on the same headline metric. WIN/LOSS means more than 5% better/worse; otherwise TIE. "
                   "**Quality** rows (accuracy, robustness) compare every system; **[cost]** rows (latency, throughput, memory, build time) compare only systems that pass the row's accuracy "
                   "gate, and NeuroStore is represented by its best system *among those that pass* -- so nothing wins on speed by returning nothing, and no row mixes two NeuroStore "
                   "operating points. Comparisons against exact-match stores on noisy-key accuracy are structural (they cannot succeed by construction). "
                   "Section-5 CIM latency is excluded: it is a model, not a measurement.\n")
        out.append(md_table(["wl", "metric", "context", "best NeuroStore", "best of each competing family (that passes the gate)"], rows))
        out.append(f"\n**Tally -- {tally_text(tally)}.**\n")
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
