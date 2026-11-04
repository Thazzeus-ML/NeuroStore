"""Phase 5: physical layout benchmarks. NumPy timings are RELATIVE, not absolute: per-call Python overhead
is a large share of small operations. Hypotheses and the decision rule are read from gates.toml."""
import gc, sys, time, tracemalloc, json
import numpy as np
from neurostore.layouts import make_store, CSRGraph, DictGraph
from neurostore.layouts.volley import VolleyStore
from eval.gates import load_gates, write_result, ROOT

G = load_gates(); g = G["phase5"]; seeds = G["global"]["gate_seeds"]
QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"
N_EV = 20000 if QUICK else g["n_events"]; K = g["k"]; NU = 1024
SPIKE = ["aos", "soa", "soa_event", "volley"]


def synth_events(n, seed):
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, NU + 1) ** g["zipf_s"]; lp = np.log(p / p.sum()).astype(np.float32)
    units = np.empty((n, K), np.int32)
    for s in range(0, n, 4000):
        e = min(n, s + 4000)
        score = lp[None, :] + rng.gumbel(size=(e - s, NU)).astype(np.float32)
        units[s:e] = np.sort(np.argpartition(-score, K - 1, axis=1)[:, :K], axis=1)
    return (units, rng.integers(0, 256, (n, K)).astype(np.uint8), rng.integers(0, 256, (n, K)).astype(np.uint8),
            np.cumsum(rng.exponential(1.0, n)).astype(np.float32))


def fill(name, ev):
    st = make_store(name, K)
    u, s, su, t = ev
    for i in range(len(t)):
        st.append(i, t[i], u[i], s[i], su[i])
    return st


def timeit(fn, repeats, inner=1):
    out = []
    gc.collect(); gc.disable()
    try:
        fn()
        for _ in range(repeats):
            a = time.perf_counter_ns()
            for _ in range(inner):
                fn()
            out.append((time.perf_counter_ns() - a) / inner)
    finally:
        gc.enable()
    return out


def run_seed(seed):
    rng = np.random.default_rng(seed + 500)
    ev = synth_events(N_EV, seed)
    res = {w: {} for w in ["append", "rebuild_single", "rebuild_batch", "history_scan", "history_indexed", "scan_field",
                           "postings_build", "footprint_bytes", "postings_bytes"]}
    ids_single = rng.integers(0, N_EV, 2000); ids_batch = rng.integers(0, N_EV, 4000)
    freq = np.bincount(ev[0].ravel(), minlength=NU); hot = np.argsort(-freq)[:3]; cold = np.argsort(freq)[300:303]
    qunits = [int(u) for u in np.concatenate([hot, cold])]
    # append (fresh stores, first n_append events)
    sub = tuple(a[:g["n_append"]] for a in ev)
    for nm in SPIKE:
        def one():
            st = make_store(nm, K); u, s, su, t = sub
            for i in range(len(t)):
                st.append(i, t[i], u[i], s[i], su[i])
        res["append"][nm] = [x / g["n_append"] for x in timeit(one, g["repeats_append"])]
    for nm in SPIKE:
        st = fill(nm, ev)
        res["rebuild_single"][nm] = [x / len(ids_single) for x in timeit(lambda: [st.event(int(i)) for i in ids_single], g["repeats"])]
        res["rebuild_batch"][nm] = timeit(lambda: st.gather(ids_batch), g["repeats"], 3)
        res["history_scan"][nm] = [x / len(qunits) for x in timeit(lambda: [st.unit_history(u) for u in qunits], max(5, g["repeats"] // 2))]
        res["scan_field"][nm] = timeit(lambda: st.field_sum("strength"), g["repeats"], 3)
        res["postings_build"][nm] = timeit(st.build_postings, 3)
        res["history_indexed"][nm] = [x / len(qunits) for x in timeit(lambda: [st.unit_history_indexed(u) for u in qunits], g["repeats"], 5)]
        res["footprint_bytes"][nm] = [st.nbytes()]
        res["postings_bytes"][nm] = [st.postings_nbytes()]
        del st
    # ---------------- synapse graph
    NE = g["graph_out_degree"]
    pre = np.repeat(np.arange(NU), NE)
    post = np.concatenate([rng.choice(NU, NE, replace=False) for _ in range(NU)])
    w = rng.random(len(pre)).astype(np.float32); lag = (rng.random(len(pre)) * 2).astype(np.float16); ep = rng.integers(0, 30, len(pre))
    csr = CSRGraph(NU, pre, post, w, lag, ep); dg = DictGraph(NU, pre, post, w, lag, ep)
    dt = 0.97 ** np.arange(256); E = 30
    acts = [np.sort(rng.choice(NU, K, replace=False)) for _ in range(100)]
    outs = [rng.choice(NU, 200, replace=False) for _ in range(5)]
    ins = [int(v) for v in rng.choice(NU, 20, replace=False)]
    res.update({w_: {} for w_ in ["sparse_accumulate", "out_edge_scan", "in_edge_scan", "graph_bytes"]})
    for nm, gr in (("csr", csr), ("dict", dg)):
        def acc():
            for a in acts:
                wh, po, we = gr.gather(a, dt, E)
                np.bincount(po, weights=we, minlength=NU)
        res["sparse_accumulate"][nm] = [x / len(acts) for x in timeit(acc, g["repeats"])]
        res["out_edge_scan"][nm] = [x / 200 for x in timeit(lambda: [gr.out_edges(u) for u in outs[0]], g["repeats"])]
        res["in_edge_scan"][nm] = [x / len(ins) for x in timeit(lambda: [gr.in_edges(v) for v in ins], max(5, g["repeats"] // 3))]
    csr.in_edges(0)                     # materialise the transposed index before measuring bytes
    res["graph_bytes"]["csr"] = [csr.nbytes()]
    tracemalloc.start(); snap0 = tracemalloc.take_snapshot()
    d2 = DictGraph(NU, pre, post, w, lag, ep)
    cur = tracemalloc.get_traced_memory()[0]; tracemalloc.stop()
    res["graph_bytes"]["dict"] = [cur]
    return res


def summarize_runs(per_seed, wl, nm):
    seed_med = [float(np.median(r[wl][nm])) for r in per_seed]
    pooled = np.concatenate([r[wl][nm] for r in per_seed])
    q1, med, q3 = np.percentile(pooled, [25, 50, 75])
    return {"median": float(med), "q1": float(q1), "q3": float(q3), "seed_medians": seed_med}


def verdict(table, wl, winner, group):
    meds = {nm: table[wl][nm]["median"] for nm in group}
    order = sorted(meds, key=meds.get)
    runner = order[1] if order[0] == winner else order[0]
    wins = sum(1 for i in range(len(seeds)) if min(group, key=lambda n: table[wl][n]["seed_medians"][i]) == winner)
    margin = 1 - meds[winner] / meds[runner] if meds[order[0]] == meds[winner] else -(meds[winner] / meds[order[0]] - 1)
    held = bool(order[0] == winner and margin >= g["min_margin"] and wins >= g["min_seed_wins"])
    return {"workload": wl, "predicted_winner": winner, "group": group, "fastest": order[0], "margin_vs_runner_up": margin,
            "seed_wins": wins, "held": held, "medians": meds}


per = []
for s in seeds:
    per.append(run_seed(s)); print("seed", s, "done", flush=True)
wls = list(per[0].keys())
table = {wl: {nm: summarize_runs(per, wl, nm) for nm in per[0][wl]} for wl in wls}
base3 = ["aos", "soa", "volley"]
H = []
for wl in ("rebuild_single", "rebuild_batch"):
    H.append({"hypothesis": "H1a volley wins event-local reads", **verdict(table, wl, "volley", base3)})
H.append({"hypothesis": "H1b volley wins footprint", **verdict({**table, "footprint_bytes": table["footprint_bytes"]}, "footprint_bytes", "volley", base3)})
H.append({"hypothesis": "H2 plain SoA wins single-field scans", **verdict(table, "scan_field", "soa", base3)})
H.append({"hypothesis": "H3a CSR wins out-edge scans", **verdict(table, "out_edge_scan", "csr", ["csr", "dict"])})
H.append({"hypothesis": "H3b CSR wins sparse accumulate", **verdict(table, "sparse_accumulate", "csr", ["csr", "dict"])})
for h in H:
    print(h["hypothesis"], h["workload"], "HELD" if h["held"] else "NOT HELD", f"fastest={h['fastest']} margin={h['margin_vs_runner_up']:.2f} wins={h['seed_wins']}/5")
out = {"quick": QUICK, "n_events": N_EV, "k": K, "table": table, "hypotheses": H, "numpy_version": np.__version__,
       "note": "NumPy timings are relative, not absolute (Python call overhead dominates small ops)."}
if QUICK:
    (ROOT / "results" / "dev").mkdir(parents=True, exist_ok=True); (ROOT / "results" / "dev" / "phase5.json").write_text(json.dumps(out, indent=1, default=float))
else:
    write_result("phase5", out)
