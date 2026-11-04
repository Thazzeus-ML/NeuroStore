"""Phase 6: remove each principle in turn and re-measure the phase 2-4 metrics (+ rate and surprise read-outs)."""
import sys, time, json
import numpy as np
from sim.generator import World, WorldConfig
from neurostore.config import Config
from neurostore.store import NeuroStore
from neurostore.readout import rate_readout
from eval.p2task import phase2_metrics, temporal_recall, encode_stream
from eval.protocol import ConceptMap
from eval.metrics import auc, summarize
from eval.gates import load_gates, write_result, ROOT

G = load_gates(); g = G["phase6"]; g2 = G["phase2"]
QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"
seeds = G["global"]["dev_seeds"][:1] if QUICK else G["global"]["gate_seeds"]
N_TR = 2000 if QUICK else g["n_train"]
C = WorldConfig().C
BASE = dict(homeo=True, gate=True, gate_tau=g["gate_tau"], pred_mode="assembly", hebbian=True, consolidate="batch",
            replay_budget=150, hot_tau=0.3, consolidate_every=200, strength_floor=0.25)
VARIANTS = {
    "full": {},
    "no_sparsity": dict(k=256, d=16, asm_match_min=40, asm_new_min=80, prune_R=256, spread_keep=256),
    "no_surprise_gate": dict(gate=False),
    "no_hebbian": dict(hebbian=False),
    "no_homeostasis": dict(homeo=False),
    "no_consolidation": dict(consolidate="online"),
    "no_learning_floor": dict(consolidate="off"),
}


def code_overlap_margin(store, stream, n=3000, seed=0):
    codes = [e[0] for e in encode_stream(store, stream)]
    single = [i for i in range(len(stream)) if len(stream.concepts[i]) == 1]
    rng = np.random.default_rng(seed); same, diff = [], []
    for _ in range(n):
        i, j = rng.choice(single, 2, replace=False)
        o = len(np.intersect1d(codes[i], codes[j])) / store.cfg.k
        (same if stream.concepts[i] == stream.concepts[j] else diff).append(o)
    return float(np.mean(same) - np.mean(diff)) if same and diff else float("nan")


def run(seed, name, overrides, wcfg):
    w = World(wcfg, seed)
    tr, va, te = w.stream("train", N_TR), w.stream("val", g["n_val"]), w.stream("test", g["n_test"])
    wa = World(wcfg.with_(anomaly_rate=g["anomaly_rate"]), seed)
    ex = wa.stream("extra", g["n_extra"])
    cfg = Config(seed=seed, **{**BASE, **overrides})
    st = NeuroStore(cfg)
    t0 = time.perf_counter(); st.ingest(tr); st.flush(); dt = time.perf_counter() - t0
    k, N = cfg.k, cfg.N
    out = {"seed": seed, "variant": name, "train_s": dt, "k": k}
    # ---- phase 2 task (assembly vote) and phase 3 task (temporal readout)
    if st.asm.n_alive == 0 or st.log.n_stored == 0:
        out.update({"f1": 0.0, "prec": 0.0, "n_assemblies": 0})
    else:
        m = phase2_metrics(st, tr, va, te, seed, g["mask"], g2, C); out.update({"f1": m["f1"], "prec": m["prec"], "n_assemblies": m["n_assemblies"]})
    mask_chance = (int(round(g["mask"] * k))) / (N - (k - int(round(g["mask"] * k))))
    out["prec_chance"] = mask_chance; out["prec_lift"] = (out["prec"] - mask_chance) / (1 - mask_chance)
    tf = temporal_recall(st, te, K=k) if st.lattice is not None and (st.lattice.csr.n_edges + st.lattice.n_delta) > 0 else float("nan")
    cnt = st.rates.total.astype(float)
    freq_top = np.argsort(-cnt, kind="stable")[:k]
    tcodes = [e[0] for e in encode_stream(st, te)]
    freq_rec = float(np.mean([len(set(freq_top) & set(c)) / k for c in tcodes[1:]]))
    tf_eff = freq_rec if np.isnan(tf) else tf              # a system without usable synapses can only guess frequent units
    out.update({"temporal_recall": tf_eff, "temporal_used_synapses": not np.isnan(tf), "temporal_freq_baseline": freq_rec,
                "temporal_lift": (tf_eff - k / N) / (1 - k / N)})
    # ---- economy
    out.update({"log_bytes": st.log.bytes_written, "log_bytes_per_event": st.log.bytes_written / N_TR, "n_stored": st.log.n_stored,
                "count_only_fraction": st.log.n_count_only / N_TR, "mean_surprise": float(np.mean(st.s_hist)),
                "delta_bytes": int(st.lattice.bytes_delta_written) if st.lattice is not None else 0})
    r = st.rates.rate(); out["rate_cv"] = float(r.std() / r.mean()); out["rate_max_over_target"] = float(r.max() / (k / N))
    out["overlap_margin"] = code_overlap_margin(st, te, seed=seed)
    # ---- surprise + rate read-outs on a continuing stream that contains anomalies (pure-noise events)
    rarity_w = -np.log(np.maximum(r, 1e-6))
    off = float(tr.t[-1]) + 1.0
    S, rar = [], []
    n0 = len(st.rates.closed)
    for i in range(len(ex)):
        info = st.write(float(ex.t[i]) + off, ex.event(i)); S.append(info["surprise"]); rar.append(rarity_w[info["units"]].mean())
    out["surprise_auc"] = auc(S, ex.anomaly); out["rarity_auc_baseline"] = auc(rar, ex.anomaly)
    rc = float("nan")
    if st.asm.n_alive and st.log.n_stored:
        codes = st.log.gather(np.arange(min(st.log.n_stored, 10 ** 9)))[0]
        # naming uses the stored TRAIN events only
        ntr = int(sum(1 for e in st.log.event_ids() if e < N_TR))
        eids = st.log.event_ids()[:ntr].astype(int)
        cmap = ConceptMap(st.asm, codes[:ntr], [tr.concepts[e] for e in eids], C)
        amap = {a: int(np.argmax(cmap.counts[a])) for a in np.flatnonzero(cmap.n > 0)}
        est_all, tru_all = [], []
        for widx, counts, nev in st.rates.closed[n0:]:
            est_a = rate_readout(st.asm, counts, nev)
            est = np.zeros(C)
            for a, c in amap.items():
                est[c] = max(est[c], est_a[a])
            tru = np.zeros(C)
            for i in range(len(ex)):
                if int((ex.t[i] + off) // cfg.rate_window) == widx:
                    for c in ex.concepts[i]:
                        tru[c] += 1
            est_all.append(est); tru_all.append(tru)
        if len(est_all) >= 3:
            e_, t_ = np.concatenate(est_all), np.concatenate(tru_all)
            rc = float(np.corrcoef(e_, t_)[0, 1]) if e_.std() > 0 and t_.std() > 0 else 0.0
    out["rate_readout_corr"] = rc
    return out


def main():
    results = {"uniform": {v: [] for v in VARIANTS}, "zipf": {v: [] for v in ("full", "no_homeostasis")}}
    worlds = {"uniform": WorldConfig(), "zipf": WorldConfig(zipf_s=g["zipf_s"])}
    for wn, wc in worlds.items():
        for s in seeds:
            for v in results[wn]:
                r = run(s, v, VARIANTS[v], wc); results[wn][v].append(r)
                print(wn, s, f"{v:18s} f1={r['f1']:.3f} prec={r['prec']:.3f} temp={r['temporal_recall']:.3f} bytes/ev={r['log_bytes_per_event']:.0f} "
                      f"cv={r['rate_cv']:.2f} sAUC={r['surprise_auc']:.2f} rateCorr={r['rate_readout_corr']:.2f} ovl={r['overlap_margin']:.2f} t={r['train_s']:.0f}s", flush=True)
    metrics = ["f1", "prec_lift", "temporal_lift", "log_bytes_per_event", "rate_cv", "overlap_margin", "surprise_auc", "rate_readout_corr", "mean_surprise", "n_assemblies"]
    table, deps = {}, {}
    for wn, vs in results.items():
        table[wn] = {v: {m: summarize([r[m] for r in rows]) for m in metrics + ["f1", "prec", "temporal_recall", "count_only_fraction", "delta_bytes", "train_s"]} for v, rows in vs.items()}
        deps[wn] = {}
        for v, rows in vs.items():
            if v == "full":
                continue
            deps[wn][v] = {}
            for m in metrics:
                a = np.array([r[m] for r in rows], float); b = np.array([x[m] for x in vs["full"]], float)
                d = a - b
                thr = g["threshold_rel"].get(m) if m in g["threshold_rel"] else None
                if thr is not None:
                    ok = np.abs(d.mean()) >= thr * max(abs(b.mean()), 1e-9)
                else:
                    ok = np.abs(d.mean()) >= g["threshold_abs"]
                same = max((d > 0).sum(), (d < 0).sum()) >= g["min_seed_agree"]
                deps[wn][v][m] = {"delta_mean": float(np.nanmean(d)), "delta_std": float(np.nanstd(d, ddof=1)) if len(d) > 1 else 0.0,
                                  "depends": bool(ok and same), "direction": "removal_hurts" if (d.mean() < 0) == (m not in g["lower_is_better"]) else "removal_helps"}
    write_result("phase6_quick" if QUICK else "phase6", {"seeds": seeds, "table": table, "dependence": deps, "variants": {k: v for k, v in VARIANTS.items()}, "base": BASE})
    print("done")


main()
