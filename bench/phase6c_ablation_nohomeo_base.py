"""POST-HOC (added after the pre-registered phase-6 run showed default-strength homeostasis hurts everything it
touches): repeat the ablation of the OTHER four principles with homeostasis OFF in the base system, so they are not
judged against a handicapped baseline. Same metrics, same dependence rule, uniform world, 5 seeds."""
import numpy as np
from sim.generator import WorldConfig
from eval.gates import load_gates, write_result
from eval.metrics import summarize
import bench.phase6_ablations as P

G = load_gates(); g = G["phase6"]; seeds = G["global"]["gate_seeds"]
H0 = dict(homeo=False)
VAR = {"base_no_homeostasis": {}, "no_sparsity": dict(P.VARIANTS["no_sparsity"]), "no_surprise_gate": dict(gate=False),
       "no_hebbian": dict(hebbian=False), "no_consolidation": dict(consolidate="online"), "no_learning_floor": dict(consolidate="off")}
metrics = ["f1", "prec_lift", "temporal_lift", "log_bytes_per_event", "overlap_margin", "surprise_auc", "rate_readout_corr", "mean_surprise", "n_assemblies"]
res = {v: [] for v in VAR}
for s in seeds:
    for v, ov in VAR.items():
        r = P.run(s, v, {**H0, **ov}, WorldConfig()); res[v].append(r)
        print(s, f"{v:22s} f1={r['f1']:.3f} prec_lift={r['prec_lift']:.3f} temp_lift={r['temporal_lift']:.3f} bytes/ev={r['log_bytes_per_event']:.0f} sAUC={r['surprise_auc']:.2f} t={r['train_s']:.0f}s", flush=True)
table = {v: {m: summarize([r[m] for r in rows]) for m in metrics + ["f1", "prec", "temporal_recall", "count_only_fraction", "train_s"]} for v, rows in res.items()}
dep = {}
for v, rows in res.items():
    if v == "base_no_homeostasis":
        continue
    dep[v] = {}
    for m in metrics:
        a = np.array([r[m] for r in rows], float); b = np.array([x[m] for x in res["base_no_homeostasis"]], float); d = a - b
        thr = g["threshold_rel"].get(m)
        big = abs(np.nanmean(d)) >= (thr * max(abs(np.nanmean(b)), 1e-9) if thr is not None else g["threshold_abs"])
        agree = max((d > 0).sum(), (d < 0).sum()) >= g["min_seed_agree"]
        dep[v][m] = {"delta_mean": float(np.nanmean(d)), "delta_std": float(np.nanstd(d, ddof=1)), "depends": bool(big and agree),
                     "direction": "removal_hurts" if (np.nanmean(d) < 0) == (m not in g["lower_is_better"]) else "removal_helps"}
write_result("phase6c_nohomeo_base", {"seeds": seeds, "table": table, "dependence": dep})
print("done")
