"""Gate 4: surprise gate bytes vs phase-2 recall, and the homeostasis test."""
import sys, time, json
import numpy as np
from sim.generator import World, WorldConfig
from neurostore.config import Config
from neurostore.store import NeuroStore
from neurostore.encoder import SparseEncoder
from neurostore.homeostasis import Homeostasis
from eval.p2task import phase2_metrics, temporal_recall
from eval.gates import load_gates, write_result, ROOT
from eval.metrics import summarize

G = load_gates(); g = G["phase4"]; g2 = G["phase2"]; gh = g["homeostasis_test"]
DEV = len(sys.argv) > 1 and sys.argv[1] == "dev"
seeds = G["global"]["dev_seeds"][:2] if DEV else G["global"]["gate_seeds"]
C = WorldConfig().C
PRED_MODE = "assembly"   # attempt 1 used "lattice" (results/phase4_attempt1.json)


def system(seed, tr, va, te, gate, tau=0.35):
    cfg = Config(seed=seed, homeo=False, gate=gate, gate_tau=tau, pred_mode=PRED_MODE, strength_floor=g["strength_floor"], hot_tau=g["hot_tau"],
                 replay_budget=g["replay_budget"], consolidate_every=200)
    st = NeuroStore(cfg)
    t0 = time.perf_counter(); st.ingest(tr); st.flush(); dt = time.perf_counter() - t0
    m = phase2_metrics(st, tr, va, te, seed, g["mask"], g2, C)
    m.update({"bytes": st.log.bytes_written, "n_stored": st.log.n_stored, "n_count_only": st.log.n_count_only,
              "delta_bytes": st.lattice.bytes_delta_written, "cache_hit_rate": st.pred.hits / max(st.pred.hits + st.pred.misses, 1),
              "temporal_recall": temporal_recall(st, te), "train_s": dt, "tau": tau if gate else None,
              "mean_surprise": float(np.mean(st.s_hist))})
    return m


def homeo_test(seed):
    w = World(WorldConfig(zipf_s=gh["zipf_s"]), seed); s = w.stream("test", gh["n_events"])
    out = {}
    for on in (True, False):
        enc = SparseEncoder(800, 1024, 32, 2, seed); h = Homeostasis(enc); cnt = np.zeros(1024); ks = set()
        for i in range(gh["n_events"]):
            u, _ = enc.encode(s.event(i)); ks.add(len(u))
            if on: h.update(u)
            if i >= gh["n_events"] - gh["last_window"]: cnt[u] += 1
        r = cnt / gh["last_window"]
        out["on" if on else "off"] = {"cv": float(r.std() / r.mean()), "max_over_target": float(r.max() / (32 / 1024)),
                                      "mean_abs_dev": float(np.abs(r - 32 / 1024).mean() / (32 / 1024)), "k_values": sorted(ks)}
    out["cv_ratio"] = out["on"]["cv"] / out["off"]["cv"]
    out["pass"] = bool(out["cv_ratio"] <= gh["cv_ratio_max"] and out["on"]["k_values"] == [32])
    return out


per = []
for seed in seeds:
    w = World(WorldConfig(), seed)
    tr, va, te = w.stream("train", g["n_train"]), w.stream("val", g["n_val"]), w.stream("test", g["n_test"])
    ung = system(seed, tr, va, te, False)
    taus = {str(t): system(seed, tr, va, te, True, t) for t in g["tau_grid"]}
    ok = [t for t in g["tau_grid"] if ung["val_f1"] - taus[str(t)]["val_f1"] <= g["recall_tolerance_abs"]
          and ung["val_prec"] - taus[str(t)]["val_prec"] <= g["recall_tolerance_abs"]]
    tau = max(ok) if ok else min(g["tau_grid"])
    ch = taus[str(tau)]
    row = {"seed": seed, "ungated": ung, "taus": taus, "chosen_tau": tau,
           "bytes_reduction": 1 - ch["bytes"] / ung["bytes"], "f1_drop": ung["f1"] - ch["f1"], "prec_drop": ung["prec"] - ch["prec"],
           "homeostasis": homeo_test(seed)}
    per.append(row)
    print(seed, f"tau={tau} bytes_red={row['bytes_reduction']:.3f} f1 {ung['f1']:.3f}->{ch['f1']:.3f} prec {ung['prec']:.3f}->{ch['prec']:.3f} "
          f"count_only={ch['n_count_only']} cache_hit={ch['cache_hit_rate']:.2f} homeoCVratio={row['homeostasis']['cv_ratio']:.2f}", flush=True)
br = np.array([r["bytes_reduction"] for r in per]); fd = np.array([r["f1_drop"] for r in per]); pd_ = np.array([r["prec_drop"] for r in per])
c_bytes = bool(br.mean() >= g["bytes_reduction_min"]); c_f1 = bool(fd.mean() <= g["recall_tolerance_abs"]); c_p = bool(pd_.mean() <= g["recall_tolerance_abs"])
curve = {}
for t in g["tau_grid"]:
    k = str(t)
    curve[k] = {"bytes_reduction": summarize([1 - r["taus"][k]["bytes"] / r["ungated"]["bytes"] for r in per]),
                "f1_drop": summarize([r["ungated"]["f1"] - r["taus"][k]["f1"] for r in per]),
                "prec_drop": summarize([r["ungated"]["prec"] - r["taus"][k]["prec"] for r in per]),
                "temporal_recall": summarize([r["taus"][k]["temporal_recall"] for r in per]),
                "count_only_fraction": summarize([r["taus"][k]["n_count_only"] / g["n_train"] for r in per])}
out = {"gate": "PASS" if (c_bytes and c_f1 and c_p) else "FAIL", "conditions": {"bytes": c_bytes, "f1": c_f1, "prec": c_p},
       "seeds": seeds, "bytes_reduction": summarize(br), "f1_drop": summarize(fd), "prec_drop": summarize(pd_),
       "ungated_f1": summarize([r["ungated"]["f1"] for r in per]), "ungated_prec": summarize([r["ungated"]["prec"] for r in per]),
       "ungated_temporal": summarize([r["ungated"]["temporal_recall"] for r in per]),
       "chosen_tau": [r["chosen_tau"] for r in per], "curve": curve,
       "homeostasis_test": {"pass_all": all(r["homeostasis"]["pass"] for r in per), "cv_ratio": summarize([r["homeostasis"]["cv_ratio"] for r in per]),
                            "cv_on": summarize([r["homeostasis"]["on"]["cv"] for r in per]), "cv_off": summarize([r["homeostasis"]["off"]["cv"] for r in per]),
                            "max_over_target_on": summarize([r["homeostasis"]["on"]["max_over_target"] for r in per]),
                            "max_over_target_off": summarize([r["homeostasis"]["off"]["max_over_target"] for r in per])},
       "per_seed": per}
if DEV:
    (ROOT / "results" / "dev").mkdir(parents=True, exist_ok=True); (ROOT / "results" / "dev" / "phase4.json").write_text(json.dumps(out, indent=1, default=float))
else:
    write_result("phase4", out)
print("GATE 4:", out["gate"], out["conditions"], f"bytes_red={br.mean():.3f} f1_drop={fd.mean():.4f} prec_drop={pd_.mean():.4f}", "(dev)" if DEV else "")
for k, v in curve.items():
    print(" tau", k, f"bytes_red={v['bytes_reduction']['mean']:.3f} f1_drop={v['f1_drop']['mean']:.4f} prec_drop={v['prec_drop']['mean']:.4f} temporal={v['temporal_recall']['mean']:.4f} count_only={v['count_only_fraction']['mean']:.2f}")
