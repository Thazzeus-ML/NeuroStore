#!/usr/bin/env python3
"""POST-HOC CONTROL (exploratory, NOT pre-registered; it cannot change any gate verdict).

Question raised by the main results: VSA degrades less than the exact reference under every injected corruption. Is that a property of
hypervector coding, or of what VSA's encoder does to each feature? Its level codes make the Hamming distance grow linearly with the level
difference (an L1-like distance) on features that are clipped to [-3, 3] z and quantised to 32 levels. Controls, all brute-force exact stores
that are tuned with the same (k, beta) grid on the same CV queries:
  exact_l2_clip3   Euclidean on features clipped to [-3, 3] z
  exact_l2_q32     Euclidean on features clipped and quantised to the 32 levels VSA uses
  exact_l1         Manhattan distance on the plain features
  exact_l1_clip3   Manhattan distance on clipped features
Same calibrated severities (sweeps.json) and corruption seeds as the main sweeps; every control is paired against the stored per-query
errors of the main run (reference and VSA).

    python3 -I studies/pdm_cmapss/posthoc_quantized.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE)]
import numpy as np  # noqa: E402
import config as C  # noqa: E402
from nsharness import bench, corruption, tasks, tune  # noqa: E402
from nsharness.domains import cmapss  # noqa: E402
from nsharness.pipeline import Pipeline  # noqa: E402
from nsharness.stats import paired_group_bootstrap  # noqa: E402

T0 = time.time()
VARIANTS = {"exact_l2_clip3": ("exact_l2", {"kind": "stats", "standardize": True, "clip": 3.0}),
            "exact_l2_q32": ("exact_l2", {"kind": "stats", "standardize": True, "clip": 3.0, "levels": 32}),
            "exact_l1": ("exact_l1", {"kind": "stats", "standardize": True}),
            "exact_l1_clip3": ("exact_l1", {"kind": "stats", "standardize": True, "clip": 3.0})}


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "pdm_cmapss"))
    ap.add_argument("--data-root", default=str(ROOT / "data" / "cmapss"))
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    if a.fast:
        C.fast()
    R = Path(a.results)
    sw = json.loads((R / "sweeps.json").read_text())
    tun = json.loads((R / f"tuning_{C.PRIMARY}.json").read_text())
    dom = cmapss.build_domain(a.data_root, C.PRIMARY, C.WINDOW, C.CAP, seed=0, min_unique=C.MIN_UNIQUE)
    mem, q = dom.memory, dom.queries["test_last"]
    plan = tune.cv_plan(mem, C.CV_FOLDS, C.CV_PER_GROUP, C.CV_SEED, ("t", C.CV_MIN_T))
    qi_all = np.concatenate([qi for _, qi in plan])
    g_all, n = mem.groups[qi_all], len(qi_all)
    offs = np.cumsum([0] + [len(qi) for _, qi in plan])
    Rn = len(C.CORRUPTION_SEEDS)
    ref = tun["reference"]
    cache = R / "sweeps_cache"
    main_se = {v: np.load(cache / f"{v}.npz") for v in (ref, "vsa")}
    out = {"variants": {}}
    test_pred = {}
    for name, (bname, fcfg) in VARIANTS.items():
        rows = tune.tune(bname, [{}], C.READOUT_GRID, mem, plan, fcfg)
        sel = {"struct": {}, "k": rows[0]["k"], "beta": rows[0]["beta"], "cv_rmse": rows[0]["cv_rmse"]}
        log(f"{name}: CV RMSE {sel['cv_rmse']:.3f} k={sel['k']} beta={sel['beta']}")
        mk = lambda bname=bname, fcfg=fcfg, sel=sel: Pipeline(bname, {"beta": sel["beta"]}, fcfg, k=sel["k"])
        pred = mk().fit(mem, 0).predict(q.Z)
        test_pred[name] = pred
        ent = {"selected": sel, "test": tasks.metric_bundle(q.y, pred[None], q.groups, C.N_BOOT["table"], y_capped=q.meta["y_capped"]), "families": {}}
        clean = np.zeros((1, n))
        cond = {(f, ln): np.zeros((Rn, n)) for f in C.FAMILIES for ln in C.LEVELS}
        for f_i, (tr, qi) in enumerate(plan):
            pipe = mk().fit(mem.take(tr), 0)
            sl, yq, Zq = slice(offs[f_i], offs[f_i + 1]), mem.y[qi], mem.Z[qi]
            clean[0, sl] = (pipe.predict(Zq) - yq) ** 2
            for fam in C.FAMILIES:
                for ln in C.LEVELS:
                    for ci, cs in enumerate(C.CORRUPTION_SEEDS):
                        Zc, _ = corruption.corrupt(Zq, fam, sw["levels"][fam][ln]["severity"], cs * 1000 + f_i)
                        cond[(fam, ln)][ci, sl] = (pipe.predict(Zc) - yq) ** 2
        ent["clean_rmse"] = float(np.sqrt(clean.mean()))
        nb = C.N_BOOT["table"]
        for fam in C.FAMILIES:
            ent["families"][fam] = {}
            for ln in C.LEVELS:
                k = f"{fam}|{ln}"
                d = tasks.degradation(clean, cond[(fam, ln)], g_all, nb, seed=1)
                d["vs_ref"] = tasks.paired_degradation_diff(clean, cond[(fam, ln)], main_se[ref]["clean"], main_se[ref][k], g_all, nb, seed=1)
                d["vs_vsa"] = tasks.paired_degradation_diff(clean, cond[(fam, ln)], main_se["vsa"]["clean"], main_se["vsa"][k], g_all, nb, seed=1)
                ra, rb = np.sqrt(cond[(fam, ln)].mean(0)), np.sqrt(main_se["vsa"][k].mean(0))
                pt, lo, hi, _ = paired_group_bootstrap(lambda i: float(np.sqrt(cond[(fam, ln)][:, i].mean())), lambda i: float(np.sqrt(main_se["vsa"][k][:, i].mean())), g_all, n_boot=nb, seed=1)
                d["abs_rmse_minus_vsa"] = {"diff": pt, "lo": lo, "hi": hi}
                ent["families"][fam][ln] = d
        t0 = time.time()
        pp = mk().fit(mem, 0)
        Zp = np.concatenate([qq for qq in [mem.Z[qi_all[:256]]]])
        ent["latency_predict_ms_p50"] = bench.latency_profile(lambda z: pp.predict(z), Zp, 100, warmup=10)["p50_ms"]
        ent["deployed_bytes"] = pp.deployed_bytes()
        out["variants"][name] = ent
        log(f"  clean {ent['clean_rmse']:.2f}; L50 ratios: " + ", ".join(f"{f}={ent['families'][f]['L50']['ratio']:.2f}" for f in C.FAMILIES if 'L50' in ent['families'][f]))
    # absolute-RMSE reference rows from the main run (same queries, same corruptions)
    out["main"] = {v: {"clean_rmse": float(np.sqrt(main_se[v]["clean"].mean())),
                       "families": {f: {ln: float(np.sqrt(main_se[v][f"{f}|{ln}"].mean())) for ln in C.LEVELS} for f in C.FAMILIES}} for v in (ref, "vsa")}
    out["reference"] = ref
    out["meta"] = {"exploratory": True, "pre_registered": False, "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                   "env": bench.environment(), "seconds": round(time.time() - T0, 1), "n_queries": n}
    name = "posthoc_quantized.json"
    (R / name).write_text(json.dumps(out, indent=1))
    log(f"wrote {R / name}")


if __name__ == "__main__":
    main()
