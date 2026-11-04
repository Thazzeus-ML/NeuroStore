#!/usr/bin/env python3
"""Predictive-maintenance study on NASA C-MAPSS (see PROTOCOL.md). One stage per invocation:

    python3 -I studies/pdm_cmapss/run_study.py --stage validate|tune|tune_others|test|sweeps|completion|ood|timing|all

Results go to results/pdm_cmapss/*.json. --fast uses tiny grids on FAKE data and only writes under results/dev/ (smoke test of the code path).
BLAS is pinned to one thread (set before NumPy is imported) so latency numbers are single-core and parallel stage processes do not contend.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE)]
import numpy as np  # noqa: E402
import config as C  # noqa: E402
from nsharness import bench, corruption, ood, tasks, tune  # noqa: E402
from nsharness.backends import make_backend  # noqa: E402
from nsharness.baselines import ConstantBaseline, RidgeBaseline  # noqa: E402
from nsharness.core import Split  # noqa: E402
from nsharness.domains import cmapss, synthetic  # noqa: E402
from nsharness.pipeline import Pipeline  # noqa: E402
from nsharness.stats import analytic_flat_recall, group_bootstrap, paired_group_bootstrap, wilson_ci  # noqa: E402

VARIANTS = ("exact", "exact_l2", "hopfield", "ann", "ann_sq8", "flyhash", "vsa")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ io / provenance
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def meta(stage, extra=None):
    proto = HERE / "PROTOCOL.md"
    snap = {k: jsonable(getattr(C, k)) for k in dir(C) if k.isupper()}
    sums = (Path(ARGS.data_root) / "SHA256SUMS.json")
    return {"stage": stage, "git_head": git("rev-parse", "HEAD"), "dirty_code": bool(git("status", "--porcelain", "src", "studies")),
            "protocol_sha256": hashlib.sha256(proto.read_bytes()).hexdigest() if proto.exists() else None,
            "data_sums": json.loads(sums.read_text()) if sums.exists() else None, "fast": ARGS.fast, "config": snap,
            "env": bench.environment(), "seconds": round(time.time() - T0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **(extra or {})}


def save(name, obj):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(jsonable(obj), indent=1, allow_nan=True))
    log(f"wrote {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------------------------------------ shared helpers
def load_domain(fd):
    return cmapss.build_domain(ARGS.data_root, fd, C.WINDOW, C.CAP, seed=0, min_unique=C.MIN_UNIQUE)


def make_plan(mem):
    return tune.cv_plan(mem, C.CV_FOLDS, C.CV_PER_GROUP, C.CV_SEED, ("t", C.CV_MIN_T))


def seeds_for(name):
    return C.SEEDS if name in C.STOCHASTIC else (0,)


def make_pipe(name, sel, use_mask=False):
    cfg = dict(sel["struct"])
    cfg["beta"] = sel["beta"]
    return Pipeline(C.BACKEND_OF[name], cfg, C.FEAT, k=int(sel["k"]), use_mask=use_mask)


def select(name, rows, exact_ref=None):
    """rows sorted by CV RMSE. ann/ann_sq8: smallest nprobe within ANN_RULE_TOL of exact's best CV RMSE, else the overall best."""
    best = dict(rows[0])
    best["rule"] = "min_cv_rmse"
    if name in ("ann", "ann_sq8") and exact_ref is not None and len({r["struct"]["nprobe"] for r in rows}) > 1:
        per = {}
        for r in rows:
            p = r["struct"]["nprobe"]
            if p not in per or r["cv_rmse"] < per[p]["cv_rmse"]:
                per[p] = r
        pick = next((per[p] for p in sorted(per) if per[p]["cv_rmse"] <= C.ANN_RULE_TOL * exact_ref), None)
        best = dict(pick if pick is not None else rows[0])
        best["rule"] = f"smallest_nprobe_within_{C.ANN_RULE_TOL}x_of_exact" if pick is not None else "rule_not_met_min_cv_rmse"
    return best


def update_reference(res):
    """The better (lower CV RMSE) of cosine-exact and Euclidean-exact is THE exact reference; hopfield/ann reuse its similarity."""
    if all(b in res["backends"] for b in C.REFERENCE_CANDIDATES):
        res["reference"] = min(C.REFERENCE_CANDIDATES, key=lambda b: res["backends"][b]["selected"]["cv_rmse"])
        res["metric"] = "l2" if res["reference"] == "exact_l2" else "cosine"


def baseline_cv(plan, mem, lam):
    err = []
    for tr, qi in plan:
        m = RidgeBaseline(lam, C.FEAT).fit(mem.take(tr))
        err.append((m.predict(mem.Z[qi]) - mem.y[qi]) ** 2)
    return float(np.sqrt(np.concatenate(err).mean()))


# ------------------------------------------------------------------------------------------------ stage: validate (E6)
def stage_validate():
    log("validate: harness vs closed-form flat-search recall")
    rows = []
    for (D, N, p) in C.VALIDATE_CONFIGS:
        hits = tot = 0
        for seed in C.SEEDS:
            dom = synthetic.pm1_world(D, N, p, C.VALIDATE_QUERIES, seed)
            pipe = Pipeline("exact", {}, {"kind": "flatten", "standardize": False}, k=1).fit(dom.memory, 0)
            q = dom.queries["queries"]
            hits += int((pipe.neighbors(q.Z, k=1)[0][:, 0] == q.y).sum())
            tot += len(q)
        ana = analytic_flat_recall(D, N, p)
        lo, hi = wilson_ci(hits, tot, z=C.VALIDATE_Z)
        rows.append({"D": D, "N": N, "p_flip": p, "observed": hits / tot, "lo": lo, "hi": hi, "analytic": ana, "inside": bool(lo <= ana <= hi), "n": tot})
        log(f"  D={D} N={N} p={p}: observed {hits / tot:.4f} [{lo:.4f},{hi:.4f}] analytic {ana:.4f}")
    P = C.RECALL_PANEL
    panel = []
    cfgs = {"exact": {}, "hopfield": {"beta": 1000.0}, "ann": {"nprobe": 8}, "flyhash": {"m": 4096, "frac": 0.05}, "vsa": {"D": 4096}}
    for p in P["p"]:
        dom = synthetic.pm1_world(P["D"], P["N"], p, P["queries"], 0)
        q = dom.queries["queries"]
        for name, cfg in cfgs.items():
            pipe = Pipeline(name, cfg, {"kind": "flatten", "standardize": False}, k=1).fit(dom.memory, 0)
            hits = int((pipe.neighbors(q.Z, k=1)[0][:, 0] == q.y).sum())
            lo, hi = wilson_ci(hits, len(q))
            panel.append({"backend": name, "p_flip": p, "recall": hits / len(q), "lo": lo, "hi": hi, "n": len(q)})
        log(f"  panel p={p}: " + ", ".join(f"{r['backend']}={r['recall']:.3f}" for r in panel[-len(cfgs):]))
    save("validate.json", {"meta": meta("validate"), "analytic": rows, "panel": panel, "panel_cfg": P})


# ------------------------------------------------------------------------------------------------ stage: tune (E0)
def stage_tune(fd):
    log(f"tune {fd}")
    dom = load_domain(fd)
    mem = dom.memory
    plan = make_plan(mem)
    name_out = f"tuning_{fd}.json"
    res = load(name_out) if not ARGS.force else None
    res = res or {"fd": fd, "backends": {}}
    res["cv"] = {"folds": len(plan), "queries": int(sum(len(q) for _, q in plan)), "engines": int(len(np.unique(mem.groups))), "n_windows": len(mem)}
    prim = None if fd == C.PRIMARY else load(f"tuning_{C.PRIMARY}.json")
    if fd != C.PRIMARY and prim is None:
        raise SystemExit(f"run the {C.PRIMARY} tuning first")
    for name in VARIANTS:
        if name in res["backends"]:
            continue
        t = time.time()
        if name == "hopfield":
            sgrid, rgrid = [{}], [(1, b) for b in C.HOPFIELD_BETAS]
        else:
            sgrid, rgrid = C.STRUCT_GRID[name], C.READOUT_GRID
        if prim is not None and name not in ("exact", "exact_l2"):
            sgrid = [prim["backends"][name]["selected"]["struct"]]           # structure transferred from the primary sub-dataset; only (k, beta) is re-tuned
        elif name in ("hopfield", "ann", "ann_sq8"):
            sgrid = [{**g, "metric": res["metric"]} for g in sgrid]
        rows = tune.tune(C.BACKEND_OF[name], sgrid, rgrid, mem, plan, C.FEAT, seeds=seeds_for(name))
        ref = res["backends"][res["reference"]]["selected"]["cv_rmse"] if "reference" in res else None
        sel = select(name, rows, ref)
        res["backends"][name] = {"selected": sel, "rows": rows, "seeds": list(seeds_for(name)), "seconds": round(time.time() - t, 1),
                                 "edge": {"k_at_grid_edge": sel["k"] in (min(C.KS), max(C.KS)) if name != "hopfield" else None,
                                          "beta_at_grid_edge": (sel["beta"] in (min(C.HOPFIELD_BETAS), max(C.HOPFIELD_BETAS))) if name == "hopfield"
                                          else (sel["beta"] in (None, max(b for b in C.BETAS if b is not None)))}}
        log(f"  {name}: CV RMSE {sel['cv_rmse']:.3f} k={sel['k']} beta={sel['beta']} struct={sel['struct']} ({time.time() - t:.0f}s)")
        update_reference(res)
        res["meta"] = meta("tune")
        save(name_out, res)
    update_reference(res)
    log(f"  exact reference = {res['reference']} (CV RMSE exact {res['backends']['exact']['selected']['cv_rmse']:.3f}, exact_l2 {res['backends']['exact_l2']['selected']['cv_rmse']:.3f})")
    if "baselines" not in res:
        const = float(np.sqrt(np.concatenate([(mem.y[qi] - mem.y[tr].mean()) ** 2 for tr, qi in plan]).mean()))
        lams = {lam: baseline_cv(plan, mem, lam) for lam in C.RIDGE_LAMBDAS}
        best = min(lams, key=lams.get)
        res["baselines"] = {"const_cv_rmse": const, "ridge_cv_by_lambda": {str(k): v for k, v in lams.items()}, "ridge_lambda": best, "ridge_cv_rmse": lams[best]}
        log(f"  baselines: const {const:.2f}, ridge {lams[best]:.2f} (lambda {best})")
    res["meta"] = meta("tune")
    save(name_out, res)


# ------------------------------------------------------------------------------------------------ stage: test (E1)
def stage_test():
    out = {"meta": None, "subsets": {}}
    for fd in C.SUBSETS:
        log(f"test panel {fd}")
        tun = load(f"tuning_{fd}.json")
        dom = load_domain(fd)
        mem, q = dom.memory, dom.queries["test_last"]
        y_raw, y_cap, g = q.y, q.meta["y_capped"], q.groups
        entry = {"n_test": len(q), "n_train_windows": len(mem), "n_train_engines": dom.info["n_train_engines"], "channels": dom.info["channels"], "variants": {}}
        preds = {}
        for name in VARIANTS:
            sel = tun["backends"][name]["selected"]
            P = np.array([make_pipe(name, sel).fit(mem, s).predict(q.Z) for s in seeds_for(name)])
            preds[name] = P
            entry["variants"][name] = {"selected": {k: sel[k] for k in ("struct", "k", "beta", "cv_rmse")}, "seeds": list(seeds_for(name)),
                                       **tasks.metric_bundle(y_raw, P, g, C.N_BOOT["table"], y_capped=y_cap)}
            log(f"  {name}: RMSE {entry['variants'][name]['rmse']:.2f} (capped {entry['variants'][name]['rmse_capped']:.2f})")
        base = tun["baselines"]
        preds["const"] = np.array([ConstantBaseline().fit(mem).predict(q.Z)])
        preds["ridge"] = np.array([RidgeBaseline(base["ridge_lambda"], C.FEAT).fit(mem).predict(q.Z)])
        for name in ("const", "ridge"):
            entry["variants"][name] = {"seeds": [0], **tasks.metric_bundle(y_raw, preds[name], g, C.N_BOOT["table"], y_capped=y_cap)}
        ref = tun["reference"]
        entry["reference"] = ref
        for name in preds:
            if name != ref:
                entry["variants"][name]["ratio_vs_ref"] = tasks.paired_rmse_ratio(y_raw, preds[name], preds[ref], g, C.N_BOOT["paired"])
        entry["ref_vs_const"] = tasks.paired_rmse_ratio(y_raw, preds[ref], preds["const"], g, C.N_BOOT["paired"])
        entry["ref_vs_ridge"] = tasks.paired_rmse_ratio(y_raw, preds[ref], preds["ridge"], g, C.N_BOOT["paired"])
        entry["y_raw"], entry["pred_ref"] = y_raw, preds[ref][0].round(2)
        out["subsets"][fd] = entry
    out["meta"] = meta("test")
    save("test_panel.json", out)


# ------------------------------------------------------------------------------------------------ stage: sweeps (E2)
SWEEP_VARIANTS = [("exact", "exact", False), ("exact_mask", "exact", True), ("exact_l2", "exact_l2", False), ("exact_l2_mask", "exact_l2", True),
                  ("hopfield", "hopfield", False), ("hopfield_mask", "hopfield", True),
                  ("ann", "ann", False), ("ann_sq8", "ann_sq8", False), ("flyhash", "flyhash", False), ("vsa", "vsa", False), ("vsa_mask", "vsa", True),
                  ("ridge", "ridge", False)]


def stage_sweeps():
    tun = load(f"tuning_{C.PRIMARY}.json")
    dom = load_domain(C.PRIMARY)
    mem = dom.memory
    plan = make_plan(mem)
    qi_all = np.concatenate([qi for _, qi in plan])
    g_all, n = mem.groups[qi_all], len(qi_all)
    offs = np.cumsum([0] + [len(qi) for _, qi in plan])
    log(f"sweeps: {n} out-of-fold queries from {len(np.unique(g_all))} engines")
    # ---- calibrate severities on the exact REFERENCE (unmasked) so that every family hurts by the same multiple
    ref = tun["reference"]
    sel = tun["backends"][ref]["selected"]
    pipes = [make_pipe(ref, sel).fit(mem.take(tr), 0) for tr, _ in plan]

    def rmse_at(fam, sev, seed):
        e = []
        for f, (tr, qi) in enumerate(plan):
            Zc, _ = corruption.corrupt(mem.Z[qi], fam, sev, seed * 1000 + f)
            e.append((pipes[f].predict(Zc) - mem.y[qi]) ** 2)
        return float(np.sqrt(np.concatenate(e).mean()))

    levels = {}
    for fam in C.FAMILIES:
        memo = {}

        def measure(s, fam=fam, memo=memo):
            key = round(float(s), 9)
            if key not in memo:
                memo[key] = rmse_at(fam, key, C.CALIB_SEED)
            return memo[key]

        levels[fam] = {}
        for ln, target in C.LEVELS.items():
            s, r, ok = corruption.calibrate_severity(measure, target)
            levels[fam][ln] = {"severity": s, "ratio": r, "reached": bool(ok), "target": target,
                               "meaning": corruption.describe_severity(fam, s, len(dom.info["channels"]))}
        log(f"  calibrated {fam}: " + ", ".join(f"{ln}: s={v['severity']:.3f} (x{v['ratio']:.2f}{'' if v['reached'] else ' NOT REACHED'})" for ln, v in levels[fam].items()))
    # ---- evaluate every variant on identical queries and identical corruptions
    R = len(C.CORRUPTION_SEEDS)
    se_clean, se_cond = {}, {}
    cache_dir = OUT / "sweeps_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for vname, bname, use_mask in SWEEP_VARIANTS:
        cache = cache_dir / f"{vname}.npz"
        if cache.exists() and not ARGS.force:
            z = np.load(cache)
            se_clean[vname] = z["clean"]
            se_cond[vname] = {tuple(k.split("|")): z[k] for k in z.files if k != "clean"}
            log(f"  {vname}: loaded from cache")
            continue
        t = time.time()
        seeds = seeds_for(bname)
        fams = [f for f in C.FAMILIES if (not use_mask or f in C.MASK_FAMILIES)]
        clean = np.zeros((len(seeds), n))
        cond = {(f, ln): np.zeros((len(seeds) * R, n)) for f in fams for ln in C.LEVELS}
        for si, s in enumerate(seeds):
            for f, (tr, qi) in enumerate(plan):
                if bname == "ridge":
                    model = RidgeBaseline(tun["baselines"]["ridge_lambda"], C.FEAT).fit(mem.take(tr))
                    pred = lambda Z, obs, model=model: model.predict(Z)
                else:
                    model = make_pipe(bname, tun["backends"][bname]["selected"], use_mask).fit(mem.take(tr), s)
                    pred = lambda Z, obs, model=model: model.predict(Z, obs if use_mask else None)
                sl, yq, Zq = slice(offs[f], offs[f + 1]), mem.y[qi], mem.Z[qi]
                clean[si, sl] = (pred(Zq, None) - yq) ** 2
                for fam in fams:
                    for ln in C.LEVELS:
                        for ci, cs in enumerate(C.CORRUPTION_SEEDS):
                            Zc, obs = corruption.corrupt(Zq, fam, levels[fam][ln]["severity"], cs * 1000 + f)
                            cond[(fam, ln)][si * R + ci, sl] = (pred(Zc, obs) - yq) ** 2
        se_clean[vname], se_cond[vname] = clean, cond
        np.savez_compressed(cache, clean=clean, **{f"{a}|{b}": v for (a, b), v in cond.items()})
        log(f"  {vname}: clean RMSE {np.sqrt(clean.mean()):.2f} ({time.time() - t:.0f}s)")
    # ---- tables
    nb = C.N_BOOT["table"]
    table = {}
    for vname, bname, use_mask in SWEEP_VARIANTS:
        ent = {"clean_rmse": float(np.sqrt(se_clean[vname].mean())), "families": {}}
        for (fam, ln), arr in se_cond[vname].items():
            d = tasks.degradation(se_clean[vname], arr, g_all, nb, seed=1)
            if vname != ref:
                d["vs_ref"] = tasks.paired_degradation_diff(se_clean[vname], arr, se_clean[ref], se_cond[ref][(fam, ln)], g_all, nb, seed=1)
            if vname.endswith("_mask"):
                base = vname[:-5]
                d["vs_unmasked"] = tasks.paired_degradation_diff(se_clean[vname], arr, se_clean[base], se_cond[base][(fam, ln)], g_all, nb, seed=1)
            ent["families"].setdefault(fam, {})[ln] = d
        table[vname] = ent
    save("sweeps.json", {"meta": meta("sweeps"), "reference": ref, "n_queries": n, "n_engines": int(len(np.unique(g_all))), "levels": levels, "variants": table})


# ------------------------------------------------------------------------------------------------ stage: completion (E3)
def stage_completion():
    tun = load(f"tuning_{C.PRIMARY}.json")
    dom = load_domain(C.PRIMARY)
    mem = dom.memory
    plan = make_plan(mem)
    qi_all = np.concatenate([qi for _, qi in plan])
    g_all, n = mem.groups[qi_all], len(qi_all)
    offs = np.cumsum([0] + [len(qi) for _, qi in plan])
    R = len(C.CORRUPTION_SEEDS)
    ref = tun["reference"]
    variants = [("exact", "exact", False), ("exact_mask", "exact", True), ("exact_l2", "exact_l2", False), ("exact_l2_mask", "exact_l2", True),
                ("hopfield", "hopfield", False), ("hopfield_mask", "hopfield", True), ("vsa", "vsa", False), ("vsa_mask", "vsa", True)]
    per_q = {}
    imp = {sev: np.zeros((R, n)) for sev in C.COMPLETION_SEVERITIES}
    for vname, bname, use_mask in variants:
        t = time.time()
        seeds = seeds_for(bname)
        arr = {sev: np.zeros((len(seeds) * R, n)) for sev in C.COMPLETION_SEVERITIES}
        for si, s in enumerate(seeds):
            for f, (tr, qi) in enumerate(plan):
                pipe = make_pipe(bname, tun["backends"][bname]["selected"], use_mask).fit(mem.take(tr), s)
                sl = slice(offs[f], offs[f + 1])
                for sev in C.COMPLETION_SEVERITIES:
                    for ci, cs in enumerate(C.CORRUPTION_SEEDS):
                        arr[sev][si * R + ci, sl] = tasks.completion_eval(pipe, mem.Z[qi], sev, cs * 1000 + f, give_mask=use_mask)["per_query"]
                        if vname == ref and si == 0:
                            imp[sev][ci, sl] = tasks.mean_imputation_nmse(pipe, mem.Z[qi], sev, cs * 1000 + f)
        per_q[vname] = {sev: np.nanmean(a, axis=0) for sev, a in arr.items()}
        log(f"  {vname}: NMSE@{C.COMPLETION_SEVERITIES[0]}={np.nanmean(per_q[vname][C.COMPLETION_SEVERITIES[0]]):.3f} ({time.time() - t:.0f}s)")
    per_q["mean_imputation"] = {sev: np.nanmean(a, axis=0) for sev, a in imp.items()}
    nb = C.N_BOOT["table"]
    res = {}
    for vname, d in per_q.items():
        res[vname] = {}
        for sev, q in d.items():
            pt, lo, hi, _ = group_bootstrap(lambda i, q=q: float(np.nanmean(q[i])), g_all, n_boot=nb, seed=2)
            ent = {"nmse": pt, "lo": lo, "hi": hi}
            if vname not in (ref, "mean_imputation"):
                qe = per_q[ref][sev]
                dp, dlo, dhi, _ = paired_group_bootstrap(lambda i, q=q: float(np.nanmean(q[i])), lambda i, qe=qe: float(np.nanmean(qe[i])), g_all, n_boot=nb, seed=2)
                ent["vs_ref"] = {"diff": dp, "lo": dlo, "hi": dhi, "excludes_zero": bool(dlo > 0 or dhi < 0)}
            res[vname][str(sev)] = ent
    chan = len(dom.info["channels"])
    save("completion.json", {"meta": meta("completion"), "reference": ref, "n_queries": n, "channels": chan,
                             "missing_channels": {str(s): int(np.ceil(s * chan - 1e-9)) for s in C.COMPLETION_SEVERITIES}, "variants": res})


# ------------------------------------------------------------------------------------------------ stage: ood (E4)
def stage_ood():
    tun = load(f"tuning_{C.PRIMARY}.json")
    dom = load_domain(C.PRIMARY)
    mem = dom.memory
    plan = make_plan(mem)
    prep = dom.info["prep"]
    foreign = {fd: cmapss.foreign_split(ARGS.data_root, fd, prep, C.WINDOW, C.CAP, stride=C.OOD_STRIDE) for fd in C.FOREIGN}
    for fd, s in foreign.items():
        log(f"  foreign {fd}: {len(s)} windows from {len(np.unique(s.groups))} engines")
    names = ["exact", "exact_l2", "hopfield", "ann", "flyhash", "vsa"]
    skipped = defaultdict(int)
    acc = {v: {"id": defaultdict(list), "idg": [], "idlate": [], "sets": defaultdict(lambda: {"s": defaultdict(list), "g": [], "ref": "all"})} for v in names}
    for f, (tr, _) in enumerate(plan):
        held = np.setdiff1d(np.unique(mem.groups), np.unique(mem.groups[tr]))
        ID = mem.take(np.where(np.isin(mem.groups, held) & (mem.meta["t"] % C.OOD_STRIDE == 0))[0])
        rng = np.random.default_rng(1000 + f)
        sets, refs = {}, {}
        for fam in ood.SYNTHETIC_FAMILIES:
            sets[fam] = Split(ood.synth_ood(ID.Z, fam, 1000 + f), ID.y, ID.groups)
        for fd in C.FOREIGN:
            sub = foreign[fd]
            sets[f"foreign_{fd}"] = sub.take(np.sort(rng.choice(len(sub), min(C.OOD_MAX_FOREIGN, len(sub)), replace=False)))
        late = np.where(foreign["FD003"].meta["rul_raw"] <= C.OOD_LATE_RUL)[0]
        sets["foreign_FD003_late"] = foreign["FD003"].take(np.sort(rng.choice(late, min(C.OOD_MAX_FOREIGN, len(late)), replace=False)))
        refs["foreign_FD003_late"] = "late"
        for kind in C.CHIMERA_KINDS:
            for band, (lo, hi) in C.CHIMERA_BANDS.items():
                sets[f"chimera_{kind}_{band}"] = ood.make_chimeras(ID, rng, C.CHIMERA_N, lo, hi, kind, space="value")
        id_late = ID.meta["rul_raw"] <= C.OOD_LATE_RUL
        empty = [k for k, sp in sets.items() if len(sp) == 0]
        for k in empty:
            log(f"  fold {f}: OOD set {k} is EMPTY (no pairs in the requested band) and is skipped")
            sets.pop(k)
            skipped[k] += 1
        log(f"  fold {f}: ID {len(ID)} windows ({len(held)} engines), {len(sets)} OOD sets")
        for v in names:
            for s in seeds_for(v):
                pipe = make_pipe(v, tun["backends"][v]["selected"]).fit(mem.take(tr), s)
                fid = pipe.familiarity(ID.Z)
                for k, a in fid.items():
                    acc[v]["id"][k].append(a)
                acc[v]["idg"].append(ID.groups)
                acc[v]["idlate"].append(id_late)
                for sname, sp in sets.items():
                    fo = pipe.familiarity(sp.Z)
                    for k, a in fo.items():
                        acc[v]["sets"][sname]["s"][k].append(a)
                    acc[v]["sets"][sname]["g"].append(sp.groups)
                    acc[v]["sets"][sname]["ref"] = refs.get(sname, "all")
    res = {}
    for v in names:
        a = acc[v]
        idg = np.concatenate(a["idg"])
        idl = np.concatenate(a["idlate"])
        idsc = {k: np.concatenate(x) for k, x in a["id"].items()}
        res[v] = {}
        for sname, d in a["sets"].items():
            osc = {k: np.concatenate(x) for k, x in d["s"].items()}
            og = np.concatenate(d["g"])
            if d["ref"] == "late":
                ids, ig = {k: x[idl] for k, x in idsc.items()}, idg[idl]
            else:
                ids, ig = idsc, idg
            res[v][sname] = ood.familiarity_auroc(ids, ig, osc, og, n_boot=C.N_BOOT["auroc"], seed=3)
        log(f"  {v}: gauss {res[v]['gauss']['max_sim']['auroc']:.3f}, FD003_late {res[v]['foreign_FD003_late']['max_sim']['auroc']:.3f}, "
            f"chimera_channel same/far {res[v]['chimera_channel_same']['max_sim']['auroc']:.3f}/{res[v]['chimera_channel_far']['max_sim']['auroc']:.3f}")
    save("ood.json", {"meta": meta("ood"), "reference": tun["reference"], "foreign_sizes": {fd: len(s) for fd, s in foreign.items()},
                       "empty_sets_skipped_per_set": dict(skipped), "variants": res})


# ------------------------------------------------------------------------------------------------ stage: timing (E5)
def stage_timing():
    tun = load(f"tuning_{C.PRIMARY}.json")
    dom = load_domain(C.PRIMARY)
    mem = dom.memory
    plan = make_plan(mem)
    Zpool = mem.Z[np.concatenate([qi for _, qi in plan])]
    raw_w, raw_f = mem.Z.nbytes, len(mem) * 4 * len(dom.info["channels"]) * 4
    rows = {}
    for name in VARIANTS:
        sel = tun["backends"][name]["selected"]
        pipe, prof = bench.build_profile(lambda: make_pipe(name, sel).fit(mem, 0))
        ent = {"selected": {k: sel[k] for k in ("struct", "k", "beta")}, "build": prof, "memory": bench.memory_table(pipe, len(mem), raw_w, raw_f),
               "latency_predict": bench.latency_profile(lambda q: pipe.predict(q), Zpool, C.LATENCY_SAMPLES),
               "latency_search10": bench.latency_profile(lambda q: pipe.neighbors(q, k=10), Zpool, C.LATENCY_SAMPLES),
               "throughput_predict": bench.throughput(lambda q: pipe.predict(q), Zpool, C.THROUGHPUT_BATCH, C.THROUGHPUT_REPEATS, C.THROUGHPUT_SECONDS),
               "search_peak_bytes_batch": bench.search_peak_bytes(lambda q: pipe.predict(q), Zpool[:C.THROUGHPUT_BATCH])}
        if name in ("ann", "ann_sq8"):
            ent["scanned_per_query"] = float(pipe.be.last_scanned)
        rows[name] = ent
        log(f"  {name}: build {prof['seconds']:.2f}s, {ent['memory']['deployed_bytes'] / 1e6:.2f} MB, p50 {ent['latency_predict']['p50_ms']:.2f} ms, {ent['throughput_predict']['qps_median']:.0f} q/s")
    # ---- scale-up on clustered synthetic vectors (d matches the PdM feature size)
    scale = {}
    for vname, sizes in C.SCALE_SIZES.items():
        scale[vname] = []
        for n in sizes:
            X, Q = synthetic.blob_world(n, C.SCALE_D, max(20, n // 200), 0.5, 300, 7)
            ex = make_backend("exact").fit(X)
            ie, _ = ex.search(Q[:200], 10)
            sel = tun["backends"][vname]["selected"] if vname in tun["backends"] else {"struct": {}}
            row = {"n": n}
            if vname in ("ann", "ann_sq8"):
                for npb in C.SCALE_NPROBES:
                    be = make_backend("ann", sq8=(vname == "ann_sq8"), nprobe=npb).fit(X, 0)
                    rec = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(be.search(Q[:200], 10)[0].tolist(), ie.tolist())]))
                    if rec >= C.SCALE_RECALL_TARGET or npb == C.SCALE_NPROBES[-1]:
                        break
                row.update({"nprobe": npb, "recall10": rec, "target_met": bool(rec >= C.SCALE_RECALL_TARGET)})
            elif vname == "exact":
                be, rec = ex, 1.0
                row["recall10"] = 1.0
            else:
                cfg = {k: v for k, v in sel["struct"].items()}
                be = make_backend(vname, **cfg).fit(X, 0)
                row["recall10"] = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(be.search(Q[:200], 10)[0].tolist(), ie.tolist())]))
                row["top1_agree"] = float(np.mean(be.search(Q[:200], 1)[0][:, 0] == ie[:, 0]))
            row["latency"] = bench.latency_profile(lambda q: be.search(q, 10), Q, C.SCALE_LATENCY_SAMPLES)
            row["throughput"] = bench.throughput(lambda q: be.search(q, 10), Q, C.THROUGHPUT_BATCH, repeats=2, min_seconds=min(0.3, C.THROUGHPUT_SECONDS))
            row["index_bytes"] = be.index_bytes()
            scale[vname].append(row)
            log(f"  scale {vname} n={n}: p50 {row['latency']['p50_ms']:.2f} ms, recall@10 {row['recall10']:.3f}, {row['index_bytes'] / 1e6:.1f} MB")
            del be, ex
    save("timing.json", {"meta": meta("timing"), "n_windows": len(mem), "d_features": 4 * len(dom.info["channels"]), "variants": rows, "scale": scale})


# ------------------------------------------------------------------------------------------------ main
def main():
    global ARGS, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--fd", default=None)
    ap.add_argument("--data-root", default=str(ROOT / "data" / "cmapss"))
    ap.add_argument("--out", default=str(ROOT / "results" / "pdm_cmapss"))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--force", action="store_true")
    ARGS = ap.parse_args()
    OUT = Path(ARGS.out)
    if ARGS.fast:
        C.fast()
        if "results/dev" not in str(OUT.resolve()).replace("\\", "/"):
            raise SystemExit("--fast writes only under results/dev/ (it must never produce reported results)")
    stages = ["validate", "tune", "tune_others", "test", "sweeps", "completion", "ood", "timing"] if ARGS.stage == "all" else [ARGS.stage]
    for st in stages:
        log(f"=== stage {st}")
        if st == "validate":
            stage_validate()
        elif st == "tune":
            stage_tune(ARGS.fd or C.PRIMARY)
        elif st == "tune_others":
            for fd in ([ARGS.fd] if ARGS.fd else [f for f in C.SUBSETS if f != C.PRIMARY]):
                stage_tune(fd)
        elif st == "test":
            stage_test()
        elif st == "sweeps":
            stage_sweeps()
        elif st == "completion":
            stage_completion()
        elif st == "ood":
            stage_ood()
        elif st == "timing":
            stage_timing()
        else:
            raise SystemExit(f"unknown stage {st}")


if __name__ == "__main__":
    main()
