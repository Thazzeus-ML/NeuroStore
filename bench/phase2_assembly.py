"""Gate 2: partial-cue recovery (assembly vote) vs exact-match and brute-force Jaccard kNN."""
import sys, time, itertools, json
from pathlib import Path
import numpy as np
from sim.generator import World, WorldConfig
from neurostore.config import Config
from neurostore.store import NeuroStore
from neurostore.readout import select_assemblies, complete_pattern
from eval.baselines import ExactMatch, JaccardNN
from eval.protocol import mask_cues, ConceptMap, ns_decode, f1_mean, prec_mean
from eval.gates import load_gates, write_result, ROOT
from eval.metrics import summarize

G = load_gates(); g = G["phase2"]
DEV = len(sys.argv) > 1 and sys.argv[1] == "dev"
seeds = G["global"]["dev_seeds"] if DEV else G["global"]["gate_seeds"]


def nn_decode(idx, sims, concepts, k, thr):
    out = []
    for ii, ss in zip(idx, sims):
        if k == 1:
            out.append(concepts[ii[0]]); continue
        w = {}
        for j, s in zip(ii[:k], ss[:k]):
            for c in concepts[j]:
                w[c] = w.get(c, 0.0) + s
        tot = float(ss[:k].sum()) or 1.0
        pred = tuple(sorted(c for c, v in w.items() if v / tot >= thr))
        out.append(pred if pred else concepts[ii[0]])
    return out


def nn_complete(idx, sims, codes, cues, ms, k, N):
    out = []
    for ii, ss, cue, m in zip(idx, sims, cues, ms):
        sc = np.zeros(N)
        for j, s in zip(ii[:k], ss[:k]):
            sc[codes[j]] += s
        sc[cue] = 0
        nz = int((sc > 0).sum()); mm = min(m, nz)
        out.append(np.argsort(-sc, kind="stable")[:mm] if mm > 0 else np.zeros(0, np.int64))
    return out


def run(seed):
    wc = WorldConfig()
    w = World(wc, seed)
    tr, va, te = w.stream("train", g["n_train"]), w.stream("val", g["n_val"]), w.stream("test", g["n_test"])
    cfg = Config(seed=seed, homeo=False, gate=False, hebbian=False, consolidate_every=200, replay_budget=200, hot_tau=0.0)
    store = NeuroStore(cfg)
    t0 = time.perf_counter(); store.ingest(tr); store.flush(); t_build = time.perf_counter() - t0
    asm = store.asm
    pos = np.arange(store.log.n_stored)
    codes = store.log.gather(pos)[0]
    cmap = ConceptMap(asm, codes, tr.concepts, wc.C)
    enc = lambda s: [store.enc.encode(s.event(i))[0] for i in range(len(s))]
    vc, tc = enc(va), enc(te)
    nn = JaccardNN().fit(codes, cfg.N); ex = ExactMatch().fit(codes, tr.concepts, cfg.N)
    res = {"seed": seed, "n_assemblies": asm.n_alive, "purity": cmap.purity(), "concepts_covered": cmap.concepts_covered(),
           "build_s": t_build, "masks": {}}
    for mf in [0.0] + list(g["masks"]):
        vcues, vmasked = mask_cues(vc, mf, seed, 1); tcues, tmasked = mask_cues(tc, mf, seed, 2)
        ms_v = [len(m) for m in vmasked]; ms_t = [len(m) for m in tmasked]
        out = {}
        # ---- NeuroStore: tune on val, apply to test
        vcos = [asm.vote(c) for c in vcues]; tcos = [asm.vote(c) for c in tcues]
        best = (-1, None)
        for rho, cm, thr in itertools.product(g["grid_ns_rho"], g["grid_ns_cos_min"], g["grid_ns_map_thr"]):
            f = f1_mean([ns_decode(c, cmap, rho, cm, thr) for c in vcos], va.concepts)
            if f > best[0]:
                best = (f, (rho, cm, thr))
        rho, cm, thr = best[1]
        ns_f1 = f1_mean([ns_decode(c, cmap, rho, cm, thr) for c in tcos], te.concepts)
        if mf > 0:
            bp = (-1, None)
            for A_c in g["grid_ns_complete_a"]:
                comps = [complete_pattern(asm, c, select_assemblies(c, rho, cm, A_c), q, m) for c, q, m in zip(vcos, vcues, ms_v)]
                p = prec_mean(comps, vmasked)
                if p > bp[0]: bp = (p, A_c)
            A_c = bp[1]
            comps = [complete_pattern(asm, c, select_assemblies(c, rho, cm, A_c), q, m) for c, q, m in zip(tcos, tcues, ms_t)]
            ns_p = prec_mean(comps, tmasked)
        else:
            ns_p, A_c = float("nan"), None
        out["ns"] = {"f1": ns_f1, "prec": ns_p, "params": {"rho": rho, "cos_min": cm, "map_thr": thr, "A_c": A_c}}
        # ---- Jaccard kNN reference
        vi, vs = nn.topk(vcues); ti, ts = nn.topk(tcues)
        bf = (-1, None)
        for k, thr2 in itertools.product(g["grid_nn_k"], g["grid_nn_vote_thr"]):
            f = f1_mean(nn_decode(vi, vs, tr.concepts, k, thr2), va.concepts)
            if f > bf[0]: bf = (f, (k, thr2))
        k1, thr2 = bf[1]
        nn_f1 = f1_mean(nn_decode(ti, ts, tr.concepts, k1, thr2), te.concepts)
        if mf > 0:
            bp = (-1, None)
            for k in g["grid_nn_k"]:
                p = prec_mean(nn_complete(vi, vs, codes, vcues, ms_v, k, cfg.N), vmasked)
                if p > bp[0]: bp = (p, k)
            nn_p = prec_mean(nn_complete(ti, ts, codes, tcues, ms_t, bp[1], cfg.N), tmasked); kc = bp[1]
        else:
            nn_p, kc = float("nan"), None
        out["nn"] = {"f1": nn_f1, "prec": nn_p, "params": {"k_concept": k1, "vote_thr": thr2, "k_complete": kc}}
        # ---- exact match
        qs = [ex.query(c, m) for c, m in zip(tcues, ms_t)]
        ex_f1 = f1_mean([q[0] for q in qs], te.concepts)
        ex_p = prec_mean([q[1] for q in qs], tmasked) if mf > 0 else float("nan")
        out["exact"] = {"f1": ex_f1, "prec": ex_p, "hit_rate": float(np.mean([q[2] for q in qs]))}
        res["masks"][str(mf)] = out
    # ---- cost (mask 0.5): per-query latency and index bytes
    tcues, tmasked = mask_cues(tc[:200], 0.5, seed, 3)
    def timeit(fn):
        ts = []
        for q, mk in zip(tcues, tmasked):
            a = time.perf_counter(); fn(q, len(mk)); ts.append(time.perf_counter() - a)
        return float(np.median(ts) * 1e6)
    def ns_q(q, m):
        c = asm.vote(q); sel = select_assemblies(c, rho, cm, A_c or 1)
        ns_decode(c, cmap, rho, cm, thr); complete_pattern(asm, c, sel, q, m)
    def nn_q(q, m):
        i, s = nn.topk([q], 10); nn_decode(i, s, tr.concepts, k1, thr2); nn_complete(i, s, codes, [q], [m], 5, cfg.N)
    res["cost"] = {"ns_query_us": timeit(ns_q), "nn_query_us": timeit(nn_q), "exact_query_us": timeit(lambda q, m: ex.query(q, m)),
                   "ns_index_bytes": asm.index_bytes() + sum(a.nbytes + b.nbytes for a, b in zip(asm.member_ids, asm.member_w)),
                   "nn_bytes_packed": nn.nbytes_packed(), "nn_bytes_dense": nn.nbytes_dense(), "exact_bytes": ex.nbytes(),
                   "n_train_events": int(nn.n)}
    return res


per = []
for s in seeds:
    r = run(s); per.append(r)
    m5 = r["masks"]["0.5"]
    print(s, f"asm={r['n_assemblies']} purity={r['purity']:.2f} cov={r['concepts_covered']}/40 | mask.5 F1 ns={m5['ns']['f1']:.3f} nn={m5['nn']['f1']:.3f} ex={m5['exact']['f1']:.3f}"
          f" | P@m ns={m5['ns']['prec']:.3f} nn={m5['nn']['prec']:.3f} ex={m5['exact']['prec']:.3f}", flush=True)

# ---- gate
cond, ok_all = {}, True
agg = {}
for mf in g["masks"]:
    for metric in ("f1", "prec"):
        v = {sy: np.array([r["masks"][str(mf)][sy][metric] for r in per]) for sy in ("ns", "nn", "exact")}
        beat = bool(v["ns"].mean() - v["exact"].mean() >= g["beat_exact_margin_abs"] and np.all(v["ns"] - v["exact"] >= g["beat_exact_margin_abs"]))
        ratio_req = g["approach_ratio"][str(mf)]
        ratio = float(v["ns"].mean() / v["nn"].mean())
        appr = bool(ratio >= ratio_req)
        cond[f"mask{mf}_{metric}"] = {"beats_exact": beat, "ratio_vs_nn": ratio, "ratio_required": ratio_req, "approaches_nn": appr,
                                      "ns": summarize(v["ns"]), "nn": summarize(v["nn"]), "exact": summarize(v["exact"])}
        ok_all &= beat and appr
out = {"gate": "PASS" if ok_all else "FAIL", "seeds": seeds, "conditions": cond, "per_seed": per,
       "n_assemblies": summarize([r["n_assemblies"] for r in per]), "purity": summarize([r["purity"] for r in per]),
       "concepts_covered": summarize([r["concepts_covered"] for r in per])}
if DEV:
    (ROOT / "results" / "dev").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "dev" / "phase2.json").write_text(json.dumps(out, indent=1, default=float))
else:
    write_result("phase2", out)
for k_, v in cond.items():
    print(k_, f"ns={v['ns']['mean']:.3f} nn={v['nn']['mean']:.3f} ex={v['exact']['mean']:.3f} ratio={v['ratio_vs_nn']:.3f}/{v['ratio_required']} beat={v['beats_exact']} appr={v['approaches_nn']}")
print("GATE 2:", out["gate"], "(dev run)" if DEV else "")
