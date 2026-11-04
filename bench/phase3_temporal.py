"""Gate 3: top-k next-unit prediction vs global-frequency and first-order bigram baselines."""
import sys, time, itertools, json
import numpy as np
from sim.generator import World, WorldConfig
from neurostore.config import Config
from neurostore.store import NeuroStore
from neurostore.readout import temporal_readout
from eval.protocol import Bigram, recall_at_k_batch, topk_rows
from eval.gates import load_gates, write_result, ROOT
from eval.metrics import summarize

G = load_gates(); g = G["phase3"]; K = 32
DEV = len(sys.argv) > 1 and sys.argv[1] == "dev"
seeds = G["global"]["dev_seeds"] if DEV else G["global"]["gate_seeds"]


def ns_eval(store, codes, strengths, nexts, hops, lagged, norm, keep):
    preds = [temporal_readout(store.lattice, c, s, hops=hops, hop_decay=store.cfg.hop_decay, keep=keep, m=K,
                              normalize=norm, lagged_only=lagged)[0] for c, s in zip(codes, strengths)]
    return recall_at_k_batch(preds, nexts, K)


def run(seed):
    w = World(WorldConfig(), seed)
    tr, va, te = w.stream("train", g["n_train"]), w.stream("val", g["n_val"]), w.stream("test", g["n_test"])
    cfg = Config(seed=seed, homeo=False, gate=False, hebbian=True, consolidate_every=200, replay_budget=200)
    store = NeuroStore(cfg)
    t0 = time.perf_counter(); store.ingest(tr); store.flush(); t_train = time.perf_counter() - t0
    codes = store.log.gather(np.arange(store.log.n_stored))[0]
    enc = lambda s: [store.enc.encode(s.event(i)) for i in range(len(s))]
    ev, et = enc(va), enc(te)
    vc, vs = [e[0] for e in ev], [e[1] for e in ev]; tc, ts = [e[0] for e in et], [e[1] for e in et]
    # ---- NeuroStore: pick read parameters on validation
    best = (-1, None)
    for hops, lag, norm in itertools.product(g["grid_hops"], g["grid_lagged_only"], g["grid_normalize"]):
        r = ns_eval(store, vc[:-1], vs[:-1], vc[1:], hops, lag, norm, cfg.spread_keep)
        if r > best[0]: best = (r, (hops, lag, norm))
    hops, lag, norm = best[1]
    ns = ns_eval(store, tc[:-1], ts[:-1], tc[1:], hops, lag, norm, cfg.spread_keep)
    # ---- baselines
    freq = np.bincount(codes.ravel(), minlength=cfg.N); top_f = np.argsort(-freq, kind="stable")[:K]
    gf = recall_at_k_batch([top_f] * (len(tc) - 1), tc[1:], K)
    bg = Bigram(cfg.N).fit(codes)
    bres = {}
    for nm in (True, False):
        Sv = bg.scores(vc[:-1], nm); bres[nm] = recall_at_k_batch(list(np.argsort(-Sv, axis=1, kind="stable")[:, :K]), vc[1:], K)
    bnorm = max(bres, key=bres.get)
    St = bg.scores(tc[:-1], bnorm); bigram = recall_at_k_batch(list(np.argsort(-St, axis=1, kind="stable")[:, :K]), tc[1:], K)
    # ---- oracle ceiling (true primary-chain transition matrix + true unit profiles from labelled train)
    prof = np.zeros((wc_C, cfg.N)); cnt = np.zeros(wc_C)
    for i, cs in enumerate(tr.concepts):
        if len(cs) == 1:
            prof[cs[0], codes[i]] += 1; cnt[cs[0]] += 1
    prof /= np.maximum(cnt, 1)[:, None]
    So = np.stack([w.P[p] @ prof for p in te.primary[:-1]])
    oracle = recall_at_k_batch(list(np.argsort(-So, axis=1, kind="stable")[:, :K]), tc[1:], K)
    lat = store.lattice
    return {"seed": seed, "ns": ns, "freq": gf, "bigram": bigram, "bigram_normalized": bool(bnorm), "oracle": oracle,
            "ns_params": {"hops": hops, "lagged_only": lag, "normalize": norm}, "val_ns": best[0],
            "train_s": t_train, "n_edges": int(lat.csr.n_edges), "lattice_bytes": int(lat.nbytes()), "epochs": int(lat.epoch),
            "n_assemblies": store.asm.n_alive}


wc_C = WorldConfig().C
per = []
for s in seeds:
    r = run(s); per.append(r)
    print(s, f"ns={r['ns']:.4f} bigram={r['bigram']:.4f} freq={r['freq']:.4f} oracle={r['oracle']:.4f} params={r['ns_params']} train={r['train_s']:.0f}s edges={r['n_edges']}", flush=True)
ns, bg, fr = (np.array([r[k] for r in per]) for k in ("ns", "bigram", "freq"))
ok = bool((ns.mean() - bg.mean() >= g["recall_margin_abs"]) and (ns.mean() - fr.mean() >= g["recall_margin_abs"])
          and (ns > bg).sum() >= g["min_seed_wins"] and (ns > fr).sum() >= g["min_seed_wins"])
out = {"gate": "PASS" if ok else "FAIL", "seeds": seeds, "per_seed": per, "ns": summarize(ns), "bigram": summarize(bg), "freq": summarize(fr),
       "oracle": summarize([r["oracle"] for r in per]), "wins_vs_bigram": int((ns > bg).sum()), "wins_vs_freq": int((ns > fr).sum()),
       "margin_vs_bigram": float(ns.mean() - bg.mean()), "margin_vs_freq": float(ns.mean() - fr.mean())}
if DEV:
    (ROOT / "results" / "dev").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "dev" / "phase3.json").write_text(json.dumps(out, indent=1, default=float))
else:
    write_result("phase3", out)
print("GATE 3:", out["gate"], "(dev run)" if DEV else "", f"ns={ns.mean():.4f} bigram={bg.mean():.4f} freq={fr.mean():.4f}")
