"""Phase 3 (reported, not gated): behaviour after the drift point.
Train stream of 8000 events, membership of every concept changes (40% of members) at event 4000.
Models are evaluated on a fixed post-drift and a fixed pre-drift test stream at checkpoints."""
import numpy as np, itertools
from sim.generator import World, WorldConfig
from neurostore.config import Config
from neurostore.store import NeuroStore
from neurostore.encoder import SparseEncoder
from neurostore.readout import temporal_readout
from eval.protocol import Bigram, recall_at_k_batch
from eval.gates import load_gates, write_result
from eval.metrics import summarize

G = load_gates(); seeds = G["global"]["gate_seeds"]; K = 32
N_TRAIN, DRIFT = 8000, 4000
CKPT = [DRIFT - 200, DRIFT, DRIFT + 200, DRIFT + 400, DRIFT + 800, DRIFT + 1600, DRIFT + 2400, N_TRAIN]


def ns_recall(store, codes_s, nexts):
    preds = [temporal_readout(store.lattice, c, s, hops=1, hop_decay=store.cfg.hop_decay, keep=store.cfg.spread_keep, m=K)[0] for c, s in codes_s]
    return recall_at_k_batch(preds, nexts, K)


def bg_recall(bg, cur, nexts):
    S = bg.scores(cur, True)
    return recall_at_k_batch(list(np.argsort(-S, axis=1, kind="stable")[:, :K]), nexts, K)


def run(seed):
    wc = WorldConfig(drift=True, drift_frac=0.4)
    w = World(wc, seed)
    tr = w.stream("train", N_TRAIN); assert tr.drift_index == DRIFT
    tests = {"post": w.stream("test", 1000, phase=1, idx=1), "pre": w.stream("test", 1000, phase=0, idx=2)}
    models = {"ns_lam0.03": Config(seed=seed, homeo=False, gate=False), "ns_lam0.15": Config(seed=seed, homeo=False, gate=False, lam=0.15)}
    stores = {k: NeuroStore(c) for k, c in models.items()}
    enc = SparseEncoder(wc.F, 1024, 32, 2, seed)
    all_codes = np.array([enc.encode(tr.event(i))[0] for i in range(N_TRAIN)])
    tenc = {k: [enc.encode(s.event(i)) for i in range(len(s))] for k, s in tests.items()}
    rows = []
    done = 0
    for ck in CKPT:
        for st in stores.values():
            st.ingest(tr, done, ck); st.flush()
        done = ck
        row = {"events": ck, "after_drift": ck - DRIFT}
        for tname in tests:
            e = tenc[tname]; cur = [x[0] for x in e[:-1]]; nxt = [x[0] for x in e[1:]]
            for name, st in stores.items():
                row[f"{name}/{tname}"] = ns_recall(st, e[:-1], nxt)
            row[f"bigram_cum/{tname}"] = bg_recall(Bigram(1024).fit(all_codes[:ck]), cur, nxt)
            row[f"bigram_win1500/{tname}"] = bg_recall(Bigram(1024).fit(all_codes[max(0, ck - 1500):ck]), cur, nxt)
        rows.append(row)
    return rows


per = [run(s) for s in seeds]
models = [k for k in per[0][0] if "/" in k]
curve = {m: {str(r["events"]): summarize([p[i][m] for p in per]) for i, r in enumerate(per[0])} for m in models}
summary = {}
for m in models:
    pre = np.array([p[1][m] for p in per])                      # at drift point
    fin = np.array([p[-1][m] for p in per])
    ev0 = np.array([p[2][m] for p in per])                      # +200 events after drift
    summary[m] = {"at_drift": summarize(pre), "plus200": summarize(ev0), "final": summarize(fin)}
    # events after drift until post-drift recall regains 95% of the model's own recall on the PRE-drift
    # test stream measured at the drift point (None = not regained within the run)
    if m.endswith("/post"):
        t95 = []
        for p in per:
            tgt = 0.95 * p[1][m.replace("/post", "/pre")]
            t95.append(next((r["after_drift"] for r in p if r["after_drift"] > 0 and r[m] >= tgt), None))
        summary[m]["events_to_regain_95pct_of_pre_drift_level"] = t95
    t90 = summary[m].get("events_to_regain_95pct_of_pre_drift_level")
    print(m, f"at drift {pre.mean():.3f} -> +200 {ev0.mean():.3f} -> final {fin.mean():.3f}  t90={t90}")
write_result("phase3_drift", {"seeds": seeds, "checkpoints": CKPT, "drift_index": DRIFT, "curve": curve, "summary": summary})
