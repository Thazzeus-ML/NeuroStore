"""Phase 3 post-hoc controls (added AFTER the gate-3 result was known; cannot change its verdict):
which ingredient of the lattice produces the advantage over the bigram baseline?"""
import itertools, time
import numpy as np
from sim.generator import World, WorldConfig
from neurostore.config import Config
from neurostore.store import NeuroStore
from neurostore.readout import temporal_readout
from eval.protocol import Bigram, recall_at_k_batch
from eval.gates import load_gates, write_result
from eval.metrics import summarize

G = load_gates(); g = G["phase3"]; seeds = G["global"]["gate_seeds"]; K = 32
VARIANTS = {
    "default": {},
    "unpruned_nodecay": dict(prune_R=None, eps=0.0, lam=0.0),
    "no_within_event_edges": dict(within_event=False),
    "one_lag": dict(n_lags=1),
    "no_within_one_lag": dict(within_event=False, n_lags=1),
}


def evalv(store, vc, vs, nx, hops, lag, norm):
    p = [temporal_readout(store.lattice, c, s, hops=hops, hop_decay=0.5, keep=store.cfg.spread_keep, m=K, normalize=norm, lagged_only=lag)[0]
         for c, s in zip(vc, vs)]
    return recall_at_k_batch(p, nx, K)


res = {v: [] for v in VARIANTS}
for seed in seeds:
    w = World(WorldConfig(), seed)
    tr, va, te = w.stream("train", g["n_train"]), w.stream("val", g["n_val"]), w.stream("test", g["n_test"])
    for name, kw in VARIANTS.items():
        cfg = Config(seed=seed, homeo=False, gate=False, replay_budget=200, **kw)
        st = NeuroStore(cfg); st.ingest(tr); st.flush()
        ev = [st.enc.encode(va.event(i)) for i in range(len(va))]; et = [st.enc.encode(te.event(i)) for i in range(len(te))]
        best = (-1, None)
        for hops, lag, norm in itertools.product(g["grid_hops"], g["grid_lagged_only"], g["grid_normalize"]):
            r = evalv(st, [e[0] for e in ev[:-1]], [e[1] for e in ev[:-1]], [e[0] for e in ev[1:]], hops, lag, norm)
            if r > best[0]: best = (r, (hops, lag, norm))
        r = evalv(st, [e[0] for e in et[:-1]], [e[1] for e in et[:-1]], [e[0] for e in et[1:]], *best[1])
        res[name].append({"seed": seed, "recall": r, "params": list(best[1]), "n_edges": int(st.lattice.csr.n_edges), "lattice_bytes": int(st.lattice.nbytes())})
        print(seed, name, f"{r:.4f}", best[1], st.lattice.csr.n_edges, flush=True)
out = {v: {"recall": summarize([x["recall"] for x in rows]), "n_edges": summarize([x["n_edges"] for x in rows]),
           "lattice_bytes": summarize([x["lattice_bytes"] for x in rows]), "rows": rows} for v, rows in res.items()}
write_result("phase3_controls", {"seeds": seeds, "variants": out})
for v in out: print(v, f"{out[v]['recall']['mean']:.4f} +- {out[v]['recall']['std']:.4f}  edges={out[v]['n_edges']['mean']:.0f}")
