"""Gate 1: exact-k, determinism, bytes/event = 8+4k independent of raw features, overlap."""
import numpy as np
from sim.generator import World, WorldConfig
from neurostore.encoder import SparseEncoder, q8
from neurostore.spikelog import SpikeLog
from neurostore.rates import RateCounters
from eval.gates import load_gates, write_result
from eval.metrics import summarize

G = load_gates(); g = G["phase1"]; seeds = G["global"]["gate_seeds"]
N, K, D = g["N"], g["k"], g["d"]


def ingest(F, k, seed, n, wcfg=None):
    w = World((wcfg or WorldConfig()).with_(F=F), seed)
    s = w.stream("test", n)
    enc = SparseEncoder(F, N, k, D, seed); log = SpikeLog(k); rc = RateCounters(N)
    codes, nfeat = [], []
    for i in range(len(s)):
        f = s.event(i)
        u, st_ = enc.encode(f)
        log.append(i, s.t[i], u, q8(st_), np.zeros(k, np.uint8))
        rc.update(s.t[i], u); codes.append(u); nfeat.append(len(f))
    return s, enc, log, rc, np.array(codes), np.array(nfeat)


per_seed, ok_all = [], True
for seed in seeds:
    s, enc, log, rc, codes, nfeat = ingest(800, K, seed, g["n_events"])
    exact = float(np.mean([len(np.unique(c)) == K for c in codes]))
    # determinism: rebuild everything from scratch, compare log contents byte-for-byte
    s2, enc2, log2, _, codes2, _ = ingest(800, K, seed, g["n_events"])
    idx = np.arange(log.n_stored)
    mism = sum(int(not np.array_equal(a, b)) for a, b in zip(log.gather(idx), log2.gather(idx)))
    mism += int(not np.array_equal(log.times(), log2.times()))
    mism += sum(int(not np.array_equal(enc.encode(s.event(i))[0], codes[i])) for i in range(0, 3000, 7))
    # bytes per event across raw-feature regimes and k
    bytes_rows = []
    for F in g["raw_feature_regimes"]:
        _, _, lg, _, _, nf = ingest(F, K, seed, 600)
        bytes_rows.append({"F": F, "k": K, "bytes_per_event": lg.bytes_written / lg.n_stored,
                           "mean_features": float(nf.mean()), "min_features": int(nf.min()), "max_features": int(nf.max())})
    for k in g["k_values"]:
        _, _, lg, _, _, nf = ingest(800, k, seed, 600)
        bytes_rows.append({"F": 800, "k": k, "bytes_per_event": lg.bytes_written / lg.n_stored,
                           "mean_features": float(nf.mean()), "min_features": int(nf.min()), "max_features": int(nf.max())})
    bytes_ok = all(abs(r["bytes_per_event"] - (8 + 4 * r["k"])) <= g["bytes_per_event_tol"] for r in bytes_rows)
    # overlap
    rng = np.random.default_rng(seed + 1000)
    single = np.array([i for i in range(len(s)) if len(s.concepts[i]) == 1])
    by_c = {}
    for i in single: by_c.setdefault(s.concepts[i][0], []).append(i)
    same, diff = [], []
    cs = [c for c, v in by_c.items() if len(v) >= 2]
    for _ in range(g["overlap_pairs"]):
        c = cs[rng.integers(len(cs))]; i, j = rng.choice(by_c[c], 2, replace=False)
        same.append(len(np.intersect1d(codes[i], codes[j])) / K)
        i = single[rng.integers(len(single))]; j = single[rng.integers(len(single))]
        if s.concepts[i] != s.concepts[j]:
            diff.append(len(np.intersect1d(codes[i], codes[j])) / K)
    rate = rc.rate()
    row = {"seed": seed, "exact_k_fraction": exact, "determinism_mismatches": int(mism),
           "bytes": bytes_rows, "overlap_same": float(np.mean(same)), "overlap_diff": float(np.mean(diff)),
           "overlap_margin": float(np.mean(same) - np.mean(diff)),
           "unit_rate_cv": float(rate.std() / rate.mean()), "mean_features": float(nfeat.mean())}
    row["pass"] = bool(exact >= g["exact_k_fraction"] and mism <= g["determinism_mismatches"] and bytes_ok
                       and row["overlap_margin"] >= g["overlap_margin_min"])
    ok_all &= row["pass"]
    per_seed.append(row)
    print(seed, f"exactk={exact:.3f} mism={mism} bytes_ok={bytes_ok} same={row['overlap_same']:.3f} diff={row['overlap_diff']:.3f} cv={row['unit_rate_cv']:.2f}")
res = {"gate": "PASS" if ok_all else "FAIL", "seeds": seeds, "per_seed": per_seed,
       "overlap_same": summarize([r["overlap_same"] for r in per_seed]),
       "overlap_diff": summarize([r["overlap_diff"] for r in per_seed]),
       "overlap_margin": summarize([r["overlap_margin"] for r in per_seed]),
       "unit_rate_cv": summarize([r["unit_rate_cv"] for r in per_seed])}
write_result("phase1", res)
print("GATE 1:", res["gate"])
