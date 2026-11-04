"""Where does the time go? Decides whether porting a kernel to Rust/C is justified (phase 5 rule)."""
import cProfile, pstats, io, time
import numpy as np
from sim.generator import World, WorldConfig
from neurostore.config import Config
from neurostore.store import NeuroStore
from eval.gates import write_result

w = World(WorldConfig(), 0); tr = w.stream("train", 4000)
st = NeuroStore(Config(seed=0, homeo=True, gate=True, gate_tau=0.55, pred_mode="assembly", replay_budget=150, hot_tau=0.3))
pr = cProfile.Profile(); t0 = time.perf_counter(); pr.enable(); st.ingest(tr); st.flush(); pr.disable(); total = time.perf_counter() - t0
s = pstats.Stats(pr); rows = []
for (fn, line, name), (cc, nc, tt, ct, callers) in s.stats.items():
    rows.append({"func": f"{fn.split('/')[-1]}:{name}", "cumulative_s": ct, "own_s": tt, "calls": nc})
rows.sort(key=lambda r: -r["cumulative_s"])
keep = [r for r in rows if any(x in r["func"] for x in ("write", "encode", "run_round", "_synapses", "learn", "vote", "expected", "gather", "accumulate", "spread", "merge", "maintain", "update", "append", "_build_index", "matmul"))]
# time of the sparse-accumulate kernel (lattice.gather + bincount) as called during ingestion
acc = sum(r["cumulative_s"] for r in rows if r["func"].endswith(":accumulate"))
write_result("phase5_profile", {"events": 4000, "total_s": total, "per_event_ms": 1000 * total / 4000, "accumulate_cumulative_s": acc,
                               "accumulate_fraction": acc / total, "top": keep[:18]})
print(f"total {total:.1f}s ({1000*total/4000:.2f} ms/event); sparse-accumulate kernel {acc:.2f}s = {100*acc/total:.1f}% of ingestion")
for r in keep[:12]: print(f"  {r['func']:40s} cum={r['cumulative_s']:.2f}s calls={r['calls']}")
