"""POST-HOC (added after seeing the phase-6 smoke run): how does the homeostasis strength (threshold clip) trade
rate uniformity against code content? clip = 0 is equivalent to homeostasis off. Not part of the pre-registered
gate; it cannot change any verdict."""
import sys
from sim.generator import WorldConfig
from eval.gates import load_gates, write_result
from eval.metrics import summarize
import bench.phase6_ablations as P

G = load_gates(); g = G["phase6"]; seeds = G["global"]["gate_seeds"]
CLIPS = [0.0, 0.25, 0.5, 1.0, 2.0]
metrics = ["f1", "prec_lift", "temporal_lift", "rate_cv", "overlap_margin", "log_bytes_per_event", "surprise_auc"]
out = {}
for wn, wc in (("uniform", WorldConfig()), ("zipf", WorldConfig(zipf_s=g["zipf_s"]))):
    out[wn] = {}
    for c in CLIPS:
        rows = [P.run(s, f"clip{c}", dict(theta_clip=c), wc) for s in seeds]
        out[wn][str(c)] = {m: summarize([r[m] for r in rows]) for m in metrics}
        print(wn, "clip", c, {m: round(out[wn][str(c)][m]["mean"], 3) for m in metrics}, flush=True)
write_result("phase6_homeostasis_sweep", {"clips": CLIPS, "seeds": seeds, "table": out})
print("done")
