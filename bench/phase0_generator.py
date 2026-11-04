"""Gate 0: generator determinism + documented statistics (thresholds from gates.toml)."""
import numpy as np
from sim.generator import World, WorldConfig
from eval.gates import load_gates, write_result
from eval.metrics import tv_distance
from tests.test_generator import same

G = load_gates(); g0 = G["phase0"]
rows, ok_all = [], True
for seed in g0["determinism_seeds"]:
    cfg = WorldConfig(zipf_s=1.0, drift=True)
    w = World(cfg, seed)
    det = same(w.stream("train", 3000), World(cfg, seed).stream("train", 3000))
    det_diff = not same(w.stream("train", 3000), World(cfg, seed + 1000).stream("train", 3000))
    # statistics on a stationary (no-drift) config with Zipf skew
    cfg2 = WorldConfig(zipf_s=1.0)
    w2 = World(cfg2, seed)
    n = g0["n_events_stats"]
    s = w2.stream("test", n)
    mc = float(np.mean([len(c) for c in s.concepts])); mc_a = w2.expected_mean_concepts()
    single = np.array([len(s.event(i)) for i in range(n) if len(s.concepts[i]) == 1])
    nf, nf_a = float(single.mean()), w2.expected_features_single()
    emp = np.bincount(s.primary, minlength=cfg2.C).astype(float)
    tv = tv_distance(emp, w2.stationary())
    st_emp = float(np.mean(s.primary[1:] == s.primary[:-1])); st_a = w2.expected_self_transition()
    wd = World(WorldConfig(drift=True, drift_frac=0.4), seed)
    n_rep = round(0.4 * 20)
    drift_exact = all(len(np.setdiff1d(wd.members[1][c], wd.members[0][c])) == n_rep for c in range(40))
    crit = {
        "deterministic": bool(det and det_diff),
        "mean_concepts": abs(mc - mc_a) / mc_a <= g0["tol_mean_concepts_rel"],
        "mean_features": abs(nf - nf_a) / nf_a <= g0["tol_mean_features_rel"],
        "chain_tv": tv <= g0["tol_chain_tv"],
        "self_transition": abs(st_emp - st_a) <= g0["tol_self_transition_abs"],
        "drift_exact": bool(drift_exact),
    }
    ok_all &= all(crit.values())
    rows.append({"seed": seed, "mean_concepts": mc, "mean_concepts_analytic": mc_a,
                 "mean_features": nf, "mean_features_analytic": nf_a, "chain_tv": tv,
                 "self_transition": st_emp, "self_transition_analytic": st_a,
                 "criteria": {k: bool(v) for k, v in crit.items()}})
    print(seed, {k: bool(v) for k, v in crit.items()}, f"tv={tv:.4f} nf={nf:.2f}/{nf_a:.2f}")
write_result("phase0", {"gate": "PASS" if ok_all else "FAIL", "rows": rows,
                        "n_events": g0["n_events_stats"]})
print("GATE 0:", "PASS" if ok_all else "FAIL")
