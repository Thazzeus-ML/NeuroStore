"""The committed gate verdicts must follow from the committed numbers and the pre-registered thresholds."""
import json, tomllib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
G = tomllib.loads((ROOT / "gates.toml").read_text())
R = lambda n: json.loads((ROOT / "results" / f"{n}.json").read_text())


def test_gate1():
    r = R("phase1"); g = G["phase1"]
    ok = all(x["exact_k_fraction"] >= g["exact_k_fraction"] and x["determinism_mismatches"] <= g["determinism_mismatches"]
             and x["overlap_margin"] >= g["overlap_margin_min"] for x in r["per_seed"])
    assert (r["gate"] == "PASS") == ok


def test_gate2():
    r = R("phase2"); g = G["phase2"]; ok = True
    for mf in g["masks"]:
        for m in ("f1", "prec"):
            c = r["conditions"][f"mask{mf}_{m}"]
            ns, ex, nn = np.array(c["ns"]["values"]), np.array(c["exact"]["values"]), np.array(c["nn"]["values"])
            beat = ns.mean() - ex.mean() >= g["beat_exact_margin_abs"] and np.all(ns - ex >= g["beat_exact_margin_abs"])
            appr = ns.mean() / nn.mean() >= g["approach_ratio"][str(mf)]
            assert c["beats_exact"] == beat and c["approaches_nn"] == appr
            ok &= bool(beat and appr)
    assert (r["gate"] == "PASS") == ok


def test_gate3():
    r = R("phase3"); g = G["phase3"]
    ns, bg, fr = (np.array(r[k]["values"]) for k in ("ns", "bigram", "freq"))
    ok = (ns.mean() - bg.mean() >= g["recall_margin_abs"] and ns.mean() - fr.mean() >= g["recall_margin_abs"]
          and (ns > bg).sum() >= g["min_seed_wins"] and (ns > fr).sum() >= g["min_seed_wins"])
    assert (r["gate"] == "PASS") == ok


def test_gate4_both_attempts():
    g = G["phase4"]
    for name, expect in (("phase4_attempt1", "FAIL"), ("phase4", "PASS")):
        r = R(name)
        ok = (np.mean(r["bytes_reduction"]["values"]) >= g["bytes_reduction_min"] and np.mean(r["f1_drop"]["values"]) <= g["recall_tolerance_abs"]
              and np.mean(r["prec_drop"]["values"]) <= g["recall_tolerance_abs"])
        assert (r["gate"] == "PASS") == ok and r["gate"] == expect


def test_gate5_hypothesis_rule():
    r = R("phase5"); g = G["phase5"]
    for h in r["hypotheses"]:
        assert h["held"] == (h["fastest"] == h["predicted_winner"] and h["margin_vs_runner_up"] >= g["min_margin"] and h["seed_wins"] >= g["min_seed_wins"])


def test_results_record_gate_commit():
    for n in ("phase0", "phase1", "phase2", "phase3", "phase4", "phase5", "phase6"):
        assert R(n)["gates_commit"] != "unknown"
