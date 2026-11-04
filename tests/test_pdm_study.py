"""The PdM study: gate decision logic on hand-built result files, and an end-to-end smoke run of the runner on fake data."""
import hashlib
import json
import subprocess
import shutil
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "studies" / "pdm_cmapss"))
import evaluate_gates as EG  # noqa: E402

ALT = EG.ALT
FAMS = ["noise", "dropout", "stuck", "drift", "spikes", "quantize", "combined"]


def build(tmp: Path, **kw):
    """A complete, internally consistent result directory in which every gate is HELD; kw overrides pieces for the failing cases."""
    tmp.mkdir(parents=True, exist_ok=True)
    proto = tmp / "PROTOCOL.md"
    proto.write_text("protocol")
    sha = hashlib.sha256(b"protocol").hexdigest()
    sums = {"a": "1"}
    meta = {"git_head": "abc", "dirty_code": False, "fast": False, "protocol_sha256": sha, "data_sums": sums, **kw.get("meta", {})}

    def w(name, obj):
        (tmp / name).write_text(json.dumps(obj))

    w("validate.json", {"meta": meta, "analytic": [{"inside": True}] * 5})
    for fd in ("FD001", "FD002", "FD003", "FD004"):
        w(f"tuning_{fd}.json", {"meta": meta})
    ratio = lambda r, hi=None: {"ratio": r, "lo": r - 0.02, "hi": hi if hi is not None else r + 0.03}
    variants = {b: {"rmse": 20.0, "ratio_vs_ref": ratio(1.02)} for b in (*ALT, "exact", "exact_l2")}
    variants.update(kw.get("g3", {}))
    variants["const"], variants["ridge"] = {"rmse": 42.0}, {"rmse": 22.0}
    w("test_panel.json", {"meta": meta, "subsets": {"FD001": {"reference": "exact_l2", "variants": variants, **kw.get("g2", {"ref_vs_const": ratio(0.5), "ref_vs_ridge": ratio(0.95)})}}})
    mem = {b: {"memory": {"deployed_bytes": 1000}} for b in ("exact", "exact_l2", "hopfield", "ann", "flyhash", "vsa")}
    mem["ann_sq8"] = {"memory": {"deployed_bytes": kw.get("sq8_bytes", 300)}}
    scale = kw.get("scale", {"exact": [{"n": 500000, "latency": {"p50_ms": 30.0}}],
                             "ann": [{"n": 500000, "latency": {"p50_ms": 6.0}, "recall10": 0.93, "nprobe": 8, "target_met": True}],
                             "ann_sq8": [{"n": 500000, "latency": {"p50_ms": 7.0}, "recall10": 0.91, "nprobe": 8, "target_met": True}]})
    w("timing.json", {"meta": meta, "variants": mem, "scale": scale})
    sweeps = {"meta": meta, "reference": "exact_l2", "levels": {f: {"L50": {"reached": True}} for f in FAMS}, "variants": {}}
    better = kw.get("better", {"vsa": FAMS[:4]})
    for b in ALT:
        sweeps["variants"][b] = {"families": {f: {"L50": {"vs_ref": {"diff": -0.2, "lo": -0.3, "hi": -0.1} if f in better.get(b, []) else {"diff": 0.0, "lo": -0.1, "hi": 0.1}}} for f in FAMS}}
    w("sweeps.json", sweeps)
    a = lambda auc, lo, hi: {"auroc": auc, "lo": lo, "hi": hi}
    ood = {"foreign_FD003_late": a(0.85, 0.80, 0.9), "chimera_channel_far": a(0.9, 0.85, 0.95), "chimera_channel_same": a(0.6, 0.55, 0.65),
           "chimera_channel_mid": a(0.75, 0.7, 0.8), "gauss": a(0.99, 0.98, 1.0)}
    ood.update(kw.get("ood", {}))
    w("ood.json", {"meta": meta, "reference": "exact_l2", "variants": {"exact_l2": {k: {"max_sim": v} for k, v in ood.items()}}})
    w("completion.json", {"meta": meta, "reference": "exact_l2", "channels": 14, "missing_channels": {"0.3": 5},
                          "variants": {"exact_l2_mask": {"0.3": {"nmse": kw.get("nmse", 0.2), "lo": 0.1, "hi": 0.3}}, "mean_imputation": {"0.3": {"nmse": 1.0, "lo": 0.9, "hi": 1.1}}}})
    return tmp, proto, sums


def verdicts(tmp, **kw):
    d, proto, sums = build(tmp, **kw)
    return {r["id"]: r["verdict"] for r in EG.evaluate(d, protocol_path=proto, data_sums=sums)}


def test_everything_held_in_the_consistent_fixture(tmp_path):
    v = verdicts(tmp_path)
    assert all(x == "HELD" for x in v.values()), v
    assert {"G0", "G1", "G2", "G4", "G5", "G6a", "G6b", "G6c", "G7"} <= set(v) and "G3[vsa]" in v and "G3[exact]" in v


def test_g3_three_way_verdict(tmp_path):
    v = verdicts(tmp_path, g3={"vsa": {"rmse": 1, "ratio_vs_ref": {"ratio": 1.05, "lo": 1.0, "hi": 1.15}},
                               "flyhash": {"rmse": 1, "ratio_vs_ref": {"ratio": 1.20, "lo": 1.1, "hi": 1.3}},
                               "ann": {"rmse": 1, "ratio_vs_ref": {"ratio": 1.10, "lo": 1.0, "hi": 1.10}}})
    assert v["G3[vsa]"] == "INCONCLUSIVE" and v["G3[flyhash]"] == "NOT HELD" and v["G3[ann]"] == "HELD" and v["G3[hopfield]"] == "HELD"


def test_g2_needs_both_conditions(tmp_path):
    ok = lambda r: {"ratio": r, "lo": r, "hi": r}
    assert verdicts(tmp_path / "a", g2={"ref_vs_const": ok(0.61), "ref_vs_ridge": ok(0.9)})["G2"] == "NOT HELD"
    assert verdicts(tmp_path / "b", g2={"ref_vs_const": ok(0.4), "ref_vs_ridge": ok(1.26)})["G2"] == "NOT HELD"
    assert verdicts(tmp_path / "c", g2={"ref_vs_const": ok(0.6), "ref_vs_ridge": ok(1.25)})["G2"] == "HELD"


def test_g4_needs_memory_and_latency_parts(tmp_path):
    assert verdicts(tmp_path / "a", sq8_bytes=600)["G4"] == "NOT HELD"                              # nothing at <= 50 % of exact's bytes
    slow = {"exact": [{"n": 500000, "latency": {"p50_ms": 30.0}}], "ann": [{"n": 500000, "latency": {"p50_ms": 15.0}, "recall10": 0.93, "nprobe": 8, "target_met": True}],
            "ann_sq8": [{"n": 500000, "latency": {"p50_ms": 16.0}, "recall10": 0.93, "nprobe": 8, "target_met": True}]}
    assert verdicts(tmp_path / "b", scale=slow)["G4"] == "NOT HELD"                                  # ANN is not 3x faster
    unmet = {"exact": slow["exact"], "ann": [{"n": 500000, "latency": {"p50_ms": 2.0}, "recall10": 0.6, "nprobe": 64, "target_met": False}], "ann_sq8": []}
    assert verdicts(tmp_path / "c", scale=unmet)["G4"] == "NOT HELD"                                 # fast but recall target not met
    assert verdicts(tmp_path / "d", scale={"exact": [], "ann": [], "ann_sq8": []})["G4"] == "NOT EVALUABLE"
    noninf = {"ann_sq8": {"rmse": 1, "ratio_vs_ref": {"ratio": 1.3, "lo": 1.2, "hi": 1.4}}}
    assert verdicts(tmp_path / "e", g3=noninf)["G4"] == "NOT HELD"                                  # cheap but not non-inferior


def test_g5_counts_families_with_ci_below_zero(tmp_path):
    assert verdicts(tmp_path / "a", better={"vsa": FAMS[:2], "flyhash": FAMS[2:4]})["G5"] == "NOT HELD"   # 2 + 2 does not add up across backends
    assert verdicts(tmp_path / "b", better={"vsa": FAMS[:3]})["G5"] == "HELD"


def test_g6_and_g7_thresholds(tmp_path):
    a = lambda auc, lo, hi: {"auroc": auc, "lo": lo, "hi": hi}
    v = verdicts(tmp_path / "a", ood={"foreign_FD003_late": a(0.8, 0.74, 0.86), "gauss": a(0.9, 0.85, 0.95)})
    assert (v["G6a"], v["G6b"], v["G6c"]) == ("NOT HELD", "HELD", "NOT HELD")          # CI lower 0.74 < 0.75; chimeras fine; gauss 0.90 < 0.95
    v = verdicts(tmp_path / "b", ood={"chimera_channel_same": a(0.6, 0.5, 0.86)})
    assert v["G6b"] == "NOT HELD" and v["G6a"] == "HELD"                               # far.lo 0.85 does not clear same.hi 0.86
    v = verdicts(tmp_path / "c", ood={"chimera_channel_far": a(0.69, 0.65, 0.73), "chimera_channel_same": a(0.5, 0.45, 0.55)})
    assert v["G6b"] == "NOT HELD"                                                        # well separated but far AUROC 0.69 < 0.70
    assert verdicts(tmp_path / "d", nmse=0.5)["G7"] == "HELD" and verdicts(tmp_path / "e", nmse=0.51)["G7"] == "NOT HELD"


def test_g0_catches_dirty_fast_or_foreign_results(tmp_path):
    assert verdicts(tmp_path / "a", meta={"dirty_code": True})["G0"] == "NOT HELD"
    assert verdicts(tmp_path / "b", meta={"fast": True})["G0"] == "NOT HELD"
    assert verdicts(tmp_path / "c", meta={"protocol_sha256": "0" * 64})["G0"] == "NOT HELD"
    assert verdicts(tmp_path / "d", meta={"data_sums": {"a": "2"}})["G0"] == "NOT HELD"


def test_missing_results_are_reported_not_invented(tmp_path):
    proto = tmp_path / "P.md"; proto.write_text("x")
    v = {r["id"]: r["verdict"] for r in EG.evaluate(tmp_path, protocol_path=proto)}
    assert v["G1"] == v["G2"] == v["G4"] == v["G5"] == v["G6a"] == v["G7"] == "NOT RUN" and v["G0"] == "NOT HELD"


def test_runner_end_to_end_on_fake_data():
    out = ROOT / "results" / "dev" / "pytest_pdm"
    shutil.rmtree(out, ignore_errors=True)
    sys.path.insert(0, str(ROOT / "src"))
    from nsharness.domains.fake_cmapss import write_fake_dataset
    write_fake_dataset(out / "data", n_train=18, n_test=6)
    r = subprocess.run([sys.executable, "-I", str(ROOT / "studies" / "pdm_cmapss" / "run_study.py"), "--stage", "all", "--fast",
                        "--out", str(out), "--data-root", str(out / "data")], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    for f in ("validate", "tuning_FD001", "tuning_FD004", "test_panel", "sweeps", "completion", "ood", "timing"):
        obj = json.loads((out / f"{f}.json").read_text())
        assert obj["meta"]["fast"] is True and obj["meta"]["stage"]
    sw = json.loads((out / "sweeps.json").read_text())
    assert set(sw["variants"]) >= {"exact", "exact_l2", "vsa", "ridge"} and sw["levels"]["noise"]["L50"]["meaning"]
    rows = {r["id"]: r["verdict"] for r in EG.evaluate(out)}
    assert rows["G0"] == "NOT HELD"                                           # fast mode can never pass the provenance gate
    ph = subprocess.run([sys.executable, "-I", str(ROOT / "studies" / "pdm_cmapss" / "posthoc_quantized.py"), "--fast", "--results", str(out), "--data-root", str(out / "data")],
                        capture_output=True, text=True, timeout=600)
    assert ph.returncode == 0, ph.stdout[-1500:] + ph.stderr[-1500:]
    pj = json.loads((out / "posthoc_quantized.json").read_text())
    assert pj["meta"]["exploratory"] is True and set(pj["variants"]) == {"exact_l2_clip3", "exact_l2_q32", "exact_l1", "exact_l1_clip3"}
    rep = subprocess.run([sys.executable, "-I", str(ROOT / "studies" / "pdm_cmapss" / "make_report.py"), "--results", str(out), "--out", str(out / "REPORT.md"),
                          "--no-narrative"], capture_output=True, text=True, timeout=300)
    assert rep.returncode == 0, rep.stderr[-2000:]
    text = (out / "REPORT.md").read_text()
    assert "## Gate verdicts" in text and "| G0 | **NOT HELD** |" in text and "## E5 Latency" in text and "## Post-hoc control" in text
    shutil.rmtree(out, ignore_errors=True)
