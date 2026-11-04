"""Mechanical verdicts for the pre-registered gates (gates.toml) from results/pdm_cmapss/*.json. No judgement lives here."""
from __future__ import annotations
import hashlib
import json
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALT = ("hopfield", "ann", "ann_sq8", "flyhash", "vsa")


def _load(d: Path, name):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def g0_provenance(d: Path, protocol_sha: str | None, data_sums: dict | None):
    files = ["validate.json", "tuning_FD001.json", "tuning_FD002.json", "tuning_FD003.json", "tuning_FD004.json", "test_panel.json", "sweeps.json",
             "completion.json", "ood.json", "timing.json"]
    seen, problems, heads = {}, [], set()
    for f in files:
        r = _load(d, f)
        if r is None:
            problems.append(f"{f}: missing")
            continue
        m = r["meta"]
        heads.add(m["git_head"])
        if m["dirty_code"]:
            problems.append(f"{f}: produced from uncommitted code")
        if m["fast"]:
            problems.append(f"{f}: produced in --fast mode")
        if protocol_sha and m["protocol_sha256"] != protocol_sha:
            problems.append(f"{f}: protocol hash differs from the committed PROTOCOL.md")
        if data_sums and m["data_sums"] != data_sums:
            problems.append(f"{f}: data checksums differ from the pinned values")
        seen[f] = m["git_head"]
    return {"id": "G0", "claim": "every result file comes from committed code, the committed protocol and the pinned data", "verdict": "HELD" if not problems else "NOT HELD",
            "evidence": {"problems": problems, "git_heads": sorted(heads)}}


def g1(d: Path, cfg):
    v = _load(d, "validate.json")
    if v is None:
        return {"id": "G1", "claim": cfg["claim"], "verdict": "NOT RUN", "evidence": {}}
    inside = sum(1 for r in v["analytic"] if r["inside"])
    return {"id": "G1", "claim": cfg["claim"], "verdict": "HELD" if inside >= cfg["min_configs_inside"] else "NOT HELD",
            "evidence": {"configs_inside": inside, "of": len(v["analytic"]), "rows": v["analytic"]}}


def g2(d: Path, cfg):
    t = _load(d, "test_panel.json")
    if t is None:
        return {"id": "G2", "claim": cfg["claim"], "verdict": "NOT RUN", "evidence": {}}
    e = t["subsets"]["FD001"]
    rc, rr = e["ref_vs_const"], e["ref_vs_ridge"]
    ok = rc["ratio"] <= cfg["max_ratio_vs_const"] and rr["ratio"] <= cfg["max_ratio_vs_ridge"]
    return {"id": "G2", "claim": cfg["claim"], "verdict": "HELD" if ok else "NOT HELD",
            "evidence": {"reference": e["reference"], "rmse_reference": e["variants"][e["reference"]]["rmse"], "rmse_const": e["variants"]["const"]["rmse"],
                         "rmse_ridge": e["variants"]["ridge"]["rmse"], "ratio_vs_const": rc, "ratio_vs_ridge": rr}}


def g3(d: Path, cfg):
    t = _load(d, "test_panel.json")
    if t is None:
        return [{"id": f"G3[{b}]", "claim": cfg["claim"], "verdict": "NOT RUN", "evidence": {}} for b in ALT]
    e = t["subsets"]["FD001"]
    out = []
    for b in ALT + tuple(x for x in ("exact", "exact_l2") if x != e["reference"]):
        r = e["variants"][b]["ratio_vs_ref"]
        verdict = "HELD" if r["hi"] <= cfg["margin"] else ("INCONCLUSIVE" if r["ratio"] <= cfg["margin"] else "NOT HELD")
        out.append({"id": f"G3[{b}]", "claim": cfg["claim"], "verdict": verdict, "evidence": {"reference": e["reference"], "ratio_vs_ref": r, "rmse": e["variants"][b]["rmse"]}})
    return out


def g4(d: Path, cfg, g3_rows):
    tm = _load(d, "timing.json")
    if tm is None:
        return {"id": "G4", "claim": cfg["claim"], "verdict": "NOT RUN", "evidence": {}}
    t = _load(d, "test_panel.json")
    ref = t["subsets"]["FD001"]["reference"]
    ref_bytes = tm["variants"][ref]["memory"]["deployed_bytes"]
    ok_g3 = {r["id"][3:-1]: r["verdict"] for r in g3_rows}
    mem = {b: tm["variants"][b]["memory"]["deployed_bytes"] / ref_bytes for b in ALT}
    a_ok = [b for b in ALT if ok_g3.get(b) in ("HELD", "INCONCLUSIVE") and mem[b] <= cfg["max_memory_fraction"]]
    # latency at scale
    ex = next((r for r in tm["scale"]["exact"] if r["n"] == cfg["scale_n"]), None)
    lat = {}
    for b in ("ann", "ann_sq8"):
        r = next((r for r in tm["scale"].get(b, []) if r["n"] == cfg["scale_n"]), None)
        if r and ex:
            lat[b] = {"p50_ms": r["latency"]["p50_ms"], "exact_p50_ms": ex["latency"]["p50_ms"], "fraction": r["latency"]["p50_ms"] / ex["latency"]["p50_ms"],
                      "recall10": r["recall10"], "nprobe": r["nprobe"], "target_met": r["target_met"]}
    b_ok = [b for b, v in lat.items() if v["target_met"] and v["recall10"] >= cfg["min_recall"] and v["fraction"] <= cfg["max_latency_fraction"]]
    verdict = "NOT EVALUABLE" if not lat else ("HELD" if (a_ok and b_ok) else "NOT HELD")
    return {"id": "G4", "claim": cfg["claim"], "verdict": verdict,
            "evidence": {"deployed_bytes_fraction_of_reference": mem, "memory_part_ok": a_ok, "latency_at_scale": lat, "latency_part_ok": b_ok, "reference": ref}}


def g5(d: Path, cfg):
    s = _load(d, "sweeps.json")
    if s is None:
        return {"id": "G5", "claim": cfg["claim"], "verdict": "NOT RUN", "evidence": {}}
    lvl, ref = cfg["level"], s["reference"]
    better, worse, unreached = {}, {}, []
    fams = list(s["levels"].keys())
    for fam in fams:
        if not s["levels"][fam][lvl]["reached"]:
            unreached.append(fam)
    for b in ALT:
        better[b], worse[b] = [], []
        for fam in fams:
            v = s["variants"][b]["families"][fam][lvl]["vs_ref"]
            if v["hi"] < 0:
                better[b].append(fam)
            elif v["lo"] > 0:
                worse[b].append(fam)
    best = max(ALT, key=lambda b: len(better[b]))
    return {"id": "G5", "claim": cfg["claim"], "verdict": "HELD" if len(better[best]) >= cfg["min_families"] else "NOT HELD",
            "evidence": {"reference": ref, "level": lvl, "families_better_than_reference": better, "families_worse_than_reference": worse, "families_where_level_not_reached": unreached}}


def g6(d: Path, c_a, c_b, c_c):
    o = _load(d, "ood.json")
    if o is None:
        return [{"id": i, "claim": c["claim"], "verdict": "NOT RUN", "evidence": {}} for i, c in (("G6a", c_a), ("G6b", c_b), ("G6c", c_c))]
    ref = o["reference"]
    v = o["variants"][ref]
    a = v["foreign_FD003_late"]["max_sim"]
    far, same = v["chimera_channel_far"]["max_sim"], v["chimera_channel_same"]["max_sim"]
    gauss = v["gauss"]["max_sim"]
    return [
        {"id": "G6a", "claim": c_a["claim"], "verdict": "HELD" if a["lo"] >= c_a["min_auroc_lower_ci"] else "NOT HELD", "evidence": {"reference": ref, "auroc": a}},
        {"id": "G6b", "claim": c_b["claim"], "verdict": "HELD" if (far["auroc"] >= c_b["min_auroc_far"] and far["lo"] > same["hi"]) else "NOT HELD",
         "evidence": {"reference": ref, "far": far, "same": same, "mid": v["chimera_channel_mid"]["max_sim"]}},
        {"id": "G6c", "claim": c_c["claim"], "verdict": "HELD" if gauss["auroc"] >= c_c["min_auroc_gauss"] else "NOT HELD", "evidence": {"reference": ref, "auroc": gauss}},
    ]


def g7(d: Path, cfg):
    c = _load(d, "completion.json")
    if c is None:
        return {"id": "G7", "claim": cfg["claim"], "verdict": "NOT RUN", "evidence": {}}
    ref = c["reference"]
    key = str(cfg["severity"])
    r = c["variants"][f"{ref}_mask"][key]
    imp = c["variants"]["mean_imputation"][key]
    return {"id": "G7", "claim": cfg["claim"], "verdict": "HELD" if r["nmse"] <= cfg["max_nmse"] else "NOT HELD",
            "evidence": {"variant": f"{ref}_mask", "nmse": r, "mean_imputation_nmse": imp, "missing_channels": c["missing_channels"][key], "channels": c["channels"]}}


def evaluate(results_dir, gates_path=None, protocol_path=None, data_sums=None):
    d = Path(results_dir)
    cfg = tomllib.loads(Path(gates_path or HERE / "gates.toml").read_text())
    pp = Path(protocol_path or HERE / "PROTOCOL.md")
    sha = hashlib.sha256(pp.read_bytes()).hexdigest() if pp.exists() else None
    rows = [g0_provenance(d, sha, data_sums), g1(d, cfg["g1"]), g2(d, cfg["g2"])]
    g3_rows = g3(d, cfg["g3"])
    rows += g3_rows
    rows.append(g4(d, cfg["g4"], g3_rows))
    rows.append(g5(d, cfg["g5"]))
    rows += g6(d, cfg["g6a"], cfg["g6b"], cfg["g6c"])
    rows.append(g7(d, cfg["g7"]))
    return rows


if __name__ == "__main__":
    import sys
    for r in evaluate(sys.argv[1] if len(sys.argv) > 1 else HERE.parents[1] / "results" / "pdm_cmapss"):
        print(f"{r['id']:<14} {r['verdict']:<14} {r['claim']}")
