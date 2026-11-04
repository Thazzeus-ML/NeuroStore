#!/usr/bin/env python3
"""Execute the v2 notebook through a real Jupyter kernel on the local JAX backend and record what happened.

Writes colab/validation_cpu.json (or --out): per run the per-cell wall time, any cell errors, the number of result rows,
the self-check table, the findings text and the number of saved plots. This is the evidence for "what was validated"
in colab/REVIEW.md; it says nothing about GPU/TPU behaviour unless run on such a machine.

usage: python colab/validate_v2.py [--runs smoke multidev quick] [--out colab/validation_cpu.json] [--workdir DIR]
needs: jax, nbclient, nbformat, ipykernel, pandas, matplotlib
"""
from __future__ import annotations
import argparse, json, os, re, sys, tempfile, time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nbtools  # noqa: E402

RUNS = {
    "smoke": dict(profile="smoke", sets={}, env={}),
    "multidev": dict(profile="smoke", sets={"SIMULATE_CPU_DEVICES": "4"}, env={}),     # virtual CPU devices exercise Section 10
    "quick": dict(profile="quick", sets={}, env={}),
}


def apply_sets(nb, sets):
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        for name, val in sets.items():
            c.source = re.sub(rf"^({re.escape(name)}\s*=\s*)[^#\n]*", lambda m: m.group(1) + val + " ", c.source, flags=re.M)


def run_one(name, spec, workdir):
    import nbformat
    from nbclient import NotebookClient
    nb_path = HERE / "NeuroStore_Colab_Benchmarks_v2.ipynb"
    nbtools.build(HERE / "src", nb_path, "NeuroStore GPU/TPU benchmark lab (v2)")
    nb = nbformat.read(nb_path, as_version=4)
    out_root = Path(workdir) / name
    out_root.mkdir(parents=True, exist_ok=True)
    apply_sets(nb, {"PROFILE": json.dumps(spec["profile"]), "OUTPUT_ROOT": json.dumps(str(out_root)), **spec["sets"]})
    for k, v in spec["env"].items():
        os.environ[k] = v
    os.environ.setdefault("MPLBACKEND", "Agg")
    client = NotebookClient(nb, timeout=7200, kernel_name="python3", allow_errors=True, resources={"metadata": {"path": str(out_root)}})
    t0 = time.time()
    client.execute()
    total = time.time() - t0
    cells, errors = [], []
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        md = c.get("metadata", {}).get("execution", {})
        dur = None
        if md.get("iopub.execute_input") and md.get("shell.execute_reply"):
            f = lambda x: datetime.fromisoformat(x.replace("Z", "+00:00"))
            dur = (f(md["shell.execute_reply"]) - f(md["iopub.execute_input"])).total_seconds()
        for o in c.get("outputs", []):
            if o.output_type == "error":
                errors.append({"cell": c.id, "error": f"{o.ename}: {o.evalue[:300]}"})
        cells.append({"cell": c.id, "seconds": dur})
    rundir = max((p for p in out_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, default=None)
    summary = {"name": name, "profile": spec["profile"], "overrides": spec["sets"], "wall_seconds": total, "cells": cells, "errors": errors}
    if rundir is not None:
        import pandas as pd
        res = pd.read_csv(rundir / "results.csv")
        chk = pd.read_csv(rundir / "checks.csv") if (rundir / "checks.csv").exists() and (rundir / "checks.csv").stat().st_size > 2 else pd.DataFrame(columns=["name", "ok", "detail"])
        meta = json.loads((rundir / "hardware_metadata.json").read_text())
        summary.update({"rows": int(len(res)), "columns": int(res.shape[1]), "experiments": sorted(res.experiment.unique().tolist()),
                        "plots": len(list((rundir / "plots").glob("*.png"))),
                        "checks": [{"name": r["name"], "ok": bool(r["ok"]), "detail": str(r["detail"])} for _, r in chk.iterrows()],
                        "findings": [l[2:] for l in (rundir / "findings.txt").read_text().splitlines() if l.startswith("- ")],
                        "environment": {k: meta.get(k) for k in ("backend", "device_kind", "local_device_count", "jax", "jaxlib", "numpy", "pandas", "python", "config_hash")}})
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["smoke", "multidev"], choices=list(RUNS))
    ap.add_argument("--out", default=str(HERE / "validation_cpu.json"))
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    workdir = a.workdir or tempfile.mkdtemp(prefix="ns_validate_")
    results = [run_one(n, RUNS[n], workdir) for n in a.runs]
    Path(a.out).write_text(json.dumps({"runs": results}, indent=1, default=str) + "\n")
    bad = False
    for r in results:
        nchk = len(r.get("checks", [])); npass = sum(c["ok"] for c in r.get("checks", []))
        print(f"{r['name']:9s} profile={r['profile']:8s} wall={r['wall_seconds']:7.1f}s errors={len(r['errors'])} rows={r.get('rows')} checks={npass}/{nchk} plots={r.get('plots')}")
        for e in r["errors"]:
            print("   ERROR", e)
        bad |= bool(r["errors"]) or npass != nchk
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
