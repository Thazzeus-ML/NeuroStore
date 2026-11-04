"""Gate thresholds are read from gates.toml (committed before the experiments ran)."""
from __future__ import annotations
import json, tomllib, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_gates() -> dict:
    with open(ROOT / "gates.toml", "rb") as f:
        return tomllib.load(f)


def gate_commit() -> str:
    """Commit that last touched gates.toml (recorded in every result file)."""
    try:
        return subprocess.check_output(["git", "log", "-1", "--format=%h", "--", "gates.toml"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_result(name: str, obj: dict) -> Path:
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    obj = dict(obj); obj["gates_commit"] = gate_commit()
    p = out / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float) + "\n")
    return p


def read_result(name: str) -> dict:
    return json.loads((ROOT / "results" / f"{name}.json").read_text())
