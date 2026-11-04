"""Result containers and JSON (de)serialisation.

Every workload returns a :class:`WorkloadResult`. A full run is a :class:`SuiteResult`
(environment + workload results). Everything is plain data so a result file can be diffed,
archived, re-reported (``neurostore report``) and compared across machines.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / dataclass / path objects to JSON-safe values.

    NaN and +-inf become ``None`` so the output is strict JSON.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int,)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


@dataclass
class WorkloadResult:
    """Outcome of one workload.

    Attributes
    ----------
    tables:
        Named tables, each a list of flat row dicts (the raw measurements).
    summary:
        Small dict of headline numbers (thresholds, capacities, ...).
    headlines:
        Standardised comparison entries used to build the cross-workload scorecard::

            {"metric": str, "unit": str, "higher_is_better": bool, "context": str,
             "systems": {system_name: {"value": float, "family": "neurostore"|"kv"|"ann"|"lsh"|...}}}
    checks:
        Explicit pass/fail checks against the targets stated in the evaluation spec.
    skipped:
        Baselines that could not run (missing library, size limit, ...), with the reason.
    """

    workload: str
    title: str
    profile: str
    seed: int
    params: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    headlines: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    elapsed_s: float = 0.0

    SELFCHECK_PREFIX = "Benchmark self-check"

    def add_check(self, name: str, target: str, measured: str, passed: bool | None, note: str = "") -> None:
        """Record a check. Names starting with 'Benchmark self-check' validate the *framework*
        (measured vs closed-form theory, ideal-hardware equivalence, task well-posedness) rather
        than a claim about NeuroStore; ``neurostore run --strict`` fails if any of them fails."""
        kind = "selfcheck" if name.startswith(self.SELFCHECK_PREFIX) else ("info" if passed is None else "spec")
        self.checks.append({"name": name, "target": target, "measured": measured, "passed": passed, "note": note, "kind": kind})

    def add_skip(self, system: str, reason: str) -> None:
        entry = {"system": system, "reason": reason}
        if entry not in self.skipped:
            self.skipped.append(entry)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkloadResult":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class SuiteResult:
    """A complete benchmark run."""

    framework_version: str
    profile: str
    seed: int
    environment: dict[str, Any]
    workloads: list[WorkloadResult] = field(default_factory=list)
    created_utc: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    schema_version: int = SCHEMA_VERSION
    run_config: dict[str, Any] = field(default_factory=dict)

    def get(self, workload: str) -> WorkloadResult | None:
        for w in self.workloads:
            if w.workload == workload:
                return w
        return None

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["workloads"] = [w.to_dict() for w in self.workloads]
        return to_jsonable(d)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, allow_nan=False) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SuiteResult":
        data = json.loads(Path(path).read_text())
        workloads = [WorkloadResult.from_dict(w) for w in data.pop("workloads", [])]
        known = {f.name for f in dataclasses.fields(cls)}
        suite = cls(**{k: v for k, v in data.items() if k in known and k != "workloads"})
        suite.workloads = workloads
        return suite
