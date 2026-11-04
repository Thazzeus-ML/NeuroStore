"""Run a set of workloads and assemble a :class:`~neurostore.results.SuiteResult`."""
from __future__ import annotations

import time
import traceback
from typing import Any, Callable, Sequence

from . import __version__
from .common import environment_info
from .profiles import get_profile
from .results import SuiteResult, WorkloadResult
from .workloads import WORKLOADS

ORDER = ["a", "b", "c", "d", "capacity", "cim"]


def parse_workloads(spec: str | Sequence[str]) -> list[str]:
    """``'a,b'`` / ``['a','cim']`` / ``'all'`` -> validated, ordered workload keys."""
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    items = [i.strip().lower() for i in items if i.strip()]
    if not items or "all" in items:
        return list(ORDER)
    bad = [i for i in items if i not in WORKLOADS]
    if bad:
        raise ValueError(f"unknown workload(s) {bad}; choose from {ORDER + ['all']}")
    return [k for k in ORDER if k in items]


def run_suite(
    workloads: str | Sequence[str] = "all",
    profile: str = "standard",
    seed: int = 0,
    overrides: dict[str, dict[str, Any]] | None = None,
    include_external: bool = True,
    threads: int | None = None,
    log: Callable[[str], None] | None = None,
    on_result: Callable[[WorkloadResult], None] | None = None,
    fail_fast: bool = False,
) -> SuiteResult:
    """Execute workloads sequentially. A crashing workload is recorded (not fatal) unless ``fail_fast``."""
    keys = parse_workloads(workloads)
    cfgs = get_profile(profile, overrides)
    log = log or (lambda _m: None)
    suite = SuiteResult(framework_version=__version__, profile=profile, seed=seed, environment=environment_info(),
                        run_config={"workloads": keys, "include_external": include_external, "threads": threads, "overrides": overrides or {}})
    for key in keys:
        cfg = cfgs[key]
        if threads is not None:
            cfg["threads"] = threads
        log(f"=== workload {key} ({profile}) ===")
        t0 = time.perf_counter()
        try:
            res = WORKLOADS[key].run(cfg, seed=seed, profile=profile, include_external=include_external, log=log)
        except Exception as exc:  # noqa: BLE001 - record and continue so a long run is not lost
            if fail_fast:
                raise
            res = WorkloadResult(workload=key, title=WORKLOADS[key].TITLE, profile=profile, seed=seed, params=cfg,
                                 elapsed_s=time.perf_counter() - t0)
            res.notes.append(f"WORKLOAD CRASHED: {type(exc).__name__}: {exc}")
            res.add_check(f"Benchmark self-check: workload {key} completes", "no exception", f"{type(exc).__name__}: {exc}", False,
                          note=traceback.format_exc(limit=3))
        suite.workloads.append(res)
        if on_result:
            on_result(res)
    return suite


def selfcheck_failures(suite: SuiteResult) -> list[tuple[str, dict[str, Any]]]:
    return [(w.workload, c) for w in suite.workloads for c in w.checks if c.get("kind") == "selfcheck" and c.get("passed") is False]
