"""Helpers shared by the workload modules."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from ..baselines.base import Unavailable
from ..results import WorkloadResult

Log = Callable[[str], None]


def null_log(_msg: str) -> None:
    return None


def noise_grid(step: float, max_noise: float) -> list[float]:
    """``[0, step, 2*step, ..., max_noise]`` without float drift."""
    n = int(round(max_noise / step))
    return [round(i * step, 6) for i in range(n + 1)]


@dataclass
class SystemSpec:
    """How to construct one system under test, plus how to tell if it can run here."""

    name: str
    make: Callable[[], Any]
    available: Callable[[], tuple[bool, str]] = lambda: (True, "")
    family: str = "other"


def spec_from_class(cls: type, *args: Any, name: str | None = None, **kwargs: Any) -> SystemSpec:
    probe = cls.available if hasattr(cls, "available") else (lambda: (True, ""))
    return SystemSpec(name or getattr(cls, "name", cls.__name__), lambda: cls(*args, **kwargs), probe, getattr(cls, "family", "other"))


def instantiate(spec: SystemSpec, result: WorkloadResult, log: Log = null_log) -> Any | None:
    """Create a system, recording a skip (not an exception) when it is unavailable."""
    ok, why = spec.available()
    if not ok:
        result.add_skip(spec.name, why)
        log(f"  skip {spec.name}: {why}")
        return None
    try:
        return spec.make()
    except Unavailable as exc:
        result.add_skip(spec.name, str(exc))
        log(f"  skip {spec.name}: {exc}")
        return None


class Stopwatch:
    def __enter__(self) -> "Stopwatch":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self.t0


def info_bits(n_items: int, bits_per_item: float) -> float:
    return float(n_items) * float(bits_per_item)


def eta(info_bits_total: float, allocated_bytes: float) -> float:
    """Storage density efficiency: information bits / allocated memory bits."""
    return float(info_bits_total) / (float(allocated_bytes) * 8.0) if allocated_bytes else float("nan")


def headline(
    metric: str,
    unit: str,
    higher_is_better: bool,
    context: str,
    systems: dict[str, tuple],
    kind: str = "quality",
    gate: str | None = None,
) -> dict[str, Any]:
    """Build a scorecard entry.

    ``systems`` maps name -> ``(value, family)`` or ``(value, family, passes_gate)``.

    ``kind`` is ``'quality'`` (accuracy / robustness: the thing being bought) or ``'cost'`` (latency, throughput, memory:
    what it costs). A cost comparison is only meaningful between systems that actually do the job, so cost headlines carry
    a ``gate`` -- a one-line description of the accuracy requirement -- and each system a ``passes_gate`` flag; the
    scorecard compares only systems that pass, which stops a system from "winning" on speed by returning nothing and stops
    NeuroStore from being represented by whichever of its variants happens to be fastest while another one provides the accuracy.
    """
    entries = {}
    for name, tup in systems.items():
        entries[name] = {"value": tup[0], "family": tup[1], "passes": (bool(tup[2]) if len(tup) > 2 and tup[2] is not None else None)}
    return {"metric": metric, "unit": unit, "higher_is_better": higher_is_better, "context": context, "kind": kind, "gate": gate, "systems": entries}
