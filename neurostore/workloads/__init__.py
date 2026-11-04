"""Workload suites. Each module exposes ``KEY``, ``TITLE`` and
``run(cfg, seed, profile, include_external, log) -> WorkloadResult``."""
from . import a_telemetry, b_entity, c_prompt_cache, capacity, cim_emulation, d_graph_walk

WORKLOADS = {m.KEY: m for m in (a_telemetry, b_entity, c_prompt_cache, d_graph_walk, capacity, cim_emulation)}

__all__ = ["WORKLOADS"]
