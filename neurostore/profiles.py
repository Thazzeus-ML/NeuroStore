"""Benchmark profiles: how large and how long each workload runs.

* ``quick``    -- seconds; used by the test-suite and for smoke runs. Numbers are *not* meaningful
                 statistically (tiny stores, few trials) but every code path is exercised.
* ``standard`` -- minutes on a laptop-class CPU; the reference profile for committed results.
* ``full``     -- larger stores, more trials and the 10^5-prompt cache; tens of minutes.

Any key can be overridden from a JSON config (``neurostore run --config cfg.json``), e.g.
``{"a": {"trials": 500}, "c": {"n_cache": [1000]}}``.
"""
from __future__ import annotations

import copy
from typing import Any

PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "quick": {
        "a": dict(dims=[256, 512], n_items=40, noise_step=0.1, noise_max=0.5, trials=80, beta=20.0, iter_steps=3, recall_target=0.99,
                  beta_sweep=[4, 20], beta_sweep_dim=256, beta_noise_step=0.1, lsh_presets=[[8, 32]],
                  scale_items=[40, 400], scale_dim=256, scale_noise_step=0.1, scale_trials=80, hnsw_ef_sweep=[128],
                  latency_dim=256, latency_items=[40, 400], latency_noise=0.1, n_read_samples=30, n_write_samples=30, threads=1),
        "b": dict(n_fields=6, vocab=60, n_records=300, dims=[256, 1024], k_values=[2, 3], corrupt=[0, 1], n_queries=60,
                  composite_k=[2], latency_dim=1024, latency_k=3, n_latency_queries=20, threads=1),
        "c": dict(dim=64, n_topics=8, topic_rho=0.5, para_cos=0.9, n_cache=[300, 1000], n_pos=100, n_neg=100, n_calib=100,
                  flyhash=[[1024, 32]], simhash_bits=[1024], target_fpr=0.05, n_latency=20, batch=64,
                  sweep_n_cache=1000, para_sweep=[0.9, 0.7], flyhash_sweep=[[1024, 32, 0.1], [1024, 64, 0.1]],
                  flyhash_sweep_para=[0.7], threads=1),
        "d": dict(n_entities=30, n_relations=8, dims=[512, 1024], edges=[30, 60], algebras=["map", "hrr"], n_graph_samples=1,
                  hops=[1, 2, 3], n_walks=60, accuracy_target=0.99, selfcheck_tol=0.15, speed_dims=[512], speed_entities=[30, 100],
                  speed_batch=32, speed_reps=5, threads=1),
        "capacity": dict(dims=[256, 512], ber_target=1e-3, min_bits=20000, max_trials=6, min_symbols=1500, vsa_max_trials=60,
                         vsa_codebook=50, hopfield_noise=[0.0, 0.2], hopfield_n=[20, 200], beta=20.0, threads=1),
        "cim": dict(dims=[256], n_keys=100, n_queries=60, query_noise=[0.0, 0.3],
                    read_noise=[0.0, 0.1, 1.0], adc_bits=[3, 8], adc_fs_fractions=[1.0], program_noise=[0.0, 0.3],
                    stuck_fraction=[0.0, 0.2], drift_time_ratio=[1e4], drift_nu=0.05, drift_nu_sigma=0.02,
                    combined=dict(read_noise=0.02, adc_bits=8, program_noise=0.05, stuck_fraction=0.01, drift_nu=0.05,
                                  drift_time_ratio=1e4, drift_nu_sigma=0.01, drift_compensation=True),
                    latency_grid=[[100, 256], [1000, 256]], latency_presets=["optimistic", "conservative"], cpu_reps=5, threads=1),
    },
    "standard": {
        "a": dict(dims=[512, 2048, 4096], n_items=100, noise_step=0.01, noise_max=0.5, trials=2000, beta=20.0, iter_steps=5, recall_target=0.99,
                  beta_sweep=[1, 4, 8, 16, 20, 32, 64], beta_sweep_dim=512, beta_noise_step=0.02, lsh_presets=[[8, 64], [12, 32], [5, 256]],
                  scale_items=[100, 1000, 10000], scale_dim=512, scale_noise_step=0.01, scale_trials=1000, hnsw_ef_sweep=[256, 1024],
                  latency_dim=512, latency_items=[100, 10000], latency_noise=0.1, n_read_samples=300, n_write_samples=300, threads=1),
        "b": dict(n_fields=8, vocab=500, n_records=5000, dims=[512, 1024, 2048, 4096], k_values=[2, 3, 4, 6], corrupt=[0, 1],
                  n_queries=500, composite_k=[2, 4], latency_dim=4096, latency_k=3, n_latency_queries=200, threads=1),
        "c": dict(dim=128, n_topics=20, topic_rho=0.5, para_cos=0.9, n_cache=[1000, 10000, 50000], n_pos=1000, n_neg=1000,
                  n_calib=1000, flyhash=[[2048, 64]], simhash_bits=[2048, 512], target_fpr=0.01, n_latency=300, batch=256,
                  sweep_n_cache=10000, para_sweep=[0.95, 0.9, 0.8, 0.7, 0.6],
                  flyhash_sweep=[[2048, 64, 0.1], [2048, 102, 0.1], [4096, 128, 0.1], [4096, 205, 0.1], [8192, 256, 0.1], [8192, 410, 0.1], [2048, 64, 0.3]],
                  flyhash_sweep_para=[0.8, 0.7, 0.6], threads=1),
        "d": dict(n_entities=100, n_relations=32, dims=[2048, 4096, 8192, 16384], edges=[100, 200, 400, 800, 1600],
                  algebras=["map", "hrr"], n_graph_samples=2, hops=[1, 2, 3, 4, 5], n_walks=500, accuracy_target=0.99, selfcheck_tol=0.03,
                  speed_dims=[2048, 8192], speed_entities=[100, 1000, 10000], speed_batch=256, speed_reps=15, threads=1),
        "capacity": dict(dims=[512, 1024, 2048, 4096], ber_target=1e-3, min_bits=200000, max_trials=40, min_symbols=20000,
                         vsa_max_trials=600, vsa_codebook=100, hopfield_noise=[0.0, 0.2], hopfield_n=[100, 1000, 10000], beta=20.0,
                         threads=1),
        "cim": dict(dims=[512, 2048], n_keys=1000, n_queries=500, query_noise=[0.0, 0.2, 0.35],
                    read_noise=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0], adc_bits=[2, 3, 4, 5, 6, 8],
                    adc_fs_fractions=[1.0, 0.25], program_noise=[0.0, 0.02, 0.05, 0.1, 0.2, 0.4],
                    stuck_fraction=[0.0, 0.01, 0.05, 0.1, 0.2], drift_time_ratio=[1e2, 1e4, 1e6], drift_nu=0.05, drift_nu_sigma=0.02,
                    combined=dict(read_noise=0.02, adc_bits=8, program_noise=0.05, stuck_fraction=0.01, drift_nu=0.05,
                                  drift_time_ratio=1e4, drift_nu_sigma=0.01, drift_compensation=True),
                    latency_grid=[[100, 512], [1000, 512], [10000, 512], [100000, 512], [1000, 4096], [10000, 4096]],
                    latency_presets=["optimistic", "conservative"], cpu_reps=20, threads=1),
    },
}

# "full" = "standard" with larger stores / more trials / the 10^5 cache.
_full = copy.deepcopy(PROFILES["standard"])
_full["a"].update(trials=5000, n_items=100, latency_items=[100, 10000, 100000], scale_items=[100, 1000, 10000, 100000])
_full["b"].update(n_records=20000, n_queries=1000)
_full["c"].update(n_cache=[1000, 10000, 100000], n_pos=2000, n_neg=2000, n_calib=2000, sweep_n_cache=100000)
_full["d"].update(n_graph_samples=4, n_walks=1000)
_full["capacity"].update(hopfield_n=[100, 1000, 10000, 100000])
_full["cim"].update(n_keys=5000, n_queries=1000)
PROFILES["full"] = _full


def get_profile(name: str, overrides: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """Deep copy of a named profile with optional per-workload overrides applied."""
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    prof = copy.deepcopy(PROFILES[name])
    for key, patch in (overrides or {}).items():
        if key not in prof:
            raise KeyError(f"unknown workload {key!r} in overrides; choose from {sorted(prof)}")
        prof[key].update(patch)
    return prof
