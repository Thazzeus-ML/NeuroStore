"""Section 5 -- compute-in-memory emulation: accuracy under analogue non-idealities, and a
*modelled* latency.

Two questions, answered with different kinds of evidence:

1. **Does hyperdimensional search survive analogue hardware?** (*emulated, checkable*)
   Keys are programmed into a tiled differential-pair crossbar; each query is one analogue
   read per tile, digitised per tile and summed digitally. One non-ideality at a time is swept
   (thermal/read noise, ADC resolution, programming error, stuck cells, conductance drift with
   and without compensation), plus one combined "realistic" scenario, at several query-noise
   levels and dimensions. Metric: top-1 identification accuracy and exact value recall,
   relative to the ideal digital result.
2. **How fast could it be?** (*modelled, NOT measured*)
   :class:`~neurostore.primitives.cim.CIMTimingModel` projects search latency from explicit
   device-timing assumptions and is compared with the **measured** time of the best CPU
   implementation here (FAISS binary flat when installed, else NumPy float32 BLAS or packed XOR+popcount; single thread). The
   ratio mixes a measurement with a projection, so it is a *what-if under stated assumptions*,
   reported for two assumption sets (a range, not a point), and excluded from the scorecard.

Tables: ``ideal_check``, ``tolerance``, ``tolerance_limits``, ``latency_model``.
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np

from ..common import (
    flip_exact,
    limit_threads,
    make_rng,
    noise_to_flips,
    pack_bipolar,
    popcount_rows,
    proportion,
    random_bipolar,
    summarize_latency,
    threshold_crossing,
    time_calls,
)
from ..primitives import CIMAssociativeMemory, CIMTimingModel, CrossbarConfig
from ..results import WorkloadResult
from .base import Log, null_log

KEY = "cim"
TITLE = "Section 5 - Compute-in-memory emulation (accuracy emulated, latency modelled)"

IDEAL = CrossbarConfig(adc_bits=None)


def _sweeps(cfg: dict[str, Any]) -> list[tuple[str, Any, str, CrossbarConfig]]:
    """``(parameter, value, variant label, crossbar config)`` -- one non-ideality at a time."""
    out: list[tuple[str, Any, str, CrossbarConfig]] = [("ideal", 0, "", IDEAL)]
    out += [("read_noise", v, "ideal ADC", replace(IDEAL, read_noise=v)) for v in cfg["read_noise"] if v]
    for fs in cfg["adc_fs_fractions"]:
        out += [("adc_bits", b, f"fs={fs:g}", replace(IDEAL, adc_bits=b, adc_fs_fraction=fs)) for b in cfg["adc_bits"]]
    out += [("program_noise", v, "ideal ADC", replace(IDEAL, program_noise=v)) for v in cfg["program_noise"] if v]
    out += [("stuck_fraction", v, "ideal ADC", replace(IDEAL, stuck_fraction=v)) for v in cfg["stuck_fraction"] if v]
    for ratio in cfg["drift_time_ratio"]:
        for comp in (False, True):
            c = replace(CrossbarConfig(adc_bits=8), drift_nu=cfg["drift_nu"], drift_nu_sigma=cfg["drift_nu_sigma"],
                        drift_time_ratio=ratio, drift_compensation=comp)
            out.append(("drift_t_over_t0", ratio, f"8-bit ADC, {'compensated' if comp else 'uncompensated'}", c))
    out.append(("combined_realistic", 1, "see params.combined", CrossbarConfig(**cfg["combined"])))
    return out


def minimal_resolution(series: list[dict[str, Any]], target: float = 0.99) -> float | None:
    """Smallest swept value from which EVERY larger swept value also reaches ``target`` (no credit for a lucky low point)."""
    series = sorted(series, key=lambda r: r["value"])
    return next((series[i]["value"] for i in range(len(series)) if all(x["ident_acc"] >= target for x in series[i:])), None)


def _tolerance(cfg: dict[str, Any], seed: int, res: WorkloadResult, log: Log) -> None:
    rows: list[dict[str, Any]] = []
    sweeps = _sweeps(cfg)
    n_keys, n_q = cfg["n_keys"], cfg["n_queries"]
    for di, dim in enumerate(cfg["dims"]):
        log(f"[CIM] tolerance sweeps: D={dim}, N={n_keys}")
        rng = make_rng([seed, 50, di])
        keys, values = random_bipolar(rng, n_keys, dim), random_bipolar(rng, n_keys, dim)
        true_idx = np.arange(n_q) % n_keys
        for qi, qn in enumerate(cfg["query_noise"]):
            q = flip_exact(make_rng([seed, 51, di, qi]), keys[true_idx], noise_to_flips(dim, qn))
            for si, (param, value, variant, xcfg) in enumerate(sweeps):
                mem = CIMAssociativeMemory(keys, values, replace(xcfg, seed=int(make_rng([seed, 52, di, qi, si]).integers(0, 2**31 - 1))))
                ident = int((mem.identify(q) == true_idx).sum())
                exact = int((mem.retrieve(q) == values[true_idx]).all(axis=1).sum())
                pi, pv = proportion(ident, n_q), proportion(exact, n_q)
                rows.append({"dim": dim, "n_keys": n_keys, "query_noise": qn, "param": param, "value": value, "variant": variant,
                             "ident_acc": pi["rate"], "ident_lo": pi["lo"], "ident_hi": pi["hi"], "value_acc": pv["rate"], "n": n_q})
    res.tables["tolerance"] = rows

    limits = []
    for dim in cfg["dims"]:
        for qn in cfg["query_noise"]:
            ideal = next(r for r in rows if r["dim"] == dim and r["query_noise"] == qn and r["param"] == "ideal")["ident_acc"]
            base = {"dim": dim, "query_noise": qn, "ideal_ident_acc": ideal}
            for param in ("read_noise", "program_noise", "stuck_fraction"):
                sel = sorted((r for r in rows if r["dim"] == dim and r["query_noise"] == qn and r["param"] == param), key=lambda r: r["value"])
                xs, ys = [0.0] + [r["value"] for r in sel], [ideal] + [r["ident_acc"] for r in sel]
                tc = threshold_crossing(xs, ys, 0.99) if ideal >= 0.99 else {"last_ok": None, "first_fail": None, "crossing": None}
                limits.append({**base, "param": param, "max_value_at_99": tc["last_ok"], "first_fail": tc["first_fail"], "unit": "fraction of tile full scale" if param == "read_noise" else "fraction"})
            for fs in cfg["adc_fs_fractions"]:
                sel = sorted((r for r in rows if r["dim"] == dim and r["query_noise"] == qn and r["param"] == "adc_bits" and r["variant"] == f"fs={fs:g}"), key=lambda r: r["value"])
                minimal = minimal_resolution(sel)
                limits.append({**base, "param": f"adc_bits (fs={fs:g})", "max_value_at_99": minimal if ideal >= 0.99 else None, "first_fail": None, "unit": "minimum bits for >=99% (and all larger)"})
    res.tables["tolerance_limits"] = limits


def _ideal_check(cfg: dict[str, Any], seed: int, res: WorkloadResult) -> None:
    rng = make_rng([seed, 53])
    dim = cfg["dims"][-1]
    keys, values = random_bipolar(rng, cfg["n_keys"], dim), random_bipolar(rng, cfg["n_keys"], dim)
    q = flip_exact(rng, keys[: cfg["n_queries"] % cfg["n_keys"] or 10], noise_to_flips(dim, 0.2))
    mem = CIMAssociativeMemory(keys, values, IDEAL)
    digital = (q.astype(np.float32) @ keys.T.astype(np.float32)) / dim
    err = float(np.abs(mem.scores(q) - digital).max())
    res.tables["ideal_check"] = [{"dim": dim, "max_abs_error": err}]
    res.add_check("Benchmark self-check: ideal crossbar (no noise, ideal ADC) reproduces the digital similarity scores", "max |analogue - digital| < 1e-4", f"{err:.2e}", err < 1e-4,
                  note="Validates the differential-pair mapping and the tiled partial-sum accumulation.")


def _cpu_us(n: int, dim: int, reps: int, rng: np.random.Generator) -> dict[str, Any]:
    """Single-thread CPU time (p50, us) of one nearest-key search, for every implementation available; the fastest is the
    fair denominator for any speed-up: NumPy float32 BLAS, NumPy packed XOR+popcount, and FAISS ``IndexBinaryFlat`` (SIMD)."""
    keys = random_bipolar(rng, n, dim)
    kf = keys.astype(np.float32)
    q = kf[0].copy()
    times = {"numpy_float32_blas": summarize_latency(time_calls(lambda _x: int((kf @ q).argmax()), [None] * reps, warmup=2))["p50_us"]}
    del kf
    kp, qp = pack_bipolar(keys), pack_bipolar(keys[0])
    times["numpy_packed_popcount"] = summarize_latency(time_calls(lambda _x: int(popcount_rows(kp ^ qp).argmin()), [None] * reps, warmup=2))["p50_us"]
    try:
        import faiss  # type: ignore

        idx = faiss.IndexBinaryFlat(kp.shape[1] * 64)
        idx.add(kp.view(np.uint8))
        qb = np.ascontiguousarray(qp.view(np.uint8)[None, :])
        times["faiss_binary_flat"] = summarize_latency(time_calls(lambda _x: idx.search(qb, 1), [None] * reps, warmup=2))["p50_us"]
    except ImportError:
        pass
    best = min(times, key=times.get)
    return {"cpu_float32_us": times["numpy_float32_blas"], "cpu_packed_us": times["numpy_packed_popcount"], "cpu_faiss_us": times.get("faiss_binary_flat"),
            "cpu_best_us": times[best], "cpu_best_impl": best}


def _latency(cfg: dict[str, Any], seed: int, res: WorkloadResult, log: Log) -> None:
    rows = []
    for n, dim in cfg["latency_grid"]:
        log(f"[CIM] latency model: N={n}, D={dim}")
        cpu = _cpu_us(n, dim, cfg["cpu_reps"], make_rng([seed, 54, n, dim]))
        for preset in cfg["latency_presets"]:
            m = CIMTimingModel.preset(preset)
            lat = m.search_latency(n, dim)
            rows.append({"n_keys": n, "dim": dim, **cpu, "preset": preset, "tiles": lat["tiles"], "waves": lat["waves"],
                         "t_array_ns": lat["t_array_ns"], "t_accum_ns": lat["t_accum_ns"], "t_wta_ns": lat["t_wta_ns"],
                         "modeled_ns": lat["total_ns"], "modeled_speedup_vs_cpu_best": cpu["cpu_best_us"] * 1e3 / lat["total_ns"],
                         "regime": lat["regime"], "kind": "MODELED (not measured)", "arrays_for_single_wave": lat["tiles"]})
    res.tables["latency_model"] = rows


def run(cfg: dict[str, Any], seed: int = 0, profile: str = "custom", include_external: bool = True, log: Log = null_log) -> WorkloadResult:
    res = WorkloadResult(workload=KEY, title=TITLE, profile=profile, seed=seed, params=dict(cfg))
    t0 = time.perf_counter()
    with limit_threads(cfg.get("threads", 1)):
        _ideal_check(cfg, seed, res)
        _tolerance(cfg, seed, res, log)
        _latency(cfg, seed, res, log)
    _checks(cfg, res)
    res.notes += [
        "ACCURACY rows are numerical emulation of a tiled differential-pair crossbar (per-tile read noise and ADC, static programming/stuck/drift errors); they are not silicon measurements.",
        "LATENCY rows are an analytical projection from the stated timing parameters (CIMTimingModel presets are illustrative placeholders); they are NOT measurements and are excluded from the scorecard.",
        "The CPU column is a real single-thread measurement of the fastest available implementation (FAISS binary flat when installed, else NumPy float32 BLAS or packed popcount); the speed-up column divides a measurement by a projection.",
        "The model covers the search stage only: no programming/write time or energy, no host I/O, and (as in real arrays) a size-independent analogue time only while all tiles fit the array budget.",
        "Read-noise sigma is relative to a tile's full-scale output (tile_rows x input_fs); with 256-row tiles a 2% setting means sigma = 5.1 in dot-product units against an impostor spread of sqrt(D).",
    ]
    res.elapsed_s = time.perf_counter() - t0
    return res


def _checks(cfg: dict[str, Any], res: WorkloadResult) -> None:
    rows = res.tables["tolerance"]
    dmax, qn = max(cfg["dims"]), cfg["query_noise"][min(1, len(cfg["query_noise"]) - 1)]
    comb = next((r for r in rows if r["dim"] == dmax and r["query_noise"] == qn and r["param"] == "combined_realistic"), None)
    ideal = next(r for r in rows if r["dim"] == dmax and r["query_noise"] == qn and r["param"] == "ideal")
    if comb:
        res.add_check(f"Combined realistic non-idealities keep >=99% identification (D={dmax}, query noise {qn * 100:g}%)", ">= 99%",
                      f"{comb['ident_acc'] * 100:.1f}% (ideal {ideal['ident_acc'] * 100:.1f}%)", comb["ident_acc"] >= 0.99,
                      note="Scenario: " + ", ".join(f"{k}={v}" for k, v in cfg["combined"].items()))
    lim = [x for x in res.tables["tolerance_limits"] if x["dim"] == dmax and x["query_noise"] == qn and x["param"] == "read_noise"]
    if lim and lim[0]["max_value_at_99"] is not None:
        res.add_check(f"Read-noise tolerance at D={dmax} (informational)", "report", (f"up to {lim[0]['max_value_at_99'] * 100:g}% of tile full scale on the swept grid" if lim[0]["max_value_at_99"] else "no non-zero swept level keeps 99%"), None)
    mods = res.tables["latency_model"]
    for preset in cfg["latency_presets"]:
        single = [r for r in mods if r["preset"] == preset and r["waves"] == 1]
        multi = [r for r in mods if r["preset"] == preset and r["waves"] > 1]
        res.add_check(f"Modelled O(1) regime holds only while tiles fit the array budget ({preset})", "report",
                      f"single wave up to N={max((r['n_keys'] for r in single), default=0)}; {len(multi)} swept size(s) need serialised waves", None,
                      note="Beyond n_arrays tiles the analogue time grows linearly with the number of waves.")
