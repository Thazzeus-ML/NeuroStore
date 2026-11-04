"""Capacity and storage-density experiment (Section 3: ``C_max`` and ``eta``).

Question: how many items fit in a *fixed-size* memory before crosstalk pushes the bit error
rate above 0.1%, and how much information is stored per allocated bit?

Memories compared, all at the same dimension ``D``:

``linear_matrix``
    Classical Hebbian ``D x D`` matrix ``W = V^T K / D`` storing N (key, value) pairs.
    Raw ``BER = Q(sqrt(D/(N-1)))`` -- closed form, so the measured capacity is checked.
``vsa_bundle``
    N role-filler pairs superposed in ONE ``D``-vector. Reading one filler *bit* back has SNR
    ``1/sqrt(N-1)``, independent of ``D``, so the raw BER exceeds 0.1% already at N=2; what
    ``D`` buys is reliable *cleanup* against a codebook of ``V`` fillers. Capacity is therefore
    defined on the cleaned-up output (BER = SER/2) and compared with
    :func:`neurostore.theory.holographic_capacity`.
``modern_hopfield``
    Explicit key/value store with softmax readout. It stores N patterns in N x D cells (no
    compression), so crosstalk only matters through the softmax blend: capacity is reported at
    query-noise levels, and is *censored* ("at least N") when no error appears at the largest
    store tested.

Statistical power: resolving a 0.1% bit error rate needs on the order of 10^5 bits (linear matrix) and
10^4 symbols (bundle, whose target is a 0.2% symbol error rate), set by ``min_bits`` / ``min_symbols``.

``eta = information bits / allocated memory bits`` with allocated memory = everything the
structure must keep to answer a query (not the query cue itself).

Tables: ``capacity_curves``, ``capacity_summary``.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .. import theory
from ..common import flip_exact, limit_threads, make_rng, noise_to_flips, random_bipolar, threshold_crossing
from ..primitives import LinearAssociativeMemory, ModernHopfieldMemory
from ..results import WorkloadResult
from .base import Log, eta, headline, null_log

KEY = "capacity"
TITLE = "Capacity bound C_max and storage density eta"


def _grid(n_theory: int, lo: float = 0.3, hi: float = 2.2, points: int = 11) -> list[int]:
    """Geometric N grid bracketing the predicted capacity on both sides."""
    return sorted({max(2, int(round(x))) for x in np.geomspace(max(2.0, lo * n_theory), max(4.0, hi * n_theory), points)})


def _linear_ber(dim: int, n: int, rng: np.random.Generator, min_bits: int, max_trials: int) -> tuple[float, int, int]:
    errs = bits = trials = 0
    while trials < max_trials and (bits < min_bits or trials < 2):  # at least 2 independent key sets
        k = random_bipolar(rng, n, dim)
        v = random_bipolar(rng, n, dim)
        mem = LinearAssociativeMemory(dim)
        mem.store(k, v)
        out = mem.retrieve(k)
        errs += int((out != v).sum())
        bits += v.size
        trials += 1
    return errs / bits, bits, trials


def _vsa_bundle(dim: int, n: int, v_size: int, rng: np.random.Generator, min_symbols: int, max_trials: int) -> dict[str, float]:
    raw_err = raw_bits = sym_err = sym = trials = 0.0
    codebook = random_bipolar(rng, v_size, dim).astype(np.float32)
    while trials < max_trials and (sym < min_symbols or trials < 2):
        roles = random_bipolar(rng, n, dim).astype(np.float32)
        idx = rng.integers(0, v_size, n)
        fillers = codebook[idx]
        trace = (roles * fillers).sum(axis=0)
        unbound = roles * trace[None, :]  # each row = filler_i + noise from the other n-1 pairs
        s = np.sign(unbound)
        raw_err += float(np.where(s == 0, 0.5, (s != fillers)).sum())
        raw_bits += fillers.size
        pred = (unbound @ codebook.T).argmax(axis=1)
        sym_err += float((pred != idx).sum())
        sym += n
        trials += 1
    ser = sym_err / sym
    return {"ber_raw": raw_err / raw_bits, "ser": ser, "ber_clean": 0.5 * ser, "symbols": sym, "trials": trials}


def _hopfield_ber(dim: int, n: int, noise: float, beta: float, rng: np.random.Generator, n_queries: int) -> float:
    keys, values = random_bipolar(rng, n, dim), random_bipolar(rng, n, dim)
    mem = ModernHopfieldMemory(dim, beta)
    mem.store(keys, values)
    pick = rng.integers(0, n, n_queries)
    q = flip_exact(rng, keys[pick], noise_to_flips(dim, noise))
    return float((mem.retrieve(q) != values[pick]).mean())


def _cmax(ns: list[int], bers: list[float], target: float) -> dict[str, float | None]:
    """Largest N with BER <= target (monotone envelope, linear interpolation to the crossing)."""
    tc = threshold_crossing(ns, [-b for b in bers], -target)
    return {"last_ok": tc["last_ok"], "crossing": tc["crossing"]}


def run(cfg: dict[str, Any], seed: int = 0, profile: str = "custom", include_external: bool = True, log: Log = null_log) -> WorkloadResult:
    res = WorkloadResult(workload=KEY, title=TITLE, profile=profile, seed=seed, params=dict(cfg))
    t0 = time.perf_counter()
    target, v_size = cfg["ber_target"], cfg["vsa_codebook"]
    curves: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    with limit_threads(cfg.get("threads", 1)):
        for di, dim in enumerate(cfg["dims"]):
            log(f"[capacity] D={dim}")
            # ---------------------------------------------------------------- linear matrix
            n_th = theory.linear_memory_capacity(dim, target)
            ns, bers = [], []
            for n in _grid(n_th):
                ber, bits, trials = _linear_ber(dim, n, make_rng([seed, 40, di, n]), cfg["min_bits"], cfg["max_trials"])
                ns.append(n)
                bers.append(ber)
                curves.append({"dim": dim, "memory": "linear_matrix", "noise": 0.0, "n_items": n, "ber": ber, "ber_theory": theory.linear_memory_ber(dim, n), "bits": bits, "trials": trials})
            cm = _cmax(ns, bers, target)
            n_eff = int(cm["last_ok"] or 0)
            alloc = dim * dim * 4
            summary.append({"dim": dim, "memory": "linear_matrix", "noise": 0.0, "metric": "raw bit error rate", "c_max_last_ok": cm["last_ok"],
                            "c_max_interpolated": cm["crossing"], "c_max_theory": n_th, "censored": False,
                            "info_bits": n_eff * 2 * dim, "allocated_bytes": alloc, "eta": eta(n_eff * 2 * dim, alloc)})
            # ---------------------------------------------------------------- VSA bundle
            n_th_v = theory.holographic_capacity(dim, v_size, 1.0 - 2 * target)
            ns, bers_c, bers_r = [], [], []
            for n in _grid(max(n_th_v, 4)):
                r = _vsa_bundle(dim, n, v_size, make_rng([seed, 41, di, n]), cfg["min_symbols"], cfg["vsa_max_trials"])
                ns.append(n)
                bers_c.append(r["ber_clean"])
                bers_r.append(r["ber_raw"])
                curves.append({"dim": dim, "memory": "vsa_bundle", "noise": 0.0, "n_items": n, "ber": r["ber_clean"], "ber_raw": r["ber_raw"], "ser": r["ser"],
                               "ber_theory": 0.5 * (1.0 - theory.holographic_p_correct(dim, n, v_size)), "bits": r["symbols"], "trials": r["trials"]})
            cm = _cmax(ns, bers_c, target)
            n_eff = int(cm["last_ok"] or 0)
            alloc = dim * 4 + v_size * dim * 4  # trace + filler codebook
            summary.append({"dim": dim, "memory": "vsa_bundle", "noise": 0.0, "metric": "bit error rate after cleanup (SER/2)",
                            "c_max_last_ok": cm["last_ok"], "c_max_interpolated": cm["crossing"], "c_max_theory": n_th_v, "censored": False,
                            "info_bits": n_eff * math.log2(v_size), "allocated_bytes": alloc, "eta": eta(n_eff * math.log2(v_size), alloc)})
            summary.append({"dim": dim, "memory": "vsa_bundle_raw", "noise": 0.0, "metric": "raw bit error rate (no cleanup)",
                            "c_max_last_ok": 1 if bers_r and bers_r[0] > target else None, "c_max_interpolated": None, "c_max_theory": 1,
                            "censored": False, "info_bits": None, "allocated_bytes": dim * 4, "eta": None,
                            "raw_ber_at_smallest_n": bers_r[0], "smallest_n": ns[0]})
            # ---------------------------------------------------------------- explicit Hopfield
            for noise in cfg["hopfield_noise"]:
                ns, bers = [], []
                for n in cfg["hopfield_n"]:
                    ber = _hopfield_ber(dim, n, noise, cfg["beta"], make_rng([seed, 42, di, n, int(noise * 100)]), 200)
                    ns.append(n)
                    bers.append(ber)
                    curves.append({"dim": dim, "memory": "modern_hopfield", "noise": noise, "n_items": n, "ber": ber, "bits": 200 * dim, "trials": 1})
                cm = _cmax(ns, bers, target)
                censored = cm["crossing"] is None and cm["last_ok"] == ns[-1]
                n_eff = int(cm["last_ok"] or 0)
                alloc = 2 * n_eff * dim * 4 + n_eff * 4 if n_eff else 0
                summary.append({"dim": dim, "memory": "modern_hopfield", "noise": noise, "metric": "bit error rate of recalled value",
                                "c_max_last_ok": cm["last_ok"], "c_max_interpolated": cm["crossing"], "c_max_theory": None, "censored": bool(censored),
                                "info_bits": n_eff * 2 * dim, "allocated_bytes": alloc, "eta": eta(n_eff * 2 * dim, alloc) if alloc else None})
    res.tables["capacity_curves"] = curves
    res.tables["capacity_summary"] = summary
    _checks_and_headlines(cfg, res)
    res.notes += [
        f"C_max = largest N with BER <= {target:g} on the evaluated grid (monotone envelope); 'interpolated' is the linear crossing between grid points.",
        "linear_matrix and vsa_bundle have a hard capacity; modern_hopfield trades memory for capacity (N x D cells) so its C_max is censored by the largest store tested.",
        "vsa_bundle raw BER is >= 25% from N=2 because one bit sees SNR 1/sqrt(N-1) regardless of D; only cleanup against a codebook is reliable.",
        "eta counts information bits (N pairs x 2D for pair memories, N x log2(V) for the bundle) over all bytes the structure must keep; float32 storage is why eta << 1.",
    ]
    res.elapsed_s = time.perf_counter() - t0
    return res


def _checks_and_headlines(cfg: dict[str, Any], res: WorkloadResult) -> None:
    summ = res.tables["capacity_summary"]
    lin = [s for s in summ if s["memory"] == "linear_matrix"]
    rel = max(abs(s["c_max_interpolated"] - s["c_max_theory"]) / s["c_max_theory"] for s in lin if s["c_max_interpolated"])
    res.add_check("Benchmark self-check: measured linear-memory capacity matches Q(sqrt(D/(N-1))) <= 0.1%", "within 20% of closed form for every D",
                  f"worst relative error {rel * 100:.1f}%", rel <= 0.20)
    raw = [s for s in summ if s["memory"] == "vsa_bundle_raw"]
    worst = min(s["raw_ber_at_smallest_n"] for s in raw)
    res.add_check("A superposed bundle cannot be read bit-by-bit: raw BER at N=2 already exceeds 0.1%", "BER(N=2) > 0.1%", f"{worst * 100:.1f}%", worst > cfg["ber_target"],
                  note="Capacity of the bundle exists only after cleanup against a codebook.")
    vs = [s for s in summ if s["memory"] == "vsa_bundle" and s["c_max_interpolated"]]
    if vs:
        rel_v = max(abs(s["c_max_interpolated"] - s["c_max_theory"]) / s["c_max_theory"] for s in vs)
        res.add_check("Benchmark self-check: measured cleaned-up bundle capacity matches the closed-form bound", "within 35% for every D", f"worst relative error {rel_v * 100:.1f}%", rel_v <= 0.35,
                      note="Looser tolerance: the bound sits on the steep part of a noisy low-BER curve.")
    dim0 = cfg["dims"][len(cfg["dims"]) // 2]
    ctx = f"D={dim0}, BER <= {cfg['ber_target'] * 100:g}%"
    caps = {}
    for s in summ:
        if s["dim"] == dim0 and s["memory"] in ("linear_matrix", "vsa_bundle") and s["c_max_last_ok"]:
            caps[s["memory"]] = (float(s["c_max_last_ok"]), "neurostore")
    for s in summ:
        if s["dim"] == dim0 and s["memory"] == "modern_hopfield" and s["noise"] == cfg["hopfield_noise"][-1] and s["c_max_last_ok"]:
            caps[f"modern_hopfield(noise {s['noise'] * 100:g}%){' >=' if s['censored'] else ''}"] = (float(s["c_max_last_ok"]), "neurostore")
    res.headlines = [
        headline("Capacity C_max (items stored before BER > 0.1%)", "items", True, ctx, caps),
        headline("Storage density eta at C_max (information bits / allocated bits)", "ratio", True, ctx,
                 {s["memory"]: (s["eta"], "neurostore") for s in summ if s["dim"] == dim0 and s["eta"] and s["noise"] == 0.0}),
    ]
    res.summary = {"linear_capacity_rel_err": rel}
