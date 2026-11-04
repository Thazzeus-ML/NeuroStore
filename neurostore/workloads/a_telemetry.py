"""Workload A -- noisy telemetry / edge-sensor key retrieval.

Question: when a stored key arrives corrupted by bit flips (lossy radio link, sensor jitter),
which systems still return the exact stored value, and at what latency / memory cost?

Protocol
--------
* ``N`` random bipolar ``D``-bit keys, each mapped to a random ``D``-bit value.
* For each noise level ``p`` in ``0 ... 50%`` every system receives the **same** queries: stored
  keys with exactly ``round(D p)`` random bit flips. Success = the system returns the stored
  value *exactly* (all ``D`` bits).
* Systems: Modern Hopfield (the system under test), exact brute-force NN (the accuracy
  ceiling), FAISS binary flat/HNSW, bit-sampling LSH, datasketch MinHash-LSH, Qdrant, and the
  exact-match stores Python dict / RocksDB / Redis.
* Alongside the measurements the closed-form brute-force identification probability
  (:func:`neurostore.theory.p_identify_bruteforce`) is reported, validating the injector.

Tables: ``recall_vs_noise``, ``thresholds``, ``scale_recall``, ``scale_thresholds``,
``beta_sweep``, ``beta_thresholds``, ``latency_memory``.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .. import theory
from ..baselines import (
    BitSamplingLSHSystem,
    BruteForceHammingSystem,
    DatasketchLSHSystem,
    DictKVSystem,
    FaissBinaryFlatSystem,
    FaissBinaryHNSWSystem,
    QdrantSystem,
    RedisKVSystem,
    RocksKVSystem,
)
from ..baselines.base import AssocSystem
from ..common import (
    flip_exact,
    limit_threads,
    make_rng,
    noise_to_flips,
    proportion,
    random_bipolar,
    summarize_latency,
    threshold_crossing,
    time_calls,
)
from ..primitives import ModernHopfieldMemory
from ..results import WorkloadResult
from .base import Log, SystemSpec, eta, headline, instantiate, noise_grid, null_log, spec_from_class

KEY = "a"
TITLE = "Workload A - Noisy telemetry / edge-sensor key retrieval"


class HopfieldSystem(AssocSystem):
    """Modern Hopfield memory behind the common Workload-A interface."""

    family = "neurostore"

    def __init__(self, beta: float = 20.0, steps: int = 1) -> None:
        self.beta, self.steps = float(beta), int(steps)
        self.name = f"hopfield_b{self.beta:g}" + (f"_s{self.steps}" if self.steps > 1 else "")
        self.label = f"Modern Hopfield (beta={self.beta:g}" + (f", {self.steps} attractor steps)" if self.steps > 1 else ", single step)")
        self.mem: ModernHopfieldMemory | None = None

    def build(self, keys: np.ndarray, values: np.ndarray) -> None:
        self.mem = ModernHopfieldMemory(keys.shape[1], self.beta)
        self.mem.store(keys, values)

    def retrieve_values(self, queries: np.ndarray) -> np.ndarray:
        return self.mem.retrieve(queries, steps=self.steps)

    def retrieve_ids(self, queries: np.ndarray) -> np.ndarray:
        return self.mem.retrieve_index(queries)

    def prepare_read(self, query: np.ndarray) -> np.ndarray:
        return query.astype(np.float32)

    def read_one(self, prepared: np.ndarray) -> Any:
        return self.mem.retrieve(prepared, steps=self.steps)

    def prepare_write(self, key: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (key.astype(np.float32), value.astype(np.float32))

    def write_one(self, prepared: tuple[np.ndarray, np.ndarray]) -> None:
        self.mem.add(*prepared)

    def memory_bytes(self) -> int:
        return self.mem.memory_bytes()


def _scale_specs(cfg: dict[str, Any], include_external: bool) -> list[SystemSpec]:
    """Systems that remain meaningful (and affordable) when the store grows to 10^4+ items."""
    specs = [
        SystemSpec(f"hopfield_b{cfg['beta']:g}", lambda: HopfieldSystem(cfg["beta"]), family="neurostore"),
        SystemSpec(f"hopfield_b{cfg['beta']:g}_s{cfg['iter_steps']}", lambda: HopfieldSystem(cfg["beta"], cfg["iter_steps"]), family="neurostore"),
        spec_from_class(BruteForceHammingSystem),
        *[spec_from_class(BitSamplingLSHSystem, k, L, name=f"lsh_k{k}_L{L}") for k, L in cfg["lsh_presets"]],
        spec_from_class(DictKVSystem),
    ]
    if include_external:
        specs.append(spec_from_class(FaissBinaryHNSWSystem))
    return specs


def _specs(cfg: dict[str, Any], include_external: bool) -> list[SystemSpec]:
    specs = [
        SystemSpec(f"hopfield_b{cfg['beta']:g}", lambda: HopfieldSystem(cfg["beta"]), family="neurostore"),
        SystemSpec(f"hopfield_b{cfg['beta']:g}_s{cfg['iter_steps']}", lambda: HopfieldSystem(cfg["beta"], cfg["iter_steps"]), family="neurostore"),
        spec_from_class(BruteForceHammingSystem),
        *[spec_from_class(BitSamplingLSHSystem, k, L, name=f"lsh_k{k}_L{L}") for k, L in cfg["lsh_presets"]],
        spec_from_class(DictKVSystem),
    ]
    if include_external:
        specs += [
            spec_from_class(RocksKVSystem),
            spec_from_class(RedisKVSystem),
            spec_from_class(FaissBinaryFlatSystem),
            spec_from_class(FaissBinaryHNSWSystem),
            spec_from_class(QdrantSystem),
            spec_from_class(DatasketchLSHSystem),
        ]
    return specs


# ----------------------------------------------------------------------------- recall vs noise
def _recall_vs_noise(cfg: dict[str, Any], seed: int, specs: list[SystemSpec], res: WorkloadResult, log: Log,
                     table: str = "recall_vs_noise") -> None:
    noise = noise_grid(cfg["noise_step"], cfg["noise_max"])
    n_items, trials = cfg["n_items"], cfg["trials"]
    rows: list[dict[str, Any]] = []
    for di, dim in enumerate(cfg["dims"]):
        log(f"[A] recall vs noise: D={dim}, N={n_items}")
        rng = make_rng([seed, 1, di, n_items])
        keys, values = random_bipolar(rng, n_items, dim), random_bipolar(rng, n_items, dim)
        systems: list[tuple[SystemSpec, AssocSystem]] = []
        for spec in specs:
            if (m := instantiate(spec, res, log)) is None:
                continue
            if m.max_items and n_items > m.max_items:
                res.add_skip(spec.name, f"store of {n_items} items exceeds this system's cap of {m.max_items}")
                m.close()
                continue
            t0 = time.perf_counter()
            m.build(keys, values)
            systems.append((spec, m))
            log(f"    built {spec.name} in {time.perf_counter() - t0:.2f}s")
        try:
            true_idx = np.arange(trials) % n_items
            for ni, p in enumerate(noise):
                qrng = make_rng([seed, 2, di, ni, n_items])
                queries = flip_exact(qrng, keys[true_idx], noise_to_flips(dim, p))
                truth = values[true_idx]
                for spec, m in systems:
                    n = min(trials, m.max_queries or trials)
                    out = m.retrieve_values(queries[:n])
                    ok = int((out == truth[:n]).all(axis=1).sum())
                    row = {"dim": dim, "n_items": n_items, "system": spec.name, "family": spec.family or m.family,
                           "noise": p, **{k: v for k, v in proportion(ok, n).items() if k != "k"}, "correct": ok}
                    if getattr(m, "supports_ids", True):
                        ids = m.retrieve_ids(queries[:n])
                        if ids is not None:
                            row["ident_rate"] = float((ids == true_idx[:n]).mean())
                    rows.append(row)
                rows.append({"dim": dim, "n_items": n_items, "system": "theory_bruteforce_ident", "family": "theory", "noise": p,
                             "rate": theory.p_identify_bruteforce(dim, n_items, noise_to_flips(dim, p)), "n": 0})
        finally:
            for _, m in systems:
                m.close()
    res.tables[table] = rows


def _thresholds(cfg: dict[str, Any], res: WorkloadResult, src: str = "recall_vs_noise", dst: str = "thresholds") -> None:
    target = cfg.get("recall_target", 0.99)
    rows = res.tables[src]
    out = []
    for dim in cfg["dims"]:
        names = list(dict.fromkeys(r["system"] for r in rows if r["dim"] == dim))
        for name in names:
            sel = sorted((r for r in rows if r["dim"] == dim and r["system"] == name), key=lambda r: r["noise"])
            tc = threshold_crossing([r["noise"] for r in sel], [r["rate"] for r in sel], target)
            out.append({"dim": dim, "n_items": cfg["n_items"], "system": name, "family": sel[0]["family"], "target": target,
                        "last_ok": tc["last_ok"], "first_fail": tc["first_fail"], "crossing": tc["crossing"]})
        th = theory.noise_threshold_bruteforce(dim, cfg["n_items"], target)
        out.append({"dim": dim, "n_items": cfg["n_items"], "system": "theory_bruteforce_ident_exact", "family": "theory", "target": target,
                    "last_ok": th["rate"], "first_fail": None, "crossing": None})
    res.tables[dst] = out


# ----------------------------------------------------------------------------- store-size sweep
def _scale_sweep(cfg: dict[str, Any], seed: int, res: WorkloadResult, log: Log, include_external: bool) -> None:
    """Noise tolerance versus the number of stored items (more impostors -> lower tolerance)."""
    specs = _scale_specs(cfg, include_external)
    rec_rows: list[dict[str, Any]] = []
    thr_rows: list[dict[str, Any]] = []
    for n in cfg["scale_items"]:
        sub = {**cfg, "dims": [cfg["scale_dim"]], "n_items": n, "noise_step": cfg["scale_noise_step"], "trials": cfg["scale_trials"]}
        tmp = WorkloadResult(workload=KEY, title=TITLE, profile=res.profile, seed=seed)
        _recall_vs_noise(sub, seed, specs, tmp, log, table="t")
        _thresholds(sub, tmp, "t", "th")
        rec_rows += tmp.tables["t"]
        thr_rows += tmp.tables["th"]
        for sk in tmp.skipped:
            res.add_skip(sk["system"], sk["reason"])
    res.tables["scale_recall"] = rec_rows
    res.tables["scale_thresholds"] = thr_rows


# ----------------------------------------------------------------------------- beta sweep
def _beta_sweep(cfg: dict[str, Any], seed: int, res: WorkloadResult, log: Log) -> None:
    dim, n_items, trials = cfg["beta_sweep_dim"], cfg["n_items"], cfg["trials"]
    noise = noise_grid(cfg["beta_noise_step"], cfg["noise_max"])
    log(f"[A] beta sweep: D={dim}")
    rng = make_rng([seed, 3])
    keys, values = random_bipolar(rng, n_items, dim), random_bipolar(rng, n_items, dim)
    mems = {}
    for beta in cfg["beta_sweep"]:
        m = ModernHopfieldMemory(dim, beta)
        m.store(keys, values)
        mems[beta] = m
    steps_list = sorted({1, cfg["iter_steps"]})
    true_idx = np.arange(trials) % n_items
    rows = []
    for ni, p in enumerate(noise):
        q = flip_exact(make_rng([seed, 4, ni]), keys[true_idx], noise_to_flips(dim, p))
        for beta, m in mems.items():
            ident = float((m.retrieve_index(q) == true_idx).mean())
            for steps in steps_list:
                ok = int((m.retrieve(q, steps=steps) == values[true_idx]).all(axis=1).sum())
                rows.append({"beta": beta, "steps": steps, "noise": p, "value_recall": ok / trials, "ident_rate": ident, "n": trials})
    res.tables["beta_sweep"] = rows
    th = []
    for beta in cfg["beta_sweep"]:
        i_sel = sorted((r for r in rows if r["beta"] == beta and r["steps"] == 1), key=lambda r: r["noise"])
        i = threshold_crossing([r["noise"] for r in i_sel], [r["ident_rate"] for r in i_sel], 0.99)
        entry = {"beta": beta, "dim": dim, "ident_last_ok": i["last_ok"]}
        for steps in steps_list:
            sel = sorted((r for r in rows if r["beta"] == beta and r["steps"] == steps), key=lambda r: r["noise"])
            v = threshold_crossing([r["noise"] for r in sel], [r["value_recall"] for r in sel], 0.99)
            entry[f"value_last_ok_s{steps}"] = v["last_ok"]
        th.append(entry)
    res.tables["beta_thresholds"] = th


# ----------------------------------------------------------------------------- latency / memory / eta
def _latency_memory(cfg: dict[str, Any], seed: int, specs: list[SystemSpec], res: WorkloadResult, log: Log) -> None:
    dim = cfg["latency_dim"]
    rows = []
    for n_items in cfg["latency_items"]:
        log(f"[A] latency/memory: D={dim}, N={n_items}")
        rng = make_rng([seed, 5, n_items])
        keys, values = random_bipolar(rng, n_items, dim), random_bipolar(rng, n_items, dim)
        n_read, n_write = cfg["n_read_samples"], cfg["n_write_samples"]
        pick = rng.integers(0, n_items, n_read)
        read_q = flip_exact(rng, keys[pick], noise_to_flips(dim, cfg["latency_noise"]))
        new_k, new_v = random_bipolar(rng, n_write, dim), random_bipolar(rng, n_write, dim)
        batch_q = read_q[: min(500, n_read)]
        for spec in specs:
            m = instantiate(spec, res, log)
            if m is None:
                continue
            if m.max_items and n_items > m.max_items:
                res.add_skip(spec.name, f"N={n_items} exceeds this system's cap of {m.max_items}")
                m.close()
                continue
            try:
                t0 = time.perf_counter()
                m.build(keys, values)
                build_s = time.perf_counter() - t0
                mem = m.memory_bytes()
                prep_r = [m.prepare_read(q) for q in read_q]
                read = summarize_latency(time_calls(m.read_one, prep_r))
                reps = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    m.retrieve_values(batch_q)
                    reps.append(time.perf_counter() - t0)
                qps_batch = len(batch_q) / float(np.median(reps))
                prep_w = [m.prepare_write(new_k[i], new_v[i]) for i in range(n_write)]
                write = summarize_latency(time_calls(m.write_one, prep_w, warmup=2))
                rows.append({
                    "dim": dim, "n_items": n_items, "system": spec.name, "family": spec.family or m.family,
                    "build_s": build_s, "memory_bytes": mem, "bytes_per_pair": mem / n_items,
                    "eta": eta(n_items * 2 * dim, mem),
                    "read_p50_us": read["p50_us"], "read_p99_us": read["p99_us"], "read_mean_us": read["mean_us"],
                    "write_p50_us": write["p50_us"], "write_p99_us": write["p99_us"],
                    "read_write_ratio": read["p50_us"] / write["p50_us"] if write["p50_us"] else float("nan"),
                    "qps_single": 1e6 / read["mean_us"], "qps_batch": qps_batch,
                })
            finally:
                m.close()
    res.tables["latency_memory"] = rows


# ----------------------------------------------------------------------------- checks / headlines
def _checks_and_headlines(cfg: dict[str, Any], res: WorkloadResult) -> None:
    rec = res.tables["recall_vs_noise"]
    thr = res.tables["thresholds"]
    dim0 = cfg["dims"][0]
    hop = f"hopfield_b{cfg['beta']:g}"

    def last_ok(system: str, dim: int = dim0) -> float:
        for r in thr:
            if r["dim"] == dim and r["system"] == system:
                return r["last_ok"] if r["last_ok"] is not None else 0.0
        return float("nan")

    r0 = next(r["rate"] for r in rec if r["dim"] == dim0 and r["system"] == hop and r["noise"] == 0.0)
    res.add_check("Hopfield recalls exactly with a clean key", "100% at 0% noise", f"{r0 * 100:.1f}%", r0 == 1.0)
    h_thr, hash_thr = last_ok(hop), last_ok("python_dict")
    res.add_check("Hopfield tolerates bit flips that defeat an exact hash table", "N_noise(Hopfield) > N_noise(hash)",
                  f"{h_thr * 100:.0f}% vs {hash_thr * 100:.0f}%", h_thr > hash_thr)
    lsh_names = [r["system"] for r in thr if r["dim"] == dim0 and r["family"] == "lsh"]
    if lsh_names:
        best_lsh = max(lsh_names, key=last_ok)
        res.add_check("Hopfield tolerates more noise than the best LSH configuration", "N_noise(Hopfield) > N_noise(best LSH)",
                      f"{h_thr * 100:.0f}% vs {last_ok(best_lsh) * 100:.0f}% ({best_lsh})", h_thr > last_ok(best_lsh))
    it = f"hopfield_b{cfg['beta']:g}_s{cfg['iter_steps']}"
    res.add_check("Attractor iteration extends the noise range of exact value recovery beyond the single-step reference",
                  "N_noise(iterated) >= N_noise(single step)", f"{last_ok(it) * 100:.0f}% vs {h_thr * 100:.0f}%", last_ok(it) >= h_thr)
    bf = last_ok("bruteforce_hamming")
    step = cfg["noise_step"]
    res.add_check("Single-step Hopfield stays within 2 noise steps of the exact-NN ceiling (information bound, not a win)",
                  "N_noise(Hopfield) >= N_noise(exact NN) - 2 steps", f"{h_thr * 100:.0f}% vs {bf * 100:.0f}%", h_thr >= bf - 2 * step - 1e-9,
                  note=f"Hopfield with beta={cfg['beta']:g} is soft exact-NN; it cannot exceed the ceiling.")
    # benchmark self-validation: measured exact-NN identification vs closed form
    worst = 0.0
    for dim in cfg["dims"]:
        meas = {r["noise"]: r["ident_rate"] for r in rec if r["dim"] == dim and r["system"] == "bruteforce_hamming" and "ident_rate" in r}
        th = {r["noise"]: r["rate"] for r in rec if r["dim"] == dim and r["system"] == "theory_bruteforce_ident"}
        if meas:
            worst = max(worst, max(abs(meas[k] - th[k]) for k in meas))
    res.add_check("Benchmark self-check: measured exact-NN identification matches closed-form theory",
                  "max |measured - theory| <= 0.05 over the grid", f"{worst:.3f}", worst <= 0.05,
                  note="Validates the noise injector and the evaluation loop, not NeuroStore.")

    heads = []
    ctx = f"D={dim0}, N={cfg['n_items']}"
    sysd = {r["system"]: (100.0 * (r["last_ok"] or 0.0), r["family"]) for r in thr if r["dim"] == dim0 and r["family"] != "theory"}
    heads.append(headline("Noise tolerance: largest bit-flip rate with >=99% exact recall", "% of bits flipped", True, ctx, sysd))
    lat = res.tables.get("latency_memory", [])
    if lat:
        n_big = max(r["n_items"] for r in lat)
        sel = [r for r in lat if r["n_items"] == n_big]
        c2 = f"D={cfg['latency_dim']}, N={n_big}, 1 thread"
        heads.append(headline("Read latency p50 (single query, 10% noise)", "us", False, c2, {r["system"]: (r["read_p50_us"], r["family"]) for r in sel}))
        heads.append(headline("Write latency p50 (single insert)", "us", False, c2, {r["system"]: (r["write_p50_us"], r["family"]) for r in sel}))
        heads.append(headline("Memory per stored (key, value) pair", "bytes", False, c2, {r["system"]: (r["bytes_per_pair"], r["family"]) for r in sel}))
    res.headlines = heads
    res.summary = {
        "primary_dim": dim0,
        "noise_tolerance_pct": {str(d): {r["system"]: 100.0 * (r["last_ok"] or 0.0) for r in thr if r["dim"] == d} for d in cfg["dims"]},
        "selfcheck_max_dev": worst,
    }


# ----------------------------------------------------------------------------- entry point
def run(cfg: dict[str, Any], seed: int = 0, profile: str = "custom", include_external: bool = True, log: Log = null_log) -> WorkloadResult:
    res = WorkloadResult(workload=KEY, title=TITLE, profile=profile, seed=seed, params=dict(cfg))
    t0 = time.perf_counter()
    specs = _specs(cfg, include_external)
    with limit_threads(cfg.get("threads", 1)):
        _recall_vs_noise(cfg, seed, specs, res, log)
        _thresholds(cfg, res)
        if cfg.get("scale_items"):
            _scale_sweep(cfg, seed, res, log, include_external)
        _beta_sweep(cfg, seed, res, log)
        _latency_memory(cfg, seed, specs, res, log)
    _checks_and_headlines(cfg, res)
    res.notes += [
        "All systems receive identical noisy queries; success means the returned value equals the stored value in all D bits.",
        "Hopfield uses a softmax readout, so exact *value* recovery can fail before *identification* does (see beta_sweep).",
        "Exact NN is the accuracy ceiling: any retriever that finds the nearest key cannot beat it on this task.",
        "Systems with a query cap (LSH, datasketch, Qdrant) are evaluated on the first n queries; n is in the 'n' column.",
        "RocksDB memory is on-disk SST bytes after flush; Redis memory is the INFO used_memory growth; Qdrant memory is a lower bound.",
    ]
    res.elapsed_s = time.perf_counter() - t0
    return res
