"""Workload D -- holographic multi-hop knowledge-graph walking.

Question: can a whole knowledge graph live in ONE hypervector and be traversed by vector
algebra alone, and how many triples fit before "ghost nodes" (wrong top-1 entities) appear?

Protocol
--------
* Random directed graphs with ``E`` entities, ``R`` relation types and ``T`` triples; every
  entity has at least one outgoing edge (``T >= E``) and each ``(subject, relation)`` pair has a
  single object, so every walk is well defined.
* **Single hop**: for every stored edge ``(s, r)``, is the cleaned-up unbound vector ``o``?
  ``ghost_rate = 1 - accuracy``.
* **Multi-hop**: random walks of ``h`` hops, success = every visited entity correct.
* **Capacity**: the largest ``T`` on the grid keeping single-hop accuracy >= target, compared with
  the closed-form bound from :func:`neurostore.theory.holographic_capacity`.
* **Speed**: hops per millisecond (single query and batched) against Python-dict pointer chasing
  and a dense numpy successor table.
* **Control**: querying an edge backwards must *not* return the subject (direction is encoded).
* Algebras: ``map`` (Hadamard, bipolar) and ``hrr`` (circular convolution, unitary).

Tables: ``capacity`` (accuracy/ghost rate/multi-hop per algebra, D, T), ``capacity_summary``,
``speed``, ``direction_control``.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .. import theory
from ..baselines import AdjacencyDictGraph, DenseSuccessorTable
from ..common import limit_threads, make_rng, proportion, summarize_latency, threshold_crossing, time_calls
from ..primitives import HolographicGraphMemory
from ..results import WorkloadResult
from .base import Log, eta, headline, null_log

KEY = "d"
TITLE = "Workload D - Holographic multi-hop knowledge-graph walking"


def make_graph(rng: np.random.Generator, n_entities: int, n_relations: int, n_triples: int) -> np.ndarray:
    """``(T, 3)`` triples; each entity has an out-edge, each (subject, relation) at most one."""
    e, r, t = n_entities, n_relations, n_triples
    if not e <= t <= e * r:
        raise ValueError(f"need E <= T <= E*R, got E={e}, T={t}, E*R={e * r}")
    rel0 = rng.integers(0, r, e)
    obj0 = (np.arange(e) + 1 + rng.integers(0, e - 1, e)) % e  # never a self loop
    free = np.ones(e * r, dtype=bool)
    free[np.arange(e) * r + rel0] = False
    pairs = rng.choice(np.flatnonzero(free), t - e, replace=False)
    s_x, r_x = pairs // r, pairs % r
    o_x = (s_x + 1 + rng.integers(0, e - 1, t - e)) % e
    triples = np.concatenate([np.stack([np.arange(e), rel0, obj0], 1), np.stack([s_x, r_x, o_x], 1)])
    return triples[rng.permutation(t)]


def make_walks(rng: np.random.Generator, triples: np.ndarray, n_entities: int, n_walks: int, hops: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random walks following existing edges -> ``(start, relations (W, h), objects (W, h))``."""
    order = np.argsort(triples[:, 0], kind="stable")
    t = triples[order]
    deg = np.bincount(t[:, 0], minlength=n_entities)
    ptr = np.concatenate(([0], np.cumsum(deg)))
    start = rng.integers(0, n_entities, n_walks)
    cur = start.copy()
    rels = np.empty((n_walks, hops), dtype=np.int64)
    objs = np.empty((n_walks, hops), dtype=np.int64)
    for h in range(hops):
        j = ptr[cur] + (rng.random(n_walks) * deg[cur]).astype(np.int64)
        rels[:, h], objs[:, h] = t[j, 1], t[j, 2]
        cur = objs[:, h]
    return start, rels, objs


# ----------------------------------------------------------------------------- capacity grid
def _capacity(cfg: dict[str, Any], seed: int, res: WorkloadResult, log: Log) -> None:
    e, r = cfg["n_entities"], cfg["n_relations"]
    max_h = max(cfg["hops"])
    rows = []
    for alg in cfg["algebras"]:
        for di, dim in enumerate(cfg["dims"]):
            for ti, t in enumerate(cfg["edges"]):
                if t > e * r:
                    continue
                ok = total = 0
                cos_sum = 0.0
                path_ok = np.zeros(max_h)
                path_total = 0
                for gi in range(cfg["n_graph_samples"]):
                    rng = make_rng([seed, 30, di, ti, gi])
                    triples = make_graph(rng, e, r, t)
                    mem = HolographicGraphMemory(e, r, dim, alg, seed=int(rng.integers(0, 2**31 - 1)))
                    mem.add_triples(triples)
                    nxt, cs = mem.hop(triples[:, 0], triples[:, 1])
                    ok += int((nxt == triples[:, 2]).sum())
                    total += t
                    cos_sum += float(cs.sum())
                    start, rels, objs = make_walks(rng, triples, e, cfg["n_walks"], max_h)
                    vis = mem.walk(start, rels)
                    good = np.logical_and.accumulate(vis == objs, axis=1)
                    path_ok += good.sum(axis=0)
                    path_total += start.size
                p = proportion(ok, total)
                row = {"algebra": alg, "dim": dim, "n_entities": e, "n_relations": r, "triples": t,
                       "single_hop_acc": p["rate"], "acc_lo": p["lo"], "acc_hi": p["hi"], "ghost_rate": 1.0 - p["rate"],
                       "mean_signal_cos": cos_sum / total, "signal_cos_theory": 1.0 / np.sqrt(t),
                       "theory_acc": theory.holographic_p_correct(dim, t, e), "n_hops_tested": total}
                for h in cfg["hops"]:
                    row[f"path_acc_h{h}"] = float(path_ok[h - 1] / path_total)
                row["path_acc_theory_h2"] = row["theory_acc"] ** 2
                rows.append(row)
            log(f"[D] capacity {alg} D={dim} done")
    res.tables["capacity"] = rows
    summ = []
    for alg in cfg["algebras"]:
        for dim in cfg["dims"]:
            sel = sorted((x for x in rows if x["algebra"] == alg and x["dim"] == dim), key=lambda x: x["triples"])
            tc = threshold_crossing([x["triples"] for x in sel], [x["single_hop_acc"] for x in sel], cfg["accuracy_target"])
            summ.append({"algebra": alg, "dim": dim, "n_entities": e, "target": cfg["accuracy_target"],
                         "t_max_grid": tc["last_ok"], "first_fail_grid": tc["first_fail"], "t_max_interpolated": tc["crossing"],
                         "t_max_theory": theory.holographic_capacity(dim, e, cfg["accuracy_target"])})
    res.tables["capacity_summary"] = summ


# ----------------------------------------------------------------------------- speed
def _speed(cfg: dict[str, Any], seed: int, res: WorkloadResult, log: Log) -> None:
    rows = []
    r_rel, batch = 8, cfg["speed_batch"]
    for alg in cfg["algebras"]:
        for dim in cfg["speed_dims"]:
            for e in cfg["speed_entities"]:
                if alg == "hrr" and e * dim > 5e7:
                    res.add_skip(f"holographic_{alg}", f"E={e}, D={dim}: complex spectra of the item memory would need >1 GB")
                    continue
                rng = make_rng([seed, 31, dim, e])
                t = min(e, 100)  # a graph well inside capacity: speed, not accuracy, is measured here
                subj = rng.choice(e, t, replace=False)
                triples = np.stack([subj, rng.integers(0, r_rel, t), rng.integers(0, e, t)], 1)
                mem = HolographicGraphMemory(e, r_rel, dim, alg, seed=int(rng.integers(0, 2**31 - 1)))
                mem.add_triples(triples)
                one = [(triples[i % len(triples), 0:1], triples[i % len(triples), 1:2]) for i in range(cfg["speed_reps"] * 4)]
                lat1 = summarize_latency(time_calls(lambda a: mem.hop(*a), one))
                bs = triples[np.arange(batch) % len(triples)]
                batches = [(bs[:, 0], bs[:, 1])] * cfg["speed_reps"]
                latb = summarize_latency(time_calls(lambda a: mem.hop(*a), batches))
                rows.append({"system": f"holographic_{alg}", "family": "neurostore", "algebra": alg, "dim": dim, "n_entities": e, "triples": len(triples),
                             "hops_per_ms_single": 1e3 / lat1["p50_us"], "hops_per_ms_batch": batch * 1e3 / latb["p50_us"],
                             "hop_p50_us": lat1["p50_us"], "memory_bytes": mem.memory_bytes(), "trace_bytes": mem.trace_bytes(),
                             "bytes_per_triple": mem.memory_bytes() / len(triples)})
        log(f"[D] speed {alg} done")
    # exact pointer-chasing baselines on the same sizes
    for e in cfg["speed_entities"]:
        rng = make_rng([seed, 32, e])
        t = min(e, 100)
        s = rng.choice(e, t, replace=False)
        triples = np.stack([s, rng.integers(0, 8, t), rng.integers(0, e, t)], 1)
        dct = AdjacencyDictGraph(triples)
        tab = DenseSuccessorTable(triples, e, 8)
        keys = [(int(triples[i % t, 0]), int(triples[i % t, 1])) for i in range(2000)]
        lat1 = summarize_latency(time_calls(lambda a: dct.hop(*a), keys))
        bs = triples[np.arange(cfg["speed_batch"]) % t]
        latb = summarize_latency(time_calls(lambda a: tab.hop_batch(*a), [(bs[:, 0], bs[:, 1])] * 50))
        rows.append({"system": "adjacency_dict", "family": "kv", "algebra": None, "dim": None, "n_entities": e, "triples": t,
                     "hops_per_ms_single": 1e3 / lat1["p50_us"], "hops_per_ms_batch": None, "hop_p50_us": lat1["p50_us"],
                     "memory_bytes": dct.memory_bytes(), "trace_bytes": None, "bytes_per_triple": dct.memory_bytes() / t})
        rows.append({"system": "successor_table", "family": "kv", "algebra": None, "dim": None, "n_entities": e, "triples": t,
                     "hops_per_ms_single": None, "hops_per_ms_batch": cfg["speed_batch"] * 1e3 / latb["p50_us"], "hop_p50_us": None,
                     "memory_bytes": tab.memory_bytes(), "trace_bytes": None, "bytes_per_triple": tab.memory_bytes() / t})
    res.tables["speed"] = rows


# ----------------------------------------------------------------------------- direction control
def _direction(cfg: dict[str, Any], seed: int, res: WorkloadResult) -> None:
    rows = []
    e, r = cfg["n_entities"], cfg["n_relations"]
    dim = max(cfg["dims"])
    cap = theory.holographic_capacity(dim, e, 0.95)
    t = max([x for x in cfg["edges"] if x <= cap] or [min(cfg["edges"])])  # a graph the memory can actually hold
    for alg in cfg["algebras"]:
        rng = make_rng([seed, 33])
        triples = make_graph(rng, e, r, t)
        mem = HolographicGraphMemory(e, r, dim, alg, seed=int(rng.integers(0, 2**31 - 1)))
        mem.add_triples(triples)
        fwd, _ = mem.hop(triples[:, 0], triples[:, 1])
        rev, _ = mem.hop(triples[:, 2], triples[:, 1])
        # a backward query can only legitimately hit the subject if the reverse edge also exists
        existing = {(int(a), int(b)): int(c) for a, b, c in triples}
        legit = np.array([existing.get((int(o), int(rr))) == int(s) for s, rr, o in triples])
        keep = ~legit
        rows.append({"algebra": alg, "dim": dim, "triples": len(triples), "forward_acc": float((fwd == triples[:, 2]).mean()),
                     "reverse_hit_rate": float((rev[keep] == triples[keep, 0]).mean()), "chance": 1.0 / e, "n": int(keep.sum())})
    res.tables["direction_control"] = rows


# ----------------------------------------------------------------------------- run
def run(cfg: dict[str, Any], seed: int = 0, profile: str = "custom", include_external: bool = True, log: Log = null_log) -> WorkloadResult:
    res = WorkloadResult(workload=KEY, title=TITLE, profile=profile, seed=seed, params=dict(cfg))
    t0 = time.perf_counter()
    with limit_threads(cfg.get("threads", 1)):
        _capacity(cfg, seed, res, log)
        _speed(cfg, seed, res, log)
        _direction(cfg, seed, res)
    _checks_and_headlines(cfg, res)
    res.notes += [
        "Capacity is a property of the *single-vector* design: all T triples share one D-dimensional trace, so signal falls as 1/sqrt(T) against a noise floor of 1/sqrt(D).",
        "The item memory (entity/relation codebooks) is needed to clean up every hop and dominates the footprint; it is included in memory_bytes.",
        "Graph speed rows time the hop primitive only (bind, unbind, cleanup); the dict baseline is exact and needs no cleanup.",
        "path_acc_hK = fraction of walks whose first K hops are all correct; errors compound because a wrong entity poisons the next query.",
    ]
    res.elapsed_s = time.perf_counter() - t0
    return res


def _checks_and_headlines(cfg: dict[str, Any], res: WorkloadResult) -> None:
    cap = res.tables["capacity"]
    # Self-check against theory. Saturated cells (theory ~0 or ~1) match trivially, so only the transition region
    # is compared, pooled per algebra (signed) to average out sampling noise. MAP follows the Gaussian model closely;
    # unitary HRR sits a consistent ~1-2 points below it (documented approximation gap, not a defect).
    devs = {}
    for alg in cfg["algebras"]:
        cells = [x for x in cap if x["algebra"] == alg and 0.02 < x["theory_acc"] < 0.98]
        devs[alg] = float(np.mean([x["single_hop_acc"] - x["theory_acc"] for x in cells])) if cells else 0.0
    dev = max(abs(v) for v in devs.values())
    tol = cfg.get("selfcheck_tol", 0.03)  # looser for tiny smoke-test profiles, where a cell is a few dozen hops
    res.add_check("Benchmark self-check: measured single-hop accuracy matches closed-form theory (transition region, pooled)",
                  f"|mean(measured - theory)| <= {tol:g} per algebra", ", ".join(f"{a}: {v:+.3f}" for a, v in devs.items()), dev <= tol,
                  note="Validates the graph generator, the algebras and the evaluation loop. HRR runs ~1-2 points below the Gaussian model.")
    for x in res.tables["direction_control"]:
        res.add_check(f"Edge direction is encoded ({x['algebra']}): a reversed query does not return the subject", "reverse hit rate <= max(5%, 4x chance)",
                      f"{x['reverse_hit_rate'] * 100:.1f}% (chance {x['chance'] * 100:.1f}%), forward acc {x['forward_acc'] * 100:.0f}%",
                      x["reverse_hit_rate"] <= max(0.05, 4 * x["chance"]))
    speed = res.tables["speed"]
    e_ref = sorted(set(x["n_entities"] for x in speed))[len(set(x["n_entities"] for x in speed)) // 2]
    d_ref = cfg["speed_dims"][0]
    holo = [x for x in speed if x["family"] == "neurostore" and x["n_entities"] == e_ref and x["dim"] == d_ref]
    dct = next((x for x in speed if x["system"] == "adjacency_dict" and x["n_entities"] == e_ref), None)
    if holo and dct:
        best = max(holo, key=lambda x: x["hops_per_ms_single"])
        res.add_check("Holographic hop rate exceeds pointer chasing (spec objective, single query)", "holographic hops/ms > dict hops/ms",
                      f"{best['hops_per_ms_single']:.1f} vs {dct['hops_per_ms_single']:.0f} hops/ms (E={e_ref}, D={d_ref})",
                      best["hops_per_ms_single"] > dct["hops_per_ms_single"],
                      note="A hop costs O(E*D) for cleanup; a dict lookup is O(1). Only parallel in-memory hardware could change this.")
    ctx = f"E={e_ref}, D={d_ref}, 1 thread"
    sel = [x for x in speed if x["n_entities"] == e_ref and (x["dim"] in (d_ref, None))]
    res.headlines = [
        headline("Hops per millisecond, single query", "hops/ms", True, ctx, {x["system"] + (f"_D{x['dim']}" if x["dim"] else ""): (x["hops_per_ms_single"], x["family"]) for x in sel if x["hops_per_ms_single"]}),
        headline("Hops per millisecond, batched", "hops/ms", True, ctx, {x["system"] + (f"_D{x['dim']}" if x["dim"] else ""): (x["hops_per_ms_batch"], x["family"]) for x in sel if x["hops_per_ms_batch"]}),
        headline("Memory per stored triple (incl. item memory)", "bytes", False, ctx, {x["system"] + (f"_D{x['dim']}" if x["dim"] else ""): (x["bytes_per_triple"], x["family"]) for x in sel}),
    ]
    res.summary = {"capacity": res.tables["capacity_summary"], "selfcheck_max_dev": dev}
