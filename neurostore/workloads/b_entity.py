"""Workload B -- unstructured record reconstruction and entity resolution.

Question: given only ``K`` of a record's ``F`` attributes (primary key missing), can the system
(1) identify which record it is and (2) reconstruct an attribute it was *not* given -- and what
does that cost compared with relational indexing?

Protocol
--------
* ``M`` records with ``F`` fields, each value uniform over a vocabulary of ``V`` symbols.
* Queries: a random record, a random ``K``-subset of its fields as the cue (optionally with
  ``c`` of the cue values *corrupted*; combinations leaving fewer than two valid clues are
  skipped as unanswerable), and one of the fields **not** in the cue as the target.
  Only cues whose uncorrupted ``K`` fields identify the record uniquely are used, so that no
  method is asked to resolve a genuinely ambiguous query.
* NeuroStore: :class:`~neurostore.primitives.vsa.VSARecordStore` (MAP, role-filler bundling) in
  two stages -- associative record lookup, then unbind + cleanup. Two cue variants: integer
  bundle and sign-binarised bundle.
* Baselines: per-column inverted index (+intersect), linear predicate scan, composite index per
  field subset (exact, O(1) lookup), and fuzzy scan (max-agreement; the exact ceiling for
  corrupted cues).
* Control: the original harness unbound a role that is not in the cue; this suite measures that
  method too and shows its output is uncorrelated with the true filler.

Tables: ``accuracy`` (identification, reconstruction, unbinding cosine, bit agreement),
``latency_memory``, ``index_blowup``, ``reference_method_control``.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np

from ..baselines import CompositeIndexSet, FuzzyScan, InvertedIndexIntersect, LinearScan
from ..common import limit_threads, make_rng, proportion, summarize_latency, time_calls
from ..primitives import VSARecordStore
from ..results import WorkloadResult
from .base import Log, eta, headline, null_log

KEY = "b"
TITLE = "Workload B - Record completion & entity resolution"


def _sample_queries(rng: np.random.Generator, records: np.ndarray, index: InvertedIndexIntersect, vocab: int, k: int,
                    n_queries: int, corrupt: int) -> dict[str, np.ndarray]:
    """Unique-cue queries with optional corrupted cue values and a missing target field."""
    m, f = records.shape
    src_l, fld_l, val_l = [], [], []
    for _ in range(100):
        need = n_queries - len(src_l)
        if need <= 0:
            break
        cand = rng.integers(0, m, 2 * need)
        flds = np.stack([rng.permutation(f)[:k] for _ in range(cand.size)])
        vals = np.take_along_axis(records[cand], flds, axis=1)
        for i in range(cand.size):
            if index.match_all(flds[i], vals[i]).size == 1:
                src_l.append(cand[i])
                fld_l.append(flds[i])
                val_l.append(vals[i])
                if len(src_l) == n_queries:
                    break
    if len(src_l) < n_queries:
        raise RuntimeError(f"could only build {len(src_l)} unique-cue queries; enlarge the vocabulary")
    src, flds, clean = np.array(src_l), np.stack(fld_l), np.stack(val_l)
    cue = clean.copy()
    if corrupt:
        for i in range(src.size):
            for j in rng.permutation(k)[:corrupt]:
                cue[i, j] = (cue[i, j] + 1 + rng.integers(0, vocab - 1)) % vocab  # guaranteed different
    target = np.array([rng.choice(np.setdiff1d(np.arange(f), flds[i])) for i in range(src.size)])
    return {"src": src, "fields": flds, "values": cue, "clean_values": clean, "target": target,
            "truth": records[src, target]}


def _vsa_metrics(store: VSARecordStore, q: dict[str, np.ndarray]) -> dict[str, float]:
    out = store.reconstruct(q["fields"], q["values"], q["target"])
    filler = store.filler(q["target"], q["truth"])
    unb = out["unbound"]
    cos = (unb * filler).sum(axis=1) / (np.linalg.norm(unb, axis=1) * math.sqrt(store.dim) + 1e-12)
    return {
        "ident_ok": int((out["ids"] == q["src"]).sum()),
        "recon_ok": int((out["values"] == q["truth"]).sum()),
        "unbind_cos": float(cos.mean()),
        "bit_agreement": float((np.sign(unb) == filler).mean()),
    }


def _baseline_metrics(sys_: Any, q: dict[str, np.ndarray]) -> dict[str, float]:
    out = sys_.reconstruct(q["fields"], q["values"], q["target"])
    return {"ident_ok": int((out["ids"] == q["src"]).sum()), "recon_ok": int((out["values"] == q["truth"]).sum())}


def run(cfg: dict[str, Any], seed: int = 0, profile: str = "custom", include_external: bool = True, log: Log = null_log) -> WorkloadResult:
    res = WorkloadResult(workload=KEY, title=TITLE, profile=profile, seed=seed, params=dict(cfg))
    t_start = time.perf_counter()
    f, v, m = cfg["n_fields"], cfg["vocab"], cfg["n_records"]
    vocab_sizes = [v] * f
    rng = make_rng([seed, 10])
    records = rng.integers(0, v, (m, f))
    table = records.astype(np.uint16 if v <= 65535 else np.int64)  # the relational baselines store compact integers
    entropy_bits = m * f * math.log2(v)

    with limit_threads(cfg.get("threads", 1)):
        index = InvertedIndexIntersect(table, vocab_sizes)
        baselines: dict[str, Any] = {
            "inverted_index": index,
            "linear_scan": LinearScan(table),
            "fuzzy_scan": FuzzyScan(table),
        }
        composites: dict[int, CompositeIndexSet] = {}
        for k in cfg["composite_k"]:
            t0 = time.perf_counter()
            composites[k] = CompositeIndexSet(table, k)
            log(f"[B] composite indexes K={k}: {composites[k].n_indexes} indexes built in {time.perf_counter() - t0:.1f}s")

        # ------------------------------------------------------------- accuracy
        queries: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        for k in cfg["k_values"]:
            for c in cfg["corrupt"]:
                if k - c < 2:  # a single valid clue matches ~M/V records: unanswerable for every method
                    continue
                queries[(k, c)] = _sample_queries(make_rng([seed, 11, k, c]), records, index, v, k, cfg["n_queries"], c)

        rows: list[dict[str, Any]] = []
        n_q = cfg["n_queries"]
        for (k, c), q in queries.items():
            for name, b in baselines.items():
                met = _baseline_metrics(b, q)
                rows.append({"system": name, "family": "relational", "dim": None, "K": k, "corrupt": c, "n": n_q,
                             **{f"ident_{a}": b_ for a, b_ in _prop(met["ident_ok"], n_q).items()},
                             **{f"recon_{a}": b_ for a, b_ in _prop(met["recon_ok"], n_q).items()}})
            if k in composites:
                met = _baseline_metrics(composites[k], q)
                rows.append({"system": "composite_indexes", "family": "relational", "dim": None, "K": k, "corrupt": c, "n": n_q,
                             **{f"ident_{a}": b_ for a, b_ in _prop(met["ident_ok"], n_q).items()},
                             **{f"recon_{a}": b_ for a, b_ in _prop(met["recon_ok"], n_q).items()}})

        stores: dict[int, VSARecordStore] = {}
        control_rows = []
        for di, dim in enumerate(cfg["dims"]):
            log(f"[B] VSA D={dim}: encoding {m} records")
            store = VSARecordStore(vocab_sizes, dim=dim, seed=int(make_rng([seed, 12, di]).integers(0, 2**31 - 1)))
            store.add_records(records)
            stores[dim] = store
            for variant, bin_cue in (("vsa_map", False), ("vsa_map_bincue", True)):
                store.binarize_cue = bin_cue
                for (k, c), q in queries.items():
                    met = _vsa_metrics(store, q)
                    rows.append({"system": variant, "family": "neurostore", "dim": dim, "K": k, "corrupt": c, "n": n_q,
                                 **{f"ident_{a}": b_ for a, b_ in _prop(met["ident_ok"], n_q).items()},
                                 **{f"recon_{a}": b_ for a, b_ in _prop(met["recon_ok"], n_q).items()},
                                 "unbind_cos": met["unbind_cos"], "bit_agreement": met["bit_agreement"]})
            store.binarize_cue = False
            # control: the original harness unbinds the target role from the CUE itself
            q = queries[(cfg["k_values"][1] if len(cfg["k_values"]) > 1 else cfg["k_values"][0], 0)]
            store.binarize_cue = True
            cue = store.encode_cue(q["fields"], q["values"])
            store.binarize_cue = False
            unbound = cue * store.roles[q["target"]]
            cos = (unbound * store.filler(q["target"], q["truth"])).sum(axis=1) / dim
            control_rows.append({"dim": dim, "mean_cos": float(cos.mean()), "std_cos": float(cos.std()),
                                 "chance_std": 1.0 / math.sqrt(dim), "frac_cos_gt_0p6": float((cos > 0.6).mean()), "n": int(cos.size)})
        res.tables["accuracy"] = rows
        res.tables["reference_method_control"] = control_rows

        # ------------------------------------------------------------- latency / memory / blow-up
        ld, lk = cfg["latency_dim"], cfg["latency_k"]
        lat_rows = []
        lat_q = queries[(lk, 0)]
        n_lat = min(cfg["n_latency_queries"], lat_q["src"].size)
        single = [(lat_q["fields"][i : i + 1], lat_q["values"][i : i + 1], lat_q["target"][i : i + 1]) for i in range(n_lat)]
        timed: dict[str, tuple[Callable, int, str, float]] = {}
        store = stores[ld]
        for variant, bin_cue in (("vsa_map", False), ("vsa_map_bincue", True)):
            def fn(args, s=store, b=bin_cue):
                s.binarize_cue = b
                return s.reconstruct(*args)
            timed[variant] = (fn, store.memory_bytes(), "neurostore", 0.0)
        for name, b in baselines.items():
            timed[name] = (lambda args, b=b: b.reconstruct(*args), b.memory_bytes(), "relational", 0.0)
        if lk in composites:
            timed["composite_indexes"] = (lambda args: composites[lk].reconstruct(*args), composites[lk].memory_bytes(), "relational", 0.0)
        for name, (fn, mem, fam, _) in timed.items():
            lat = summarize_latency(time_calls(fn, single))
            store.binarize_cue = False
            info = entropy_bits
            lat_rows.append({"system": name, "family": fam, "dim": ld if fam == "neurostore" else None, "K": lk, "n_records": m,
                             "memory_bytes": mem, "bytes_per_record": mem / m, "eta": eta(info, mem),
                             "p50_us": lat["p50_us"], "p99_us": lat["p99_us"], "mean_us": lat["mean_us"], "qps": 1e6 / lat["mean_us"]})
        lat_rows.append({"system": "vsa_map(bit-packed, ideal)", "family": "neurostore", "dim": ld, "K": lk, "n_records": m,
                         "memory_bytes": store.ideal_packed_bytes(), "bytes_per_record": store.ideal_packed_bytes() / m,
                         "eta": eta(entropy_bits, store.ideal_packed_bytes()), "p50_us": None, "p99_us": None, "mean_us": None, "qps": None})
        res.tables["latency_memory"] = lat_rows

        blow = []
        per_index = {k: composites[k].index_bytes() / composites[k].n_indexes for k in composites}
        itemsize = table.dtype.itemsize
        for k in cfg["k_values"]:
            n_idx = math.comb(f, k)
            blow.append({"F": f, "K": k, "n_indexes_for_K": n_idx, "n_indexes_any_subset": 2**f - 1,
                         "composite_bytes": composites[k].index_bytes() if k in composites else None,
                         "composite_bytes_extrapolated": (per_index[k] * n_idx) if k in composites else
                         (np.mean(list(per_index.values())) * n_idx if per_index else None),
                         "composite_measured": k in composites,
                         # the same indexes as sorted arrays of (K field-width keys + int32 row id): representation-independent scaling
                         "composite_compact_bytes": n_idx * m * (k * itemsize + 4),
                         "inverted_bytes": index.index_bytes(), "table_bytes": table.nbytes,
                         "vsa_record_bytes_by_dim": {str(d): s_.record_bytes() for d, s_ in stores.items()}})
        res.tables["index_blowup"] = blow
        for s in stores.values():
            s.binarize_cue = False

    _checks_and_headlines(cfg, res, entropy_bits)
    res.notes += [
        "Queries are restricted to cues whose clean K fields are unique in the table, so exact methods are never asked an ambiguous question.",
        "'corrupt' = number of cue values replaced by a wrong value; exact-match methods cannot recover from it, fuzzy scan and VSA can.",
        "VSA record lookup is a dense O(M*D) scan: 'O(1) query steps' is a statement about operation count, not CPU time.",
        "Relational baselines use hash/inverted structures in Python; their memory is the deep size of those objects as allocated.",
        "VSA memory is reported as allocated float32 (what runs here) and as the ideal bit-packed size; the item memories (roles + filler codebooks) are included.",
        "reference_method_control reproduces the original harness (unbind a role absent from the cue); its cosine is pure noise of scale 1/sqrt(D).",
    ]
    res.elapsed_s = time.perf_counter() - t_start
    return res


def _prop(k: int, n: int) -> dict[str, float]:
    p = proportion(k, n)
    return {"rate": p["rate"], "lo": p["lo"], "hi": p["hi"]}


def _checks_and_headlines(cfg: dict[str, Any], res: WorkloadResult, entropy_bits: float) -> None:
    rows = res.tables["accuracy"]
    dmax = max(cfg["dims"])

    def get(system: str, k: int, c: int, key: str, dim: int | None = None) -> float | None:
        for r in rows:
            if r["system"] == system and r["K"] == k and r["corrupt"] == c and (dim is None or r["dim"] in (dim, None)):
                return r.get(key)
        return None

    ctrl = res.tables["reference_method_control"]
    worst = max(abs(r["mean_cos"]) / r["chance_std"] for r in ctrl)
    res.add_check("Original-harness unbinding (role absent from the cue) carries no information",
                  "|mean cosine| < 4/sqrt(D) and 0% success at cos>0.6", f"max |mean|/(1/sqrt(D)) = {worst:.2f}; success {max(r['frac_cos_gt_0p6'] for r in ctrl) * 100:.0f}%",
                  worst < 4.0 and all(r["frac_cos_gt_0p6"] == 0 for r in ctrl),
                  note="Why the reference Workload B always printed 'Exact Key Recovery Success: NO'.")
    k3 = 3 if 3 in cfg["k_values"] else cfg["k_values"][-1]
    ra = get("vsa_map", k3, 0, "recon_rate", dmax)
    res.add_check(f"Two-stage VSA reconstructs the missing field (K={k3}, clean cue, D={dmax})", ">= 99% accuracy",
                  f"{ra * 100:.1f}%", ra is not None and ra >= 0.99)
    if 1 in cfg["corrupt"] and k3 > 1:
        va, fa = get("vsa_map", k3, 1, "ident_rate", dmax), get("fuzzy_scan", k3, 1, "ident_rate")
        res.add_check(f"VSA identification with 1 corrupted cue field is within 5 points of the exact fuzzy-scan ceiling (K={k3}, D={dmax})",
                      "VSA >= fuzzy - 5 pts", f"{va * 100:.1f}% vs {fa * 100:.1f}%", va >= fa - 0.05)
        ex = get("inverted_index", k3, 1, "ident_rate")
        res.add_check("Exact-match relational methods cannot recover from a corrupted cue field", "~0%", f"{ex * 100:.1f}%", ex <= 0.02)

    heads = []
    ctx = f"K={k3}, D={dmax}, M={cfg['n_records']}, F={cfg['n_fields']}, V={cfg['vocab']}"
    for c, label in ((0, "clean cue"), (1, "1 corrupted cue field")):
        if c not in cfg["corrupt"]:
            continue
        sysd = {}
        for r in rows:
            if r["K"] == k3 and r["corrupt"] == c and (r["dim"] in (dmax, None)):
                sysd[r["system"]] = (100 * r["ident_rate"], r["family"])
        heads.append(headline(f"Record identification accuracy ({label})", "%", True, ctx, sysd, kind="quality"))
    sysd = {r["system"]: (100 * r["recon_rate"], r["family"]) for r in rows if r["K"] == k3 and r["corrupt"] == 0 and r["dim"] in (dmax, None)}
    heads.append(headline("Missing-field reconstruction accuracy (clean cue)", "%", True, ctx, sysd, kind="quality"))
    lat = [r for r in res.tables["latency_memory"] if r["p50_us"] is not None]
    c2 = f"K={cfg['latency_k']}, D={cfg['latency_dim']}, M={cfg['n_records']}, 1 thread"
    # Cost is compared only between systems that can serve a cue with one wrong field: >= 95% reconstruction accuracy there.
    kq = cfg["latency_k"]
    robust = {r["system"]: r["recon_rate"] for r in rows if r["K"] == kq and r["corrupt"] == 1 and r["dim"] in (cfg["latency_dim"], None)}
    gate = f"reconstructs the unknown field with >= 95% accuracy when one of the K={kq} cue values is wrong"
    ok = {r["system"]: robust.get(r["system"], 0.0) >= 0.95 for r in lat}
    heads.append(headline("Query latency p50 (identify + reconstruct)", "us", False, c2, {r["system"]: (r["p50_us"], r["family"], ok[r["system"]]) for r in lat}, kind="cost", gate=gate))
    heads.append(headline("Memory per stored record (as allocated, incl. item memories / base table)", "bytes", False, c2,
                          {r["system"]: (r["bytes_per_record"], r["family"], ok[r["system"]]) for r in lat}, kind="cost", gate=gate))
    res.headlines = heads
    res.summary = {"entropy_bits_per_record": entropy_bits / cfg["n_records"], "primary_dim": dmax, "K": k3}
