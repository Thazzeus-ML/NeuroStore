"""Workload C -- high-throughput semantic prompt caching.

Question: can a binary sparse code (FlyHash) recognise a *paraphrase* of a cached prompt faster
and in less memory than float vectors / HNSW graphs / classical hashing, without false hits?

Data
----
No LLM is run offline, so prompt embeddings are synthesised with the two properties that decide
cache behaviour: *topics* (unrelated prompts of one topic have cosine ~``topic_rho``, which
makes negatives hard) and *paraphrases* (cosine exactly ``para_cos`` to their source prompt).
Real embeddings can be supplied instead through ``embeddings_npz`` (arrays ``cache``, ``queries``,
``labels`` with -1 = novel prompt) -- see :func:`load_embeddings`.

Protocol (per cache size N)
---------------------------
* ``n_pos`` paraphrase queries (ground truth = the source prompt) and ``n_neg`` novel prompts.
* **Recall@1**: fraction of paraphrases whose top-1 candidate is the source prompt.
* **TPR @ FPR**: a hit is declared when the top-1 score exceeds a threshold calibrated on a
  separate set of novel prompts so that at most ``target_fpr`` of novel prompts hit; reported on
  held-out queries together with the realised false-hit rate.
* **Throughput**: single-query latency/QPS (encode + search, 1 thread) and batched QPS.
* **Memory**: bytes of the per-prompt index, N-independent state reported separately.
* **ANN agreement**: how often the system returns the same top-1 as exact cosine scan,
  separating *approximation error* from *encoding error*.

Tables: ``systems`` (one row per system and N), ``hardness_sweep`` (recall vs paraphrase cosine, same corpus and
topics, new queries) and ``flyhash_sweep`` (FlyHash expansion / winners / fan-in vs SimHash at equal bits).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..baselines import (
    BruteForceCosineCache,
    ExactHashCache,
    FaissFlatCache,
    FaissHNSWCache,
    FlyHashBinaryHNSWCache,
    FlyHashInvertedCache,
    FlyHashScanCache,
    QdrantCache,
    SimHashCache,
)
from ..baselines.cache import CacheSystem
from ..common import limit_threads, make_rng, proportion, summarize_latency, time_calls
from ..results import WorkloadResult
from .base import Log, SystemSpec, headline, instantiate, null_log, spec_from_class

KEY = "c"
TITLE = "Workload C - High-throughput semantic prompt caching"


# ----------------------------------------------------------------------------- data
def _unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def make_prompts(rng: np.random.Generator, n: int, centers: np.ndarray, topic_rho: float) -> np.ndarray:
    """Unit-norm "prompt embeddings": a topic component plus an independent random one."""
    topics = rng.integers(0, centers.shape[0], n)
    g = _unit(rng.standard_normal((n, centers.shape[1])))
    return _unit(np.sqrt(topic_rho) * centers[topics] + np.sqrt(1.0 - topic_rho) * g)


def make_paraphrases(rng: np.random.Generator, base: np.ndarray, cos: float) -> np.ndarray:
    """Vectors at cosine exactly ``cos`` from each row of ``base`` (random orthogonal displacement)."""
    u = rng.standard_normal(base.shape).astype(np.float32)
    u -= (u * base).sum(axis=1, keepdims=True) * base
    return _unit(cos * base + np.sqrt(max(0.0, 1.0 - cos * cos)) * _unit(u))


def load_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    """Load user-supplied embeddings: ``cache`` (N, d), ``queries`` (Q, d), ``labels`` (Q,).

    ``labels[i]`` is the index of the cached prompt that query ``i`` paraphrases, or ``-1`` if the
    query is a novel prompt. Vectors are L2-normalised.
    """
    z = np.load(path)
    cache, queries, labels = _unit(z["cache"]), _unit(z["queries"]), np.asarray(z["labels"], dtype=np.int64)
    if not (len(queries) == len(labels) and (labels < len(cache)).all()):
        raise ValueError("inconsistent embeddings file")
    return {"cache": cache, "queries": queries, "labels": labels}


def _dataset(cfg: dict[str, Any], seed: int, n_cache: int, para_cos: float, tag: int = 0) -> dict[str, np.ndarray]:
    if cfg.get("embeddings_npz"):
        d = load_embeddings(cfg["embeddings_npz"])
        pos = d["labels"] >= 0
        neg_q = d["queries"][~pos]
        half = len(neg_q) // 2
        return {"cache": d["cache"], "pos_q": d["queries"][pos], "pos_src": d["labels"][pos], "neg_q": neg_q[half:], "calib_q": neg_q[:half]}
    # The corpus (topic centres + cached prompts) depends only on (seed, n_cache); the queries on (seed, n_cache, tag).
    # Sweeps vary `tag`/`para_cos` and so get new queries drawn from the SAME topics as the cache -- novel prompts
    # must come from the cache's own topics or they are artificially easy negatives.
    crng = make_rng([seed, 20, n_cache])
    centers = _unit(crng.standard_normal((cfg["n_topics"], cfg["dim"])))
    cache = make_prompts(crng, n_cache, centers, cfg["topic_rho"])
    qrng = make_rng([seed, 22, n_cache, tag])
    src = qrng.integers(0, n_cache, cfg["n_pos"])
    return {
        "cache": cache,
        "pos_q": make_paraphrases(qrng, cache[src], para_cos),
        "pos_src": src,
        "neg_q": make_prompts(qrng, cfg["n_neg"], centers, cfg["topic_rho"]),
        "calib_q": make_prompts(qrng, cfg["n_calib"], centers, cfg["topic_rho"]),
    }


# ----------------------------------------------------------------------------- systems
def _specs(cfg: dict[str, Any], include_external: bool) -> list[SystemSpec]:
    specs = [spec_from_class(BruteForceCosineCache), spec_from_class(ExactHashCache)]
    for bits in cfg["simhash_bits"]:
        specs.append(SystemSpec(f"simhash_{bits}", lambda b=bits: SimHashCache(b), family="lsh"))
    for out_dim, k in cfg["flyhash"]:
        specs.append(SystemSpec(f"flyhash_scan_{out_dim}_{k}", lambda o=out_dim, kk=k: FlyHashScanCache(o, kk), family="neurostore"))
        specs.append(SystemSpec(f"flyhash_inv_{out_dim}_{k}", lambda o=out_dim, kk=k: FlyHashInvertedCache(o, kk), family="neurostore"))
        if include_external:
            specs.append(SystemSpec(f"flyhash_binhnsw_{out_dim}_{k}", lambda o=out_dim, kk=k: FlyHashBinaryHNSWCache(o, kk),
                                    FlyHashBinaryHNSWCache.available, "neurostore"))
    if include_external:
        specs += [spec_from_class(FaissFlatCache), spec_from_class(FaissHNSWCache), spec_from_class(QdrantCache)]
    return specs


def _calibrated_metrics(sys_: CacheSystem, data: dict[str, np.ndarray], target_fpr: float, brute_ids: np.ndarray | None) -> dict[str, Any]:
    ids_p, sc_p = sys_.search(data["pos_q"])
    _, sc_n = sys_.search(data["neg_q"])
    _, sc_c = sys_.search(data["calib_q"])
    n_pos = len(ids_p)
    correct = ids_p == data["pos_src"]
    thr = float(np.quantile(sc_c, 1.0 - target_fpr, method="higher"))
    hit_ok = (sc_p > thr) & correct
    out = {
        "recall_at_1": proportion(int(correct.sum()), n_pos)["rate"],
        "recall_lo": proportion(int(correct.sum()), n_pos)["lo"],
        "recall_hi": proportion(int(correct.sum()), n_pos)["hi"],
        "tpr_at_fpr": float(hit_ok.mean()),
        "realised_fpr": float((sc_n > thr).mean()),
        "threshold": thr,
        "n_pos": n_pos,
        "n_neg": len(sc_n),
    }
    if brute_ids is not None:
        out["ann_agreement"] = float((ids_p == brute_ids).mean())
    return out


# ----------------------------------------------------------------------------- FlyHash hyper-parameter sweep
def _flyhash_sweep(cfg: dict[str, Any], cache: np.ndarray, sweep_sets: list[tuple[float, dict[str, np.ndarray]]], log: Log) -> list[dict[str, Any]]:
    """Give FlyHash its best shot: expansion m, winners k (incl. the ~5% of the original paper) and fan-in density,
    each next to SimHash with the same number of bits, on the hardest paraphrase levels.

    Uses the inverted-index implementation, which returns exactly the same top-1 as the popcount scan (tested) but is
    far cheaper to evaluate on thousands of queries.
    """
    rows: list[dict[str, Any]] = []
    brute = BruteForceCosineCache()
    brute.build(cache)
    for m_bits, k, density in cfg["flyhash_sweep"]:
        log(f"[C] flyhash sweep m={m_bits} k={k} density={density}")
        fh = FlyHashInvertedCache(m_bits, k, density=density)
        fh.build(cache)
        for pc, sd in sweep_sets:
            met = _calibrated_metrics(fh, sd, cfg["target_fpr"], None)
            rows.append({"kind": "flyhash", "bits": m_bits, "k": k, "k_frac": k / m_bits, "density": density, "para_cos": pc,
                         "recall_at_1": met["recall_at_1"], "tpr_at_fpr": met["tpr_at_fpr"], "bytes_per_prompt": fh.memory_bytes() / len(cache)})
    for bits in sorted({m for m, _, _ in cfg["flyhash_sweep"]}):
        sh = SimHashCache(bits)
        sh.build(cache)
        for pc, sd in sweep_sets:
            met = _calibrated_metrics(sh, sd, cfg["target_fpr"], None)
            rows.append({"kind": "simhash", "bits": bits, "k": None, "k_frac": None, "density": None, "para_cos": pc,
                         "recall_at_1": met["recall_at_1"], "tpr_at_fpr": met["tpr_at_fpr"], "bytes_per_prompt": sh.memory_bytes() / len(cache)})
    return rows


# ----------------------------------------------------------------------------- run
def run(cfg: dict[str, Any], seed: int = 0, profile: str = "custom", include_external: bool = True, log: Log = null_log) -> WorkloadResult:
    res = WorkloadResult(workload=KEY, title=TITLE, profile=profile, seed=seed, params=dict(cfg))
    t_start = time.perf_counter()
    specs = _specs(cfg, include_external)
    rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    fly_rows: list[dict[str, Any]] = []
    sizes = [None] if cfg.get("embeddings_npz") else list(cfg["n_cache"])
    with limit_threads(cfg.get("threads", 1)):
        for n_cache in sizes:
            data = _dataset(cfg, seed, n_cache or 0, cfg["para_cos"])
            n = len(data["cache"])
            log(f"[C] N={n}, d={data['cache'].shape[1]}")
            brute_ids = None
            built: list[tuple[SystemSpec, CacheSystem, float]] = []
            for spec in specs:
                m = instantiate(spec, res, log)
                if m is None:
                    continue
                if m.max_items and n > m.max_items:
                    res.add_skip(spec.name, f"N={n} exceeds this system's cap of {m.max_items}")
                    m.close()
                    continue
                t0 = time.perf_counter()
                m.build(data["cache"])
                build_s = time.perf_counter() - t0
                built.append((spec, m, build_s))
                log(f"    built {spec.name} in {build_s:.2f}s")
                if m.name == "bruteforce_cosine":
                    brute_ids = m.search(data["pos_q"])[0]
            try:
                for spec, m, build_s in built:
                    met = _calibrated_metrics(m, data, cfg["target_fpr"], brute_ids)
                    pick = np.arange(cfg["n_latency"]) % len(data["pos_q"])
                    prep = [m.prepare_one(data["pos_q"][i]) for i in pick]
                    lat = summarize_latency(time_calls(m.search_one, prep))
                    batch = data["pos_q"][np.arange(cfg["batch"]) % len(data["pos_q"])]
                    reps = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        m.search(batch)
                        reps.append(time.perf_counter() - t0)
                    mem = m.memory_bytes()
                    rows.append({
                        "n_cache": n, "system": spec.name, "family": spec.family or m.family, "dim": int(data["cache"].shape[1]),
                        **met, "build_s": build_s, "index_bytes": mem, "bytes_per_prompt": mem / n, "fixed_bytes": m.fixed_bytes(),
                        "p50_us": lat["p50_us"], "p99_us": lat["p99_us"], "qps_single": 1e6 / lat["mean_us"],
                        "qps_batch": cfg["batch"] / float(np.median(reps)),
                    })
                if not cfg.get("embeddings_npz") and n_cache == cfg.get("sweep_n_cache"):
                    sweep_sets = [(pc, _dataset(cfg, seed, n, pc, tag=100 + pi)) for pi, pc in enumerate(cfg["para_sweep"])]
                    assert all(np.array_equal(sd["cache"], data["cache"]) for _, sd in sweep_sets)  # same corpus, new queries
                    brute = next(m for _, m, _ in built if m.name == "bruteforce_cosine")
                    for pc, sd in sweep_sets:
                        b_ids = brute.search(sd["pos_q"])[0]
                        for spec, m, _ in built:
                            met = _calibrated_metrics(m, sd, cfg["target_fpr"], b_ids)
                            sweep_rows.append({"n_cache": n, "para_cos": pc, "system": spec.name, "family": spec.family or m.family,
                                               "recall_at_1": met["recall_at_1"], "tpr_at_fpr": met["tpr_at_fpr"],
                                               "realised_fpr": met["realised_fpr"]})
                    fly_rows.extend(_flyhash_sweep(cfg, data["cache"], [x for x in sweep_sets if x[0] in cfg["flyhash_sweep_para"]], log))
            finally:
                for _, m, _ in built:
                    m.close()
    res.tables["systems"] = rows
    res.tables["hardness_sweep"] = sweep_rows
    if fly_rows:
        res.tables["flyhash_sweep"] = fly_rows
    _checks_and_headlines(cfg, res)
    res.notes += [
        "Embeddings are synthetic (topic structure + controlled-cosine paraphrases); absolute recalls depend on topic_rho/para_cos and must be re-measured on real embeddings (embeddings_npz).",
        "Latency is end-to-end for one query in one thread: it includes FlyHash/SimHash encoding. NumPy call overhead alone is several microseconds.",
        "Hit decisions use a per-system score threshold calibrated on separate novel prompts at the target false-hit rate.",
        "bytes_per_prompt excludes N-independent state (fixed_bytes: the random projection); FAISS sizes are the serialised index size.",
        "Qdrant runs in embedded local mode (a NumPy scan behind a client API) and is capped at 20k prompts.",
    ]
    res.elapsed_s = time.perf_counter() - t_start
    return res


def _checks_and_headlines(cfg: dict[str, Any], res: WorkloadResult) -> None:
    rows = res.tables["systems"]
    if not rows:
        return
    n_head = cfg.get("sweep_n_cache") if cfg.get("sweep_n_cache") in {r["n_cache"] for r in rows} else max(r["n_cache"] for r in rows)
    sel = {r["system"]: r for r in rows if r["n_cache"] == n_head}
    ns = {k: v for k, v in sel.items() if v["family"] == "neurostore"}
    d = next(iter(sel.values()))["dim"]

    bf = sel.get("bruteforce_cosine")
    if bf:
        res.add_check("Benchmark self-check: task is well-posed (exact cosine scan finds the source prompt)", ">= 99% recall@1", f"{bf['recall_at_1'] * 100:.1f}%", bf["recall_at_1"] >= 0.99,
                      note="If this fails the synthetic data is too hard to measure anything else.")
    if ns:
        fast = min(ns.values(), key=lambda r: r["p50_us"])
        res.add_check("Sub-microsecond semantic lookup (spec objective)", "< 1 us p50", f"{fast['p50_us']:.1f} us ({fast['system']})", fast["p50_us"] < 1.0,
                      note="Measured in NumPy/Python at N=%d including encoding; only specialised hardware could reach this." % n_head)
        fly = ns.get(next((k for k in ns if k.startswith("flyhash_inv")), ""), next(iter(ns.values())))
        res.add_check("FlyHash code is smaller than the dense float embedding", f"< {d * 4} bytes/prompt", f"{fly['bytes_per_prompt']:.0f} bytes ({fly['system']})",
                      fly["bytes_per_prompt"] < d * 4)
        sim = [v for k, v in sel.items() if k.startswith("simhash_") and v["family"] == "lsh"]
        fscan = next((v for k, v in ns.items() if k.startswith("flyhash_scan")), None)
        if sim and fscan:
            # compare against SimHash at the same number of bits
            same_bits = [s for s in sim if s["system"] == f"simhash_{cfg['flyhash'][0][0]}"]
            if same_bits:
                res.add_check("FlyHash recall@1 >= SimHash recall@1 at the same bit budget", "FlyHash >= SimHash",
                              f"{fscan['recall_at_1'] * 100:.1f}% vs {same_bits[0]['recall_at_1'] * 100:.1f}%", fscan["recall_at_1"] >= same_bits[0]["recall_at_1"])
        sw = res.tables.get("hardness_sweep", [])
        if sw:
            hardest = min(r["para_cos"] for r in sw)
            f_h = next((r for r in sw if r["system"].startswith("flyhash_scan") and r["para_cos"] == hardest), None)
            s_h = next((r for r in sw if r["system"] == f"simhash_{cfg['flyhash'][0][0]}" and r["para_cos"] == hardest), None)
            if f_h and s_h:
                res.add_check(f"FlyHash recall@1 >= SimHash recall@1 at the same bit budget on the hardest paraphrases (cosine {hardest:g})",
                              "FlyHash >= SimHash", f"{f_h['recall_at_1'] * 100:.1f}% vs {s_h['recall_at_1'] * 100:.1f}%", f_h["recall_at_1"] >= s_h["recall_at_1"],
                              note="The headline-level comparison saturates at 100% for both; the hardness sweep is where codes differ.")
        if bf:
            res.add_check("FlyHash recall@1 is within 5 points of exact cosine scan", "FlyHash >= exact - 5 pts",
                          f"{fly['recall_at_1'] * 100:.1f}% vs {bf['recall_at_1'] * 100:.1f}%", fly["recall_at_1"] >= bf["recall_at_1"] - 0.05)

    ctx = f"N={n_head}, d={d}, 1 thread"
    pick = lambda key: {k: (v[key], v["family"]) for k, v in sel.items()}  # noqa: E731
    res.headlines = [
        headline("Cache-hit recall@1 on paraphrases", "fraction", True, ctx, pick("recall_at_1")),
        headline(f"True-hit rate at <= {cfg['target_fpr'] * 100:g}% false hits", "fraction", True, ctx, pick("tpr_at_fpr")),
        headline("Query throughput, single query (encode + search)", "queries/s", True, ctx, pick("qps_single")),
        headline("Query throughput, batched", "queries/s", True, ctx, pick("qps_batch")),
        headline("Index memory per cached prompt", "bytes", False, ctx, pick("bytes_per_prompt")),
        headline("Index build time", "s", False, ctx, pick("build_s")),
    ]
    res.summary = {"headline_n_cache": n_head, "dim": d}
