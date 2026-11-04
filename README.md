# NeuroStore

Empirical Evaluation of a Neural-Coding Data Architecture.

NeuroStore stores and retrieves data with neural-coding primitives — **modern Hopfield attractor
memories**, **FlyHash expansive sparse codes**, **vector-symbolic (VSA) role–filler binding** and
**holographic knowledge graphs** — instead of location-addressed tables. This repository is the
**benchmark and practical-evaluation framework** that measures whether those primitives actually
beat standard computer-science structures (KV stores, vector databases, LSH, relational indexes) on
the workloads where location-addressed systems are weak.

It is built to be *trusted*: every system sees identical data, the strongest simple baseline is always
included, results carry confidence intervals, and quantities with a closed form are checked against
theory on every run. The full evaluation plan is in [`benchmark_strategies.md`](benchmark_strategies.md);
the reference run is in [`results/standard/REPORT.md`](results/standard/REPORT.md).

## Quick start

```bash
pip install -e '.[all]'      # core: numpy + threadpoolctl (Python >= 3.11); extras add the real baselines
pytest                        # validates the framework itself (~15 s)

neurostore list               # workloads, profiles, which baselines are installed here
neurostore run --profile quick                                   # seconds: exercises every code path
neurostore run -w a,c --profile standard --out results/my-run    # chosen workloads
neurostore run --profile standard --out results/standard --strict   # everything + fail if a self-check fails
neurostore report results/standard/results.json                  # re-render a report from stored data
```

`python -m neurostore ...` works without installing. Equivalent to the commands above:

| Workload | Id | What it measures |
|---|---|---|
| A — noisy telemetry key retrieval | `a` | exact-recall vs 0–50 % bit flips; tolerance vs store size; Hopfield β/attractor sweep; latency, memory, density |
| B — record completion / entity resolution | `b` | identify a record and reconstruct unknown fields from `K` of `F` fields (clean and corrupted cues) |
| C — semantic prompt cache | `c` | recall@1, hit rate at fixed false-hit rate, QPS, bytes/prompt: FlyHash vs HNSW/FAISS/Qdrant/SimHash/exact |
| D — holographic graph walking | `d` | single/multi-hop accuracy, ghost-node capacity, hops/ms vs pointer chasing |
| Capacity & density | `capacity` | `C_max` at BER ≤ 0.1 %, storage density η |
| Section 5 — compute-in-memory | `cim` | accuracy under analogue non-idealities (emulated) + latency (**modelled**, not measured) |

Profiles: `quick` (seconds, smoke test), `standard` (the reference run), `full` (larger stores, 10⁵ prompts).
`neurostore scaffold` writes an editable `config.json` for custom sweeps.

## What the reference run found (standard profile, one 4-core CPU, 1 thread)

Details, caveats and every number: [`benchmark_strategies.md` §10](benchmark_strategies.md) and [`results/standard/REPORT.md`](results/standard/REPORT.md).

* **Robustness is real, but it is similarity search, not neural coding.** A Hopfield attractor recovers stored keys at 42–47 % bit flips
  (D = 512 – 4096) where Python dict, RocksDB and Redis recover nothing at 1 % — and exact nearest-neighbour search tolerates exactly the same
  amount (it is the information-theoretic ceiling). The iterated attractor reaches it; the single softmax step of the original harness stops 4–7 points short.
* **Entity resolution (VSA)** matches an exact fuzzy scan on cues with a wrong field (97 % vs 96 %) where exact indexes get 0 % — at ~100× the latency and
  ~1 800× the memory as implemented.
* **Semantic caching (FlyHash)**: 100 % recall@1 and ≥ 99.6 % true hits at 1 % false hits, 1.4–2× smaller than float vectors (with / without its shared projection), up to 10× faster to build than HNSW (at 50 000 prompts) —
  but 2.7× slower to query than FAISS HNSW, and a lossier code than SimHash at equal bits (61 % vs 99.8 % recall@1 on hard paraphrases), on synthetic embeddings.
  Sub-microsecond lookup is not met (best semantic p50: 44 µs).
* **Holographic graph walking** follows its capacity law to within 7 % (≈ D/26 triples per vector at 100 entities) and encodes direction, but is
  ~1 400× slower than a dict per hop and ~1 600× larger.
* **Compute-in-memory (emulated)**: hyperdimensional search tolerates 20 % read noise, 40 % programming noise, 20 % stuck cells and a 2-bit ADC. The *latency* gain is
  a model: 170–10 000× under optimistic assumptions, **6–9×** under conservative ones, against the fastest CPU path.
* **Scorecard**: accuracy rows 3 wins / 9 ties / 0 losses; accuracy-gated cost rows 5 wins / 14 losses. The benchmark validates itself — seven closed-form self-checks pass.

## What is compared with what

Real systems, auto-detected at run time (a missing one is recorded as *skipped, with the reason*):

* **Exact KV** — Python `dict`, embedded **RocksDB**, a real **Redis** server
* **Exact / approximate NN** — packed-Hamming and float brute force (the accuracy ceiling), **FAISS** flat and **HNSW** (float and binary), **Qdrant** (embedded)
* **LSH** — bit-sampling LSH (tunable `k`, `L`), **datasketch** MinHash-LSH, SimHash
* **Relational** — per-column inverted index, linear scan, composite index per field subset, fuzzy scan

## Layout

```
neurostore/
  primitives/   hopfield.py  flyhash.py  vsa.py  holographic_graph.py  cim.py     # the systems under test
  baselines/    kv.py  ann.py  lsh.py  relational.py  graph.py  cache.py         # what they are compared with
  workloads/    a_telemetry.py  b_entity.py  c_prompt_cache.py  d_graph_walk.py  capacity.py  cim_emulation.py
  theory.py     closed-form oracles used to validate the benchmark itself
  common.py     seeded RNG, exact-count noise, bit packing/popcount, Wilson CIs, timing, memory sizing
  profiles.py   report.py   runner.py   results.py   cli.py
tests/          200+ tests: utilities, theory-vs-Monte-Carlo, every primitive and baseline, workloads, CLI
experiments/    one-off exploratory scripts (not part of the reference run), e.g. FlyHash vs SimHash across intrinsic dimension
results/standard/   committed reference run (results.json + REPORT.md) — specific to the machine named in it; results/README.md has the provenance
benchmark_strategies.md   the evaluation plan, protocols, defect audit of the reference harness, limitations
```

## Reading the results honestly

* The **scorecard** in each report compares the best NeuroStore system with the best system of each
  competing family on the same metric and labels it WIN / TIE / LOSS (> 5 % difference). It is computed
  from the stored data, not written by hand. *Quality* rows (accuracy) compare every system; *cost* rows
  (latency, memory, ...) compare only systems that pass an accuracy gate, so nothing wins on speed by
  returning nothing and NeuroStore is represented by a system that actually does the job.
* A softmax associative memory with large β **is** exact nearest-neighbour search, so on retrieval
  tasks the neural primitive cannot beat the exact-NN ceiling; the interesting questions are cost
  (latency, memory) and behaviour where exact methods break (corrupted cues, exact-match stores).
* **Latency** is single-thread Python/NumPy on one machine and includes interpreter overhead
  (the empty-call overhead is printed in every report). Memory is *as allocated*, which penalises
  `float32` storage of ±1 patterns.
* **Section 5 latency is a model**, not a measurement; only its accuracy sweeps are emulation of values.
* Workload C uses **synthetic** embeddings unless you supply real ones (`embeddings_npz`).

See `benchmark_strategies.md` §8 for the full list of limitations.
