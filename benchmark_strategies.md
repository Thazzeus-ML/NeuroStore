# NeuroStore Benchmarking & Practical Evaluation Framework

Practical strategies, workloads, baselines, metrics and reference code for evaluating the
NeuroStore neural-coding data architecture against standard computer-science primitives
(KV stores, vector databases, LSH, relational indexes).

This document is both the **evaluation plan** and the **map from that plan to the code**: every
numbered requirement below says where it is implemented and how to run it. The framework lives in
the `neurostore/` package; `README.md` has the quick start; `results/standard/REPORT.md` is the
generated report of the reference run.

> **Ground rules the framework enforces** (they are what make a result a *result*):
> 1. Every system sees **identical** data and queries (paired comparison, explicit seeds).
> 2. The strongest simple baseline is always present — for retrieval that is **exact brute-force
>    nearest neighbour**, because a softmax associative memory is that algorithm.
> 3. Proportions carry **95% intervals that respect clustering**: in Workload A each stored key is queried
>    many times and failures are largely key-driven, so the interval counts *keys*, not queries (conservative
>    Wilson on the number of keys); Workload D uses a t-interval across independent graphs. Thresholds are read
>    off a monotone envelope.
> 4. The benchmark **validates itself**: measured quantities with a closed form (identification
>    probability, linear-memory BER, capacity, graph-hop accuracy) are compared with theory, and
>    `neurostore run --strict` fails if they disagree.
> 5. **Measured** and **modelled** numbers are never combined into one score. The Section-5 table shows
>    a measured CPU time and a modelled hardware time side by side, in separately labelled columns, with the
>    assumptions printed beneath; it is excluded from the scorecard.
> 6. Optional baselines (FAISS, Qdrant, RocksDB, Redis, datasketch) are detected at run time and
>    recorded as *skipped, with the reason* when absent — never silently dropped.
> 7. **Cost is only compared between systems that do the job.** Latency, throughput, memory and build-time
>    rows in the scorecard carry an accuracy *gate* (e.g. "noise tolerance ≥ 90 % of the exact-NN ceiling at
>    this store size"); only systems passing it compete, and NeuroStore is represented by its best system
>    *among those that pass*. A system therefore cannot win on speed by returning nothing, and no row mixes
>    two NeuroStore operating points (a fast variant that fails the gate while a slower one supplies the
>    accuracy). Accuracy comparisons against exact-match stores on noisy keys are *structural* — they cannot
>    succeed by construction — and are reported, but cost comparisons are what decide the practical case.

---

## 1. Concrete application workloads

Location-addressed stores struggle where the key is *corrupted*, *partial* or only *similar*. Four
workloads probe exactly that.

| | Scenario | Objective | NeuroStore mechanism | Evaluation metric | Code |
|---|---|---|---|---|---|
| **A** | Noisy telemetry / edge-sensor stream: keys arrive with bit errors | Retrieve the exact stored metadata without retransmission or fuzzy join scans | Modern continuous Hopfield network (`beta >= 16`), softmax attractor | Exact-recovery rate over 0–50 % bit flips vs hash tables and LSH | `workloads/a_telemetry.py` |
| **B** | Entity resolution: primary key missing or redacted, only some attributes known | Identify the record and reconstruct unknown fields without `2^F` composite indexes | VSA / MAP role–filler binding, superposition, cleanup | Unbinding accuracy (cosine, bit agreement, top-1) with `K` of `F` fields | `workloads/b_entity.py` |
| **C** | LLM prompt caching: has a *paraphrase* of this prompt been seen? | Sub-microsecond semantic lookup | FlyHash expansive sparse k-WTA code + Hamming/overlap match | QPS, bytes per cached prompt, cache-hit recall@1 | `workloads/c_prompt_cache.py` |
| **D** | Multi-hop knowledge-graph walking `(Alice, WorksAt, Lab) ∧ (Lab, LocatedIn, City)` | Traverse by vector algebra, not pointer chasing | Binding / permutation algebra (MAP and HRR) | Hops per ms; capacity before ghost nodes | `workloads/d_graph_walk.py` |

### Protocol per workload (as implemented)

**A — noisy key retrieval.** `N` random bipolar `D`-bit keys → random `D`-bit values. At each noise
level `p` every system receives the same queries: stored keys with **exactly** `round(D p)` random
flips (exact count removes the variance of the flip count). Success = the returned value equals the
stored value in *all* `D` bits. Dimensions `D ∈ {512, 2048, 4096}` (spec next-step 1), noise grid
0–50 % in 1 % steps, 2000 queries/level. Extra tables: tolerance vs store size `N`, a Hopfield
`beta` × attractor-steps sweep, and latency / memory / density at `N = 100` and `N = 10 000`.
Alongside the measurements the **closed-form brute-force identification probability** is reported.

**B — record completion.** `M` records × `F` fields, vocabulary `V`. A query is a random record, a
random `K`-subset of its fields as the cue (optionally with `c` cue values *corrupted*), and one
*unknown* field as the target. Only cues that identify the record **uniquely** are used, so no
method is asked an ambiguous question; cases with fewer than two valid clues are excluded as
unanswerable. The VSA store works in **two stages** — associative lookup of the stored record
(`identify`), then unbind the target role from that *stored* record and clean up against the field's
filler codebook (`reconstruct`). Metrics: identification accuracy, reconstruction accuracy, cosine
between the unbound vector and the true filler, bit agreement. Dimensions 512–4096, `K ∈ {2,3,4,6}`.

**C — semantic prompt cache.** Synthetic embeddings with *topics* (unrelated prompts of one topic
have cosine `topic_rho`, so negatives are hard) and *paraphrases* at controlled cosine. Metrics per
cache size: recall@1 on paraphrases; **true-hit rate at a fixed false-hit rate** (threshold calibrated
on separate novel prompts, evaluated on held-out ones); single-query and batched throughput; bytes
per cached prompt; build time; agreement with exact cosine scan (separates approximation error from
encoding error). The **true-hit rate is read off each system's ROC at exactly the target false-hit rate**
(randomised handling of tied scores — a plain `score > quantile` rule lands far below the target for coarse
integer scores like FlyHash overlaps and would penalise them); the deployment protocol (threshold fitted on
separate calibration prompts) is stored beside it. A hardness sweep varies the paraphrase cosine over the **same corpus and topics** (only
the queries change). A FlyHash hyper-parameter sweep (expansion `m`, winners `k` including the ≈5 % of the
original paper, fan-in density) runs next to SimHash with the same number of bits, so FlyHash is judged
at its best configuration and not at one setting. Real embeddings can be supplied
(`embeddings_npz`, see `workloads/c_prompt_cache.py:load_embeddings`).

**D — holographic graph walking.** All `T` triples are superposed in **one** `D`-vector; a hop unbinds
`subject ⊛ ρ(relation)` and cleans up against the entity codebook; a walk feeds the result into the
next query. Direction is encoded with a **random permutation** `ρ` (see §4). Metrics: single-hop
accuracy (`ghost rate = 1 − accuracy`), multi-hop path accuracy, capacity `T_max` at 99 %, hops per
millisecond, and a direction control (a reversed query must not return the subject).

---

## 2. Standardised baselines

Real systems, not stand-ins. "family" is the key the scorecard groups by.

| Class | System | Family | What it is here | Benchmark focus |
|---|---|---|---|---|
| Exact KV | Python `dict` | `kv` | packed-bit `bytes` → `bytes` | 0 % noise lookups, latency, memory per pair |
| Exact KV | **RocksDB** (`rocksdict`) | `kv` | embedded, raw bytes; memory = persisted SST bytes | idem |
| Exact KV | **Redis** (`redis-server`) | `kv` | real server subprocess on a free port, localhost TCP; memory = `INFO used_memory` growth | idem |
| Exact search | brute-force Hamming / cosine | `ann` | XOR+popcount scan; float32 BLAS scan — **the accuracy ceiling** | what any retriever can at best achieve |
| ANN / vector DB | **FAISS** `IndexBinaryFlat/HNSW`, `IndexFlatIP`, `IndexHNSWFlat` | `ann` | `M=32`, `efConstruction=200`, `efSearch=64`; binary HNSW is also run at `efSearch` 256 and 1024 in the store-size sweep and latency table | accuracy vs latency vs build time |
| ANN / vector DB | **Qdrant** (embedded local mode) | `ann` | NumPy scan behind a client API; capped at 20 k items | client/DB overhead |
| LSH | custom **bit-sampling LSH** | `lsh` | `k` sampled bits × `L` tables + exact re-rank; `P(found) = 1 − (1 − (1−p)^k)^L`; three presets from fast `(12,32)` to noise-robust `(5,256)` | noise tolerance is a tuning knob paid for in candidate-set size and latency |
| LSH | **datasketch** MinHash-LSH | `lsh` | Jaccard over the set of `+1` coordinates | idem |
| LSH | SimHash | `lsh` | random-hyperplane signatures, Hamming scan, same bit budget as FlyHash | the FlyHash-vs-LSH comparison |
| Relational | per-column inverted index + intersect; linear scan; **composite index per field subset**; fuzzy scan | `relational` | exact; fuzzy scan is the "expensive fuzzy join" and the ceiling for corrupted cues | memory blow-up `C(F,K)` vs `F`, query cost |
| Graph | dict adjacency, dense successor table | `kv` | exact pointer chasing | hops/ms |

`neurostore list` prints which of these are installed on the current machine.

---

## 3. Key metrics & benchmarking protocol

**Attractor recovery threshold `N_noise`.** The largest bit-flip rate at which exact recall stays
≥ 99 % (`P_fail` is the next grid point). Computed on the *running minimum* of the recall curve so a
lucky point after the first failure cannot move it. For exact brute force it equals the closed-form
value `theory.noise_threshold_bruteforce`, which is reported next to every measurement.

**Superposition capacity `C_max`.** The largest `N` stored in a fixed `D` before the bit error rate
exceeds 0.1 %, measured by `workloads/capacity.py` for three memories:

* *linear (Hebbian) `D×D` matrix* — the textbook "fixed matrix". `BER = Q(√(D/(N−1)))`, so
  `C_max ≈ D / 9.55` (measured: 54 at `D=512`, 215 at `D=2048`, matching theory to < 1 %).
* *VSA bundle* — `N` role–filler pairs in one `D`-vector. A single bit has SNR `1/√(N−1)`
  **independent of `D`**, so the raw BER exceeds 0.1 % already at `N = 2`. `D` buys reliable
  *cleanup*, so capacity is defined on the cleaned-up output (`BER = SER/2`).
* *explicit modern Hopfield* — stores `N × D` cells, so it is never crosstalk-limited at clean
  keys; its capacity is *censored* ("at least the largest `N` tested") and bounded by memory.

**Execution-latency ratio `τ_read / τ_write`.** Median single-operation latency (`perf_counter_ns`
per call, GC off, 1 thread, inputs pre-converted outside the timed region) for a read of a 10 %-noisy
key and for an incremental write, reported with p99. All systems have amortised-O(1) appends
(`common.GrowableRows`), so a write measures the algorithm, not an `O(N)` array copy. The empty-call
overhead is stored in every result file (`environment.timer_overhead_ns`); latencies near it are
interpreter overhead.

**Storage density `η`.**

```
η = (uncompressed data bits) / (allocated memory bits)
```

*Data bits* are `N·2D` for key–value pairs (`N·F·log2 V` for records, `N·log2 V` for a bundle).
*Allocated* is everything the structure must keep to answer a query, as it actually exists in
memory: Python-object overhead for dicts, float32 for NumPy matrices, serialised size for FAISS,
SST bytes for RocksDB, `used_memory` for Redis. Query cues are not counted. Note what this does to
the neural primitives: a ±1 pattern stored in `float32` has `η = 1/32`; bit-packing would make it 1.

---

## 4. Reference code: what was wrong and what replaced it

The benchmark harness this project started from (Workloads A–C and a CLI) was run and audited before
being extended. The defects below would have produced misleading numbers; each has a regression test.

| # | Reference behaviour | Consequence | Replaced by | Test |
|---|---|---|---|---|
| 1 | CLI: `run_parser subparsers.add_parser(...)` (missing `=`) | `SyntaxError`; the suite could not run at all | `neurostore/cli.py` | `test_REGRESSION_module_entry_point_parses_and_runs` |
| 2 | Workload B unbound the **ID** role from a cue made of `{dept, title, loc}` | The ID is not in the cue, so the output is pure noise (cosine ≈ 0 ± 1/√D); the script printed `Exact Key Recovery Success: NO` for *any* architecture | Two-stage lookup: identify the stored record, then unbind from it | `test_REGRESSION_original_harness_unbound_a_role_that_is_not_in_the_cue` |
| 3 | Workload A compared Hopfield only with an exact-match hash table | A strawman: it scores 0 % at any flip. Hopfield with `beta → ∞` *is* brute-force NN, so the honest comparison is against that | Exact NN ceiling, FAISS, LSH, datasketch, Qdrant, RocksDB, Redis; closed-form curve | `test_beta_infinity_limit_is_exact_nearest_neighbour_search` |
| 4 | Workload C only timed the *encoding* of 500 prompts | No retrieval, no paraphrases, no recall, no false hits, no baselines — none of the spec's metrics | Full protocol (§1) with calibrated hit decisions | `test_workload_c_ranks_the_systems_as_expected` |
| 5 | Workload D listed in the spec but absent from the code | — | `workloads/d_graph_walk.py` | `test_workload_d_capacity_physics` |
| 6 | Global `np.random` state; `time.time()`; per-query Python loops | Irreproducible; coarse timing; slow | Seeded `Generator`s, paired queries, `perf_counter_ns` per call, vectorised batches | `test_results_are_reproducible_from_the_seed_and_change_with_it` |
| 7 | `bundled[bundled == 0] = 1.0` | Ties resolved towards +1 (a bias that grows with an even number of terms) | Uniformly random tie-break | `test_map_bundle_resembles_its_parts_and_binarisation_breaks_ties_without_bias` |
| 8 | `int(len * rate)` flips; i.i.d. choice per query | Floor instead of round; slow | `round`, exact count, vectorised `argpartition` | `test_flip_exact_flips_exactly_n_per_row` |
| 9 | Hopfield re-normalised all keys on every query; hash table keyed by tuples of NumPy floats | Extra `O(N·D)` per read; inflated hash-table memory | Cached key norms; packed-`bytes` keys | `test_matches_the_reference_implementation` |
| 10 | Single softmax readout only | The spec's "continuous auto-associative **attractors**" were never iterated; exact value recovery breaks before identification does | `retrieve(steps=k)` attractor iteration, reported as its own system and in a `beta × steps` sweep | `test_iterated_attractor_dynamics_complete_a_noisy_pattern` |
| 11 | `scaffold` wrote `./neurostore/benchmarks/` | Collides with the importable package | `neurostore scaffold` writes a config + results workspace to `./benchmark_workspace`, never overwriting without `--force` | `test_cli_scaffold_is_non_destructive` |

`neurostore.primitives.hopfield.ModernHopfieldMemory.retrieve(steps=1)` is verified to be output-identical
to a verbatim port of the reference `ContinuousHopfieldMemory`.

**Pitfalls found while building the framework itself** (not in the reference code, but each cost a bug
here and each is now covered by a test):

* *Direction needs a random permutation.* The first graph memory used a cyclic shift as the permutation
  `ρ`. For circular-convolution (HRR) binding a shift is a convolution with a delta vector, so it commutes
  with binding and the edge direction is lost: reversed queries returned the subject 86 % of the time.
  A fixed random permutation restores it (reverse hit rate at chance).
  Tests: `test_cyclic_shift_commutes_with_circular_convolution_but_a_random_permutation_does_not`,
  `test_edge_direction_is_encoded`.
* *Exactly D/2 flips is not "chance".* A query with exactly `D/2` flips lies at the median distance from
  its own key, while the nearest of `N−1` random impostors lies ≈ 2.5σ below the median, so nearest-neighbour
  search returns the truth with probability ≈ 0.5^(N−1) — worse than the 1/N of an independent random query.
  The 50 % end of the noise sweep therefore reads ≈ 0 % for every nearest-neighbour method.
  Test: `test_identification_formula_edge_cases_and_monotonicity`.
* *Physical clipping makes programming noise one-sided.* A device programmed to `g_max` cannot overshoot,
  so ±1 weights see truncated, biased noise (spread × 0.87, mean ∓0.4σ). Test:
  `test_programming_noise_is_one_sided_at_the_conductance_limits`.
* *Report generation must survive an empty workload.* A crashed workload is recorded and the run
  continues; the renderer used to raise on the empty result and would have discarded the report of a
  multi-minute run. Test: `test_report_survives_a_crashed_workload_and_a_renderer_bug`.

---

## 5. Software emulation of compute-in-memory (non-von-Neumann)

Two different kinds of evidence, kept apart on purpose.

**What software *can* establish — accuracy under non-idealities (emulated, checkable).**
`primitives/cim.py` implements a tiled crossbar with differential conductance pairs
`(G⁺ − G⁻)`: inputs on word lines, one analogue read per tile, Kirchhoff summation on the bit lines,
per-tile ADC, digital accumulation across tiles. Non-idealities, each switchable and unit-tested:
programming noise (with physical clipping at `g_max`), stuck-at cells, power-law conductance drift
with device-to-device spread and global compensation, per-read thermal/read noise (a fraction of the
tile full scale), ADC resolution and range, DAC resolution. `workloads/cim_emulation.py` sweeps one
non-ideality at a time, plus a combined "realistic" scenario, at several dimensions and
query-noise levels, and reports identification accuracy and exact value recall relative to the ideal
digital result. The ideal configuration is verified to reproduce the digital scores exactly.

**What software *cannot* establish — hardware latency.** `CIMTimingModel` projects search latency
from explicit device-timing parameters (DAC, array read, ADC, converters per array, arrays that run
concurrently, adder-tree and winner-take-all costs). The presets are **illustrative placeholders**,
to be replaced with numbers from a real array characterisation. Under the model the analogue time
is size-independent *only while all tiles fit the array budget*; beyond it the waves serialise.
The report puts the projection next to the **measured** best single-thread CPU time and labels the
column "speed-up if the model holds". It is never entered into the scorecard.

Not modelled: programming/write time and energy, IR drop and sneak paths, host I/O, non-linear
device I–V, temperature.

---

## 6. Execution plan

```bash
pip install -e '.[all]'          # core: numpy + threadpoolctl (Python >= 3.11); extras add the real baselines
pytest                            # 200+ tests, ~15 s: validates the framework itself
neurostore list                   # which baselines are installed here

neurostore run --profile quick                          # seconds: smoke test of every code path
neurostore run --profile standard --out results/standard --strict   # reference run
neurostore run --profile full --out results/full        # larger stores, 10^5-prompt cache
```

Results are written to `results.json` **after every workload** (atomically), so an interrupted run keeps
what finished and `neurostore report` can render it. Every result file records the machine, library
versions, whether single-thread timing was enforced, and the git revision of the code that produced it.

The three suggested next steps map to single commands:

1. **Attractor breakdown curves at `D = 512, 2048, 4096`** — `neurostore run -w a` (the standard
   profile sweeps exactly these dimensions and also reports `N_noise` against the closed form,
   the `beta` × attractor-steps sweep, and the dependence on store size).
2. **FlyHash k-WTA vs HNSW for semantic caching** — `neurostore run -w c` (FAISS `IndexHNSWFlat`,
   FAISS binary HNSW *over FlyHash codes*, FlyHash popcount scan and FlyHash inverted index, SimHash,
   exact scans, Qdrant, exact hash).
3. **Memristive execution** — `neurostore run -w cim` (tolerance sweeps + modelled latency).
   `-w capacity` gives `C_max` and `η`; `-w b` and `-w d` give the entity-resolution and graph results.

To use your own data: `neurostore scaffold`, edit `benchmark_workspace/config.json`, then
`neurostore run --config benchmark_workspace/config.json`. For Workload C supply real embeddings with
`{"c": {"embeddings_npz": "emb.npz"}}` (arrays `cache`, `queries`, `labels`; label `-1` = novel prompt).

---

## 7. Statistical methodology and self-validation

* **Randomness.** Every random draw comes from an explicit `numpy.random.Generator` seeded from
  `(seed, workload, cell)`; a run is exactly reproducible and changing the seed changes the data.
* **Intervals.** Binomial proportions use the Wilson score interval; the JSON stores `rate, lo, hi, n`.
* **Thresholds.** `threshold_crossing` applies a running minimum to the (noisy) recall curve before
  locating the 99 % crossing, then interpolates linearly.
* **Calibration vs test.** Cache hit thresholds are set on a *separate* set of novel prompts.
* **Oracles.** Each of these is compared with an independent closed form or Monte-Carlo, both in
  `tests/` and as a `Benchmark self-check` inside every run:

  | Quantity measured | Oracle | Agreement observed |
  |---|---|---|
  | brute-force identification vs flips | exact binomial formula `((b+c)^N − c^N)/(N b)` | within sampling error |
  | linear-memory BER and `C_max` | `Q(√(D/(N−1)))` | 54.1 vs 54 (`D=512`), 215.0 vs 215 (`D=2048`) |
  | bundle cleanup SER / capacity | `E[Φ(X)^(V−1)]`, `X ~ N(√(D/N), (N−1)/N)` | 16.0 vs 16, 64.2 vs 63 |
  | holographic single-hop accuracy | same law | MAP within 1 s.e.; HRR consistently 1–2 points *below* it |
  | ideal crossbar | digital matrix product | max error 0 |

* **Known model gap.** Unitary-HRR sits 1–2 accuracy points under the Gaussian capacity law (MAP does
  not); the self-check tolerance (±0.03 pooled) accommodates it and the report states it.

---

## 8. Threats to validity and limitations

* **Synthetic data.** Workload C uses synthetic embeddings. Absolute recalls and false-hit trade-offs
  depend on `topic_rho` and `para_cos`; re-measure on real sentence embeddings (`embeddings_npz`)
  before drawing deployment conclusions. Workloads A, B and D use i.i.d. random hypervectors — the
  regime the theory covers — not structured real telemetry.
* **Python/NumPy implementations.** Latencies measure *these* implementations on *this* CPU with
  interpreter overhead. They say nothing about what an optimised C++/SIMD/GPU implementation of the
  neural primitives could reach; FAISS, RocksDB and Redis are native code and are favoured by that.
* **One random key set per dimension.** Noise thresholds in Workload A come from a single draw of the
  stored keys; near the 99 % crossing the recall curve is steep (tens of points per noise step), so read a
  threshold to ±1 grid step. The closed-form rows are the key-set average, and the self-check compares the
  two over the whole grid. Systems evaluated on a prefix of the queries (datasketch, Qdrant) have coarser
  99 % thresholds (at most 1 error in 100); the report lists the count per system.
* **Single machine, 1 thread.** Throughput is per core. Multi-core scaling, NUMA and cache effects of
  stores larger than RAM are out of scope.
* **Memory accounting differs by system** (deep object size, serialised size, SST bytes, server
  `used_memory`); the report names the method. Storing ±1 patterns as `float32` is how these
  primitives are implemented here, not a property of the idea.
* **Scale.** The standard profile stops at 10⁴ items (A) and 5·10⁴ prompts (C). The `full` profile
  adds 10⁵; the spec's 10⁶ regime is not exercised.
* **Section 5 latency** is a model (§5).

---

## 9. Extending the framework

*Add a baseline.* Subclass `baselines.base.AssocSystem` (or `IdBasedSystem` if it maps a key to an
item index) / `baselines.cache.CacheSystem`; implement `build`, the batch `retrieve_values`/`search`,
and the single-op `prepare_*`/`*_one` pair for latency; set `family`; add an `available()` classmethod
if it needs an optional library; register it in the workload's `_specs`. The harness handles
timing, memory, skipping and reporting.

*Add a workload.* Create `workloads/<name>.py` exposing `KEY`, `TITLE` and
`run(cfg, seed, profile, include_external, log) -> WorkloadResult`; add per-profile settings to
`profiles.py`; add `WORKLOADS`/`RENDERERS` entries; return `headlines` (standardised comparison
entries) and `checks` (spec targets, or `Benchmark self-check: ...` for oracle comparisons).

---

## 10. Results of the reference run

See [`results/standard/REPORT.md`](results/standard/REPORT.md) (generated) and the interpretation
below.

<!-- RESULTS-SUMMARY-PLACEHOLDER -->
