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
* **Intervals.** Workload A counts *keys*, not queries (conservative Wilson on the number of stored keys, since the
  20 noisy replicates of a key share its geometry); Workload D uses a t-interval across independent graphs; elsewhere
  the Wilson score interval. The JSON stores `rate, lo, hi, n` (and `n_clusters` / `n_graphs`).
* **Thresholds.** `threshold_crossing` applies a running minimum to the (noisy) recall curve before
  locating the 99 % crossing, then interpolates linearly.
* **Matched operating point.** Workload C reads each system's true-hit rate off its ROC at *exactly* the target
  false-hit rate (randomised Neyman–Pearson tie handling). The deployment-style threshold, fitted on separate
  calibration prompts, is reported beside it (`deployed FPR`) to show how well a threshold transfers.
* **Oracles.** Each of these is compared with an independent closed form or Monte-Carlo, both in
  `tests/` and as a `Benchmark self-check` inside every run:

  | Quantity measured | Oracle | Agreement observed |
  |---|---|---|
  | brute-force identification vs flips | exact binomial formula `((b+c)^N − c^N)/(N b)` | max deviation 0.028 over the grid; thresholds 42/45/47 % vs exact 41.8/45.8/47.1 % (`D=512/2048/4096`) |
  | exact-NN noise tolerance vs store size | same formula | 42/40/39 % measured vs 42/40/39 % (`N=100/1000/10000`) |
  | linear-memory BER and `C_max` | `Q(√(D/(N−1)))` | 54/109/213/429 vs 54/108/215/429 (`D=512…4096`); worst error 0.9 % |
  | bundle cleanup SER / capacity | `E[Φ(X)^(V−1)]`, `X ~ N(√(D/N), (N−1)/N)` | 16/32/63/125 vs 16/31/63/126; worst error 4.1 % |
  | holographic single-hop accuracy | same law | pooled over the transition region: MAP +0.000, HRR −0.010 |
  | holographic capacity `T_max` (bisection on simulations) | same law, solved for 99 % | measured/theory 0.94–1.07 over 8 cells; mean error 4.2 %, max 7.5 % |
  | ideal crossbar | digital matrix product | max error 0 |

* **Known model gap.** Unitary-HRR sits 1–2 accuracy points under the Gaussian capacity law (MAP does
  not), i.e. about 6 % lower capacity (597 vs 637 at `D=16384`); the self-check tolerances accommodate it and
  the report states it.
* **Independent review.** The code was read, cold and read-only, by a separate reviewer who was given the
  source but none of the conclusions; each finding was verified before acting on it. It led to the gated
  scorecard, the matched-FPR comparison, the capacity bisection, the honest memory accounting, and the discovery
  that the inverse permutation was being recomputed on every hop (71 % of a hop at `D=8192`).

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

Raw data and the generated report: [`results/standard/`](results/standard/) (`results.json`, `REPORT.md`; provenance in
[`results/README.md`](results/README.md)). One machine (4 × Xeon @ 2.1 GHz), one thread, Python/NumPy implementations of
the neural primitives against native FAISS/RocksDB/Redis. Revision `3a14086`, ~25 minutes, every self-check passed.
Everything below is a statement about *this* evidence; §8 lists what it cannot show.

### The short version

* **Where NeuroStore's primitives work, they work as the theory says.** All seven closed-form self-checks agree with the
  measurements (§7). The claims in the spec about robustness are real: a Hopfield attractor recovers keys at 42–47 % bit
  flips where RocksDB, Redis and a dict recover nothing at 1 %; VSA completes records from a cue with a wrong field
  where exact indexes return nothing; HD search survives large analogue noise.
* **But the robustness is a property of *similarity search*, not of neural coding.** Exact nearest-neighbour search has the
  same noise tolerance (it is the information-theoretic ceiling: 41.8 / 45.8 / 47.1 % at `D=512 / 2048 / 4096`, `N=100`)
  and the Hopfield memory can only match it (the iterated attractor does; the single-step reference readout falls
  4–7 points short). On the scorecard the quality rows are 3 wins (all against exact-match stores, which cannot succeed
  by construction), 9 ties and **0 losses**.
* **On cost, NeuroStore loses on a CPU.** Among systems that do the job, the cost rows are 5 wins and 14 losses. The
  wins are each over *one* competing family in a row where another family is still ahead (robust LSH on write latency and
  memory, but exact packed search beats both; SimHash scans on throughput, but HNSW is faster; float32 vectors on cache memory,
  but SimHash-512 is 4× smaller). Against the best system in each row the losses run from
  **2.7× (cache throughput vs HNSW) to ~4 800× (batched graph hops)** in latency/throughput and from **4× to ~1 850×** in memory (the build-time rows are also lost, trivially: an exact scan has no index to build).
  Nothing here supports "outperforms" claims on speed or density in software.

### Workload A — noisy telemetry key retrieval

* **Noise tolerance (≥ 99 % exact recall, `N=100`)**: exact NN / FAISS flat / Qdrant **42 / 45 / 47 %**; Hopfield
  iterated (5 attractor steps) **42 / 45 / 47 %** (ties the ceiling); Hopfield single-step (the reference) 38 / 40 / 40 %;
  bit-sampling LSH 28 % at `k=8, L=64` but **42 / 45 / 46 %** at `k=5, L=256`; datasketch 22–25 %; dict / RocksDB / Redis 0 %.
  LSH noise tolerance is a tuning knob — the tolerant setting reaches the ceiling and pays for it (below).
* **Softmax blending is the single-step weakness.** Identification is β-independent (40 % on the 2 % grid) but exact value recovery needs
  β ≥ 32 in one step, or β ≥ 16 with five attractor steps (β=20, 40 % noise: 28 % → 100 %). The reference harness's
  single readout never exercises the attractor dynamics the spec describes.
* **Scale hurts tolerance, as the formula says**: 42 → 40 → 39 % for `N=100 → 1 000 → 10 000`. Approximate indexes
  lose more: FAISS binary HNSW at `efSearch=64` drops to **26 %** at `N=10 000`; it takes `efSearch=256` (250 µs/query) to
  recover the 39 % ceiling — 9× slower than exact FAISS flat (27 µs) at that size.
* **Cost at `N=10 000`, `D=512` (reads of 10 %-noisy keys, 1 thread)**: dict 0.29 µs, RocksDB 4.9 µs, FAISS flat **27 µs**,
  Redis 65 µs (TCP), brute-force packed Hamming 274 µs, Hopfield single-step 1.6 ms, robust LSH 5.9 ms, **Hopfield
  iterated 7.8 ms**, Qdrant (embedded) 9.2 ms. Memory per pair: packed exact NN / FAISS flat 128 B (η = 1.00), dict 223 B,
  RocksDB 139 B, Redis 224 B, HNSW 400 B, **Hopfield 4 100 B (η = 0.031 — float32 storage of ±1 bits)**, LSH 2.6–9.4 KB.
  Writes: Hopfield 4 µs, FAISS flat 2 µs, HNSW 330 µs, Redis 75 µs.
* **Reading**: the accuracy-gated comparison pits iterated Hopfield (7.8 ms) against exact FAISS flat (27 µs): **~290× slower**,
  and 32× larger than a bit-packed store. (Without gating, the single-step variant looks 5× faster than that — and 6 points
  less tolerant; the earlier reference-run scorecard mixed the two.)

### Workload B — record completion and entity resolution

* **Accuracy** (`M=5 000`, `F=8`, vocab 500, `D=4096`): with clean cues VSA identifies and reconstructs 100 %, identical to
  every exact method. With **one wrong cue field** (`K=3`) VSA reaches **97.4–97.6 %** vs fuzzy scan 96.0 % (tie; ties in
  the table break arbitrarily for both) vs **0 %** for inverted index, linear scan and composite indexes. VSA needs
  `D ≥ 1024` for this (`D=512`: 91 % / 82 %). The unbound-filler cosine is ≈ 0.27 independent of `D`.
* **The original harness's method is chance** (cosine 0 ± 1/√D; 0 % pass the 0.6 bar at every `D`).
* **Cost**: 3.4 ms vs 18–34 µs per query (100–194×); **29.5 KB per record** vs 16–54 B (the item memories are included;
  bit-packed it would be 922 B/record, η = 0.0097 vs 0.56 for the table). The spec's "O(1) query steps" is true of the operation
  count; the dense `O(M·D)` scan is what runs.
* **The `2^F` index claim is real but overstated.** All-subset composite indexes would need 255 indexes; for `K=2…4` they cost
  18–61 MiB as Python dicts (1.1–4.0 MiB as compact arrays) vs **188 KiB** for one inverted index per column. A standard
  engine never builds them; the inverted index + intersection is the practical rival and it is cheap.

### Workload C — semantic prompt cache (FlyHash vs the rest), `N = 10 000`, `d = 128`, synthetic embeddings

* **Recall@1 on paraphrases (cosine 0.9) is 100 % for every semantic system, true-hit rate at exactly 1 % false hits ≥ 99.6 %**
  (98.4 % at `N=50 000`); the exact-match hash finds 0 %.
* **Speed**: FAISS HNSW **21 100 q/s** (44 µs); FlyHash inverted index 7 900 q/s (117 µs); FlyHash + binary HNSW 6 800; exact cosine
  scan 5 400 single-query (47 700 batched); SimHash-512 scan 3 000. At `N=50 000`: HNSW 10 700, FlyHash inverted 2 000. FlyHash's
  sparsity pays off only as an inverted index (6× over its own popcount scan at 10k); it never beats HNSW (2.7× faster at 10k, 5× at 50k).
* **Memory**: FlyHash 258 B index / 361 B with its projection vs 512 B float vs HNSW 784 B vs SimHash-512 90 B. Build: FlyHash inverted
  1.0 s vs HNSW 10.2 s at `N=50 000`.
* **Sub-microsecond lookup is not met** (at `N=10 000` the best semantic p50 is 44 µs, FAISS HNSW; FlyHash 117 µs). The only sub-µs system is the exact hash
  (0.3 µs), which recognises 0 % of paraphrases.
* **FlyHash is a lossier code than SimHash at equal bits.** Recall@1 at paraphrase cosine 0.6: FlyHash `m=2048,k=64` 61.4 % vs
  SimHash-2048 99.8 % (SimHash-512, one quarter of the bits, 89.3 %). The hyper-parameter sweep gives FlyHash every chance: a larger
  expansion and 5 % winners reach 92.6–97.2 % but at 3–6× the memory of SimHash-2048. A difficulty-matched check across intrinsic
  dimensions 8–128 ([`results/exploratory/flyhash_intrinsic_dim.md`](results/exploratory/flyhash_intrinsic_dim.md)) does not reverse the
  ordering. The caveat is the data: embeddings are synthetic Gaussian mixtures; FlyHash may behave differently on real sentence embeddings
  (use `embeddings_npz`).
* **Ill-posed regime**: below paraphrase cosine ≈ 0.8 even exact cosine scan cannot separate paraphrases from same-topic novel prompts at 1 %
  false hits (the report explains why); recall@1 remains meaningful.

### Workload D — holographic graph walking

* **A single trace holds `T_max ≈ D / 26` triples** at 99 % single-hop accuracy for `E=100` entities: measured (bisection) 86 / 164 / 328 / 627 (MAP)
  and 76 / 160 / 298 / 597 (HRR) for `D = 2 048 … 16 384`, against 80 / 159 / 318 / 637 from the closed form. Beyond capacity the
  ghost-node rate rises smoothly (`T=1 600, D=2 048`: 10 % correct) and multi-hop accuracy compounds (`T=400, D=8 192`: 97 % → 84–88 % over 5 hops).
* **Direction is encoded** (reverse hit rate 0.9–1.1 % ≈ chance 1 %) — after fixing the cyclic-shift trap (§4).
* **Speed and size**: 3.1 hops/ms vs 4 425 for a dict (`E=1 000, D=2 048`; **~1 400× slower**), 0.34 hops/ms at `E=10 000`; batched 26.8 vs 127 872
  for a numpy successor table. Memory ≈ **267 KB per triple at capacity** vs 161 B (item memory dominates). Only parallel in-memory hardware could change this.

### Capacity and density (`C_max`, η)

* Linear matrix: **54 / 109 / 213 / 429** items at BER ≤ 0.1 % for `D = 512 … 4 096` (= D / 9.55), η ≈ 0.0065. VSA bundle read *with cleanup*: **16 / 32 / 63 / 125**
  (η ≈ 6·10⁻⁵); read bit-by-bit it is already at 31 % BER with 5 items. The explicit Hopfield store never saturates (≥ 10 000 tested) because it spends
  `N × D` float32 cells (η = 1/32).

### Section 5 — compute-in-memory

* **Accuracy (emulated)**: with 1 000 keys, `D=2 048` and 20 % query noise, identification stays ≥ 99 % up to read noise of 20 % of a tile's full scale,
  programming noise of 40 % and 20 % stuck cells (the largest values swept) and a 2-bit ADC; a combined realistic scenario (2 % read noise, 8-bit ADC, 5 % programming noise, 1 % stuck cells,
  drift with compensation) is 100 %. Tolerance to *per-read* noise grows with `D`; at `D=512` with 35 % query noise it needs a 4-bit ADC.
* **Latency (modelled, not measured)**: against the *fastest* CPU path here (FAISS binary flat), the model projects 170–10 000× under optimistic device
  timings but **6–9×** under conservative ones, because once tiles outnumber arrays the waves serialise and the advantage stops growing with size.
  The assumptions, not the physics, decide the answer; replace the illustrative timing parameters with array characterisation data before quoting a number.

### What this does and does not show

* **Shown**: the primitives are correct and behave as theory predicts; similarity search tolerates corrupted keys and cues, which exact-match stores cannot;
  the benchmark itself is validated against closed forms; the originally supplied harness could not have shown any of this (§4).
* **Not shown**: a speed, memory or accuracy advantage over exact or approximate nearest-neighbour search in software. In every cost comparison among
  systems that do the job, the specialised native baseline wins by a wide margin, and FlyHash does not beat SimHash or HNSW on this data.
* **Not tested**: real embeddings and real telemetry; stores beyond 10⁵; multi-threaded or GPU implementations; actual crossbar hardware. The
  neural primitives' unrealised potential (bit-packed storage, popcount kernels, parallel analogue search) is exactly what the CIM model gestures at and
  what a hardware evaluation would have to measure.
