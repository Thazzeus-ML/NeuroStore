# NeuroStore benchmark report

- framework `0.1.0` | profile `standard` | seed `0` | generated 2026-11-04T19:50:14+00:00 | code revision `3a140866b7`
- machine: Intel(R) Xeon(R) Processor @ 2.10GHz x4, 15.7 GiB RAM, Linux-6.18.44-fc-v233-x86_64-with-glibc2.39
- software: Python 3.13.18, NumPy 2.5.3 (scipy-openblas); optional libraries: faiss-cpu 1.15.1, qdrant-client 1.20.0, datasketch 2.0.0, rocksdict 0.3.29, redis 8.1.0, threadpoolctl 3.7.0
- timing: per-call `perf_counter_ns`, GC disabled, **1 thread unless stated** (enforced with threadpoolctl); empty-call overhead 62 ns (latencies below ~1 us are dominated by Python call overhead).
- total benchmark time: 1510 s

## Scorecard (computed from the data)

Best NeuroStore system vs the best system of each competing family on the same headline metric. WIN/LOSS means more than 5% better/worse; otherwise TIE. **Quality** rows (accuracy, robustness) compare every system; **[cost]** rows (latency, throughput, memory, build time) compare only systems that pass the row's accuracy gate, and NeuroStore is represented by its best system *among those that pass* -- so nothing wins on speed by returning nothing, and no row mixes two NeuroStore operating points. Comparisons against exact-match stores on noisy-key accuracy are structural (they cannot succeed by construction). Section-5 CIM latency is excluded: it is a model, not a measurement.

| wl       | metric                                                                                                                                      | context                         | best NeuroStore                        | best of each competing family (that passes the gate)                                                                                                                                                                                                                                                                                               |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A        | Noise tolerance: largest bit-flip rate with >=99% exact recall [% of bits flipped]                                                          | D=512, N=100                    | 42 (hopfield_b20_s5)                   | ann: 42 (bruteforce_hamming) **TIE**<br>lsh: 42 (lsh_k5_L256) **TIE**<br>kv: 0 (python_dict) **WIN**                                                                                                                                                                                                                                               |
| A        | [cost] Read latency p50 (single query, 10% noise) [us, lower=better]                                                                        | D=512, N=10000, 1 thread        | 7,837 (hopfield_b20_s5)                | ann: 27 (faiss_binary_flat) **LOSS**<br>lsh: 5,853 (lsh_k5_L256) **LOSS**<br>_excluded by the gate: hopfield_b20, lsh_k8_L64, lsh_k12_L32, python_dict, rocksdb, redis, faiss_binary_hnsw_<br>_gate: exact-recall noise tolerance >= 90% of the exact-NN ceiling (39%) at N=10000 (systems absent from the store-size sweep: their N=100 result)_  |
| A        | [cost] Write latency p50 (single insert) [us, lower=better]                                                                                 | D=512, N=10000, 1 thread        | 3.96 (hopfield_b20_s5)                 | ann: 1.19 (bruteforce_hamming) **LOSS**<br>lsh: 30.7 (lsh_k5_L256) **WIN**<br>_excluded by the gate: hopfield_b20, lsh_k8_L64, lsh_k12_L32, python_dict, rocksdb, redis, faiss_binary_hnsw_<br>_gate: exact-recall noise tolerance >= 90% of the exact-NN ceiling (39%) at N=10000 (systems absent from the store-size sweep: their N=100 result)_ |
| A        | [cost] Memory per stored (key, value) pair [bytes, lower=better]                                                                            | D=512, N=10000, 1 thread        | 4,100 (hopfield_b20_s5)                | ann: 128 (bruteforce_hamming) **LOSS**<br>lsh: 9,379 (lsh_k5_L256) **WIN**<br>_excluded by the gate: hopfield_b20, lsh_k8_L64, lsh_k12_L32, python_dict, rocksdb, redis, faiss_binary_hnsw_<br>_gate: exact-recall noise tolerance >= 90% of the exact-NN ceiling (39%) at N=10000 (systems absent from the store-size sweep: their N=100 result)_ |
| B        | Record identification accuracy (clean cue) [%]                                                                                              | K=3, D=4096, M=5000, F=8, V=500 | 100 (vsa_map)                          | relational: 100 (inverted_index) **TIE**                                                                                                                                                                                                                                                                                                           |
| B        | Record identification accuracy (1 corrupted cue field) [%]                                                                                  | K=3, D=4096, M=5000, F=8, V=500 | 97.6 (vsa_map_bincue)                  | relational: 96 (fuzzy_scan) **TIE**                                                                                                                                                                                                                                                                                                                |
| B        | Missing-field reconstruction accuracy (clean cue) [%]                                                                                       | K=3, D=4096, M=5000, F=8, V=500 | 100 (vsa_map)                          | relational: 100 (inverted_index) **TIE**                                                                                                                                                                                                                                                                                                           |
| B        | [cost] Query latency p50 (identify + reconstruct) [us, lower=better]                                                                        | K=3, D=4096, M=5000, 1 thread   | 3,431 (vsa_map)                        | relational: 34.3 (fuzzy_scan) **LOSS**<br>_excluded by the gate: inverted_index, linear_scan_<br>_gate: reconstructs the unknown field with >= 95% accuracy when one of the K=3 cue values is wrong_                                                                                                                                               |
| B        | [cost] Memory per stored record (as allocated, incl. item memories / base table) [bytes, lower=better]                                      | K=3, D=4096, M=5000, 1 thread   | 29,521 (vsa_map)                       | relational: 16 (fuzzy_scan) **LOSS**<br>_excluded by the gate: inverted_index, linear_scan_<br>_gate: reconstructs the unknown field with >= 95% accuracy when one of the K=3 cue values is wrong_                                                                                                                                                 |
| C        | Cache-hit recall@1 on paraphrases [fraction]                                                                                                | N=10000, d=128, 1 thread        | 1 (flyhash_scan_2048_64)               | ann: 1 (bruteforce_cosine) **TIE**<br>kv: 0 (exact_hash) **WIN**<br>lsh: 1 (simhash_2048) **TIE**                                                                                                                                                                                                                                                  |
| C        | True-hit rate at exactly 1% false hits [fraction]                                                                                           | N=10000, d=128, 1 thread        | 0.996 (flyhash_scan_2048_64)           | ann: 1 (bruteforce_cosine) **TIE**<br>kv: 0 (exact_hash) **WIN**<br>lsh: 1 (simhash_2048) **TIE**                                                                                                                                                                                                                                                  |
| C        | [cost] Query throughput, single query (encode + search) [queries/s]                                                                         | N=10000, d=128, 1 thread        | 7,930 (flyhash_inv_2048_64)            | ann: 21,108 (faiss_hnsw) **LOSS**<br>lsh: 2,961 (simhash_512) **WIN**<br>_excluded by the gate: exact_hash_<br>_gate: recall@1 >= 99% and true-hit rate >= 95% at 1% false hits (paraphrase cosine 0.9)_                                                                                                                                           |
| C        | [cost] Query throughput, batched [queries/s]                                                                                                | N=10000, d=128, 1 thread        | 12,547 (flyhash_binhnsw_2048_64)       | ann: 47,684 (bruteforce_cosine) **LOSS**<br>lsh: 2,111 (simhash_512) **WIN**<br>_excluded by the gate: exact_hash_<br>_gate: recall@1 >= 99% and true-hit rate >= 95% at 1% false hits (paraphrase cosine 0.9)_                                                                                                                                    |
| C        | [cost] Memory per cached prompt (index + N-independent state) [bytes, lower=better]                                                         | N=10000, d=128, 1 thread        | 361 (flyhash_scan_2048_64)             | ann: 512 (bruteforce_cosine) **WIN**<br>lsh: 90.2 (simhash_512) **LOSS**<br>_excluded by the gate: exact_hash_<br>_gate: recall@1 >= 99% and true-hit rate >= 95% at 1% false hits (paraphrase cosine 0.9)_                                                                                                                                        |
| C        | [cost] Index build time [s, lower=better]                                                                                                   | N=10000, d=128, 1 thread        | 0.429 (flyhash_inv_2048_64)            | ann: 3.95e-06 (bruteforce_cosine) **LOSS**<br>lsh: 0.0109 (simhash_512) **LOSS**<br>_excluded by the gate: exact_hash_<br>_gate: recall@1 >= 99% and true-hit rate >= 95% at 1% false hits (paraphrase cosine 0.9)_                                                                                                                                |
| D        | [cost] Hops per millisecond, single query [hops/ms]                                                                                         | E=1000, D=2048, 1 thread        | 3.09 (holographic_map_D2048)           | kv: 4,425 (adjacency_dict) **LOSS**<br>_gate: single-hop accuracy >= 99% on the benchmark graph (sized to 70% of the closed-form capacity); the pointer-chasing baselines are exact_                                                                                                                                                               |
| D        | [cost] Hops per millisecond, batched [hops/ms]                                                                                              | E=1000, D=2048, 1 thread        | 26.8 (holographic_map_D2048)           | kv: 127,872 (successor_table) **LOSS**<br>_gate: single-hop accuracy >= 99% on the benchmark graph (sized to 70% of the closed-form capacity); the pointer-chasing baselines are exact_                                                                                                                                                            |
| D        | [cost] Memory per stored triple (holographic: all allocated arrays, spread over the triples it holds at 99% accuracy) [bytes, lower=better] | E=1000, D=2048, 1 thread        | 266,569 (holographic_map_D2048)        | kv: 161 (adjacency_dict) **LOSS**<br>_gate: single-hop accuracy >= 99% on the benchmark graph (sized to 70% of the closed-form capacity); the pointer-chasing baselines are exact_                                                                                                                                                                 |
| CAPACITY | Capacity C_max (items stored before BER > 0.1%) [items]                                                                                     | D=2048, BER <= 0.1%             | 10,000 (modern_hopfield(noise 20%) >=) | no baseline                                                                                                                                                                                                                                                                                                                                        |
| CAPACITY | Storage density eta at C_max (information bits / allocated bits) [ratio]                                                                    | D=2048, BER <= 0.1%             | 0.0312 (modern_hopfield)               | no baseline                                                                                                                                                                                                                                                                                                                                        |


**Tally -- quality (accuracy / robustness): 3 win, 9 tie, 0 loss over 12 comparisons; cost (latency / throughput / memory, accuracy-gated): 5 win, 0 tie, 14 loss over 19 comparisons.**

## Checks against the evaluation spec and benchmark self-validation

| wl       | check                                                                                                          | target                                                | measured                                                          | result   |
| -------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------- | -------- |
| A        | Hopfield recalls exactly with a clean key                                                                      | 100% at 0% noise                                      | 100.0%                                                            | PASS     |
| A        | Hopfield tolerates bit flips that defeat an exact hash table                                                   | N_noise(Hopfield) > N_noise(hash)                     | 38% vs 0%                                                         | PASS     |
| A        | Hopfield tolerates more noise than the best LSH configuration                                                  | N_noise(Hopfield) > N_noise(best LSH)                 | 38% vs 42% (lsh_k5_L256)                                          | **FAIL** |
| A        | Attractor iteration extends the noise range of exact value recovery beyond the single-step reference           | N_noise(iterated) >= N_noise(single step)             | 42% vs 38%                                                        | PASS     |
| A        | Single-step Hopfield stays within 2 noise steps of the exact-NN ceiling (information bound, not a win)         | N_noise(Hopfield) >= N_noise(exact NN) - 2 steps      | 38% vs 42%                                                        | **FAIL** |
| A        | Benchmark self-check: measured exact-NN identification matches closed-form theory                              | max \|measured - theory\| <= 0.05 over the grid       | 0.028                                                             | PASS     |
| B        | Original-harness unbinding (role absent from the cue) carries no information                                   | \|mean cosine\| < 4/sqrt(D) and 0% success at cos>0.6 | max \|mean\|/(1/sqrt(D)) = 0.06; success 0%                       | PASS     |
| B        | Two-stage VSA reconstructs the missing field (K=3, clean cue, D=4096)                                          | >= 99% accuracy                                       | 100.0%                                                            | PASS     |
| B        | VSA identification with 1 corrupted cue field is within 5 points of the exact fuzzy-scan ceiling (K=3, D=4096) | VSA >= fuzzy - 5 pts                                  | 97.4% vs 96.0%                                                    | PASS     |
| B        | Exact-match relational methods cannot recover from a corrupted cue field                                       | ~0%                                                   | 0.0%                                                              | PASS     |
| C        | Benchmark self-check: task is well-posed (exact cosine scan finds the source prompt)                           | >= 99% recall@1                                       | 100.0%                                                            | PASS     |
| C        | Sub-microsecond semantic lookup (spec objective)                                                               | < 1 us p50                                            | 116.9 us (flyhash_inv_2048_64)                                    | **FAIL** |
| C        | FlyHash code is smaller than the dense float embedding                                                         | < 512 bytes/prompt                                    | 258 bytes (flyhash_inv_2048_64)                                   | PASS     |
| C        | FlyHash recall@1 >= SimHash recall@1 at the same bit budget                                                    | FlyHash >= SimHash                                    | 100.0% vs 100.0%                                                  | PASS     |
| C        | FlyHash recall@1 >= SimHash recall@1 at the same bit budget on the hardest paraphrases (cosine 0.6)            | FlyHash >= SimHash                                    | 61.4% vs 99.8%                                                    | **FAIL** |
| C        | FlyHash recall@1 is within 5 points of exact cosine scan                                                       | FlyHash >= exact - 5 pts                              | 100.0% vs 100.0%                                                  | PASS     |
| D        | Benchmark self-check: measured single-hop accuracy matches closed-form theory (transition region, pooled)      | \|mean(measured - theory)\| <= 0.03 per algebra       | map: +0.000, hrr: -0.010                                          | PASS     |
| D        | Benchmark self-check: measured capacity T_max (bisection on simulations) matches the closed-form bound         | mean \|T_max/T_theory - 1\| <= 0.15 and max <= 0.3    | mean 4.2%, max 7.5%                                               | PASS     |
| D        | Edge direction is encoded (map): a reversed query does not return the subject                                  | reverse hit rate <= max(5%, 4x chance)                | 0.9% (chance 1.0%), forward acc 97%                               | PASS     |
| D        | Edge direction is encoded (hrr): a reversed query does not return the subject                                  | reverse hit rate <= max(5%, 4x chance)                | 1.1% (chance 1.0%), forward acc 96%                               | PASS     |
| D        | Holographic hop rate exceeds pointer chasing (spec objective, single query)                                    | holographic hops/ms > dict hops/ms                    | 3.1 vs 4425 hops/ms (E=1000, D=2048)                              | **FAIL** |
| CAPACITY | Benchmark self-check: measured linear-memory capacity matches Q(sqrt(D/(N-1))) <= 0.1%                         | within 20% of closed form for every D                 | worst relative error 0.9%                                         | PASS     |
| CAPACITY | A superposed bundle cannot be read bit-by-bit: raw BER at N=2 already exceeds 0.1%                             | BER(N=2) > 0.1%                                       | 31.3%                                                             | PASS     |
| CAPACITY | Benchmark self-check: measured cleaned-up bundle capacity matches the closed-form bound                        | within 35% for every D                                | worst relative error 4.1%                                         | PASS     |
| CIM      | Benchmark self-check: ideal crossbar (no noise, ideal ADC) reproduces the digital similarity scores            | max \|analogue - digital\| < 1e-4                     | 0.00e+00                                                          | PASS     |
| CIM      | Combined realistic non-idealities keep >=99% identification (D=2048, query noise 20%)                          | >= 99%                                                | 100.0% (ideal 100.0%)                                             | PASS     |
| CIM      | Read-noise tolerance at D=2048 (informational)                                                                 | report                                                | up to 20% of tile full scale on the swept grid                    | info     |
| CIM      | Modelled O(1) regime holds only while tiles fit the array budget (optimistic)                                  | report                                                | single wave up to N=100000; 0 swept size(s) need serialised waves | info     |
| CIM      | Modelled O(1) regime holds only while tiles fit the array budget (conservative)                                | report                                                | single wave up to N=1000; 4 swept size(s) need serialised waves   | info     |


Notes on checks:

- *Hopfield tolerates more noise than the best LSH configuration*: LSH noise tolerance is a tuning knob (P(found) = 1-(1-(1-p)^k)^L); the more tolerant presets pay with much larger candidate sets and latency (see latency table).
- *Single-step Hopfield stays within 2 noise steps of the exact-NN ceiling (information bound, not a win)*: A softmax read-out with finite beta (=20) blends neighbouring patterns, so exact value recovery fails a few noise steps before the exact-NN decision does (see the beta sweep); attractor iteration closes the gap. It can never exceed the ceiling.
- *Benchmark self-check: measured exact-NN identification matches closed-form theory*: Validates the noise injector and the evaluation loop, not NeuroStore.
- *Original-harness unbinding (role absent from the cue) carries no information*: Why the reference Workload B always printed 'Exact Key Recovery Success: NO'.
- *Benchmark self-check: task is well-posed (exact cosine scan finds the source prompt)*: If this fails the synthetic data is too hard to measure anything else.
- *Sub-microsecond semantic lookup (spec objective)*: Measured in NumPy/Python at N=10000 including encoding; only specialised hardware could reach this.
- *FlyHash recall@1 >= SimHash recall@1 at the same bit budget on the hardest paraphrases (cosine 0.6)*: The headline-level comparison saturates at 100% for both; the hardness sweep is where codes differ.
- *Benchmark self-check: measured single-hop accuracy matches closed-form theory (transition region, pooled)*: Validates the graph generator, the algebras and the evaluation loop. HRR runs ~1-2 points below the Gaussian model.
- *Benchmark self-check: measured capacity T_max (bisection on simulations) matches the closed-form bound*: Resolution 3%, noise of a few % from finite graphs; unitary HRR sits ~1-2 accuracy points below the Gaussian model, i.e. up to ~10% lower capacity.
- *Holographic hop rate exceeds pointer chasing (spec objective, single query)*: A hop costs O(E*D) for cleanup; a dict lookup is O(1). Only parallel in-memory hardware could change this.
- *A superposed bundle cannot be read bit-by-bit: raw BER at N=2 already exceeds 0.1%*: Capacity of the bundle exists only after cleanup against a codebook.
- *Benchmark self-check: measured cleaned-up bundle capacity matches the closed-form bound*: Looser tolerance: the bound sits on the steep part of a noisy low-BER curve.
- *Benchmark self-check: ideal crossbar (no noise, ideal ADC) reproduces the digital similarity scores*: Validates the differential-pair mapping and the tiled partial-sum accumulation.
- *Combined realistic non-idealities keep >=99% identification (D=2048, query noise 20%)*: Scenario: read_noise=0.02, adc_bits=8, program_noise=0.05, stuck_fraction=0.01, drift_nu=0.05, drift_time_ratio=10000.0, drift_nu_sigma=0.01, drift_compensation=True
- *Modelled O(1) regime holds only while tiles fit the array budget (optimistic)*: Beyond n_arrays tiles the analogue time grows linearly with the number of waves.
- *Modelled O(1) regime holds only while tiles fit the array budget (conservative)*: Beyond n_arrays tiles the analogue time grows linearly with the number of waves.

## Workload A - Noisy telemetry / edge-sensor key retrieval

_968 s; seed 0; profile standard_

**Noise tolerance** -- largest bit-flip rate with >= 99% exact value recall (N=100 stored pairs, 2000 queries/level; grid step 1%). Cell = last passing level (first failing level).

| system                        | D=512     | D=2048    | D=4096    |
| ----------------------------- | --------- | --------- | --------- |
| hopfield_b20                  | 38% (39%) | 40% (41%) | 40% (41%) |
| hopfield_b20_s5               | 42% (43%) | 45% (46%) | 47% (48%) |
| bruteforce_hamming            | 42% (43%) | 45% (46%) | 47% (48%) |
| faiss_binary_flat             | 42% (43%) | 45% (46%) | 47% (48%) |
| faiss_binary_hnsw             | 42% (43%) | 45% (46%) | 47% (48%) |
| lsh_k8_L64                    | 28% (29%) | 28% (29%) | 28% (29%) |
| lsh_k12_L32                   | 15% (16%) | 14% (15%) | 15% (16%) |
| lsh_k5_L256                   | 42% (43%) | 45% (46%) | 46% (47%) |
| datasketch_minhash_lsh        | 22% (23%) | 25% (26%) | 25% (26%) |
| qdrant_local                  | 42% (43%) | 45% (46%) | 47% (48%) |
| python_dict                   | 0% (1%)   | 0% (1%)   | 0% (1%)   |
| rocksdb                       | 0% (1%)   | 0% (1%)   | 0% (1%)   |
| redis                         | 0% (1%)   | 0% (1%)   | 0% (1%)   |
| theory_bruteforce_ident       | 41% (42%) | 45% (46%) | 47% (48%) |
| theory_bruteforce_ident_exact | 41.8%     | 45.8%     | 47.1%     |


Queries per noise level: 2000 for every system except lsh_k8_L64 (500), lsh_k12_L32 (500), lsh_k5_L256 (500), qdrant_local (200), datasketch_minhash_lsh (100) (slow Python systems are evaluated on a prefix of the same queries). Thresholds come from one random key set per dimension, so read them to +-1 grid step; the closed-form rows are the key-set average.


**Exact value recall vs noise** (D=512; Wilson 95% CI in the JSON). `theory_bruteforce_ident` is the closed-form exact-NN identification probability.

| system                  | 0      | 0.05   | 0.1    | 0.2    | 0.3    | 0.35   | 0.4    | 0.45  | 0.5  |
| ----------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ----- | ---- |
| hopfield_b20            | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 21.3%  | 0.0%  | 0.0% |
| hopfield_b20_s5         | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 31.8% | 0.0% |
| bruteforce_hamming      | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 33.9% | 0.0% |
| faiss_binary_flat       | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 33.9% | 0.0% |
| faiss_binary_hnsw       | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 33.6% | 0.0% |
| lsh_k8_L64              | 100.0% | 100.0% | 100.0% | 100.0% | 97.0%  | 89.4%  | 69.6%  | 25.0% | 0.0% |
| lsh_k12_L32             | 100.0% | 100.0% | 100.0% | 91.0%  | 35.6%  | 17.8%  | 7.8%   | 1.0%  | 0.2% |
| lsh_k5_L256             | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 35.6% | 0.0% |
| datasketch_minhash_lsh  | 100.0% | 100.0% | 100.0% | 100.0% | 97.0%  | 81.0%  | 60.0%  | 35.0% | 0.0% |
| qdrant_local            | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 36.5% | 0.0% |
| python_dict             | 100.0% | 0.0%   | 0.0%   | 0.0%   | 0.0%   | 0.0%   | 0.0%   | 0.0%  | 0.0% |
| rocksdb                 | 100.0% | 0.0%   | 0.0%   | 0.0%   | 0.0%   | 0.0%   | 0.0%   | 0.0%  | 0.0% |
| redis                   | 100.0% | 0.0%   | 0.0%   | 0.0%   | 0.0%   | 0.0%   | 0.0%   | 0.0%  | 0.0% |
| theory_bruteforce_ident | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 34.2% | 0.0% |


**Tolerance vs store size** (D=512): more stored items = more impostors = lower tolerance.

| system                        | N=100 | N=1000 | N=10000 |
| ----------------------------- | ----- | ------ | ------- |
| hopfield_b20                  | 38%   | 35%    | 33%     |
| hopfield_b20_s5               | 42%   | 40%    | 39%     |
| bruteforce_hamming            | 42%   | 40%    | 39%     |
| faiss_binary_flat             | 42%   | 40%    | 39%     |
| faiss_binary_hnsw             | 42%   | 40%    | 26%     |
| faiss_binary_hnsw_ef256       | 42%   | 40%    | 39%     |
| faiss_binary_hnsw_ef1024      | 42%   | 40%    | 39%     |
| lsh_k8_L64                    | 28%   | 28%    | 27%     |
| lsh_k12_L32                   | 15%   | 15%    | 15%     |
| lsh_k5_L256                   | 42%   | 40%    | 39%     |
| python_dict                   | 0%    | 0%     | 0%      |
| theory_bruteforce_ident       | 41%   | 40%    | 39%     |
| theory_bruteforce_ident_exact | 42%   | 40%    | 39%     |


**Hopfield beta and attractor-iteration sweep** (D=512, N=100): a single softmax step blends neighbouring patterns, so exact *value* recovery breaks before *identification* does; extra attractor steps (key-space cleanup, then read-out) pull the state into one pattern's basin.

| beta | identification tolerance | value-recall tolerance, 1 step | value-recall tolerance, 5 steps | value recall @ 40%, 1 step | value recall @ 40%, 5 steps |
| ---- | ------------------------ | ------------------------------ | ------------------------------- | -------------------------- | --------------------------- |
| 1    | 40%                      | n/a                            | n/a                             | 0.0%                       | 0.0%                        |
| 4    | 40%                      | 2%                             | 16%                             | 0.0%                       | 0.0%                        |
| 8    | 40%                      | 24%                            | 38%                             | 0.0%                       | 98.5%                       |
| 16   | 40%                      | 36%                            | 40%                             | 0.0%                       | 100.0%                      |
| 20   | 40%                      | 38%                            | 40%                             | 27.7%                      | 100.0%                      |
| 32   | 40%                      | 40%                            | 40%                             | 100.0%                     | 100.0%                      |
| 64   | 40%                      | 40%                            | 40%                             | 100.0%                     | 100.0%                      |


**Latency / memory / density** (D=512, N=100; reads use 10% noisy keys; 1 thread; timer+call overhead is in the environment block). `eta` = information bits / allocated bits; `R/W` = read p50 / write p50.

| system                   | read p50 us | read p99 us | write p50 us | R/W  | batch q/s | bytes/pair | eta    | build s |
| ------------------------ | ----------- | ----------- | ------------ | ---- | --------- | ---------- | ------ | ------- |
| hopfield_b20             | 22.53       | 47.5        | 3.41         | 6.6  | 319,479   | 4,100      | 0.0312 | 0.000   |
| hopfield_b20_s5          | 96.96       | 239.7       | 3.15         | 30.8 | 85,490    | 4,100      | 0.0312 | 0.000   |
| bruteforce_hamming       | 5.74        | 13.3        | 0.80         | 7.2  | 223,255   | 128        | 1      | 0.000   |
| lsh_k8_L64               | 19.00       | 47.3        | 8.49         | 2.2  | 43,419    | 6,628      | 0.0193 | 0.002   |
| lsh_k12_L32              | 10.57       | 45.9        | 6.07         | 1.7  | 80,104    | 5,322      | 0.0241 | 0.001   |
| lsh_k5_L256              | 52.07       | 102.7       | 23.44        | 2.2  | 16,762    | 10,719     | 0.0119 | 0.006   |
| python_dict              | 0.12        | 0.2         | 0.20         | 0.6  | 1.73e+06  | 241        | 0.531  | 0.000   |
| rocksdb                  | 1.09        | 3.4         | 2.03         | 0.5  | 607,024   | 151        | 0.849  | 0.002   |
| redis                    | 61.15       | 94.4        | 71.40        | 0.9  | 468,530   | 465        | 0.275  | 0.001   |
| faiss_binary_flat        | 6.48        | 14.6        | 3.26         | 2.0  | 1.13e+06  | 128        | 0.997  | 0.000   |
| faiss_binary_hnsw        | 17.74       | 38.3        | 82.02        | 0.2  | 79,321    | 401        | 0.32   | 0.003   |
| qdrant_local             | 450.64      | 574.7       | 70.93        | 6.4  | 2,672     | 2,112      | 0.0606 | 0.019   |
| datasketch_minhash_lsh   | 79.19       | 129.4       | 35.98        | 2.2  | 2,224     | 7,967      | 0.0161 | 0.060   |
| faiss_binary_hnsw_ef256  | 14.84       | 34.2        | 71.71        | 0.2  | 79,305    | 401        | 0.32   | 0.002   |
| faiss_binary_hnsw_ef1024 | 14.94       | 33.4        | 75.16        | 0.2  | 72,619    | 401        | 0.32   | 0.002   |


**Latency / memory / density** (D=512, N=10000; reads use 10% noisy keys; 1 thread; timer+call overhead is in the environment block). `eta` = information bits / allocated bits; `R/W` = read p50 / write p50.

| system                   | read p50 us | read p99 us | write p50 us | R/W   | batch q/s | bytes/pair | eta    | build s |
| ------------------------ | ----------- | ----------- | ------------ | ----- | --------- | ---------- | ------ | ------- |
| hopfield_b20             | 1,578       | 2,472       | 4.38         | 360.5 | 4,854     | 4,100      | 0.0312 | 0.028   |
| hopfield_b20_s5          | 7,837       | 11,964      | 3.96         | 1,980 | 1,151     | 4,100      | 0.0312 | 0.023   |
| bruteforce_hamming       | 273.85      | 352.8       | 1.19         | 229.4 | 1,994     | 128        | 1      | 0.004   |
| lsh_k8_L64               | 411.53      | 938.4       | 14.97        | 27.5  | 2,663     | 2,601      | 0.0492 | 0.120   |
| lsh_k12_L32              | 50.41       | 174.2       | 16.49        | 3.1   | 19,495    | 2,885      | 0.0444 | 0.094   |
| lsh_k5_L256              | 5,853       | 11,550      | 30.68        | 190.8 | 180       | 9,379      | 0.0136 | 0.417   |
| python_dict              | 0.29        | 0.5         | 0.40         | 0.7   | 1.07e+06  | 223        | 0.573  | 0.016   |
| rocksdb                  | 4.88        | 11.6        | 3.08         | 1.6   | 342,667   | 139        | 0.92   | 0.055   |
| redis                    | 65.19       | 98.7        | 75.05        | 0.9   | 511,744   | 224        | 0.573  | 0.183   |
| faiss_binary_flat        | 26.96       | 46.1        | 1.95         | 13.8  | 41,816    | 128        | 1      | 0.005   |
| faiss_binary_hnsw        | 63.38       | 104.6       | 327.51       | 0.2   | 16,774    | 400        | 0.32   | 2.484   |
| qdrant_local             | 9,242       | 11,148      | 73.69        | 125.4 | 104       | 2,112      | 0.0606 | 1.066   |
| faiss_binary_hnsw_ef256  | 249.67      | 423.6       | 324.13       | 0.8   | 3,764     | 400        | 0.32   | 2.491   |
| faiss_binary_hnsw_ef1024 | 917.50      | 1,278       | 379.53       | 2.4   | 1,168     | 400        | 0.32   | 2.579   |


Notes:

- All systems receive identical noisy queries; success means the returned value equals the stored value in all D bits.
- Hopfield uses a softmax readout, so exact *value* recovery can fail before *identification* does (see beta_sweep).
- Exact NN is the accuracy ceiling: any retriever that finds the nearest key cannot beat it on this task.
- Systems with a query cap (LSH, datasketch, Qdrant) are evaluated on the first n queries; n is in the 'n' column.
- RocksDB memory is on-disk SST bytes after flush; Redis memory is the INFO used_memory growth; Qdrant memory is a lower bound.

## Workload B - Record completion & entity resolution

_12 s; seed 0; profile standard_

**Record identification accuracy** (M=5000 records, F=8 fields, vocab 500, D=4096 for VSA; 500 unique-cue queries per cell). Columns: K known fields / c corrupted.

| system            | K=2, c=0 | K=3, c=0 | K=3, c=1 | K=4, c=0 | K=4, c=1 | K=6, c=0 | K=6, c=1 |
| ----------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| vsa_map           | 100.0%   | 100.0%   | 97.4%    | 100.0%   | 100.0%   | 100.0%   | 100.0%   |
| vsa_map_bincue    | 100.0%   | 100.0%   | 97.6%    | 100.0%   | 100.0%   | 100.0%   | 100.0%   |
| fuzzy_scan        | 100.0%   | 100.0%   | 96.0%    | 100.0%   | 100.0%   | 100.0%   | 100.0%   |
| inverted_index    | 100.0%   | 100.0%   | 0.0%     | 100.0%   | 0.0%     | 100.0%   | 0.0%     |
| linear_scan       | 100.0%   | 100.0%   | 0.0%     | 100.0%   | 0.0%     | 100.0%   | 0.0%     |
| composite_indexes | 100.0%   |          |          | 100.0%   | 0.0%     |          |          |


**Missing-field reconstruction accuracy** (M=5000 records, F=8 fields, vocab 500, D=4096 for VSA; 500 unique-cue queries per cell). Columns: K known fields / c corrupted.

| system            | K=2, c=0 | K=3, c=0 | K=3, c=1 | K=4, c=0 | K=4, c=1 | K=6, c=0 | K=6, c=1 |
| ----------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| vsa_map           | 100.0%   | 100.0%   | 97.4%    | 100.0%   | 100.0%   | 100.0%   | 100.0%   |
| vsa_map_bincue    | 100.0%   | 100.0%   | 97.6%    | 100.0%   | 100.0%   | 100.0%   | 100.0%   |
| fuzzy_scan        | 100.0%   | 100.0%   | 96.0%    | 100.0%   | 100.0%   | 100.0%   | 100.0%   |
| inverted_index    | 100.0%   | 100.0%   | 0.0%     | 100.0%   | 0.0%     | 100.0%   | 0.0%     |
| linear_scan       | 100.0%   | 100.0%   | 0.0%     | 100.0%   | 0.0%     | 100.0%   | 0.0%     |
| composite_indexes | 100.0%   |          |          | 100.0%   | 0.0%     |          |          |


**VSA vs dimension** (K=3): identification / reconstruction accuracy, unbound-filler cosine and bit agreement.

| system         | D    | corrupted | identify | reconstruct | unbind cosine | bit agreement |
| -------------- | ---- | --------- | -------- | ----------- | ------------- | ------------- |
| vsa_map        | 512  | 0         | 100.0%   | 100.0%      | 0.277         | 63.9%         |
| vsa_map        | 1024 | 0         | 100.0%   | 100.0%      | 0.273         | 63.6%         |
| vsa_map        | 2048 | 0         | 100.0%   | 100.0%      | 0.275         | 63.8%         |
| vsa_map        | 4096 | 0         | 100.0%   | 100.0%      | 0.273         | 63.7%         |
| vsa_map        | 512  | 1         | 91.0%    | 91.0%       | 0.247         | 62.3%         |
| vsa_map        | 1024 | 1         | 98.0%    | 98.0%       | 0.266         | 63.3%         |
| vsa_map        | 2048 | 1         | 96.8%    | 96.8%       | 0.265         | 63.3%         |
| vsa_map        | 4096 | 1         | 97.4%    | 97.4%       | 0.266         | 63.3%         |
| vsa_map_bincue | 512  | 0         | 100.0%   | 100.0%      | 0.277         | 63.9%         |
| vsa_map_bincue | 1024 | 0         | 100.0%   | 100.0%      | 0.273         | 63.6%         |
| vsa_map_bincue | 2048 | 0         | 100.0%   | 100.0%      | 0.275         | 63.8%         |
| vsa_map_bincue | 4096 | 0         | 100.0%   | 100.0%      | 0.273         | 63.7%         |
| vsa_map_bincue | 512  | 1         | 81.6%    | 81.6%       | 0.221         | 61.1%         |
| vsa_map_bincue | 1024 | 1         | 96.6%    | 96.6%       | 0.262         | 63.1%         |
| vsa_map_bincue | 2048 | 1         | 96.6%    | 96.6%       | 0.265         | 63.2%         |
| vsa_map_bincue | 4096 | 1         | 97.6%    | 97.6%       | 0.267         | 63.3%         |


**Control -- the original harness's method** (unbind a role that is *not* in the cue). Cosine to the true filler is pure noise of scale 1/sqrt(D); it can never exceed the 0.6 success bar.

| D    | mean cosine | std    | 1/sqrt(D) | success (cos>0.6) |
| ---- | ----------- | ------ | --------- | ----------------- |
| 512  | 0.0007      | 0.0462 | 0.0442    | 0%                |
| 1024 | 0.0020      | 0.0313 | 0.0312    | 0%                |
| 2048 | 0.0004      | 0.0217 | 0.0221    | 0%                |
| 4096 | -0.0002     | 0.0157 | 0.0156    | 0%                |


**Latency / memory** (K=3, D=4096, M=5000, 1 thread). Information content = 71.7 bits/record.

| system                     | p50 us | p99 us | queries/s | bytes/record | eta      |
| -------------------------- | ------ | ------ | --------- | ------------ | -------- |
| vsa_map                    | 3,431  | 4,220  | 288       | 29,521       | 0.000304 |
| vsa_map_bincue             | 3,581  | 4,291  | 276       | 29,521       | 0.000304 |
| inverted_index             | 17.7   | 50.9   | 50,291    | 54           | 0.165    |
| linear_scan                | 23.5   | 45.2   | 40,478    | 16           | 0.56     |
| fuzzy_scan                 | 34.3   | 58.5   | 27,740    | 16           | 0.56     |
| vsa_map(bit-packed, ideal) | n/a    | n/a    | n/a       | 922          | 0.00972  |


**Index blow-up** -- composite (multi-column) indexes needed to serve any K-of-F cue in O(1) vs one inverted index per column. 'Python dicts' is what is implemented here; 'compact arrays' is the same index as sorted (key, row id) arrays, which is the fair size to set against the numpy posting lists -- the C(F,K)-vs-F scaling holds in both.

| K | indexes for this K | indexes for any subset | composite, Python dicts | composite, compact arrays | inverted index | base table |
| - | ------------------ | ---------------------- | ----------------------- | ------------------------- | -------------- | ---------- |
| 2 | 28                 | 255                    | 18.5 MiB                | 1.1 MiB                   | 187.6 KiB      | 78.1 KiB   |
| 3 | 56                 | 255                    | 42.8 MiB (extrap.)      | 2.7 MiB                   | 187.6 KiB      | 78.1 KiB   |
| 4 | 70                 | 255                    | 61.0 MiB                | 4.0 MiB                   | 187.6 KiB      | 78.1 KiB   |
| 6 | 28                 | 255                    | 21.4 MiB (extrap.)      | 2.1 MiB                   | 187.6 KiB      | 78.1 KiB   |


Notes:

- Queries are restricted to cues whose clean K fields are unique in the table, so exact methods are never asked an ambiguous question.
- 'corrupt' = number of cue values replaced by a wrong value; exact-match methods cannot recover from it, fuzzy scan and VSA can.
- VSA record lookup is a dense O(M*D) scan: 'O(1) query steps' is a statement about operation count, not CPU time.
- Relational baselines use hash/inverted structures in Python; their memory is the deep size of those objects as allocated.
- VSA memory is reported as allocated float32 (what runs here) and as the ideal bit-packed size; the item memories (roles + filler codebooks) are included.
- reference_method_control reproduces the original harness (unbind a role absent from the cue); its cosine is pure noise of scale 1/sqrt(D).

## Workload C - High-throughput semantic prompt caching

_355 s; seed 0; profile standard_

**N = 1,000 cached prompts** (d=128, paraphrase cosine 0.9, topic rho 0.5; true-hit rate read at exactly 1% false hits; 1 thread).

| system                  | recall@1 | TPR@1%FPR | deployed FPR | ANN agree | p50 us | q/s (1)  | q/s (batch) | B/prompt index | B/prompt total | build s |
| ----------------------- | -------- | --------- | ------------ | --------- | ------ | -------- | ----------- | -------------- | -------------- | ------- |
| bruteforce_cosine       | 100.0%   | 100.0%    | 0.8%         | 100.0%    | 8.0    | 119,442  | 487,438     | 512            | 512            | 0.00    |
| exact_hash              | 0.0%     | 0.0%      | 1.0%         | 0.0%      | 0.2    | 3.14e+06 | 3.20e+06    | 610            | 610            | 0.00    |
| simhash_2048            | 100.0%   | 100.0%    | 0.6%         | 100.0%    | 85.1   | 7,467    | 8,698       | 256            | 1,305          | 0.01    |
| simhash_512             | 100.0%   | 100.0%    | 0.3%         | 100.0%    | 47.3   | 19,315   | 25,668      | 64             | 326            | 0.00    |
| flyhash_scan_2048_64    | 100.0%   | 99.9%     | 1.1%         | 100.0%    | 101.5  | 8,987    | 8,289       | 256            | 1,305          | 0.01    |
| flyhash_inv_2048_64     | 100.0%   | 99.9%     | 1.1%         | 100.0%    | 54.1   | 15,221   | 22,737      | 272            | 1,321          | 0.02    |
| flyhash_binhnsw_2048_64 | 100.0%   | 99.9%     | 1.1%         | 100.0%    | 121.0  | 7,048    | 17,166      | 528            | 1,576          | 0.10    |
| faiss_flat_ip           | 100.0%   | 100.0%    | 0.8%         | 100.0%    | 13.1   | 76,136   | 116,490     | 512            | 512            | 0.00    |
| faiss_hnsw              | 100.0%   | 100.0%    | 0.8%         | 100.0%    | 26.4   | 35,825   | 42,782      | 784            | 784            | 0.07    |
| qdrant_local            | 100.0%   | 100.0%    | 0.8%         | 100.0%    | 337.5  | 2,527    | 2,516       | 512            | 512            | 0.03    |


**N = 10,000 cached prompts** (d=128, paraphrase cosine 0.9, topic rho 0.5; true-hit rate read at exactly 1% false hits; 1 thread).

| system                  | recall@1 | TPR@1%FPR | deployed FPR | ANN agree | p50 us | q/s (1)  | q/s (batch) | B/prompt index | B/prompt total | build s |
| ----------------------- | -------- | --------- | ------------ | --------- | ------ | -------- | ----------- | -------------- | -------------- | ------- |
| bruteforce_cosine       | 100.0%   | 100.0%    | 1.4%         | 100.0%    | 180.5  | 5,400    | 47,684      | 512            | 512            | 0.00    |
| exact_hash              | 0.0%     | 0.0%      | 1.0%         | 0.0%      | 0.3    | 2.22e+06 | 2.87e+06    | 602            | 602            | 0.01    |
| simhash_2048            | 100.0%   | 100.0%    | 1.0%         | 100.0%    | 689.6  | 1,350    | 774         | 256            | 361            | 0.06    |
| simhash_512             | 100.0%   | 100.0%    | 1.1%         | 100.0%    | 327.2  | 2,961    | 2,111       | 64             | 90             | 0.01    |
| flyhash_scan_2048_64    | 100.0%   | 99.6%     | 1.2%         | 100.0%    | 756.1  | 1,308    | 744         | 256            | 361            | 0.66    |
| flyhash_inv_2048_64     | 100.0%   | 99.6%     | 1.2%         | 100.0%    | 116.9  | 7,930    | 11,431      | 258            | 362            | 0.43    |
| flyhash_binhnsw_2048_64 | 100.0%   | 99.6%     | 1.2%         | 100.0%    | 133.5  | 6,840    | 12,547      | 528            | 633            | 1.75    |
| faiss_flat_ip           | 100.0%   | 100.0%    | 1.4%         | 100.0%    | 170.5  | 5,464    | 6,572       | 512            | 512            | 0.00    |
| faiss_hnsw              | 100.0%   | 100.0%    | 1.4%         | 100.0%    | 44.4   | 21,108   | 20,478      | 784            | 784            | 1.17    |
| qdrant_local            | 100.0%   | 100.0%    | 1.4%         | 100.0%    | 2,591  | 376      | 346         | 512            | 512            | 0.33    |


**N = 50,000 cached prompts** (d=128, paraphrase cosine 0.9, topic rho 0.5; true-hit rate read at exactly 1% false hits; 1 thread).

| system                  | recall@1 | TPR@1%FPR | deployed FPR | ANN agree | p50 us | q/s (1)  | q/s (batch) | B/prompt index | B/prompt total | build s |
| ----------------------- | -------- | --------- | ------------ | --------- | ------ | -------- | ----------- | -------------- | -------------- | ------- |
| bruteforce_cosine       | 100.0%   | 100.0%    | 1.4%         | 100.0%    | 926.5  | 1,068    | 8,016       | 512            | 512            | 0.00    |
| exact_hash              | 0.0%     | 0.0%      | 1.0%         | 0.0%      | 0.3    | 2.82e+06 | 3.26e+06    | 625            | 625            | 0.02    |
| simhash_2048            | 100.0%   | 100.0%    | 1.1%         | 100.0%    | 3,408  | 274      | 147         | 256            | 277            | 1.51    |
| simhash_512             | 100.0%   | 100.0%    | 1.2%         | 100.0%    | 1,473  | 655      | 400         | 64             | 69             | 0.08    |
| flyhash_scan_2048_64    | 100.0%   | 98.4%     | 1.8%         | 100.0%    | 3,106  | 313      | 137         | 256            | 277            | 3.43    |
| flyhash_inv_2048_64     | 100.0%   | 98.4%     | 1.8%         | 100.0%    | 475.7  | 2,046    | 2,400       | 256            | 277            | 1.04    |
| flyhash_binhnsw_2048_64 | 100.0%   | 98.4%     | 1.8%         | 100.0%    | 222.6  | 4,383    | 6,784       | 528            | 549            | 14.27   |
| faiss_flat_ip           | 100.0%   | 100.0%    | 1.4%         | 100.0%    | 900.0  | 1,092    | 1,184       | 512            | 512            | 0.02    |
| faiss_hnsw              | 100.0%   | 100.0%    | 1.4%         | 100.0%    | 89.8   | 10,651   | 11,663      | 784            | 784            | 10.19   |


**Recall@1 vs paraphrase strength** (N=10,000): how fast each system degrades as paraphrases drift from their source.

| system                  | 0.95   | 0.9    | 0.8    | 0.7    | 0.6    |
| ----------------------- | ------ | ------ | ------ | ------ | ------ |
| bruteforce_cosine       | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| exact_hash              | 0.0%   | 0.0%   | 0.0%   | 0.0%   | 0.0%   |
| simhash_2048            | 100.0% | 100.0% | 100.0% | 100.0% | 99.8%  |
| simhash_512             | 100.0% | 100.0% | 100.0% | 98.7%  | 89.3%  |
| flyhash_scan_2048_64    | 100.0% | 100.0% | 99.2%  | 89.7%  | 61.4%  |
| flyhash_inv_2048_64     | 100.0% | 100.0% | 99.2%  | 89.7%  | 61.4%  |
| flyhash_binhnsw_2048_64 | 100.0% | 100.0% | 99.1%  | 89.2%  | 60.7%  |
| faiss_flat_ip           | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| faiss_hnsw              | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| qdrant_local            | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |


**True-hit rate at exactly 1% false hits** (same sweep):

| system                  | 0.95   | 0.9    | 0.8    | 0.7   | 0.6  |
| ----------------------- | ------ | ------ | ------ | ----- | ---- |
| bruteforce_cosine       | 100.0% | 100.0% | 100.0% | 0.0%  | 0.0% |
| exact_hash              | 0.0%   | 0.0%   | 0.0%   | 0.0%  | 0.0% |
| simhash_2048            | 100.0% | 100.0% | 100.0% | 16.0% | 0.0% |
| simhash_512             | 100.0% | 100.0% | 82.1%  | 5.9%  | 0.0% |
| flyhash_scan_2048_64    | 100.0% | 99.7%  | 45.0%  | 2.3%  | 0.1% |
| flyhash_inv_2048_64     | 100.0% | 99.7%  | 45.0%  | 2.3%  | 0.1% |
| flyhash_binhnsw_2048_64 | 100.0% | 99.7%  | 45.0%  | 2.3%  | 0.1% |
| faiss_flat_ip           | 100.0% | 100.0% | 100.0% | 0.0%  | 0.0% |
| faiss_hnsw              | 100.0% | 100.0% | 100.0% | 0.0%  | 0.0% |
| qdrant_local            | 100.0% | 100.0% | 100.0% | 0.0%  | 0.0% |


Read this table with care below paraphrase cosine ~0.8. Novel prompts from the cache's own topics sit at cosine ~topic_rho = 0.5 to each other, so the nearest cached neighbour of a novel prompt is already quite similar; here the exact-cosine true-hit rate collapses between paraphrase cosine 0.7 and 0.8. Once a paraphrase is no more similar to its source than a novel prompt is to its nearest neighbour, **no system can separate them at this false-hit rate** (the task is ill-posed there; it is not a defect of any one system), and a noisier code can score slightly above an exact one by chance because every positive sits at exactly the stated cosine. Recall@1 (previous table) is unaffected: the nearest cached prompt is still the right one.


**FlyHash hyper-parameter sweep vs SimHash at the same number of bits** (expansion m, winners k incl. ~5% of the original paper, fan-in density; N=10,000). `B/prompt` is the inverted-index / packed-code size. FlyHash is given every chance; SimHash is the plain random-hyperplane code.

| code                                   | B/prompt | recall@1 @0.8 | recall@1 @0.7 | recall@1 @0.6 | TPR@FPR @0.8 | TPR@FPR @0.7 | TPR@FPR @0.6 |
| -------------------------------------- | -------- | ------------- | ------------- | ------------- | ------------ | ------------ | ------------ |
| flyhash m=2048 k=64 (3.1%) fan-in 0.1  | 258      | 99.2%         | 89.7%         | 61.4%         | 45.0%        | 2.3%         | 0.1%         |
| flyhash m=2048 k=102 (5.0%) fan-in 0.1 | 410      | 100.0%        | 97.4%         | 75.7%         | 70.7%        | 3.8%         | 0.1%         |
| flyhash m=4096 k=128 (3.1%) fan-in 0.1 | 515      | 100.0%        | 98.6%         | 83.1%         | 81.3%        | 4.9%         | 0.1%         |
| flyhash m=4096 k=205 (5.0%) fan-in 0.1 | 823      | 100.0%        | 99.7%         | 92.6%         | 95.6%        | 7.6%         | 0.0%         |
| flyhash m=8192 k=256 (3.1%) fan-in 0.1 | 1,031    | 100.0%        | 99.7%         | 94.6%         | 95.6%        | 6.5%         | 0.0%         |
| flyhash m=8192 k=410 (5.0%) fan-in 0.1 | 1,647    | 100.0%        | 100.0%        | 97.2%         | 99.2%        | 9.8%         | 0.0%         |
| flyhash m=2048 k=64 (3.1%) fan-in 0.3  | 258      | 99.6%         | 91.7%         | 64.1%         | 49.7%        | 1.9%         | 0.0%         |
| simhash 2048 bits                      | 256      | 100.0%        | 100.0%        | 99.8%         | 100.0%       | 16.0%        | 0.0%         |
| simhash 4096 bits                      | 512      | 100.0%        | 100.0%        | 99.7%         | 100.0%       | 21.8%        | 0.0%         |
| simhash 8192 bits                      | 1,024    | 100.0%        | 100.0%        | 100.0%        | 100.0%       | 30.9%        | 0.0%         |


Notes:

- Embeddings are synthetic (topic structure + controlled-cosine paraphrases); absolute recalls depend on topic_rho/para_cos and must be re-measured on real embeddings (embeddings_npz).
- Latency is end-to-end for one query in one thread: it includes FlyHash/SimHash encoding. NumPy call overhead alone is several microseconds.
- TPR@FPR is read off each system's ROC at exactly the target false-hit rate (randomised ties); 'deployed FPR' is the false-hit rate of a threshold fit on separate calibration prompts and shows how well it transfers.
- bytes/prompt (index) excludes N-independent state; 'total' adds it (the dense float32 projection of FlyHash/SimHash); FAISS sizes are the serialised index size.
- Qdrant runs in embedded local mode (a NumPy scan behind a client API) and is capped at 20k prompts.

## Workload D - Holographic multi-hop knowledge-graph walking

_127 s; seed 0; profile standard_

**Single-hop accuracy** (E=100 entities, R=32 relations; all T triples superposed in ONE D-vector; 3 graph(s) per cell). `ghost rate` = 1 - accuracy. Cell = measured (closed-form theory).

_MAP_

| D \ triples T | 100           | 200           | 400           | 800         | 1600        |
| ------------- | ------------- | ------------- | ------------- | ----------- | ----------- |
| 2048          | 98.0% (97%)   | 73.7% (74%)   | 41.9% (41%)   | 21.1% (20%) | 10.4% (10%) |
| 4096          | 100.0% (100%) | 96.7% (97%)   | 74.5% (74%)   | 40.0% (41%) | 19.7% (20%) |
| 8192          | 100.0% (100%) | 100.0% (100%) | 97.2% (97%)   | 74.6% (74%) | 40.5% (41%) |
| 16384         | 100.0% (100%) | 100.0% (100%) | 100.0% (100%) | 96.4% (97%) | 74.5% (74%) |


_HRR_

| D \ triples T | 100           | 200           | 400           | 800         | 1600        |
| ------------- | ------------- | ------------- | ------------- | ----------- | ----------- |
| 2048          | 96.3% (97%)   | 73.8% (74%)   | 40.2% (41%)   | 19.8% (20%) | 10.4% (10%) |
| 4096          | 100.0% (100%) | 97.2% (97%)   | 71.3% (74%)   | 41.0% (41%) | 18.9% (20%) |
| 8192          | 100.0% (100%) | 99.8% (100%)  | 95.8% (97%)   | 72.5% (74%) | 40.0% (41%) |
| 16384         | 100.0% (100%) | 100.0% (100%) | 100.0% (100%) | 96.1% (97%) | 71.5% (74%) |


**Capacity** -- largest T with single-hop accuracy >= 99% (E=100), found by bisection on fresh random graphs (4 per evaluation, 3% resolution), against the closed-form bound ~ D / (2 ln E + margin). The coarse grid above only resolves a factor of two.

| algebra | D     | T_max measured | T_max theory | measured / theory | grid: last pass / first fail |
| ------- | ----- | -------------- | ------------ | ----------------- | ---------------------------- |
| map     | 2048  | 86             | 80           | 1.07              | n/a / 100                    |
| map     | 4096  | 164            | 159          | 1.03              | 100 / 200                    |
| map     | 8192  | 328            | 318          | 1.03              | 200 / 400                    |
| map     | 16384 | 627            | 637          | 0.98              | 400 / 800                    |
| hrr     | 2048  | 76             | 80           | 0.95              | n/a / 100                    |
| hrr     | 4096  | 160            | 159          | 1.01              | 100 / 200                    |
| hrr     | 8192  | 298            | 318          | 0.94              | 200 / 400                    |
| hrr     | 16384 | 597            | 637          | 0.94              | 400 / 800                    |


**Multi-hop path accuracy** (D=8192; success = every hop of the walk correct; errors compound):

| algebra | T    | 1-hop  | 2-hop  | 3-hop  | 4-hop  | 5-hop  |
| ------- | ---- | ------ | ------ | ------ | ------ | ------ |
| hrr     | 100  | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| hrr     | 200  | 99.3%  | 99.1%  | 99.1%  | 99.1%  | 99.1%  |
| hrr     | 400  | 96.5%  | 93.1%  | 89.8%  | 86.9%  | 84.1%  |
| hrr     | 800  | 71.1%  | 53.7%  | 39.1%  | 29.4%  | 22.5%  |
| hrr     | 1600 | 40.6%  | 17.2%  | 6.8%   | 2.7%   | 1.5%   |
| map     | 100  | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| map     | 200  | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| map     | 400  | 97.1%  | 94.7%  | 92.3%  | 90.3%  | 87.5%  |
| map     | 800  | 73.7%  | 56.2%  | 41.7%  | 29.6%  | 21.5%  |
| map     | 1600 | 40.1%  | 15.2%  | 6.3%   | 2.2%   | 1.3%   |


**Traversal speed and memory** (graph sized to 70% of the closed-form capacity at each (D, E) and at most 100 triples, so the timed memory is an accurate one; hop time itself does not depend on the number of triples):

| system          | D    | E     | triples | hops/ms (1 query) | hops/ms (batch) | bytes/triple (as built) | bytes/triple at 99% capacity | trace bytes | allocated |
| --------------- | ---- | ----- | ------- | ----------------- | --------------- | ----------------------- | ---------------------------- | ----------- | --------- |
| holographic_map | 2048 | 100   | 56      | 31.9              | 87.2            | 31,751                  | 22,226                       | 8,192       | 1.7 MiB   |
| holographic_map | 2048 | 1000  | 43      | 3.09              | 26.8            | 384,355                 | 266,569                      | 8,192       | 15.8 MiB  |
| holographic_map | 2048 | 10000 | 35      | 0.343             | 3.4             | 4.69e+06                | 3.22e+06                     | 8,192       | 156.4 MiB |
| holographic_map | 8192 | 100   | 100     | 5.62              | 17.7            | 71,111                  | 22,362                       | 32,768      | 6.8 MiB   |
| holographic_map | 8192 | 1000  | 100     | 0.746             | 6.18            | 660,971                 | 265,450                      | 32,768      | 63.0 MiB  |
| holographic_map | 8192 | 10000 | 100     | 0.0515            | 0.852           | 6.56e+06                | 3.18e+06                     | 32,768      | 625.6 MiB |
| holographic_hrr | 2048 | 100   | 56      | 17.1              | 38.4            | 62,501                  | 43,751                       | 8,192       | 3.3 MiB   |
| holographic_hrr | 2048 | 1000  | 43      | 2.4               | 20.1            | 767,657                 | 532,407                      | 8,192       | 31.5 MiB  |
| holographic_hrr | 2048 | 10000 | 35      | 0.327             | 3.24            | 9.37e+06                | 6.43e+06                     | 8,192       | 312.9 MiB |
| holographic_hrr | 8192 | 100   | 100     | 3.62              | 8.4             | 139,940                 | 44,006                       | 32,768      | 13.3 MiB  |
| holographic_hrr | 8192 | 1000  | 100     | 0.76              | 4.32            | 1.32e+06                | 530,027                      | 32,768      | 125.9 MiB |
| adjacency_dict  | -    | 100   | 100     | 5,025             | n/a             | 131                     | -                            | n/a         | 12.8 KiB  |
| successor_table | -    | 100   | 100     | n/a               | 127,713         | 32                      | -                            | n/a         | 3.1 KiB   |
| adjacency_dict  | -    | 1000  | 100     | 4,425             | n/a             | 161                     | -                            | n/a         | 15.7 KiB  |
| successor_table | -    | 1000  | 100     | n/a               | 127,872         | 320                     | -                            | n/a         | 31.2 KiB  |
| adjacency_dict  | -    | 10000 | 100     | 4,926             | n/a             | 161                     | -                            | n/a         | 15.7 KiB  |
| successor_table | -    | 10000 | 100     | n/a               | 125,459         | 3,200                   | -                            | n/a         | 312.5 KiB |


**Direction control** -- querying an edge backwards must not return the subject:

| algebra | D     | T   | forward accuracy | reverse hit rate | chance |
| ------- | ----- | --- | ---------------- | ---------------- | ------ |
| map     | 16384 | 800 | 96.9%            | 0.9%             | 1.0%   |
| hrr     | 16384 | 800 | 95.5%            | 1.1%             | 1.0%   |


Notes:

- Capacity is a property of the *single-vector* design: all T triples share one D-dimensional trace, so signal falls as 1/sqrt(T) against a noise floor of 1/sqrt(D).
- The item memory (entity/relation codebooks) is needed to clean up every hop and dominates the footprint; it is included in memory_bytes.
- Graph speed rows time the hop primitive only (bind, unbind, cleanup); the dict baseline is exact and needs no cleanup.
- path_acc_hK = fraction of walks whose first K hops are all correct; errors compound because a wrong entity poisons the next query.

## Capacity bound C_max and storage density eta

_28 s; seed 0; profile standard_

**Capacity C_max** -- items stored before BER > 0.1%, and storage density eta at that point.

| D    | memory                      | BER measured on                      | C_max     | C_max theory | eta      | allocated |
| ---- | --------------------------- | ------------------------------------ | --------- | ------------ | -------- | --------- |
| 512  | linear_matrix               | raw bit error rate                   | 54        | 54           | 0.00659  | 1.0 MiB   |
| 512  | vsa_bundle                  | bit error rate after cleanup (SER/2) | 16        | 16           | 6.42e-05 | 202.0 KiB |
| 512  | modern_hopfield (noise 0%)  | bit error rate of recalled value     | >= 10,000 | -            | 0.0312   | 39.1 MiB  |
| 512  | modern_hopfield (noise 20%) | bit error rate of recalled value     | >= 10,000 | -            | 0.0312   | 39.1 MiB  |
| 1024 | linear_matrix               | raw bit error rate                   | 109       | 108          | 0.00653  | 4.0 MiB   |
| 1024 | vsa_bundle                  | bit error rate after cleanup (SER/2) | 32        | 31           | 6.22e-05 | 404.0 KiB |
| 1024 | modern_hopfield (noise 0%)  | bit error rate of recalled value     | >= 10,000 | -            | 0.0312   | 78.2 MiB  |
| 1024 | modern_hopfield (noise 20%) | bit error rate of recalled value     | >= 10,000 | -            | 0.0312   | 78.2 MiB  |
| 2048 | linear_matrix               | raw bit error rate                   | 213       | 215          | 0.0065   | 16.0 MiB  |
| 2048 | vsa_bundle                  | bit error rate after cleanup (SER/2) | 63        | 63           | 6.22e-05 | 808.0 KiB |
| 2048 | modern_hopfield (noise 0%)  | bit error rate of recalled value     | >= 10,000 | -            | 0.0312   | 156.3 MiB |
| 2048 | modern_hopfield (noise 20%) | bit error rate of recalled value     | >= 10,000 | -            | 0.0312   | 156.3 MiB |
| 4096 | linear_matrix               | raw bit error rate                   | 429       | 429          | 0.00648  | 64.0 MiB  |
| 4096 | vsa_bundle                  | bit error rate after cleanup (SER/2) | 125       | 126          | 6.27e-05 | 1.6 MiB   |
| 4096 | modern_hopfield (noise 0%)  | bit error rate of recalled value     | >= 10,000 | -            | 0.0312   | 312.5 MiB |
| 4096 | modern_hopfield (noise 20%) | bit error rate of recalled value     | >= 10,000 | -            | 0.0312   | 312.5 MiB |


A superposed bundle read **bit by bit** (no cleanup) already has BER 31.3% at N=5 for every D -- the per-bit SNR is 1/sqrt(N-1), independent of D.


Notes:

- C_max = largest N with BER <= 0.001 on the evaluated grid (monotone envelope); 'interpolated' is the linear crossing between grid points.
- linear_matrix and vsa_bundle have a hard capacity; modern_hopfield trades memory for capacity (N x D cells) so its C_max is censored by the largest store tested.
- vsa_bundle raw BER is >= 25% from N=2 because one bit sees SNR 1/sqrt(N-1) regardless of D; only cleanup against a codebook is reliable.
- eta counts information bits (N pairs x 2D for pair memories, N x log2(V) for the bundle) over all bytes the structure must keep; float32 storage is why eta << 1.

## Section 5 - Compute-in-memory emulation (accuracy emulated, latency modelled)

_21 s; seed 0; profile standard_

**Accuracy under analogue non-idealities -- EMULATED** (top-1 identification; value recall in the JSON). One non-ideality at a time, others ideal (N=1000 keys, 500 queries). Read noise is a fraction of a 256-row tile's full-scale output.

_D=512, query bit-flip noise 0% (ideal digital accuracy 100.0%)_

| non-ideality       | variant                  | value: identification accuracy                                                      |
| ------------------ | ------------------------ | ----------------------------------------------------------------------------------- |
| read_noise         | ideal ADC                | 0.01: 100%, 0.02: 100%, 0.05: 100%, 0.1: 100%, 0.2: 100%, 0.3: 90%, 0.5: 34%, 1: 6% |
| adc_bits           | fs=1                     | 2: 100%, 3: 100%, 4: 100%, 5: 100%, 6: 100%, 8: 100%                                |
| adc_bits           | fs=0.25                  | 2: 9%, 3: 99%, 4: 100%, 5: 100%, 6: 100%, 8: 100%                                   |
| program_noise      | ideal ADC                | 0.02: 100%, 0.05: 100%, 0.1: 100%, 0.2: 100%, 0.4: 100%                             |
| stuck_fraction     | ideal ADC                | 0.01: 100%, 0.05: 100%, 0.1: 100%, 0.2: 100%                                        |
| drift_t_over_t0    | 8-bit ADC, uncompensated | 100: 100%, 10,000: 100%, 1,000,000: 100%                                            |
| drift_t_over_t0    | 8-bit ADC, compensated   | 100: 100%, 10,000: 100%, 1,000,000: 100%                                            |
| combined_realistic | see params.combined      | 1: 100%                                                                             |


_D=512, query bit-flip noise 20% (ideal digital accuracy 100.0%)_

| non-ideality       | variant                  | value: identification accuracy                                                    |
| ------------------ | ------------------------ | --------------------------------------------------------------------------------- |
| read_noise         | ideal ADC                | 0.01: 100%, 0.02: 100%, 0.05: 100%, 0.1: 100%, 0.2: 78%, 0.3: 33%, 0.5: 7%, 1: 1% |
| adc_bits           | fs=1                     | 2: 100%, 3: 100%, 4: 100%, 5: 100%, 6: 100%, 8: 100%                              |
| adc_bits           | fs=0.25                  | 2: 10%, 3: 99%, 4: 100%, 5: 100%, 6: 100%, 8: 100%                                |
| program_noise      | ideal ADC                | 0.02: 100%, 0.05: 100%, 0.1: 100%, 0.2: 100%, 0.4: 100%                           |
| stuck_fraction     | ideal ADC                | 0.01: 100%, 0.05: 100%, 0.1: 100%, 0.2: 100%                                      |
| drift_t_over_t0    | 8-bit ADC, uncompensated | 100: 100%, 10,000: 100%, 1,000,000: 100%                                          |
| drift_t_over_t0    | 8-bit ADC, compensated   | 100: 100%, 10,000: 100%, 1,000,000: 100%                                          |
| combined_realistic | see params.combined      | 1: 100%                                                                           |


_D=512, query bit-flip noise 35% (ideal digital accuracy 100.0%)_

| non-ideality       | variant                  | value: identification accuracy                                                  |
| ------------------ | ------------------------ | ------------------------------------------------------------------------------- |
| read_noise         | ideal ADC                | 0.01: 100%, 0.02: 100%, 0.05: 100%, 0.1: 67%, 0.2: 14%, 0.3: 5%, 0.5: 1%, 1: 1% |
| adc_bits           | fs=1                     | 2: 100%, 3: 94%, 4: 100%, 5: 100%, 6: 100%, 8: 100%                             |
| adc_bits           | fs=0.25                  | 2: 8%, 3: 100%, 4: 100%, 5: 100%, 6: 100%, 8: 100%                              |
| program_noise      | ideal ADC                | 0.02: 100%, 0.05: 100%, 0.1: 100%, 0.2: 100%, 0.4: 100%                         |
| stuck_fraction     | ideal ADC                | 0.01: 100%, 0.05: 100%, 0.1: 100%, 0.2: 100%                                    |
| drift_t_over_t0    | 8-bit ADC, uncompensated | 100: 100%, 10,000: 100%, 1,000,000: 100%                                        |
| drift_t_over_t0    | 8-bit ADC, compensated   | 100: 100%, 10,000: 100%, 1,000,000: 100%                                        |
| combined_realistic | see params.combined      | 1: 100%                                                                         |


_D=2048, query bit-flip noise 0% (ideal digital accuracy 100.0%)_

| non-ideality       | variant                  | value: identification accuracy                                                        |
| ------------------ | ------------------------ | ------------------------------------------------------------------------------------- |
| read_noise         | ideal ADC                | 0.01: 100%, 0.02: 100%, 0.05: 100%, 0.1: 100%, 0.2: 100%, 0.3: 100%, 0.5: 99%, 1: 38% |
| adc_bits           | fs=1                     | 2: 100%, 3: 100%, 4: 100%, 5: 100%, 6: 100%, 8: 100%                                  |
| adc_bits           | fs=0.25                  | 2: 100%, 3: 100%, 4: 100%, 5: 100%, 6: 100%, 8: 100%                                  |
| program_noise      | ideal ADC                | 0.02: 100%, 0.05: 100%, 0.1: 100%, 0.2: 100%, 0.4: 100%                               |
| stuck_fraction     | ideal ADC                | 0.01: 100%, 0.05: 100%, 0.1: 100%, 0.2: 100%                                          |
| drift_t_over_t0    | 8-bit ADC, uncompensated | 100: 100%, 10,000: 100%, 1,000,000: 100%                                              |
| drift_t_over_t0    | 8-bit ADC, compensated   | 100: 100%, 10,000: 100%, 1,000,000: 100%                                              |
| combined_realistic | see params.combined      | 1: 100%                                                                               |


_D=2048, query bit-flip noise 20% (ideal digital accuracy 100.0%)_

| non-ideality       | variant                  | value: identification accuracy                                                      |
| ------------------ | ------------------------ | ----------------------------------------------------------------------------------- |
| read_noise         | ideal ADC                | 0.01: 100%, 0.02: 100%, 0.05: 100%, 0.1: 100%, 0.2: 100%, 0.3: 98%, 0.5: 56%, 1: 8% |
| adc_bits           | fs=1                     | 2: 100%, 3: 100%, 4: 100%, 5: 100%, 6: 100%, 8: 100%                                |
| adc_bits           | fs=0.25                  | 2: 100%, 3: 100%, 4: 100%, 5: 100%, 6: 100%, 8: 100%                                |
| program_noise      | ideal ADC                | 0.02: 100%, 0.05: 100%, 0.1: 100%, 0.2: 100%, 0.4: 100%                             |
| stuck_fraction     | ideal ADC                | 0.01: 100%, 0.05: 100%, 0.1: 100%, 0.2: 100%                                        |
| drift_t_over_t0    | 8-bit ADC, uncompensated | 100: 100%, 10,000: 100%, 1,000,000: 100%                                            |
| drift_t_over_t0    | 8-bit ADC, compensated   | 100: 100%, 10,000: 100%, 1,000,000: 100%                                            |
| combined_realistic | see params.combined      | 1: 100%                                                                             |


_D=2048, query bit-flip noise 35% (ideal digital accuracy 100.0%)_

| non-ideality       | variant                  | value: identification accuracy                                                    |
| ------------------ | ------------------------ | --------------------------------------------------------------------------------- |
| read_noise         | ideal ADC                | 0.01: 100%, 0.02: 100%, 0.05: 100%, 0.1: 100%, 0.2: 79%, 0.3: 33%, 0.5: 8%, 1: 1% |
| adc_bits           | fs=1                     | 2: 100%, 3: 100%, 4: 100%, 5: 100%, 6: 100%, 8: 100%                              |
| adc_bits           | fs=0.25                  | 2: 100%, 3: 100%, 4: 100%, 5: 100%, 6: 100%, 8: 100%                              |
| program_noise      | ideal ADC                | 0.02: 100%, 0.05: 100%, 0.1: 100%, 0.2: 100%, 0.4: 100%                           |
| stuck_fraction     | ideal ADC                | 0.01: 100%, 0.05: 100%, 0.1: 100%, 0.2: 100%                                      |
| drift_t_over_t0    | 8-bit ADC, uncompensated | 100: 100%, 10,000: 100%, 1,000,000: 100%                                          |
| drift_t_over_t0    | 8-bit ADC, compensated   | 100: 100%, 10,000: 100%, 1,000,000: 100%                                          |
| combined_realistic | see params.combined      | 1: 100%                                                                           |


**Tolerance limits** (largest value keeping >= 99% identification; '-' = ideal accuracy itself is below 99% here, or never reached):

| D    | query noise | parameter          | limit | unit                                    |
| ---- | ----------- | ------------------ | ----- | --------------------------------------- |
| 512  | 0%          | read_noise         | 0.2   | fraction of tile full scale             |
| 512  | 0%          | program_noise      | 0.4   | fraction                                |
| 512  | 0%          | stuck_fraction     | 0.2   | fraction                                |
| 512  | 0%          | adc_bits (fs=1)    | 2     | minimum bits for >=99% (and all larger) |
| 512  | 0%          | adc_bits (fs=0.25) | 3     | minimum bits for >=99% (and all larger) |
| 512  | 20%         | read_noise         | 0.1   | fraction of tile full scale             |
| 512  | 20%         | program_noise      | 0.4   | fraction                                |
| 512  | 20%         | stuck_fraction     | 0.2   | fraction                                |
| 512  | 20%         | adc_bits (fs=1)    | 2     | minimum bits for >=99% (and all larger) |
| 512  | 20%         | adc_bits (fs=0.25) | 3     | minimum bits for >=99% (and all larger) |
| 512  | 35%         | read_noise         | 0.05  | fraction of tile full scale             |
| 512  | 35%         | program_noise      | 0.4   | fraction                                |
| 512  | 35%         | stuck_fraction     | 0.2   | fraction                                |
| 512  | 35%         | adc_bits (fs=1)    | 4     | minimum bits for >=99% (and all larger) |
| 512  | 35%         | adc_bits (fs=0.25) | 3     | minimum bits for >=99% (and all larger) |
| 2048 | 0%          | read_noise         | 0.5   | fraction of tile full scale             |
| 2048 | 0%          | program_noise      | 0.4   | fraction                                |
| 2048 | 0%          | stuck_fraction     | 0.2   | fraction                                |
| 2048 | 0%          | adc_bits (fs=1)    | 2     | minimum bits for >=99% (and all larger) |
| 2048 | 0%          | adc_bits (fs=0.25) | 2     | minimum bits for >=99% (and all larger) |
| 2048 | 20%         | read_noise         | 0.2   | fraction of tile full scale             |
| 2048 | 20%         | program_noise      | 0.4   | fraction                                |
| 2048 | 20%         | stuck_fraction     | 0.2   | fraction                                |
| 2048 | 20%         | adc_bits (fs=1)    | 2     | minimum bits for >=99% (and all larger) |
| 2048 | 20%         | adc_bits (fs=0.25) | 2     | minimum bits for >=99% (and all larger) |
| 2048 | 35%         | read_noise         | 0.1   | fraction of tile full scale             |
| 2048 | 35%         | program_noise      | 0.4   | fraction                                |
| 2048 | 35%         | stuck_fraction     | 0.2   | fraction                                |
| 2048 | 35%         | adc_bits (fs=1)    | 2     | minimum bits for >=99% (and all larger) |
| 2048 | 35%         | adc_bits (fs=0.25) | 2     | minimum bits for >=99% (and all larger) |


**Search latency -- MODELED projection vs MEASURED CPU** (not a measured hardware result; see assumptions below). CPU = the fastest single-thread implementation available here (FAISS binary flat if installed, else NumPy).

| N keys | D    | CPU best us (measured) | CPU implementation | preset       | tiles | waves | modeled ns | speed-up if model holds | regime                                       |
| ------ | ---- | ---------------------- | ------------------ | ------------ | ----- | ----- | ---------- | ----------------------- | -------------------------------------------- |
| 100    | 512  | 3.4                    | faiss_binary_flat  | optimistic   | 2     | 1     | 20.0       | 170x                    | single wave (size-independent analogue time) |
| 100    | 512  | 3.4                    | faiss_binary_flat  | conservative | 2     | 1     | 386.0      | 9x                      | single wave (size-independent analogue time) |
| 1000   | 512  | 5.4                    | faiss_binary_flat  | optimistic   | 8     | 1     | 25.5       | 212x                    | single wave (size-independent analogue time) |
| 1000   | 512  | 5.4                    | faiss_binary_flat  | conservative | 8     | 1     | 772.0      | 7x                      | single wave (size-independent analogue time) |
| 10000  | 512  | 24.6                   | faiss_binary_flat  | optimistic   | 80    | 1     | 27.5       | 893x                    | single wave (size-independent analogue time) |
| 10000  | 512  | 24.6                   | faiss_binary_flat  | conservative | 80    | 5     | 3,780      | 6x                      | 5 serialised waves                           |
| 100000 | 512  | 299.3                  | faiss_binary_flat  | optimistic   | 782   | 1     | 29.0       | 10322x                  | single wave (size-independent analogue time) |
| 100000 | 512  | 299.3                  | faiss_binary_flat  | conservative | 782   | 49    | 36,786     | 8x                      | 49 serialised waves                          |
| 1000   | 4096 | 19.3                   | faiss_binary_flat  | optimistic   | 64    | 1     | 27.0       | 715x                    | single wave (size-independent analogue time) |
| 1000   | 4096 | 19.3                   | faiss_binary_flat  | conservative | 64    | 4     | 3,028      | 6x                      | 4 serialised waves                           |
| 10000  | 4096 | 253.1                  | faiss_binary_flat  | optimistic   | 640   | 1     | 29.0       | 8729x                   | single wave (size-independent analogue time) |
| 10000  | 4096 | 253.1                  | faiss_binary_flat  | conservative | 640   | 40    | 30,036     | 8x                      | 40 serialised waves                          |


Timing-model parameters (illustrative placeholders, ns):

| preset       | t_dac_ns | t_read_ns | t_adc_ns | adc_per_array | n_arrays | t_accum_ns | t_wta_ns |
| ------------ | -------- | --------- | -------- | ------------- | -------- | ---------- | -------- |
| optimistic   | 2.0      | 10.0      | 2.0      | 64            | 1024     | 0.5        | 0.5      |
| conservative | 10.0     | 100.0     | 20.0     | 8             | 16       | 2.0        | 2.0      |


Notes:

- ACCURACY rows are numerical emulation of a tiled differential-pair crossbar (per-tile read noise and ADC, static programming/stuck/drift errors); they are not silicon measurements.
- LATENCY rows are an analytical projection from the stated timing parameters (CIMTimingModel presets are illustrative placeholders); they are NOT measurements and are excluded from the scorecard.
- The CPU column is a real single-thread measurement of the fastest available implementation (FAISS binary flat when installed, else NumPy float32 BLAS or packed popcount); the speed-up column divides a measurement by a projection.
- The model covers the search stage only: no programming/write time or energy, no host I/O, and (as in real arrays) a size-independent analogue time only while all tiles fit the array budget.
- Read-noise sigma is relative to a tile's full-scale output (tile_rows x input_fs); with 256-row tiles a 2% setting means sigma = 5.1 in dot-product units against an impostor spread of sqrt(D).

## Skipped systems

| wl | system                 | reason                                                               |
| -- | ---------------------- | -------------------------------------------------------------------- |
| A  | datasketch_minhash_lsh | N=10000 exceeds this system's cap of 2000                            |
| C  | qdrant_local           | N=50000 exceeds this system's cap of 20000                           |
| D  | holographic_hrr        | E=10000, D=8192: complex spectra of the item memory would need >1 GB |

