# %% [markdown] id=intro
# # NeuroStore: GPU/TPU Benchmark & Simulation Lab (v2: reviewed and extended)
#
# **Purpose.** Evaluate the computational ideas behind *NeuroStore: Empirical Evaluation of a Neural-Coding Data
# Architecture* on whatever accelerator Colab assigns, and say honestly how much each measurement can support.
#
# This is an **independent, conceptual implementation**. It does not execute the report's original `coding.py`,
# `memory_sdm.py`, `memory_vsa.py`, `memory_hopfield.py` or `benchmarks.py`, so nothing here should be described as
# reproducing the report until sources, seeds, datasets and configurations have been reconciled.
#
# ### What changed relative to v1 (full review in `colab/REVIEW.md`)
#
# | Problem found in v1 (each one measured, see `colab/review_evidence.json`) | What v2 does |
# |---|---|
# | OOD AUROC = 1.0 and 14/14 fault conditions at recall 1.0: the tests could not fail. Noisy recall at ≤ 25 % flips has an error probability near 1e-11. | Operating points chosen from an exact analytic model so baselines sit near 95 % recall; harder OOD families; many more fault models. |
# | One draw of 128 queries, no intervals. The published Hopfield-vs-exact gaps were not significant (McNemar p = 0.45 and 0.69) yet over 300 independent batches exact search is better at N = 512 and 40 % flips. | Several seeds, Wilson intervals, paired exact McNemar tests, analytic cross-check. |
# | p95/p99 reported from 8 samples; with so few samples the "p99" is essentially the maximum and is biased low (quantified in `REVIEW.md`). | Percentiles only when enough samples exist; separate single-request latency study with hundreds of samples. |
# | Sizes (N ≤ 2048, 128 queries) are at most ~0.5 GFLOP, so on an accelerator launch and synchronisation overhead can dominate the measured milliseconds. | A measured dispatch floor, scaling sweeps in N and batch size, achieved GFLOP/s and GB/s, a launch-bound flag. |
# | Global `highest` matmul precision (multi-pass on TPU, no TF32 on GPU) although ±1 data is exact in bf16/int8. | Explicit precision and kernel variants, including int8 and bit-packed popcount. |
# | Results list not idempotent (re-running a cell duplicated rows), CSV lacked profile/seed/version columns. | Keyed records, full provenance columns, config hash. |
# | VSA top-1 equals its oracle by construction; SDM compared without a memory-efficiency view. | Tuple-level and decode-capacity metrics validated against theory; SDM with K/M sweeps and bytes per reliable pattern. |
#
# ### Experiments
#
# 1. Noisy-key retrieval: exact flat search vs Modern-Hopfield cleanup, with analytic validation and paired statistics.
# 2. Hopfield deep dive: inverse temperature, iteration, denoising error, capacity.
# 3. Kernel and precision variants: f32 / bf16 / int8 / bit-packed popcount / approximate top-k.
# 4. Systems scaling: N sweep, batch sweep, single-request latency distribution, compile cost, device memory.
# 5. FlyHash-style sparse projection vs SimHash vs exact, with code-size accounting.
# 6. Role-filler VSA: record retrieval and bundle-decode capacity vs theory.
# 7. Sparse distributed memory: capacity, K/M sweeps, dense vs scatter/gather implementation, bytes per pattern.
# 8. Out-of-distribution rejection: random, chimera and partial-overlap families, several scores.
# 9. Fault injection: bit flips, stuck-at, dead rows/columns, variation, read noise, ADC quantisation.
# 10. Multi-device scaling (data parallel and memory sharded) when more than one device is present.
# 11. The NeuroStore Python prototype's core read kernel (sparse accumulate) on the accelerator.
# 12. Self-checks, summary tables, plots; 13. export and cross-hardware comparison.
#
# ### How to run
#
# 1. **Runtime → Change runtime type**: choose CPU, a GPU or a TPU. Run the notebook once per runtime you want to compare.
# 2. Run all cells in order. Start with `PROFILE = "quick"`. `"smoke"` is a tiny profile for checking the notebook
#    itself; `"extended"` takes much longer and uses more memory.
# 3. Download the ZIP from the last section and use the comparison cell to merge runs.
#
# ### Rules for reading the results
#
# * Each run benchmarks **one JAX device**, even when a TPU exposes several cores (Section 10 adds explicit multi-device runs).
# * Device-resident timings exclude host transfers and index construction; transfer-inclusive latency is reported separately.
# * Batch throughput is not single-request latency; both are measured and labelled.
# * FlyHash, VSA and SDM are **algorithmic proxies**; SDM uses fixed-K activation, not the report's exact variant.
# * Fault injection is **functional**, not a calibrated FeNAND / crossbar / compute-in-memory device model. Energy,
#   endurance, ADC/DAC power and area are **not measured** here.
# * Every claim should be backed by an interval or a test; the final cell prints a claims ledger with the evidence.
#
# References: [Colab FAQ](https://research.google.com/colaboratory/faq.html),
# [JAX installation](https://docs.jax.dev/en/latest/installation.html),
# [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html).
