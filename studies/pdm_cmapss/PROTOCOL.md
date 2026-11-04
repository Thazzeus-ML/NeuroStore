# PdM study protocol: NeuroStore-style memories on NASA C-MAPSS turbofan degradation

**Status: PRE-REGISTERED.** This file, `gates.toml`, `config.py`, `run_study.py` and `evaluate_gates.py` are committed before the first
real-data result exists. After that point no threshold, grid, band, severity target, metric or gate may change; a bug fix is allowed only
if it is listed in the report's deviation log, the affected stage is re-run, and the earlier result file is kept next to the new one.

## 0. What this study can and cannot show

* **C-MAPSS is a simulation** (NASA's commercial modular aero-propulsion system simulation). Its readme states "the data is contaminated
  with sensor noise"; that noise is *simulated*, so nothing here measures real field corruption (dead sensors, drift, bit errors, gaps).
  The robustness results come from corruption **we inject** with the harness' standardized families. A real-sensor follow-up (for example
  NASA's IMS bearing run-to-failure data) would be needed to claim anything about natural corruption.
* The question asked is architectural: *if remaining useful life (RUL) is read out from a memory of past run-to-failure windows, how do
  exact, approximate (IVF/SQ8), FlyHash, Hopfield and VSA stores compare on accuracy, robustness, familiarity (OOD/chimera), completion,
  latency, throughput and memory?* It is not an attempt to beat the deep-learning state of the art on C-MAPSS.
* One workstation CPU, NumPy only, BLAS pinned to one thread. Timings are machine-specific and reported with the environment; accelerator
  behaviour is **not** measured. 100 test engines (FD001) bound the precision of every official-test number; intervals are reported.

Before this protocol was committed the only real-data inspection was: archive sizes and SHA-256 values, row and engine counts, the number of
distinct values and the standard deviation of each sensor in the *training* files, and the operating-condition cluster sizes of FD002/FD004.
No model was fitted to, or scored on, real data. (The NASA readme swaps the FD004 train/test engine counts relative to the files; the files
are authoritative: 249 train / 248 test.) Smoke tests of the runner used fake files in the same format (`--fast`, `results/dev/`).

## 1. Data

NASA Prognostics Center of Excellence, "Turbofan Engine Degradation Simulation Data Set" (C-MAPSS), FD001-FD004, fetched by
`fetch_data.py` from the pinned URL with SHA-256 checks on the outer archive, the nested archive and all 13 extracted files; files are
extracted by explicit name only. Four sub-datasets: FD001 (1 operating condition, HPC degradation), FD002 (6 conditions, HPC), FD003
(1 condition, HPC + fan), FD004 (6 conditions, HPC + fan). Each has run-to-failure **train** engines, **test** engines truncated before failure
and an official **RUL** file (remaining cycles at the last test cycle).

## 2. Preprocessing (all parameters fitted on the training file only)

* **Regimes:** FD001/FD003 one regime. FD002/FD004: k-means (k = 6, 3 initialisations, seeded) on standardized operating settings; test rows
  are assigned to the nearest training centre.
* **Channels:** sensors s1-s21; keep a sensor iff it has >= 10 distinct training values in every regime (`MIN_UNIQUE`). Operating settings
  are not channels (they only define the regime). Expected FD001 result: 14 channels (s2,3,4,7,8,9,11,12,13,14,15,17,20,21).
* **Normalisation:** per regime and channel z-score with training mean/std.
* **Windows:** W = 30 consecutive cycles ending at cycle t, left-padded with the first cycle (some test engines are shorter than 30 cycles).
* **Memory:** every cycle of every training engine (stride 1). **Stored target:** capped RUL = min(L - t, 125).
* **Official queries:** the final window of each test engine; scored against the official RUL (*raw*, primary, comparable with the literature)
  and against min(RUL, 125) (*capped*, secondary).
* **Features** (all backends): per channel last value, mean, least-squares slope and std over the window -> 4C features (56 for FD001), then
  per-feature z-score fitted on the memory. No PCA.

## 3. Systems under test

| name | what it is | similarity |
|---|---|---|
| `exact` | brute-force cosine search | cosine on features |
| `exact_l2` | brute-force Euclidean search (textbook kNN) | -(mean squared difference per feature) |
| `hopfield` | softmax attention over **all** stored patterns, energy familiarity, clamped completion | metric of the reference (below) |
| `ann` / `ann_sq8` | IVF (k-means coarse quantiser, nlist = 2 sqrt(n), nprobe lists), fp32 / 8-bit scalar quantised vectors | metric of the reference |
| `flyhash` | sparse random expansion (fan-in 10 %) -> winner-take-all -> overlap of active units via an inverted index | overlap / k |
| `vsa` | 32-level thermometer-style level vectors (+) role vectors, majority bundle, packed popcount Hamming | 1 - 2 Hamming / D |

*Readout* (RUL estimate): weighted mean of the stored capped RUL of the k retrieved items, weights softmax(beta * similarity)
(uniform if beta = none); `hopfield` attends over all items (k unused).
*Exact reference:* the better (lower CV RMSE, section 4) of `exact` and `exact_l2` is **the exact reference** for every comparison below;
`hopfield`, `ann`, `ann_sq8` use the reference's similarity so that they re-read or approximate the same exact store. `flyhash` and `vsa` keep
their native similarity (that difference is part of the question). Context baselines: `const` (mean stored target) and `ridge` (closed form on
the same features, lambda by CV). `*_mask` variants receive the observed-sensor mask where the corruption family has one (dropout, combined);
without a mask a lost channel is zero (the z-scored mean).

*Seeds:* `ann`, `ann_sq8`, `flyhash`, `vsa` use model seeds 0, 1, 2 (metrics are averaged as sqrt of the mean squared error over seeds);
`exact`, `exact_l2`, `hopfield` are deterministic.

## 4. Experiments

All confidence intervals resample **whole engines** (windows of one engine are correlated); paired comparisons use the same resamples.
Bootstrap sizes: 500 (tables), 1000 (paired official-test ratios), 300 (AUROC). No multiplicity correction: the gates below are the
confirmatory claims, everything else is descriptive.

**E6 - harness validity.** Exact search on random +-1 patterns of five (D, N, p_flip) settings (4000 queries x 3 seeds each) against the
closed-form tie-aware recall; Wilson interval at +-3 sigma. A secondary panel runs all five backends on the noisy-pattern recall task of the
Colab benchmark (D = 512, N = 2048).

**E0 - tuning (training engines only, never the official test).** Engine-grouped 5-fold CV, 10 random truncation points per held-out engine
(window must end at cycle >= 30), identical queries for every backend. Grids: k in {1,3,5,10,20,50}; beta in {none,3,10,30,100,300};
Hopfield beta in {3,10,30,100,300,1000,3000}; ann nprobe in {1,2,4,8,16}; flyhash m in {1024,2048,4096} x active fraction in {0.02,0.05,0.10};
vsa D in {2048,4096}. Selection = lowest CV RMSE against the capped target (mean over seeds), except `ann`/`ann_sq8`: the *smallest* nprobe
whose CV RMSE is within 2 % of the reference's. FD001 selects structure and readout; FD002-FD004 reuse FD001's structure and re-tune only (k, beta).
A selection at the edge of its grid is reported, not extended.

**E1 - official test.** For FD001-FD004 fit on all training windows with the selected configuration; RMSE (raw and capped), MAE, NASA PHM08 score
(late predictions penalised more), engine-bootstrap CIs, paired RMSE ratios against the reference, against `const` and against `ridge`.

**E2 - corruption sweeps (FD001, out-of-fold CV queries, 1000 queries from 100 engines).** Families: noise, dropout, stuck, drift, spikes,
quantize, combined (definitions in `corruption.py`). For each family the severity that makes the *reference* degrade to x1.25 / x1.5 / x2.0
of its clean RMSE (levels L25/L50/L100; grid of 20 plus 6 bisection steps, calibration corruption seed 100) is found once and then applied to
every backend with corruption seeds 0 and 1. If a level is unreachable at severity 1 the maximum is used and flagged. Reported: RMSE ratio
corrupted/clean per backend and the paired difference of that ratio against the reference.

**E3 - completion (FD001, same queries).** Channel dropout at severity 0.1 / 0.3 / 0.5 (2 / 5 / 7 of 14 channels); the memory reconstructs the
feature vector; NMSE on the *missing* features only (features are z-scored, mean imputation scores ~1.0). `exact`, `exact_l2`, `hopfield`, `vsa`
with and without the mask. FlyHash and IVF have no decoder and are not scored.

**E4 - familiarity (FD001 memory, held-out-engine ID windows at stride 5 pooled over the 5 CV folds).** Scores (higher = more familiar):
max similarity, margin, mean of top-10, and Hopfield neg-energy. AUROC with ID as the positive class; OOD sets: Gaussian windows, time-shuffled
windows, channel-permuted windows, foreign sub-datasets FD002/FD003/FD004 pushed through FD001's preprocessing, FD003 late-life windows
(RUL <= 50) against FD001 late-life windows, and **chimeras** (random half of the channels, or first half of the window, from window A and the
rest from window B) built from held-out windows with a capped-RUL gap between A and B in bands *same* [0,5], *mid* [30,60], *far* [80,125] cycles.
The same-stage band is the control for splicing artefacts. Groups for the bootstrap are engines.

**E5 - latency, throughput, memory (FD001, full memory, single thread).** Build time and tracemalloc peak; deployed bytes by component
(alternative encodings and caches are listed but not counted); single-query latency over 1000 queries (p95 reported only with >= 100 samples,
p99 only with >= 1000); throughput at batch 256; plus a scale-up on clustered synthetic vectors (d = 56) at n up to 500k for exact / ann / ann_sq8
(nprobe = smallest in {1,2,4,8,16,32,64} reaching recall@10 >= 0.9 against exact) and up to 100k for flyhash / vsa.

## 5. Gates (thresholds in `gates.toml`; verdicts computed by `evaluate_gates.py`)

| gate | claim | rule |
|---|---|---|
| G0 | provenance | every result file from committed code, this protocol (hash) and the pinned data |
| G1 | harness validity | all 5 analytic configurations inside the +-3 sigma Wilson interval |
| G2 | utility | FD001 official test: RMSE(reference) <= 0.60 x RMSE(const) and <= 1.25 x RMSE(ridge) (point estimates, raw labels) |
| G3[b] | non-inferiority of backend b | paired RMSE ratio vs reference: HELD if CI upper <= 1.10; INCONCLUSIVE if point <= 1.10 < upper; else NOT HELD |
| G4 | efficiency | some b with G3 HELD/INCONCLUSIVE has <= 50 % of the reference's deployed bytes AND ANN (recall@10 >= 0.9) has p50 <= 1/3 of exact at n = 500k |
| G5 | robustness | at level L50 some *alternative* backend (hopfield, ann, ann_sq8, flyhash, vsa) has a paired degradation CI entirely below the reference's in >= 3 of 7 families (the non-reference exact variant is reported, not counted) |
| G6a | near-OOD | reference max-sim AUROC for FD003 late-life windows: CI lower bound >= 0.75 |
| G6b | chimera dose-response | AUROC(far) >= 0.70 and CI lower(far) > CI upper(same), channel chimeras, reference max-sim |
| G6c | sanity | AUROC(Gaussian windows) >= 0.95 |
| G7 | completion | `<reference>_mask` NMSE at dropout 0.3 <= 0.50 |

Verdict words: HELD, NOT HELD, INCONCLUSIVE (G3 only), NOT EVALUABLE (a needed measurement is absent). A failed gate is a finding, not a
reason to change the gate.

## 6. Predictions (recorded now, scored in the report; not gates)

P1. At their CV-selected configurations FlyHash and VSA need at least as many deployed bytes as the exact store (the features are only 56 floats).
P2. SQ8 costs <= 2 % RMSE relative to fp32 IVF on the official test.
P3. Hopfield with the reference similarity is within +-3 % RMSE of the reference.
P4. Knowing which sensors are dead lowers completion NMSE for `exact`/`exact_l2`/`hopfield` at all three severities (paired CI excludes zero).
P5. The same-stage chimera control has AUROC > 0.55 (splicing leaves detectable traces), which is why G6b requires a dose-response.
P6. Euclidean exact has a lower CV RMSE than cosine exact (the reference will be `exact_l2`).

## 7. Reporting rules

REPORT.md is generated from the result JSON by `make_report.py`; every number in its prose comes from that script. It states per gate the
verdict and the numbers, scores P1-P6, lists every deviation from this protocol, and repeats the section-0 caveats. Result files:
`results/pdm_cmapss/{validate,tuning_FD00x,test_panel,sweeps,completion,ood,timing}.json`, each with the git commit, protocol hash, data
checksums, configuration snapshot and environment.
