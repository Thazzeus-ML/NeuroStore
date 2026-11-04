# NeuroStore
Empirical Evaluation of a Neural-Coding Data Architecture

Events as sparse, timestamped activity over feature units; meaning in learned weights. See `REPORT.md` for results,
`DECISIONS.md` for every design choice, `gates.toml` for the pre-registered gates.

```
make test     # pytest + hypothesis, ~10 s
make bench    # all experiments (~2 h, 4 cores); each script reads its thresholds from gates.toml and writes results/*.json
make report   # regenerate REPORT.md from results/*.json (eval/report.py + eval/report_narrative.md)
```

Layout: `src/neurostore/` (encoder, spikelog, rates, assembly, synapse, predict, homeostasis, consolidate, readout, store,
`layouts/` aos/soa/volley/csr/dictgraph), `sim/generator.py` (synthetic world), `eval/` (metrics, baselines, protocol, report),
`bench/phase*.py` (one script per experiment), `tests/`.
Seeds 0-4 are gate seeds; seeds >=100 are dev seeds. `results/dev/` is git-ignored.

## nsharness: a domain-neutral test harness (`src/nsharness/`) and the PdM study (`studies/pdm_cmapss/`)

`nsharness` (NumPy only) turns the Colab benchmark into a reusable harness. A **domain adapter** returns windows `(n, T, C)`, targets and
group ids (`Split`, `DomainData`); everything else is domain-neutral:

| piece | what it does |
|---|---|
| `features.Featurizer` | window statistics (last / mean / slope / std per channel) or raw flatten, optional PCA; channel-aligned feature masks |
| `backends/` | `exact` (cosine or Euclidean, mask-aware), `ann` (IVF, optional SQ8), `flyhash`, `hopfield`, `vsa` behind one interface: `fit / search / readout / familiarity / reconstruct / memory_report` |
| `corruption` | seven standardized families (noise, dropout, stuck, drift, spikes, quantize, combined), severity in [0, 1], calibration to a target degradation |
| `ood` | Gaussian / shuffled / channel-permuted windows, chimeras (channel or time splice) with a target-gap knob, group-bootstrap AUROC |
| `tune`, `tasks` | engine-grouped CV, paired group bootstrap, degradation ratios, completion NMSE |
| `bench` | latency percentiles gated by sample count, throughput, tracemalloc build profile, deployed-bytes tables (alternatives and caches listed, not counted) |
| `domains/` | `synthetic` (+-1 patterns, validated against the closed-form recall), `cmapss` (NASA turbofan, all preprocessing fitted on training files only) |

To add a domain write one function that returns a `DomainData` (see `domains/cmapss.py`); the backends, corruption families, OOD tests and
benchmarks need no change.

The first study is predictive maintenance on NASA C-MAPSS. It is **pre-registered**: `studies/pdm_cmapss/PROTOCOL.md` and `gates.toml` were
committed before any real-data result existed. C-MAPSS is a simulation, so its "natural" sensor noise is simulated; real field corruption is
out of scope (see the protocol).

```
make pdm-fetch    # download + SHA-256 verify the NASA files into data/ (git-ignored)
make pdm-study    # all stages -> results/pdm_cmapss/*.json  (single-thread BLAS; the timing stage should run alone)
make pdm-gates    # mechanical HELD / NOT HELD verdicts
make pdm-report   # studies/pdm_cmapss/REPORT.md from the result files
```

## Colab benchmark notebook (`colab/`)

`colab/REVIEW.md` reviews an uploaded GPU/TPU Colab notebook for the report's FlyHash / SDM / VSA / Hopfield ideas, with every
finding measured (`colab/review_evidence.json`), and `colab/NeuroStore_Colab_Benchmarks_v2.ipynb` is the reinforced version
(sources in `colab/src/`, built by `colab/build_notebook.py`). It has been executed on CPU JAX only; `make colab-validate`
reruns that validation. The notebook is a separate design from the Python prototype in `src/neurostore`; its Section 11 runs
the prototype's sparse-accumulate kernel on an accelerator.

