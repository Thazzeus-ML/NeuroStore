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

## Colab benchmark notebook (`colab/`)

`colab/REVIEW.md` reviews an uploaded GPU/TPU Colab notebook for the report's FlyHash / SDM / VSA / Hopfield ideas, with every
finding measured (`colab/review_evidence.json`), and `colab/NeuroStore_Colab_Benchmarks_v2.ipynb` is the reinforced version
(sources in `colab/src/`, built by `colab/build_notebook.py`). It has been executed on CPU JAX only; `make colab-validate`
reruns that validation. The notebook is a separate design from the Python prototype in `src/neurostore`; its Section 11 runs
the prototype's sparse-accumulate kernel on an accelerator.

