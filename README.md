# NeuroStore (research prototype)

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
