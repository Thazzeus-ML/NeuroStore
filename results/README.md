# Reference results

`standard/` is the committed reference run of the **standard** profile:

| | |
|---|---|
| command | `neurostore run --profile standard --seed 0 --strict --out results/standard` |
| code revision | `3a14086` (recorded inside `results.json` as `environment.git_commit`; the tree was clean) |
| machine | 4 x Intel Xeon @ 2.10 GHz, 15.7 GiB, Linux, Python 3.13, NumPy 2.5.3 |
| libraries | faiss-cpu 1.15.1, qdrant-client 1.20.0, datasketch 2.0.0, rocksdict 0.3.29, redis 8.1.0 (+ a real `redis-server`) |
| threads | 1 (enforced with threadpoolctl) |
| run time | about 25 minutes (`total benchmark time` in the report header) |
| result | exit status 0 under `--strict`: every benchmark self-check passed |

* `results.json` is the raw measurement record (all tables, checks, notes, environment).
* `REPORT.md` is **derived**: `neurostore report results/standard/results.json` regenerates it. It was rendered with a later
  version of the report code than the one that produced the measurements (only presentation changed: two captions, the
  FlyHash sweep table and the explanation of the ill-posed paraphrase regime).
* Numbers are specific to this machine and to these Python/NumPy implementations. Latencies are single-thread and include
  interpreter overhead (the empty-call overhead is in the header); the CIM latency table is a **model**, not a measurement.

`exploratory/` holds one-off experiments that are *not* part of the reference run (each has a script under `experiments/`):

* `flyhash_intrinsic_dim.md` -- does the FlyHash-vs-SimHash ordering depend on the intrinsic dimension of the embeddings?
