# DECISIONS

Every design choice made without asking, in the order made. Format: **D<n>** decision — reason.

**D1** Work in `./neurostore` (created under the session working directory `/home/user`). Python 3.13, NumPy 2.5, pytest, hypothesis; no other dependencies.
**D2** Gates live in `gates.toml` and are *read* by the experiment scripts (`eval/gates.py`), so a gate verdict cannot silently drift from the committed threshold. Each phase has two commits: a `prereg:` commit containing only the gate, then the phase commit with code, tests, results and the verdict.
**D3** Gate runs use seeds 0-4. Seeds >=100 are dev seeds used to debug and to choose hyper-parameters on a validation stream; they never appear in gate numbers. Hyper-parameter sweeps are done on validation streams (never the held-out test stream) and are logged in `results/`.
