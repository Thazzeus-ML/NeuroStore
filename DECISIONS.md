# DECISIONS

Every design choice made without asking, in the order made. Format: **D<n>** decision — reason.

**D1** Work in `./neurostore` (created under the session working directory `/home/user`). Python 3.13, NumPy 2.5, pytest, hypothesis; no other dependencies.
**D2** Gates live in `gates.toml` and are *read* by the experiment scripts (`eval/gates.py`), so a gate verdict cannot silently drift from the committed threshold. Each phase has two commits: a `prereg:` commit containing only the gate, then the phase commit with code, tests, results and the verdict.
**D3** Gate runs use seeds 0-4. Seeds >=100 are dev seeds used to debug and to choose hyper-parameters on a validation stream; they never appear in gate numbers. Hyper-parameter sweeps are done on validation streams (never the held-out test stream) and are logged in `results/`.
**D4** World defaults: F=800 raw features, C=40 concepts, M=20 members, p=0.7, Poisson(3) noise, 1/2/3 concepts with prob .6/.3/.1, chain stay=0.5 + 3 preferred successors (mass 0.3) + popularity background. Chosen so concepts overlap by chance in raw features (F is small relative to C*M=800) and the task is not trivially separable. Not tuned against any system result.
**D5** Noise features are Poisson(noise_mean) uniform draws with replacement. Reason: per-feature presence is then independent Poisson, which gives an exact analytic mean feature count to test against.
**D6** Extra (2nd/3rd) concepts are drawn from the popularity distribution, not the chain; only the primary concept has sequential structure. Anomaly events (optional) are pure-noise events with no concepts and do not advance the chain.
**D7** Streams are addressed as (split, idx): train/val/test/extra use disjoint SeedSequence streams. With drift enabled, a train stream switches membership at n//2; test streams are generated with a fixed phase (0 = pre-drift, 1 = post-drift).
**D8** Phase 0 gate: PASS on seeds 0-4 (results/phase0.json).
