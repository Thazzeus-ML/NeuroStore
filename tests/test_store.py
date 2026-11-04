import numpy as np
from neurostore.config import Config
from neurostore.store import NeuroStore
from sim.generator import World, WorldConfig


def run(cfg, n=700, seed=0):
    w = World(WorldConfig(), seed); s = w.stream("train", n)
    st = NeuroStore(cfg); st.ingest(s); st.flush()
    return st, s


def test_end_to_end_invariants_and_determinism():
    cfg = Config(seed=1, gate=True, gate_tau=0.55, homeo=True)
    a, s = run(cfg); b, _ = run(cfg)
    assert a.log.n_stored + a.log.n_count_only == len(s) == a.n_events
    assert a.log.bytes_written == a.log.n_stored * (8 + 4 * cfg.k)             # bytes depend on k only
    idx = np.arange(a.log.n_stored)
    for x, y in zip(a.log.gather(idx), b.log.gather(idx)):
        assert np.array_equal(x, y)                                              # deterministic
    assert np.array_equal(a.rates.total, b.rates.total) and a.rates.total.sum() == len(s) * cfg.k   # rates count every event
    units = a.log.gather(idx)[0]
    assert all(len(np.unique(u)) == cfg.k for u in units)
    assert a.asm.n_alive > 0 and a.lattice.csr.n_edges > 0
    assert np.all((a.homeo.rate >= 0) & (a.homeo.rate <= 1))


def test_gate_off_stores_everything_and_gate_on_saves_bytes():
    off, _ = run(Config(seed=2, gate=False, homeo=False, pred_mode="assembly", gate_tau=0.9))
    on, _ = run(Config(seed=2, gate=True, homeo=False, pred_mode="assembly", gate_tau=0.9))
    assert off.log.n_count_only == 0 and off.log.n_stored == 700
    assert on.log.bytes_written < off.log.bytes_written and on.log.n_count_only > 0


def test_online_mode_runs_and_learns():
    st, _ = run(Config(seed=3, consolidate="online", homeo=False, gate=False, consolidate_every=100), n=300)
    assert st.asm.n_alive > 0 and st.lattice.csr.n_edges > 0 and not st.hot
