import numpy as np
from neurostore.encoder import SparseEncoder
from neurostore.homeostasis import Homeostasis
from neurostore.predict import surprise
from sim.generator import World, WorldConfig


def _rates(homeo, seed=0, n=8000, last=2000):
    w = World(WorldConfig(zipf_s=1.5), seed); s = w.stream("test", n)
    enc = SparseEncoder(800, 1024, 32, 2, seed); h = Homeostasis(enc)
    cnt = np.zeros(1024); ks = set()
    for i in range(n):
        u, _ = enc.encode(s.event(i)); ks.add(len(u))
        if homeo: h.update(u)
        if i >= n - last: cnt[u] += 1
    return cnt / last, ks


def test_homeostasis_equalises_rates_under_skew_and_keeps_k_fixed():
    r_on, k_on = _rates(True); r_off, k_off = _rates(False)
    assert k_on == {32} and k_off == {32}
    cv = lambda r: r.std() / r.mean()
    assert abs(r_on.mean() - 32 / 1024) < 1e-9
    assert cv(r_on) < 0.7 * cv(r_off)
    assert r_on.max() < r_off.max()


def test_thresholds_bounded():
    enc = SparseEncoder(800, 1024, 32, 2, 0); h = Homeostasis(enc, clip=1.5)
    for _ in range(3000):
        h.update(np.arange(32))
    assert enc.theta.max() <= 1.5 + 1e-6 and enc.theta.min() >= -1.5 - 1e-6


def test_surprise_definition():
    fired = np.array([1, 2, 3, 4])
    S, per = surprise(fired, np.array([1, 2, 3, 4]), np.ones(4, np.float32), 10)
    assert S == 0 and np.all(per == 0)
    S, per = surprise(fired, np.array([1, 2, 7, 8]), np.ones(4, np.float32), 10)
    assert np.isclose(S, 4 / 8) and list(per) == [0, 0, 1, 1]       # 2 unexpected + 2 silent
    assert surprise(fired, np.zeros(0, np.int64), np.zeros(0, np.float32), 10)[0] == 1.0
