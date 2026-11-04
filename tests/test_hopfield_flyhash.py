"""Modern Hopfield memory, the linear associative matrix and the FlyHash encoder."""
import numpy as np
import pytest

from neurostore.common import flip_exact, make_rng, popcount_rows, random_bipolar
from neurostore.primitives import FlyHashEncoder, LinearAssociativeMemory, ModernHopfieldMemory


class ReferenceHopfield:
    """Verbatim port of the ContinuousHopfieldMemory in the original evaluation spec."""

    def __init__(self, dim, beta=16.0):
        self.dim, self.beta = dim, beta

    def store(self, keys, values):
        self.keys = keys.astype(np.float32)
        self.values = values.astype(np.float32)

    def retrieve(self, query):
        q_norm = query / (np.linalg.norm(query) + 1e-8)
        k_norm = self.keys / (np.linalg.norm(self.keys, axis=1, keepdims=True) + 1e-8)
        sims = np.dot(k_norm, q_norm)
        weights = np.exp(self.beta * (sims - np.max(sims)))
        weights /= np.sum(weights)
        return np.sign(np.dot(weights, self.values))


# ----------------------------------------------------------------------------- Hopfield
@pytest.fixture
def memory(rng):
    keys, values = random_bipolar(rng, 60, 256), random_bipolar(rng, 60, 256)
    m = ModernHopfieldMemory(256, beta=20.0)
    m.store(keys, values)
    return m, keys, values


def test_matches_the_reference_implementation(rng):
    keys, values = random_bipolar(rng, 40, 256), random_bipolar(rng, 40, 256)
    ref, ours = ReferenceHopfield(256, 20.0), ModernHopfieldMemory(256, 20.0)
    ref.store(keys, values)
    ours.store(keys, values)
    for noise in (0, 40, 80, 100):
        q = flip_exact(rng, keys, noise)
        got = ours.retrieve(q)
        for i in range(len(q)):
            assert np.array_equal(got[i], ref.retrieve(q[i].astype(np.float32))), (noise, i)


def test_clean_recall_is_exact_for_hetero_and_auto(memory, rng):
    m, keys, values = memory
    assert (m.retrieve(keys) == values).all()
    auto = ModernHopfieldMemory(256, 20.0)
    auto.store(keys)
    assert (auto.retrieve(keys) == keys).all()


def test_single_vector_and_batch_agree(memory, rng):
    m, keys, _ = memory
    q = flip_exact(rng, keys[:5], 60)
    batch = m.retrieve(q)
    for i in range(5):
        assert np.array_equal(m.retrieve(q[i]), batch[i])
    assert m.retrieve(q[0]).shape == (256,)


def test_weights_are_a_stable_softmax_even_for_huge_beta(rng):
    keys = random_bipolar(rng, 20, 128)
    m = ModernHopfieldMemory(128, beta=1e4)
    m.store(keys, keys)
    w = m.weights(m.similarities(flip_exact(rng, keys, 30)))
    assert np.isfinite(w).all() and np.allclose(w.sum(axis=1), 1, atol=1e-5)


def test_identification_is_nearest_key_and_robust_to_noise(memory, rng):
    m, keys, _ = memory
    q = flip_exact(rng, keys, 80)  # 31% flips, far inside the tolerance of D=256, N=60
    assert (m.retrieve_index(q) == np.arange(60)).mean() > 0.99


def test_low_beta_blends_neighbours_and_breaks_exact_value_recovery(rng):
    keys, values = random_bipolar(rng, 60, 256), random_bipolar(rng, 60, 256)
    soft, sharp = ModernHopfieldMemory(256, 1.0), ModernHopfieldMemory(256, 64.0)
    soft.store(keys, values)
    sharp.store(keys, values)
    q = flip_exact(rng, keys, 60)
    assert (soft.retrieve(q) == values).all(axis=1).mean() < 0.2
    assert (sharp.retrieve(q) == values).all(axis=1).mean() > 0.95


def test_beta_infinity_limit_is_exact_nearest_neighbour_search(rng):
    keys, values = random_bipolar(rng, 80, 128), random_bipolar(rng, 80, 128)
    m = ModernHopfieldMemory(128, beta=500.0)
    m.store(keys, values)
    q = flip_exact(rng, keys, 40)
    nn = ((q[:, None, :] != keys[None, :, :]).sum(axis=2)).argmin(axis=1)
    assert np.array_equal(m.retrieve_index(q), nn)
    assert np.array_equal(m.retrieve(q), values[nn].astype(np.float32))


def test_add_appends_and_is_retrievable(memory, rng):
    m, keys, values = memory
    k, v = random_bipolar(rng, 1, 256)[0], random_bipolar(rng, 1, 256)[0]
    idx = m.add(k, v)
    assert idx == 60 and len(m) == 61
    assert np.array_equal(m.retrieve(k), v)
    assert (m.retrieve(keys) == values).all()  # old content untouched


def test_memory_bytes_accounts_keys_values_and_norms(memory):
    m, _, _ = memory
    assert m.memory_bytes() == 60 * 256 * 4 * 2 + 60 * 4
    auto = ModernHopfieldMemory(256)
    auto.store(random_bipolar(make_rng(0), 60, 256))
    assert auto.memory_bytes() == 60 * 256 * 4 + 60 * 4  # values are the keys: stored once


def test_input_validation(rng):
    m = ModernHopfieldMemory(64)
    with pytest.raises(ValueError):
        m.store(random_bipolar(rng, 5, 32))
    with pytest.raises(ValueError):
        m.store(random_bipolar(rng, 5, 64), random_bipolar(rng, 4, 64))
    m.store(random_bipolar(rng, 5, 64), random_bipolar(rng, 5, 64))
    assert m.retrieve(random_bipolar(rng, 1, 64), steps=3).shape == (1, 64)  # iteration works for hetero memories too


def test_iterated_attractor_dynamics_complete_a_noisy_pattern(rng):
    """At beta=8 and 39% flips a single softmax step leaves a blend (0% exact); a few attractor steps recover it."""
    keys = np.tile(random_bipolar(rng, 30, 256), (10, 1))
    m = ModernHopfieldMemory(256, beta=8.0)
    m.store(keys[:30])
    q = flip_exact(rng, keys, 100)
    one = (m.retrieve(q, steps=1) == keys).all(axis=1).mean()
    five = (m.retrieve(q, steps=5) == keys).all(axis=1).mean()
    assert one < 0.3 and five > 0.85


def test_iterated_retrieval_in_hetero_mode_cleans_the_key_then_reads_the_value(rng):
    keys, values = random_bipolar(rng, 30, 256), random_bipolar(rng, 30, 256)
    m = ModernHopfieldMemory(256, beta=8.0)
    m.store(keys, values)
    q = flip_exact(rng, np.tile(keys, (10, 1)), 100)
    truth = np.tile(values, (10, 1))
    assert (m.retrieve(q, steps=1) == truth).all(axis=1).mean() < 0.3
    assert (m.retrieve(q, steps=5) == truth).all(axis=1).mean() > 0.85
    assert np.array_equal(m.retrieve(q[:3], steps=1), m.retrieve(q[:3]))  # steps=1 is the reference single shot


def test_iteration_is_a_noop_for_clean_keys(memory):
    m, keys, values = memory
    assert (m.retrieve(keys, steps=4) == values).all()


# ----------------------------------------------------------------------------- linear matrix memory
def test_linear_memory_recalls_a_few_pairs_and_degrades_with_load(rng):
    def ber(n):
        k, v = random_bipolar(rng, n, 256), random_bipolar(rng, n, 256)
        mem = LinearAssociativeMemory(256)
        mem.store(k, v)
        return float((mem.retrieve(k) != v).mean())

    assert ber(5) < 1e-3 < ber(120)
    assert LinearAssociativeMemory(256).memory_bytes() == 256 * 256 * 4  # constant, independent of N


# ----------------------------------------------------------------------------- FlyHash
def test_every_code_has_exactly_k_active_bits_and_is_deterministic(rng):
    x = rng.standard_normal((50, 96)).astype(np.float32)
    a, b = FlyHashEncoder(96, 1024, 40, seed=5), FlyHashEncoder(96, 1024, 40, seed=5)
    dense = a.encode_dense(x)
    assert (dense.sum(axis=1) == 40).all() and np.array_equal(dense, b.encode_dense(x))
    assert not np.array_equal(dense, FlyHashEncoder(96, 1024, 40, seed=6).encode_dense(x))


def test_projection_is_binary_with_fixed_fan_in():
    enc = FlyHashEncoder(100, 512, 16, density=0.1, seed=1)
    assert set(np.unique(enc.projection)) == {0.0, 1.0}
    assert (enc.projection.sum(axis=0) == 10).all()  # every output unit samples exactly 10 inputs
    bern = FlyHashEncoder(100, 512, 16, density=0.1, seed=1, fixed_fan_in=False)  # reference-harness style
    assert abs(bern.projection.mean() - 0.1) < 0.01 and len(set(bern.projection.sum(axis=0))) > 3


def test_packed_and_dense_codes_agree_and_single_vector_shape(rng):
    from neurostore.common import unpack_bits

    enc = FlyHashEncoder(64, 1000, 30, seed=2)
    x = rng.standard_normal((6, 64)).astype(np.float32)
    assert np.array_equal(unpack_bits(enc.encode(x), 1000), enc.encode_dense(x))
    assert enc.encode(x[0]).shape == (enc.code_bytes // 8,)
    assert enc.encode_indices(x).shape == (6, 30)


def test_codes_preserve_locality(rng):
    """Shared active bits fall monotonically as paraphrases drift away; unrelated inputs share ~k^2/m."""
    enc = FlyHashEncoder(128, 2048, 64, seed=3)
    x = rng.standard_normal((300, 128)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    base = enc.encode(x)
    overlaps = {}
    for cos in (0.99, 0.9, 0.7, 0.5):
        u = rng.standard_normal(x.shape).astype(np.float32)
        u -= (u * x).sum(axis=1, keepdims=True) * x
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        y = cos * x + np.sqrt(1 - cos**2) * u
        overlaps[cos] = popcount_rows(base & enc.encode(y)).mean()
    chance = popcount_rows(base & np.roll(base, 1, axis=0)).mean()
    assert overlaps[0.99] > overlaps[0.9] > overlaps[0.7] > overlaps[0.5] > chance
    assert chance == pytest.approx(64**2 / 2048, rel=0.25)
    assert overlaps[0.9] > 8 * chance


def test_codes_are_invariant_to_positive_scaling_and_centering_changes_them(rng):
    x = rng.standard_normal((20, 64)).astype(np.float32)
    enc = FlyHashEncoder(64, 512, 20, seed=4)
    assert np.array_equal(enc.encode_dense(x), enc.encode_dense(3.7 * x))  # k-WTA depends only on ranking
    before = enc.encode_dense(x)
    enc.fit_center(x + 5.0)
    assert not np.array_equal(before, enc.encode_dense(x))


def test_rejects_invalid_sparsity():
    with pytest.raises(ValueError):
        FlyHashEncoder(10, 100, 0)
    with pytest.raises(ValueError):
        FlyHashEncoder(10, 100, 101)
