"""Crossbar emulation: each non-ideality has a measurable, individually-checkable effect."""
import math
from dataclasses import replace

import numpy as np
import pytest

from neurostore.common import flip_exact, make_rng, random_bipolar
from neurostore.primitives import CIMAssociativeMemory, CIMTimingModel, CrossbarArray, CrossbarConfig

IDEAL = CrossbarConfig(adc_bits=None)


def _xbar(w, **kw):
    return CrossbarArray(replace(IDEAL, **kw)).program(w)


# ----------------------------------------------------------------------------- ideal behaviour
@pytest.mark.parametrize("rows,cols,tile", [(64, 10, 64), (300, 20, 128), (1000, 7, 256), (5, 3, 256)])
def test_ideal_crossbar_equals_matrix_product_for_any_tiling(rng, rows, cols, tile):
    w = rng.standard_normal((rows, cols)).astype(np.float32)  # arbitrary real weights incl. negatives
    x = rng.standard_normal((9, rows)).astype(np.float32)
    got = _xbar(w, tile_rows=tile).matvec(x)
    assert got.shape == (9, cols) and np.allclose(got, x @ w, atol=2e-4 * max(1, np.abs(x @ w).max()))


def test_weight_scale_is_normalised_and_restored(rng):
    w = rng.standard_normal((100, 8)).astype(np.float32)
    x = rng.standard_normal((4, 100)).astype(np.float32)
    assert np.allclose(_xbar(5.0 * w).matvec(x), 5.0 * (x @ w), rtol=1e-3, atol=1e-3)


def test_differential_pair_stores_signs(rng):
    w = np.array([[1.0, -1.0], [-1.0, 1.0], [0.0, 0.5]], dtype=np.float32)
    assert np.allclose(_xbar(w, tile_rows=4).matvec(np.eye(3, dtype=np.float32)), w, atol=1e-6)


def test_input_validation(rng):
    xb = CrossbarArray(IDEAL)
    with pytest.raises(RuntimeError):
        xb.matvec(np.zeros((1, 4)))
    xb.program(np.zeros((8, 2)))
    with pytest.raises(ValueError):
        xb.matvec(np.zeros((1, 9)))
    with pytest.raises(ValueError):
        xb.program(np.zeros(5))


# ----------------------------------------------------------------------------- ADC
def test_adc_quantises_each_tile_to_its_step_and_clips_at_full_scale(rng):
    rows, bits = 64, 4
    w = random_bipolar(rng, rows, 40).astype(np.float32)
    x = random_bipolar(rng, 30, rows).astype(np.float32)
    ideal = x @ w
    out = _xbar(w, tile_rows=rows, adc_bits=bits, adc_fs_fraction=1.0).matvec(x)
    step = 2 * rows / 2**bits  # = 8
    assert np.allclose(out / step, np.rint(out / step))  # every output sits on the ADC grid
    inside = np.abs(ideal) < rows - step
    assert np.abs(out - ideal)[inside].max() <= step / 2 + 1e-4
    assert out.max() <= rows - step + 1e-4 and out.min() >= -rows - 1e-4  # saturates at the last code


def test_more_adc_bits_means_less_error(rng):
    w = random_bipolar(rng, 512, 50).astype(np.float32)
    x = random_bipolar(rng, 40, 512).astype(np.float32)
    ideal = x @ w
    rms = [np.sqrt(np.mean((_xbar(w, adc_bits=b, adc_fs_fraction=0.5).matvec(x) - ideal) ** 2)) for b in (3, 5, 8)]
    assert rms[0] > rms[1] > rms[2]


def test_per_tile_conversion_makes_error_grow_with_the_number_of_tiles(rng):
    """Each row-tile is digitised separately, so quantisation noise adds across tiles."""
    cols = 200
    errs = []
    for rows in (256, 1024):
        w = random_bipolar(rng, rows, cols).astype(np.float32)
        x = random_bipolar(rng, 60, rows).astype(np.float32)
        out = _xbar(w, tile_rows=256, adc_bits=5, adc_fs_fraction=1.0).matvec(x)
        errs.append(np.sqrt(np.mean((out - x @ w) ** 2)))
    assert errs[1] > 1.5 * errs[0]


# ----------------------------------------------------------------------------- read noise
def test_read_noise_scale_is_a_fraction_of_tile_full_scale():
    sigma, rows, cols = 0.02, 256, 4000
    xb = _xbar(np.zeros((rows, cols), dtype=np.float32), tile_rows=rows, read_noise=sigma)
    out = xb.matvec(np.ones((1, rows), dtype=np.float32))
    assert out.std() == pytest.approx(sigma * rows, rel=0.08)
    assert abs(out.mean()) < 0.5


def test_read_noise_is_independent_across_reads_but_reproducible_from_the_seed(rng):
    w = random_bipolar(rng, 256, 20).astype(np.float32)
    x = random_bipolar(rng, 3, 256).astype(np.float32)
    a, b = _xbar(w, read_noise=0.05, seed=1), _xbar(w, read_noise=0.05, seed=1)
    assert np.array_equal(a.matvec(x), b.matvec(x))
    assert not np.array_equal(a.matvec(x), a.matvec(x))  # a second read of the same array differs
    assert not np.array_equal(_xbar(w, read_noise=0.05, seed=2).matvec(x), b.matvec(x))


def test_more_tiles_average_the_noise_in_relative_terms(rng):
    """Same relative read noise: a long vector's signal grows ~linearly in D while noise grows ~sqrt(tiles)."""
    snr = []
    for rows in (256, 4096):
        w = np.ones((rows, 1), dtype=np.float32)
        x = np.ones((400, rows), dtype=np.float32)
        out = _xbar(w, read_noise=0.1).matvec(x)
        snr.append(out.mean() / out.std())
    assert snr[1] > 3 * snr[0]


# ----------------------------------------------------------------------------- static non-idealities
def test_programming_noise_has_the_expected_spread():
    rng = make_rng(0)
    # Weights kept well inside (-1, 1) with one anchor at 1.0 (fixes the scale), and g_min = 0.5, so neither
    # device of a pair reaches a physical conductance limit: the Gaussian programming error is untruncated.
    w = rng.uniform(-0.8, 0.8, (256, 200)).astype(np.float32)
    w[0, 0] = 1.0
    xb = _xbar(w, g_min=0.5, program_noise=0.05, tile_rows=256, seed=3)
    err = xb.G.reshape(256, 200) - w
    assert err.std() == pytest.approx(math.sqrt(2) * 0.05, rel=0.08)  # two independent devices per weight
    assert abs(err.mean()) < 0.003
    assert np.abs(_xbar(w, program_noise=0.0).G.reshape(256, 200) - w).max() < 1e-6


def test_programming_noise_is_one_sided_at_the_conductance_limits():
    """A device programmed to g_max cannot overshoot: clipping makes the error of +-1 weights smaller and biased."""
    rng = make_rng(1)
    w = random_bipolar(rng, 256, 200).astype(np.float32)
    err = _xbar(w, g_min=0.5, program_noise=0.05, tile_rows=256, seed=3).G.reshape(256, 200) - w
    assert err.std() < math.sqrt(2) * 0.05 * 0.95  # less than the unclipped two-device spread
    assert err[w > 0].mean() < -0.01 and err[w < 0].mean() > 0.01  # pulled towards the interior


def test_stuck_cells_hit_the_requested_fraction_of_devices():
    f = 0.1
    xb = _xbar(np.zeros((256, 400), dtype=np.float32), stuck_fraction=f, tile_rows=256, seed=5)
    changed = (np.abs(xb.G.reshape(256, 400)) > 0.5).mean()
    expected = f - 2 * (f / 2) ** 2  # either device of the pair stuck-high, minus the cancelling case
    assert changed == pytest.approx(expected, abs=0.01)


def test_uniform_drift_scales_the_output_and_global_compensation_cancels_it(rng):
    w = random_bipolar(rng, 128, 16).astype(np.float32)
    x = random_bipolar(rng, 5, 128).astype(np.float32)
    kw = dict(drift_nu=0.1, drift_time_ratio=1e4, tile_rows=128)
    f = (1e4) ** -0.1
    assert np.allclose(_xbar(w, **kw).matvec(x), f * (x @ w), rtol=1e-3, atol=1e-2)
    assert np.allclose(_xbar(w, drift_compensation=True, **kw).matvec(x), x @ w, rtol=1e-3, atol=1e-2)


def test_device_to_device_drift_spread_survives_compensation(rng):
    w = random_bipolar(rng, 256, 32).astype(np.float32)
    x = random_bipolar(rng, 20, 256).astype(np.float32)
    kw = dict(drift_nu=0.1, drift_nu_sigma=0.05, drift_time_ratio=1e5, drift_compensation=True)
    err = np.abs(_xbar(w, seed=1, **kw).matvec(x) - x @ w)
    assert err.mean() > 1.0  # a global gain cannot cancel per-device variation


# ----------------------------------------------------------------------------- associative search
def test_cim_search_finds_the_stored_key_and_degrades_gracefully(rng):
    keys, values = random_bipolar(rng, 200, 512), random_bipolar(rng, 200, 512)
    q = flip_exact(rng, keys, 150)  # 29% flips
    base = CIMAssociativeMemory(keys, values, IDEAL)
    assert (base.identify(q) == np.arange(200)).mean() > 0.99
    assert (base.retrieve(q) == values).all(axis=1).mean() > 0.97
    mild = CIMAssociativeMemory(keys, values, CrossbarConfig(adc_bits=8, read_noise=0.02, program_noise=0.05, stuck_fraction=0.01))
    assert (mild.identify(q) == np.arange(200)).mean() > 0.97
    wrecked = CIMAssociativeMemory(keys, values, replace(IDEAL, read_noise=2.0))
    assert (wrecked.identify(q) == np.arange(200)).mean() < 0.5


def test_ideal_cim_scores_are_cosine_similarities(rng):
    keys, values = random_bipolar(rng, 30, 256), random_bipolar(rng, 30, 256)
    q = flip_exact(rng, keys, 40)
    scores = CIMAssociativeMemory(keys, values, IDEAL).scores(q)
    assert np.allclose(scores, q.astype(np.float32) @ keys.T.astype(np.float32) / 256, atol=1e-5)
    assert scores.max() <= 1.0 + 1e-6


# ----------------------------------------------------------------------------- timing model
def test_timing_model_matches_hand_calculation():
    opt = CIMTimingModel.preset("optimistic").search_latency(1000, 4096)  # 16 x 4 = 64 tiles <= 1024 arrays
    assert (opt["tiles"], opt["waves"]) == (64, 1)
    assert opt["t_array_ns"] == 2 + 10 + 4 * 2  # dac + read + ceil(256/64) conversions
    assert opt["t_accum_ns"] == 4 * 0.5 and opt["t_wta_ns"] == 10 * 0.5 and opt["total_ns"] == 27
    con = CIMTimingModel.preset("conservative").search_latency(1000, 4096)  # only 16 arrays -> 4 waves
    assert con["waves"] == 4 and con["t_array_ns"] == 4 * (10 + 100 + 32 * 20) and con["total_ns"] == 3028
    assert con["total_ns"] > opt["total_ns"]


def test_analogue_time_is_size_independent_only_while_tiles_fit_the_array_budget():
    m = CIMTimingModel.preset("optimistic")  # 1024 arrays
    small, big = m.search_latency(256, 256), m.search_latency(256 * 16, 256 * 32)  # 1 tile vs 512 tiles
    assert small["waves"] == big["waves"] == 1 and small["t_array_ns"] == big["t_array_ns"]
    over = m.search_latency(256 * 64, 256 * 64)  # 4096 tiles -> 4 waves
    assert over["waves"] == 4 and over["t_array_ns"] == 4 * big["t_array_ns"] and "serialised" in over["regime"]
    lat = [m.search_latency(n, 4096)["total_ns"] for n in (10, 100, 1000, 10_000, 100_000, 1_000_000)]
    assert all(a <= b for a, b in zip(lat, lat[1:])) and lat[-1] > lat[0]


def test_unknown_preset_is_rejected():
    with pytest.raises(KeyError):
        CIMTimingModel.preset("magic")
