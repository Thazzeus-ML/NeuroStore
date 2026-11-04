"""Software emulation of analogue compute-in-memory (RRAM / FeFET crossbar) -- Section 5.

Two *separate* things are modelled here and must not be conflated:

* **Accuracy under non-idealities** (:class:`CrossbarArray`, :class:`CIMAssociativeMemory`):
  a numerical emulation of what an analogue array would compute given device noise, drift,
  stuck cells and finite ADC resolution. This is a simulation of *values*; it can be checked.
* **Latency** (:class:`CIMTimingModel`): a transparent *analytical model* with explicit
  device-timing assumptions. Software cannot measure the speed of hardware that is not there,
  so every latency produced by this model is a projection under stated parameters, never a
  measurement. The presets are illustrative placeholders to be replaced with numbers from a
  real process/array characterisation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..common import make_rng


@dataclass(frozen=True)
class CrossbarConfig:
    """Non-ideality and geometry parameters of one emulated crossbar.

    Conductances are normalised so ``g_max = 1``. A weight ``w`` in ``[-1, 1]`` (after
    scaling by ``max|W|``) is stored as a differential pair
    ``G+ = g_min + (1-g_min) max(w,0)``, ``G- = g_min + (1-g_min) max(-w,0)``; the bit-line
    current is proportional to ``sum_i V_i (G+_ij - G-_ij)`` (Kirchhoff's current law), in
    which the ``g_min`` offset cancels for ideal devices.

    Attributes
    ----------
    tile_rows, tile_cols:
        Physical array size. Matrices larger than a tile are split into tiles whose partial
        sums are digitised *separately* (one ADC conversion per tile column per read) and
        added digitally -- this is where ADC and read noise enter per tile.
    program_noise:
        Std-dev of the programming error of each device, as a fraction of ``g_max - g_min``.
    read_noise:
        Std-dev of additive bit-line noise (thermal/shot/RTN) per tile read, as a fraction of
        the tile's full-scale output (``tile_rows * input_fs``).
    drift_nu, drift_time_ratio, drift_nu_sigma:
        Power-law conductance drift ``G(t) = G0 (t/t0)^(-nu)`` with ``t/t0 = drift_time_ratio``
        and device-to-device spread ``drift_nu_sigma`` of the exponent.
    drift_compensation:
        Apply a global digital gain ``(t/t0)^nu`` after the ADC (cancels the mean decay but not
        the device-to-device spread).
    stuck_fraction:
        Fraction of devices stuck at ``g_min`` or ``g_max`` (half each), independent per device.
    adc_bits, adc_fs_fraction:
        ADC resolution (``None`` = ideal) and the saturation level as a fraction of the tile's
        full scale. Smaller fractions spend bits on the likely range but clip large outputs.
    dac_bits, input_fs:
        Input DAC resolution (``None`` = ideal) and full-scale input amplitude.
    """

    tile_rows: int = 256
    tile_cols: int = 256
    g_min: float = 0.02
    program_noise: float = 0.0
    read_noise: float = 0.0
    drift_nu: float = 0.0
    drift_time_ratio: float = 1.0
    drift_nu_sigma: float = 0.0
    drift_compensation: bool = False
    stuck_fraction: float = 0.0
    adc_bits: int | None = 8
    adc_fs_fraction: float = 1.0
    dac_bits: int | None = None
    input_fs: float = 1.0
    seed: int = 0


class CrossbarArray:
    """Emulated (tiled, differential-pair) crossbar computing ``y = x @ W``.

    ``program(W)`` takes ``W`` of shape ``(rows_total, cols)`` -- word lines carry the input
    vector, each bit line accumulates one output -- applies the *static* non-idealities once
    (programming noise, stuck cells, drift) and caches the effective differential
    conductance. ``matvec(x)`` applies the *per-read* ones (DAC, thermal noise, ADC).
    """

    def __init__(self, cfg: CrossbarConfig) -> None:
        self.cfg = cfg
        self.rng = make_rng(cfg.seed)
        self.G: np.ndarray | None = None
        self.rows_total = self.cols = 0
        self.w_scale = 1.0
        self.n_row_tiles = self.n_col_tiles = 0

    # ------------------------------------------------------------------ programming
    def program(self, weights: np.ndarray) -> "CrossbarArray":
        cfg, rng = self.cfg, self.rng
        w = np.asarray(weights, dtype=np.float32)
        if w.ndim != 2:
            raise ValueError("weights must be 2-D (rows_total, cols)")
        self.rows_total, self.cols = w.shape
        self.w_scale = float(np.abs(w).max()) or 1.0
        wn = w / np.float32(self.w_scale)
        g_min = np.float32(cfg.g_min)
        span = np.float32(1.0 - cfg.g_min)
        gp = g_min + span * np.clip(wn, 0, None)
        gn = g_min + span * np.clip(-wn, 0, None)

        if cfg.program_noise > 0:
            for g in (gp, gn):
                g += rng.standard_normal(g.shape, dtype=np.float32) * np.float32(cfg.program_noise) * span
                np.clip(g, 0.0, 1.0, out=g)
        if cfg.stuck_fraction > 0:
            for g in (gp, gn):
                stuck = rng.random(g.shape, dtype=np.float32) < cfg.stuck_fraction
                high = rng.random(g.shape, dtype=np.float32) < 0.5
                g[stuck & high] = 1.0
                g[stuck & ~high] = g_min
        if cfg.drift_nu > 0 or cfg.drift_nu_sigma > 0:
            ratio = max(float(cfg.drift_time_ratio), 1.0)
            for g in (gp, gn):
                nu = cfg.drift_nu + cfg.drift_nu_sigma * rng.standard_normal(g.shape, dtype=np.float32)
                np.clip(nu, 0.0, None, out=nu)
                g *= np.power(np.float32(ratio), -nu)

        g_eff = (gp - gn) / span
        tr = cfg.tile_rows
        self.n_row_tiles = math.ceil(self.rows_total / tr)
        self.n_col_tiles = math.ceil(self.cols / cfg.tile_cols)
        pad = self.n_row_tiles * tr - self.rows_total
        if pad:
            g_eff = np.pad(g_eff, ((0, pad), (0, 0)))
        self.G = np.ascontiguousarray(g_eff.reshape(self.n_row_tiles, tr, self.cols))
        return self

    # ------------------------------------------------------------------ reading
    @staticmethod
    def _quantize(x: np.ndarray, bits: int, fs: float) -> np.ndarray:
        step = 2.0 * fs / (2**bits - 1)
        return (np.rint(np.clip(x, -fs, fs) / step) * step).astype(np.float32)

    def matvec(self, x: np.ndarray, max_temp_bytes: int = 96 << 20) -> np.ndarray:
        """Analogue ``x @ W`` for a batch ``x`` of shape ``(Q, rows_total)`` -> ``(Q, cols)``."""
        if self.G is None:
            raise RuntimeError("program() the array first")
        cfg = self.cfg
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        if x.shape[1] != self.rows_total:
            raise ValueError(f"expected {self.rows_total} inputs, got {x.shape[1]}")
        if cfg.dac_bits:
            x = self._quantize(x, cfg.dac_bits, cfg.input_fs)
        tr, nrt = cfg.tile_rows, self.n_row_tiles
        pad = nrt * tr - self.rows_total
        if pad:
            x = np.pad(x, ((0, 0), (0, pad)))
        fs_tile = np.float32(tr * cfg.input_fs)
        per_query = max(1, nrt * self.cols * 4)
        chunk = max(1, max_temp_bytes // per_query)
        out = np.empty((x.shape[0], self.cols), dtype=np.float32)
        gain = np.float32(max(cfg.drift_time_ratio, 1.0) ** cfg.drift_nu) if cfg.drift_compensation else np.float32(1.0)
        for s in range(0, x.shape[0], chunk):
            xb = x[s : s + chunk]
            xt = xb.reshape(xb.shape[0], nrt, tr).transpose(1, 0, 2)  # (tiles, Q, rows)
            partial = np.matmul(xt, self.G)  # (tiles, Q, cols): one analogue read per tile
            if cfg.read_noise > 0:
                partial += self.rng.standard_normal(partial.shape, dtype=np.float32) * np.float32(cfg.read_noise) * fs_tile
            if cfg.adc_bits:
                n_lv = 2**cfg.adc_bits
                step = np.float32(2.0 * cfg.adc_fs_fraction * fs_tile / n_lv)
                partial = np.clip(np.rint(partial / step), -n_lv // 2, n_lv // 2 - 1) * step
            out[s : s + chunk] = partial.sum(axis=0) * np.float32(self.w_scale) * gain
        return out

    def memory_bytes(self) -> int:
        return int(self.G.nbytes) if self.G is not None else 0


class CIMAssociativeMemory:
    """Modern-Hopfield style key search whose similarity stage runs on an emulated crossbar.

    Keys (bipolar, ``(N, D)``) are programmed with ``D`` word lines and ``N`` bit lines, so one
    analogue read yields all ``N`` dot products at once. The digital periphery then picks the
    top-``k`` bit lines, applies the softmax over just those ``k`` scores and reads the ``k``
    stored values from conventional memory (a dense analogue read-out of a one-hot-like
    softmax vector has terrible signal-to-noise on a shared bit line, so the read-out stays
    digital).
    """

    def __init__(self, keys: np.ndarray, values: np.ndarray, cfg: CrossbarConfig, beta: float = 20.0, topk: int = 4) -> None:
        self.keys = np.asarray(keys)
        self.values = np.asarray(values)
        self.dim = self.keys.shape[1]
        self.beta, self.topk = float(beta), int(min(topk, self.keys.shape[0]))
        self.xbar = CrossbarArray(cfg).program(self.keys.T.astype(np.float32))

    def scores(self, queries: np.ndarray) -> np.ndarray:
        """Analogue similarity (cosine for bipolar vectors) of each query to every key."""
        return self.xbar.matvec(np.asarray(queries, dtype=np.float32)) / np.float32(self.dim)

    def identify(self, queries: np.ndarray) -> np.ndarray:
        return self.scores(queries).argmax(axis=1)

    def retrieve(self, queries: np.ndarray) -> np.ndarray:
        s = self.scores(queries)
        k = self.topk
        idx = np.argpartition(s, s.shape[1] - k, axis=1)[:, -k:]
        top = np.take_along_axis(s, idx, axis=1)
        w = np.exp(self.beta * (top - top.max(axis=1, keepdims=True)))
        w /= w.sum(axis=1, keepdims=True)
        vals = self.values[idx].astype(np.float32)  # (Q, k, Dv)
        return np.sign(np.einsum("qk,qkd->qd", w, vals))


@dataclass(frozen=True)
class CIMTimingModel:
    """Analytical latency model of a tiled crossbar similarity search (**not a measurement**).

    One search over ``N`` keys of ``D`` dimensions needs ``ceil(D/tile_rows) * ceil(N/tile_cols)``
    tile activations. With at least that many arrays running concurrently the analogue part is a
    single "wave" whose duration does not depend on ``N`` or ``D`` -- the O(1) claim -- plus a
    logarithmic digital reduction (partial-sum adder tree over row tiles, winner-take-all tree
    over keys). Beyond ``n_arrays`` tiles the waves serialise and latency grows linearly.

    All times are in nanoseconds and are placeholders for device characterisation data.
    """

    t_dac_ns: float
    t_read_ns: float
    t_adc_ns: float
    adc_per_array: int
    n_arrays: int
    t_accum_ns: float
    t_wta_ns: float
    label: str = "custom"

    PRESETS = {
        "optimistic": dict(t_dac_ns=2.0, t_read_ns=10.0, t_adc_ns=2.0, adc_per_array=64, n_arrays=1024, t_accum_ns=0.5, t_wta_ns=0.5),
        "conservative": dict(t_dac_ns=10.0, t_read_ns=100.0, t_adc_ns=20.0, adc_per_array=8, n_arrays=16, t_accum_ns=2.0, t_wta_ns=2.0),
    }

    @classmethod
    def preset(cls, name: str) -> "CIMTimingModel":
        return cls(label=name, **cls.PRESETS[name])

    def search_latency(self, n_keys: int, dim: int, tile_rows: int = 256, tile_cols: int = 256) -> dict[str, float | int | str]:
        tiles_r = math.ceil(dim / tile_rows)
        tiles_c = math.ceil(n_keys / tile_cols)
        n_tiles = tiles_r * tiles_c
        waves = math.ceil(n_tiles / self.n_arrays)
        conversions = math.ceil(min(tile_cols, n_keys) / self.adc_per_array)
        t_wave = self.t_dac_ns + self.t_read_ns + conversions * self.t_adc_ns
        t_array = waves * t_wave
        t_accum = math.ceil(math.log2(tiles_r)) * self.t_accum_ns if tiles_r > 1 else 0.0
        t_wta = math.ceil(math.log2(max(n_keys, 2))) * self.t_wta_ns
        return {
            "model": self.label,
            "tiles": n_tiles,
            "waves": waves,
            "t_array_ns": t_array,
            "t_accum_ns": t_accum,
            "t_wta_ns": t_wta,
            "total_ns": t_array + t_accum + t_wta,
            "regime": "single wave (size-independent analogue time)" if waves == 1 else f"{waves} serialised waves",
        }
