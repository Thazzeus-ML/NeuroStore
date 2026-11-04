"""Standardized corruption families, defined on z-scored windows Z (n, T, C) so that severity means the same thing in every domain.

severity s in [0, 1]; s = 0 is the identity. Every family returns (Z_corrupted, observed) where `observed` (n, C) is False
for channels a sensor-health layer would know are missing (only dropout); stuck sensors, drift, spikes, noise and
quantisation are SILENT corruptions (observed stays True).

| family    | meaning of severity s                                                          |
|-----------|---------------------------------------------------------------------------------|
| noise     | additive Gaussian, sigma = s (in z units)                                       |
| dropout   | ceil(s*C) random channels per window are lost (set to the memory mean = 0)      |
| stuck     | ceil(s*C) random channels flatline at their first value in the window (T >= 2)  |
| drift     | 30 % of channels get a linear ramp 0 -> +-2 s z over the window (T >= 2)         |
| spikes    | each sample is hit with probability 0.1 s by a +-5 z impulse                    |
| quantize  | ADC with 8 - round(6 s) bits over [-3, 3] z                                      |
| combined  | dropout(0.4 s) then noise(0.5 s) then drift(s)                                  |
"""
from __future__ import annotations
import math
import numpy as np

FAMILIES = ("noise", "dropout", "stuck", "drift", "spikes", "quantize", "combined")


def _pick_channels(n: int, C: int, m: int, rng) -> np.ndarray:
    """(n, C) boolean mask with exactly m True per row, uniformly random."""
    out = np.zeros((n, C), bool)
    if m > 0:
        order = np.argsort(rng.random((n, C)), axis=1)[:, :m]
        out[np.arange(n)[:, None], order] = True
    return out


def _noise(Z, s, rng):
    return Z + rng.normal(0.0, s, Z.shape).astype(np.float32), None


def _dropout(Z, s, rng):
    n, T, C = Z.shape
    drop = _pick_channels(n, C, int(math.ceil(s * C - 1e-9)), rng)
    return Z * (~drop)[:, None, :], ~drop


def _stuck(Z, s, rng):
    n, T, C = Z.shape
    st = _pick_channels(n, C, int(math.ceil(s * C - 1e-9)), rng)[:, None, :]
    return np.where(st, Z[:, :1, :], Z), None


def _drift(Z, s, rng):
    n, T, C = Z.shape
    ch = _pick_channels(n, C, int(round(0.3 * C)), rng)
    sign = rng.choice(np.array([-1.0, 1.0], np.float32), size=(n, 1, C))
    ramp = (np.arange(T, dtype=np.float32) / max(T - 1, 1))[None, :, None]
    return Z + ch[:, None, :] * sign * (2.0 * s) * ramp, None


def _spikes(Z, s, rng):
    hit = rng.random(Z.shape) < 0.1 * s
    sign = rng.choice(np.array([-1.0, 1.0], np.float32), size=Z.shape)
    return Z + hit * sign * 5.0, None


def _quantize(Z, s, rng):
    bits = max(2, 8 - int(round(6 * s)))
    step = 6.0 / (2 ** bits - 1)
    return (np.round((np.clip(Z, -3, 3) + 3.0) / step) * step - 3.0).astype(np.float32), None


def _combined(Z, s, rng):
    z1, obs = _dropout(Z, 0.4 * s, rng)
    z2, _ = _noise(z1, 0.5 * s, rng)
    z3, _ = _drift(z2, s, rng)
    return z3, obs


_FN = {"noise": _noise, "dropout": _dropout, "stuck": _stuck, "drift": _drift, "spikes": _spikes, "quantize": _quantize, "combined": _combined}


def corrupt(Z: np.ndarray, family: str, severity: float, seed) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic given (family, severity, seed). Never modifies Z. Returns (Z', observed (n, C))."""
    Z = np.asarray(Z, np.float32)
    n, T, C = Z.shape
    ones = np.ones((n, C), bool)
    if severity <= 0:
        return Z.copy(), ones
    if family not in _FN:
        raise KeyError(f"unknown corruption {family!r}; known: {FAMILIES}")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), sum(map(ord, family)), int(round(severity * 1e6))]))
    Zc, obs = _FN[family](Z, float(severity), rng)
    return np.asarray(Zc, np.float32), (ones if obs is None else obs)


def calibrate_severity(measure, target_ratio: float, lo: float = 0.0, hi: float = 1.0, grid: int = 20, bisect: int = 6):
    """Find the severity at which measure(severity) / measure(0) reaches target_ratio (measure: lower is better, e.g. RMSE).
    Scans a coarse grid, then bisects inside the first bracket. Returns (severity, achieved_ratio, reached)."""
    base = measure(0.0)
    prev_s, prev_r = lo, 1.0
    for s in np.linspace(lo, hi, grid + 1)[1:]:
        r = measure(float(s)) / base
        if r >= target_ratio:
            a, ra, b, rb = prev_s, prev_r, float(s), r
            for _ in range(bisect):
                m = 0.5 * (a + b)
                rm = measure(m) / base
                if rm >= target_ratio:
                    b, rb = m, rm
                else:
                    a, ra = m, rm
            return float(b), float(rb), True
        prev_s, prev_r = float(s), r
    return float(hi), float(prev_r), False
