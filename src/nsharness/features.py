"""Featurizer: windows (n, T, C) -> fixed-length vectors shared by every backend."""
from __future__ import annotations
import numpy as np


class Featurizer:
    """kind="stats": per channel [last value, mean, least-squares slope, std] over the window (4C features, channel aligned).
    kind="flatten": the T*C raw values. Then optional per-feature standardisation, optional bounding (clip to [-clip, clip] z, optionally
    quantised to `levels` evenly spaced values: the per-feature level quantiser a VSA encoder applies) and optional PCA (fit on the memory only)."""

    def __init__(self, kind: str = "stats", standardize: bool = True, pca: int | None = None, clip: float | None = None, levels: int | None = None):
        assert kind in ("stats", "flatten")
        assert levels is None or clip is not None, "levels needs clip (the quantiser spans [-clip, clip])"
        self.kind, self.standardize, self.pca, self.clip, self.levels = kind, standardize, pca, clip, levels
        self.mu = self.sd = self.comp = None
        self.T = self.C = None

    def _raw(self, Z: np.ndarray) -> np.ndarray:
        n, T, C = Z.shape
        if self.kind == "flatten":
            return Z.reshape(n, T * C).astype(np.float32)
        if T < 2:
            raise ValueError("kind='stats' needs windows with T >= 2; use kind='flatten' for non-temporal data")
        t = np.arange(T, dtype=np.float32) - (T - 1) / 2.0
        slope = np.einsum("ntc,t->nc", Z, t) / float((t ** 2).sum())
        return np.concatenate([Z[:, -1, :], Z.mean(1), slope, Z.std(1)], axis=1).astype(np.float32)

    def _bound(self, F: np.ndarray) -> np.ndarray:
        if self.clip is None:
            return F
        c = float(self.clip)
        F = np.clip(F, -c, c)
        if self.levels:
            step = 2.0 * c / (int(self.levels) - 1)
            F = np.rint((F + c) / step) * step - c
        return F

    def fit(self, Z: np.ndarray) -> "Featurizer":
        self.T, self.C = Z.shape[1], Z.shape[2]
        F = self._raw(Z)
        if self.standardize:
            self.mu, self.sd = F.mean(0), np.maximum(F.std(0), 1e-6)
            F = (F - self.mu) / self.sd
        F = self._bound(F)
        if self.pca:
            Fc = F - F.mean(0)
            _, _, Vt = np.linalg.svd(Fc, full_matrices=False)
            self.comp = Vt[: self.pca].astype(np.float32)
            self.pca_mean = F.mean(0)
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        F = self._raw(Z)
        if self.standardize:
            F = (F - self.mu) / self.sd
        F = self._bound(F)
        if self.pca:
            F = (F - self.pca_mean) @ self.comp.T
        return F.astype(np.float32)

    def fit_transform(self, Z):
        return self.fit(Z).transform(Z)

    @property
    def n_features(self) -> int:
        if self.pca:
            return int(self.pca)
        return (4 * self.C) if self.kind == "stats" else self.T * self.C

    def feature_channel(self):
        """Channel index of every feature (None after PCA, which mixes channels)."""
        if self.pca:
            return None
        return np.tile(np.arange(self.C), 4) if self.kind == "stats" else np.tile(np.arange(self.C), self.T)

    def feature_mask(self, channel_mask: np.ndarray):
        """(n, C) observed-channel mask -> (n, d) observed-feature mask (None after PCA)."""
        ch = self.feature_channel()
        return None if ch is None else np.asarray(channel_mask, bool)[:, ch]

    def param_bytes(self) -> int:
        return int(sum(a.nbytes for a in (self.mu, self.sd, self.comp) if a is not None))
