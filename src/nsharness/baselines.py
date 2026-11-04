"""Reference predictors with the same fit/predict surface as Pipeline, for context (not part of the NeuroStore claim)."""
from __future__ import annotations
import numpy as np
from .core import Split
from .features import Featurizer


class ConstantBaseline:
    """Predict the mean stored target."""
    name = "const"

    def fit(self, memory: Split, seed: int = 0):
        self.c = float(np.mean(memory.y))
        return self

    def predict(self, Z, observed=None):
        return np.full(len(Z), self.c, np.float64)


class RidgeBaseline:
    """Closed-form ridge regression on the same featurised windows the backends see."""
    name = "ridge"

    def __init__(self, lam: float = 1.0, feat_cfg: dict | None = None):
        self.lam, self.feat_cfg = float(lam), dict(feat_cfg or {})

    def fit(self, memory: Split, seed: int = 0):
        self.feat = Featurizer(**self.feat_cfg).fit(memory.Z)
        F = self.feat.transform(memory.Z).astype(np.float64)
        y = np.asarray(memory.y, np.float64)
        self.ybar = y.mean()
        self.w = np.linalg.solve(F.T @ F + self.lam * np.eye(F.shape[1]), F.T @ (y - self.ybar))
        return self

    def predict(self, Z, observed=None):
        return self.feat.transform(Z).astype(np.float64) @ self.w + self.ybar
