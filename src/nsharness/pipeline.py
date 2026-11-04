"""Pipeline = Featurizer + one Backend + the stored targets. This is the unit the studies evaluate."""
from __future__ import annotations
import numpy as np
from .core import Split
from .features import Featurizer
from .backends import make_backend


class Pipeline:
    def __init__(self, backend: str, backend_cfg: dict | None = None, feat_cfg: dict | None = None, k: int = 10, use_mask: bool = False):
        self.backend_name, self.backend_cfg = backend, dict(backend_cfg or {})
        self.feat_cfg, self.k, self.use_mask = dict(feat_cfg or {}), int(k), bool(use_mask)
        self.feat = self.be = self.Y = None

    def fit(self, memory: Split, seed: int = 0) -> "Pipeline":
        self.feat = Featurizer(**self.feat_cfg).fit(memory.Z)
        self.be = make_backend(self.backend_name, **self.backend_cfg).fit(self.feat.transform(memory.Z), seed)
        self.Y = np.asarray(memory.y, np.float32)
        return self

    def features(self, Z):
        return self.feat.transform(Z)

    def _mask(self, observed):
        if observed is None or not self.use_mask or not self.be.supports_mask:
            return None
        return self.feat.feature_mask(observed)

    def neighbors(self, Z, observed=None, k=None):
        return self.be.search(self.features(Z), k or self.k, self._mask(observed))

    def predict(self, Z, observed=None):
        return self.be.readout(self.features(Z), self.Y, self.k, self._mask(observed))

    def familiarity(self, Z, observed=None):
        return self.be.familiarity(self.features(Z), self._mask(observed))

    def reconstruct(self, Z, observed=None):
        return self.be.reconstruct(self.features(Z), self.k, self._mask(observed))

    def memory_report(self) -> dict:
        rep = {f"{self.backend_name}.{k}": v for k, v in self.be.memory_report().items()}
        rep["labels_f32"] = self.Y.nbytes
        rep["featurizer_params"] = self.feat.param_bytes()
        return rep

    def deployed_bytes(self) -> int:
        return int(sum(v for k, v in self.memory_report().items() if ".alt:" not in k and ".cache:" not in k))
