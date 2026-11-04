"""FlyHash: fruit-fly olfactory-inspired expansive sparse coding (Dasgupta et al., 2017)."""
from __future__ import annotations

import numpy as np

from ..common import make_rng, pack_bits


class FlyHashEncoder:
    """Dense float vector -> very sparse binary code.

    1. **Expand**: a fixed random *sparse binary* projection maps ``in_dim`` inputs to
       ``out_dim >> in_dim`` units. By default every output unit samples a fixed number of
       inputs (``fixed_fan_in=True``, as the Kenyon cells do); ``fixed_fan_in=False`` gives
       the i.i.d. Bernoulli(``density``) matrix used by the original reference harness.
    2. **Sparsify**: k-winners-take-all keeps the ``k`` largest responses -> a code with
       exactly ``k`` active bits out of ``out_dim``.

    Similarity between two codes is the number of shared active bits (``overlap``); for
    equal-``k`` codes ``hamming = 2 * (k - overlap)`` so both rank neighbours identically.

    ``fit_center(X)`` stores the data mean, which is subtracted before projection. Real
    embedding models are anisotropic, so centering matters there; synthetic zero-mean data
    does not need it.
    """

    name = "flyhash"

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 2048,
        k: int = 64,
        density: float = 0.1,
        seed: int | None = 0,
        fixed_fan_in: bool = True,
    ) -> None:
        if not 0 < k <= out_dim:
            raise ValueError("need 0 < k <= out_dim")
        self.in_dim, self.out_dim, self.k = int(in_dim), int(out_dim), int(k)
        self.density = float(density)
        rng = make_rng(seed)
        if fixed_fan_in:
            fan_in = max(1, int(round(density * in_dim)))
            chosen = np.argsort(rng.random((out_dim, in_dim)), axis=1)[:, :fan_in]
            proj = np.zeros((in_dim, out_dim), dtype=np.float32)
            proj[chosen, np.arange(out_dim)[:, None]] = 1.0
        else:
            proj = (rng.random((in_dim, out_dim)) < density).astype(np.float32)
        self.projection = proj
        self.mean: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit_center(self, data: np.ndarray) -> "FlyHashEncoder":
        self.mean = np.asarray(data, dtype=np.float32).mean(axis=0)
        return self

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        if self.mean is not None:
            x = x - self.mean
        return x @ self.projection

    def encode_indices(self, x: np.ndarray) -> np.ndarray:
        """Indices of the ``k`` active units per input, sorted -> ``(B, k)`` integer array."""
        proj = self.project(x)
        idx = np.argpartition(proj, self.out_dim - self.k, axis=1)[:, self.out_dim - self.k :]
        idx.sort(axis=1)
        return idx.astype(np.uint16 if self.out_dim <= 65535 else np.uint32)

    def encode_dense(self, x: np.ndarray) -> np.ndarray:
        """``(B, out_dim)`` uint8 array of {0,1} with exactly ``k`` ones per row."""
        idx = self.encode_indices(x)
        bits = np.zeros((idx.shape[0], self.out_dim), dtype=np.uint8)
        bits[np.arange(idx.shape[0])[:, None], idx] = 1
        return bits

    def encode(self, x: np.ndarray, packed: bool = True) -> np.ndarray:
        """Binary codes; ``packed=True`` -> uint64 words ``(B, ceil(out_dim/64))``."""
        single = np.ndim(x) == 1
        bits = self.encode_dense(x)
        out = pack_bits(bits) if packed else bits
        return out[0] if single else out

    @property
    def code_bytes(self) -> int:
        """Bytes of one bit-packed code."""
        return ((self.out_dim + 63) // 64) * 8

    def memory_bytes(self) -> int:
        """Encoder state (projection matrix) in bytes."""
        extra = self.mean.nbytes if self.mean is not None else 0
        return int(self.projection.nbytes + extra)
