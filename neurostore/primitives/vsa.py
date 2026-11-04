"""Vector Symbolic Architectures: MAP and HRR algebras plus a role-filler record store."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..common import make_rng, random_bipolar


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of ``a`` and every row of ``b``."""
    a = np.atleast_2d(a).astype(np.float32, copy=False)
    b = np.atleast_2d(b).astype(np.float32, copy=False)
    an = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    bn = np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
    return (a @ b.T) / (an * bn.T)


class _AlgebraBase:
    """Shared state: dimension, RNG and a fixed random permutation ``rho``.

    ``rho`` is a *random* index permutation, not a cyclic shift. For circular-convolution
    (HRR) binding a cyclic shift is itself a convolution with a delta vector, so it commutes
    with binding and cannot make a binding order-sensitive; a random permutation can. It
    also works for MAP. ``permute(a, k)`` applies ``rho`` k times.
    """

    def __init__(self, dim: int, seed: int | None = 0) -> None:
        self.dim = int(dim)
        self.rng = make_rng(seed)
        self._perm = self.rng.permutation(self.dim)
        self._perm_cache: dict[int, np.ndarray] = {1: self._perm}

    def _perm_power(self, times: int) -> np.ndarray:
        if times not in self._perm_cache:
            p = self._perm_power(times - 1)
            self._perm_cache[times] = p[self._perm]
        return self._perm_cache[times]

    def permute(self, a: np.ndarray, times: int = 1) -> np.ndarray:
        return a[..., self._perm_power(times)]

    def inverse_permute(self, a: np.ndarray, times: int = 1) -> np.ndarray:
        return a[..., np.argsort(self._perm_power(times))]

    def cosine(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return cosine_matrix(a, b)


class MAPAlgebra(_AlgebraBase):
    """MAP (Multiply-Add-Permute) over bipolar {-1, +1} hypervectors.

    * ``bind``/``unbind``: Hadamard product (self-inverse).
    * ``bundle``: element-wise sum, optionally thresholded with ``sign`` (ties broken
      uniformly at random so an even number of terms is not biased towards +1).
    * ``permute``: fixed random permutation, used to make binding order-sensitive
      (directed edges).
    """

    name = "map"

    def random_vectors(self, n: int) -> np.ndarray:
        return random_bipolar(self.rng, n, self.dim).astype(np.float32)

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    def unbind(self, trace: np.ndarray, a: np.ndarray) -> np.ndarray:
        return trace * a

    def binarize(self, x: np.ndarray) -> np.ndarray:
        out = np.sign(x).astype(np.float32)
        zeros = out == 0
        if zeros.any():
            out[zeros] = random_bipolar(self.rng, 1, int(zeros.sum()))[0]
        return out

    def bundle(self, vectors: Sequence[np.ndarray] | np.ndarray, binarize: bool = True) -> np.ndarray:
        s = np.sum(np.asarray(vectors, dtype=np.float32), axis=0)
        return self.binarize(s) if binarize else s


class HRRAlgebra(_AlgebraBase):
    """Holographic Reduced Representations: circular convolution binding (Plate, 1995).

    Vectors are *unitary* (unit-magnitude Fourier spectrum, random phase, real-valued), for
    which the involution ``a*[i] = a[-i mod D]`` is the exact inverse. Plain Gaussian HRR
    vectors only have an approximate inverse and lose a constant factor of signal per bind.
    """

    name = "hrr"

    def random_vectors(self, n: int) -> np.ndarray:
        nfreq = self.dim // 2 + 1
        spec = np.exp(1j * self.rng.uniform(-np.pi, np.pi, size=(n, nfreq)))
        spec[:, 0] = self.rng.choice([-1.0, 1.0], size=n)  # DC bin must be real
        if self.dim % 2 == 0:
            spec[:, -1] = self.rng.choice([-1.0, 1.0], size=n)  # Nyquist bin must be real
        return np.fft.irfft(spec, n=self.dim, axis=-1).astype(np.float32)

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=self.dim, axis=-1).astype(np.float32)

    def unbind(self, trace: np.ndarray, a: np.ndarray) -> np.ndarray:
        return np.fft.irfft(np.fft.rfft(trace) * np.conj(np.fft.rfft(a)), n=self.dim, axis=-1).astype(np.float32)

    def bundle(self, vectors: Sequence[np.ndarray] | np.ndarray, binarize: bool = False) -> np.ndarray:
        return np.sum(np.asarray(vectors, dtype=np.float32), axis=0)


class VSARecordStore:
    """Role-filler record memory for entity resolution / record completion (Workload B).

    Every record is the bundle (superposition) of ``role_f (*) filler_{f, value}`` over its
    fields, stored as one hypervector. A query is a *partial cue*: the same bundle built from
    only the ``K`` known fields. Two stages, because a field that is **not in the cue carries
    no information about itself** (unbinding a missing role from the cue alone yields
    chance-level similarity -- the defect in the original harness):

    1. :meth:`identify` -- associative lookup: the stored record most similar to the cue.
    2. :meth:`reconstruct` -- unbind the target role from that *stored* record and clean up
       against the item memory of that field's fillers.

    Cost note: stage 1 is a dense similarity scan, ``O(M * D)`` for ``M`` records; it is
    "O(1) query steps" only in the sense of a fixed number of vector operations, and
    constant-time only if that scan is done by parallel analogue hardware (see
    :mod:`neurostore.primitives.cim`).
    """

    name = "vsa_map"

    def __init__(
        self,
        vocab_sizes: Sequence[int],
        dim: int = 4096,
        seed: int | None = 0,
        binarize_cue: bool = False,
        binarize_records: bool = True,
    ) -> None:
        self.vocab_sizes = np.asarray(vocab_sizes, dtype=np.int64)
        self.n_fields = len(self.vocab_sizes)
        self.dim = int(dim)
        self.binarize_cue = binarize_cue
        self.binarize_records = binarize_records
        rng = make_rng(seed)
        self.roles = random_bipolar(rng, self.n_fields, self.dim).astype(np.float32)
        self.offsets = np.concatenate(([0], np.cumsum(self.vocab_sizes)[:-1])).astype(np.int64)
        self.codebook = random_bipolar(rng, int(self.vocab_sizes.sum()), self.dim).astype(np.float32)
        self._alg = MAPAlgebra(self.dim, seed=int(rng.integers(0, 2**31 - 1)))
        self.records = np.empty((0, self.dim), dtype=np.float32)
        self._record_norms = np.empty(0, dtype=np.float32)

    # ------------------------------------------------------------------ encoding
    def filler(self, field: np.ndarray | int, value: np.ndarray | int) -> np.ndarray:
        """Filler hypervector(s) for (field, value) pair(s)."""
        return self.codebook[self.offsets[np.asarray(field)] + np.asarray(value)]

    def _terms(self, fields: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Sum over the given fields of ``role * filler`` -> ``(Q, D)``."""
        fields = np.atleast_2d(fields)
        values = np.atleast_2d(values)
        acc = np.zeros((fields.shape[0], self.dim), dtype=np.float32)
        for j in range(fields.shape[1]):
            f = fields[:, j]
            acc += self.roles[f] * self.codebook[self.offsets[f] + values[:, j]]
        return acc

    def add_records(self, values: np.ndarray, chunk: int = 2048) -> None:
        """Append records given as an ``(M, n_fields)`` integer array of value indices."""
        values = np.asarray(values, dtype=np.int64)
        new = []
        for s in range(0, values.shape[0], chunk):
            blk = values[s : s + chunk]
            fields = np.broadcast_to(np.arange(self.n_fields), blk.shape)
            acc = self._terms(fields, blk)
            new.append(self._alg.binarize(acc) if self.binarize_records else acc)
        self.records = np.concatenate([self.records] + new, axis=0)
        self._record_norms = np.linalg.norm(self.records, axis=1).astype(np.float32) + 1e-8

    def encode_cue(self, fields: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Hypervector for a partial cue (``(Q, K)`` field ids and value ids)."""
        acc = self._terms(fields, values)
        return self._alg.binarize(acc) if self.binarize_cue else acc

    # ------------------------------------------------------------------ queries
    def identify(self, fields: np.ndarray, values: np.ndarray, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        """Best-matching stored record for each cue -> ``(ids, cosine_scores)``."""
        cue = self.encode_cue(fields, values)
        cue_norm = np.linalg.norm(cue, axis=1, keepdims=True) + 1e-8
        ids = np.empty(cue.shape[0], dtype=np.int64)
        scores = np.empty(cue.shape[0], dtype=np.float32)
        for s in range(0, cue.shape[0], chunk):
            sims = (cue[s : s + chunk] @ self.records.T) / (cue_norm[s : s + chunk] * self._record_norms[None, :])
            j = sims.argmax(axis=1)
            ids[s : s + chunk] = j
            scores[s : s + chunk] = sims[np.arange(sims.shape[0]), j]
        return ids, scores

    def unbind_field(self, record_ids: np.ndarray, target_fields: np.ndarray | int) -> np.ndarray:
        """Unbind the target role from stored records -> noisy filler estimates ``(Q, D)``."""
        return self.records[record_ids] * self.roles[np.asarray(target_fields)]

    def cleanup_field(self, unbound: np.ndarray, target_fields: np.ndarray | int) -> tuple[np.ndarray, np.ndarray]:
        """Nearest filler in the target field's item memory -> ``(value_ids, cosine)``."""
        tf = np.broadcast_to(np.asarray(target_fields), (unbound.shape[0],))
        un = np.linalg.norm(unbound, axis=1) + 1e-8
        values = np.empty(unbound.shape[0], dtype=np.int64)
        scores = np.empty(unbound.shape[0], dtype=np.float32)
        root_d = np.float32(np.sqrt(self.dim))
        for f in np.unique(tf):
            sel = np.flatnonzero(tf == f)
            lo = int(self.offsets[f])
            book = self.codebook[lo : lo + int(self.vocab_sizes[f])]
            sims = (unbound[sel] @ book.T) / (un[sel, None] * root_d)
            j = sims.argmax(axis=1)
            values[sel] = j
            scores[sel] = sims[np.arange(sel.size), j]
        return values, scores

    def reconstruct(self, fields: np.ndarray, values: np.ndarray, target_fields: np.ndarray | int) -> dict[str, np.ndarray]:
        """Full two-stage query: identify the record, then recover ``target_fields``."""
        ids, id_scores = self.identify(fields, values)
        unbound = self.unbind_field(ids, target_fields)
        vals, val_scores = self.cleanup_field(unbound, target_fields)
        return {"ids": ids, "id_scores": id_scores, "values": vals, "value_scores": val_scores, "unbound": unbound}

    # ------------------------------------------------------------------ memory
    def memory_bytes(self) -> int:
        """Allocated bytes: stored records plus the item memories needed to decode them."""
        return int(self.records.nbytes + self._record_norms.nbytes + self.roles.nbytes + self.codebook.nbytes)

    def record_bytes(self) -> int:
        """Bytes of the stored record hypervectors only (as allocated, float32)."""
        return int(self.records.nbytes)

    def ideal_packed_bytes(self) -> int:
        """Bytes if every bipolar vector were bit-packed (1 bit per dimension)."""
        n_vec = self.records.shape[0] + self.roles.shape[0] + self.codebook.shape[0]
        return int(n_vec * ((self.dim + 7) // 8))
