"""Associative memories: Modern (softmax) Hopfield and the classical Hebbian matrix memory."""
from __future__ import annotations

import numpy as np

from ..common import GrowableRows


class ModernHopfieldMemory:
    """Modern continuous Hopfield network / dense associative memory.

    Retrieval is cosine similarity between the query and every stored key, a softmax with
    inverse temperature ``beta`` over those similarities, and a weighted sum of the stored
    values; the output is the sign of that sum (bipolar patterns). This is the update rule
    of Ramsauer et al. (2020) and of the reference harness this project started from.

    Two modes:

    * **hetero-associative** (``values`` given): key -> different value.
    * **auto-associative** (``values=None``): the values are the keys, retrieval is a pattern
      completion attractor.

    Retrieval can be iterated (``steps > 1``) in either mode, see :meth:`retrieve`.

    Keys are stored once; their norms are cached, so a query costs one ``(N, D)`` mat-vec.

    Note the identity this class makes explicit: with ``beta -> infinity`` the softmax becomes
    an argmax and the memory is *exact brute-force nearest-neighbour search*. Any
    robustness the memory shows is therefore bounded by (and must be compared with) that
    exact-search ceiling.
    """

    name = "hopfield"

    def __init__(self, dim: int, beta: float = 16.0, dtype: type = np.float32) -> None:
        self.dim = int(dim)
        self.beta = float(beta)
        self.dtype = np.dtype(dtype)
        self._keys = GrowableRows(self.dim, self.dtype)
        self._norms = GrowableRows(1, self.dtype)
        self._values: GrowableRows | None = None
        self.auto = True

    # ------------------------------------------------------------------ storage
    def __len__(self) -> int:
        return len(self._keys)

    def store(self, keys: np.ndarray, values: np.ndarray | None = None) -> None:
        """Replace the contents with ``keys`` (``(N, D)``) and optional ``values`` (``(N, Dv)``)."""
        keys = np.asarray(keys, dtype=self.dtype)
        if keys.ndim != 2 or keys.shape[1] != self.dim:
            raise ValueError(f"keys must be (N, {self.dim}), got {keys.shape}")
        self._keys = GrowableRows(self.dim, self.dtype, capacity=keys.shape[0])
        self._keys.extend(keys)
        self._norms = GrowableRows(1, self.dtype, capacity=keys.shape[0])
        self._norms.extend(np.linalg.norm(keys, axis=1, keepdims=True).astype(self.dtype))
        if values is None:
            self._values, self.auto = None, True
        else:
            values = np.asarray(values, dtype=self.dtype)
            if values.shape[0] != keys.shape[0]:
                raise ValueError("keys and values must have the same number of rows")
            self._values = GrowableRows(values.shape[1], self.dtype, capacity=values.shape[0])
            self._values.extend(values)
            self.auto = False

    def add(self, key: np.ndarray, value: np.ndarray | None = None) -> int:
        """Append one pattern (amortised O(D)); returns its index."""
        key = np.asarray(key, dtype=self.dtype)
        idx = self._keys.append(key)
        self._norms.append(np.linalg.norm(key))
        if value is not None:
            if self._values is None:
                if len(self._keys) > 1:
                    raise ValueError("cannot mix auto- and hetero-associative patterns")
                self._values = GrowableRows(len(value), self.dtype)
                self.auto = False
            self._values.append(np.asarray(value, dtype=self.dtype))
        elif self._values is not None:
            raise ValueError("a value is required for a hetero-associative memory")
        return idx

    def memory_bytes(self) -> int:
        """Allocated bytes of the stored matrices (keys, cached norms, values if separate)."""
        total = self._keys.nbytes + self._norms.nbytes
        if self._values is not None:
            total += self._values.nbytes
        return int(total)

    # ------------------------------------------------------------------ retrieval
    def similarities(self, queries: np.ndarray) -> np.ndarray:
        """Cosine similarity of every query to every stored key -> ``(Q, N)``."""
        q = np.atleast_2d(np.asarray(queries, dtype=self.dtype))
        qn = np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
        norms = self._norms.view[:, 0] + 1e-8
        return (q @ self._keys.view.T) / (qn * norms[None, :])

    def weights(self, sims: np.ndarray) -> np.ndarray:
        """Softmax(beta * sims) along the pattern axis (numerically stabilised)."""
        z = self.beta * (sims - sims.max(axis=1, keepdims=True))
        w = np.exp(z)
        w /= w.sum(axis=1, keepdims=True)
        return w

    def _value_matrix(self) -> np.ndarray:
        return self._keys.view if self._values is None else self._values.view

    def retrieve_index(self, queries: np.ndarray) -> np.ndarray:
        """Index of the most similar stored key (the ``beta -> inf`` limit), shape ``(Q,)``."""
        return self.similarities(queries).argmax(axis=1)

    def retrieve(self, queries: np.ndarray, steps: int = 1, chunk: int = 512) -> np.ndarray:
        """Softmax-weighted recall of the stored values, thresholded with ``sign``.

        ``steps=1`` is the single-shot readout of the original reference harness. With
        ``steps > 1`` the query is first *cleaned up by attractor dynamics in key space*
        (``xi <- sum_i softmax(beta cos(xi, k_i)) k_i``, ``steps - 1`` times) and the value is
        read out with the final weights. For an auto-associative memory the values are the
        keys, so this is the classical iterated pattern completion. Iteration extends the noise
        range in which a stored pattern is recovered exactly (a state the single step leaves as
        a blend is pulled into the basin of one pattern), at ``steps`` times the cost.

        Accepts one vector or a ``(Q, D)`` batch.
        """
        single = np.ndim(queries) == 1
        q = np.atleast_2d(np.asarray(queries, dtype=self.dtype))
        keys, vals = self._keys.view, self._value_matrix()
        out = np.empty((q.shape[0], vals.shape[1]), dtype=self.dtype)
        for s in range(0, q.shape[0], chunk):
            state = q[s : s + chunk]
            for _ in range(max(1, steps) - 1):
                state = self.weights(self.similarities(state)) @ keys
            out[s : s + chunk] = np.sign(self.weights(self.similarities(state)) @ vals)
        return out[0] if single else out


class LinearAssociativeMemory:
    """Classical correlation-matrix (Hebbian) hetero-associative memory ``W = V^T K / D``.

    The textbook "fixed D x D matrix" memory: constant size however many pairs are stored,
    at the price of crosstalk. For N random bipolar pairs the raw bit error rate of
    ``sign(W k)`` is ``Q(sqrt(D / (N - 1)))`` (see :func:`neurostore.theory.linear_memory_ber`).
    """

    name = "linear_matrix"

    def __init__(self, key_dim: int, value_dim: int | None = None, dtype: type = np.float32) -> None:
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim or key_dim)
        self.dtype = np.dtype(dtype)
        self.matrix = np.zeros((self.value_dim, self.key_dim), dtype=self.dtype)
        self.n_items = 0

    def store(self, keys: np.ndarray, values: np.ndarray) -> None:
        k = np.asarray(keys, dtype=self.dtype)
        v = np.asarray(values, dtype=self.dtype)
        self.matrix = (v.T @ k) / self.key_dim
        self.n_items = k.shape[0]

    def retrieve(self, queries: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(np.asarray(queries, dtype=self.dtype))
        return np.sign(q @ self.matrix.T)

    def memory_bytes(self) -> int:
        return int(self.matrix.nbytes)
