"""Holographic knowledge-graph memory: multi-hop traversal by vector algebra (Workload D)."""
from __future__ import annotations

import numpy as np

from ..common import make_rng
from .vsa import HRRAlgebra, MAPAlgebra


class HolographicGraphMemory:
    """All triples ``(subject, relation, object)`` superposed in ONE ``D``-dimensional vector.

    Trace of a triple (direction-sensitive thanks to a fixed random permutation ``rho``)::

        t = s (*) rho(r) (*) rho^2(o)          M = sum_t t

    A hop answers ``(s, r, ?)``: unbind ``s (*) rho(r)`` from ``M``, undo ``rho^2`` and clean
    up against the entity item memory (top-1 cosine). A multi-hop walk feeds the cleaned-up
    entity into the next query -- no pointers, no adjacency lists.

    ``(*)`` is Hadamard product for ``algebra='map'`` (bipolar vectors, integer-valued trace)
    or circular convolution for ``algebra='hrr'`` (unitary vectors, evaluated in the
    frequency domain).

    After unbinding, the target is one signal term among ``T`` superposed terms, so its
    cosine is ``~1/sqrt(T)`` against an impostor noise floor of ``~1/sqrt(D)``: the memory
    breaks down ("ghost nodes", i.e. wrong top-1 entities) when ``T`` approaches
    ``D / (2 ln E)``. :func:`neurostore.theory.holographic_capacity` gives the analytic bound.

    The item memory (entity/relation codebooks) is part of the system: it is what makes the
    cleanup possible and it dominates the footprint -- :meth:`memory_bytes` reports everything allocated.
    """

    name = "holographic_graph"

    def __init__(self, n_entities: int, n_relations: int, dim: int = 4096, algebra: str = "map", seed: int | None = 0) -> None:
        if algebra not in ("map", "hrr"):
            raise ValueError("algebra must be 'map' or 'hrr'")
        self.n_entities, self.n_relations = int(n_entities), int(n_relations)
        self.dim, self.algebra = int(dim), algebra
        alg = MAPAlgebra(dim, seed) if algebra == "map" else HRRAlgebra(dim, seed)
        self.entities = alg.random_vectors(n_entities)
        self.relations = alg.random_vectors(n_relations)
        self._alg = alg
        self._rel_p = alg.permute(self.relations, 1)  # rho(r)
        self._ent_p2 = alg.permute(self.entities, 2)  # rho^2(o)
        self._ent_norms = np.linalg.norm(self.entities, axis=1).astype(np.float32)
        self.memory = np.zeros(self.dim, dtype=np.float32)
        self.n_triples = 0
        if algebra == "hrr":
            self._f_ent = np.fft.rfft(self.entities, axis=-1)
            self._f_rel_p = np.fft.rfft(self._rel_p, axis=-1)
            self._f_ent_p2 = np.fft.rfft(self._ent_p2, axis=-1)
            self._f_mem = np.zeros(self.dim // 2 + 1, dtype=np.complex128)

    # ------------------------------------------------------------------ writing
    def add_triples(self, triples: np.ndarray, chunk: int = 1024) -> None:
        """Superpose triples given as an ``(T, 3)`` integer array of (subject, relation, object)."""
        triples = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
        for s0 in range(0, triples.shape[0], chunk):
            s, r, o = triples[s0 : s0 + chunk].T
            if self.algebra == "map":
                self.memory += (self.entities[s] * self._rel_p[r] * self._ent_p2[o]).sum(axis=0)
            else:
                self._f_mem += (self._f_ent[s] * self._f_rel_p[r] * self._f_ent_p2[o]).sum(axis=0)
        if self.algebra == "hrr":
            self.memory = np.fft.irfft(self._f_mem, n=self.dim).astype(np.float32)
        self.n_triples += triples.shape[0]

    # ------------------------------------------------------------------ reading
    def query_vectors(self, subjects: np.ndarray, relations: np.ndarray) -> np.ndarray:
        """Noisy estimate of the object vector for each ``(s, r)`` -> ``(B, D)``."""
        s = np.atleast_1d(subjects)
        r = np.atleast_1d(relations)
        if self.algebra == "map":
            u = self.memory[None, :] * self.entities[s] * self._rel_p[r]
        else:
            spec = self._f_mem[None, :] * np.conj(self._f_ent[s]) * np.conj(self._f_rel_p[r])
            u = np.fft.irfft(spec, n=self.dim, axis=-1).astype(np.float32)
        return self._alg.inverse_permute(u, 2)

    def cleanup(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Top-1 entity and its cosine for each noisy vector."""
        v = np.atleast_2d(vectors)
        vn = np.linalg.norm(v, axis=1) + 1e-8
        sims = (v @ self.entities.T) / (vn[:, None] * self._ent_norms[None, :])
        idx = sims.argmax(axis=1)
        return idx, sims[np.arange(v.shape[0]), idx]

    def hop(self, subjects: np.ndarray, relations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One traversal step for a batch of ``(s, r)`` pairs -> ``(next_entity, cosine)``."""
        return self.cleanup(self.query_vectors(subjects, relations))

    def walk(self, start: np.ndarray, relation_path: np.ndarray) -> np.ndarray:
        """Follow ``relation_path`` (``(B, h)``) from ``start`` (``(B,)``) -> visited entities ``(B, h)``."""
        cur = np.atleast_1d(start).astype(np.int64)
        rels = np.atleast_2d(relation_path)
        visited = np.empty(rels.shape, dtype=np.int64)
        for h in range(rels.shape[1]):
            cur, _ = self.hop(cur, rels[:, h])
            visited[:, h] = cur
        return visited

    # ------------------------------------------------------------------ memory
    def _array_names(self, query_path_only: bool = False) -> list[str]:
        if query_path_only:  # what answering a query reads (write-path copies such as rho^2(entities) are not needed)
            base = ["memory", "entities", "_rel_p", "_ent_norms"]
            return base + (["_f_ent", "_f_rel_p", "_f_mem"] if self.algebra == "hrr" else [])
        base = ["memory", "entities", "relations", "_rel_p", "_ent_p2", "_ent_norms"]
        return base + (["_f_ent", "_f_rel_p", "_f_ent_p2", "_f_mem"] if self.algebra == "hrr" else [])

    def trace_bytes(self) -> int:
        """Bytes of the single superposed memory vector (the 'graph' itself)."""
        return int(self.memory.nbytes)

    def memory_bytes(self) -> int:
        """Every array this object holds (trace, item memories, permuted copies, HRR spectra): as allocated."""
        return int(sum(getattr(self, n).nbytes for n in self._array_names()))

    def query_path_bytes(self) -> int:
        """Lower bound: only the arrays a hop reads (a leaner implementation could drop the write-path copies)."""
        return int(sum(getattr(self, n).nbytes for n in self._array_names(query_path_only=True)))
