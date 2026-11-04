"""Relational baselines for Workload B (record completion / entity resolution).

All of them answer: "given K known (field, value) pairs, which record is it, and what is the
value of a field I do not know?". They differ in what they pay:

====================  ======================  ==========================  =====================
baseline              build memory            query                       tolerates a wrong cue?
====================  ======================  ==========================  =====================
CompositeIndexSet     C(F,K) indexes x M      one hash lookup             no (exact match)
InvertedIndexIntersect  F x M row ids         intersect K posting lists   no (exact match)
LinearScan            the table               O(M K) predicate scan       no (exact match)
FuzzyScan             the table               O(M K) overlap scan         yes (counts matches)
====================  ======================  ==========================  =====================

``FuzzyScan`` is the fair rival for corrupted cues: it is the "expensive fuzzy join" the
evaluation spec mentions, and it is exact, so it is the accuracy ceiling for noisy cues.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..common import deep_sizeof


class _Base:
    name = "relational"
    label = ""

    def __init__(self, records: np.ndarray) -> None:
        self.records = np.asarray(records)
        self.n_records, self.n_fields = self.records.shape

    def identify(self, fields: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Record id per cue (-1 if no record matches)."""
        raise NotImplementedError

    def reconstruct(self, fields: np.ndarray, values: np.ndarray, target_fields: np.ndarray) -> dict[str, np.ndarray]:
        ids = self.identify(fields, values)
        tf = np.broadcast_to(np.asarray(target_fields), ids.shape)
        found = self.records[np.maximum(ids, 0), tf].astype(np.int64)  # records may be a compact dtype (e.g. uint16)
        vals = np.where(ids >= 0, found, -1)
        return {"ids": ids, "values": vals}

    def memory_bytes(self) -> int:
        raise NotImplementedError


class LinearScan(_Base):
    """Exact predicate scan: the record equal to the cue on every known field."""

    name = "linear_scan"
    label = "Linear scan (exact predicate)"

    def identify(self, fields: np.ndarray, values: np.ndarray) -> np.ndarray:
        out = np.full(fields.shape[0], -1, dtype=np.int64)
        for q in range(fields.shape[0]):
            m = np.ones(self.n_records, dtype=bool)
            for j in range(fields.shape[1]):
                m &= self.records[:, fields[q, j]] == values[q, j]
            hit = np.flatnonzero(m)
            if hit.size:
                out[q] = hit[0]
        return out

    def memory_bytes(self) -> int:
        return int(self.records.nbytes)


class FuzzyScan(_Base):
    """Best-overlap scan: the record agreeing with the cue on the most known fields."""

    name = "fuzzy_scan"
    label = "Fuzzy scan (max field agreement)"

    def identify(self, fields: np.ndarray, values: np.ndarray) -> np.ndarray:
        out = np.empty(fields.shape[0], dtype=np.int64)
        for q in range(fields.shape[0]):
            score = (self.records[:, fields[q]] == values[q][None, :]).sum(axis=1)
            out[q] = int(score.argmax())
        return out

    def memory_bytes(self) -> int:
        return int(self.records.nbytes)


class InvertedIndexIntersect(_Base):
    """One posting list per (field, value); a query intersects the K relevant lists.

    This is how a real engine serves arbitrary field subsets *without* a composite index per
    subset: memory is ``F x M`` row ids regardless of which subsets are queried.
    """

    name = "inverted_index"
    label = "Per-column inverted index + intersect"

    def __init__(self, records: np.ndarray, vocab_sizes: np.ndarray | list[int]) -> None:
        super().__init__(records)
        self._order: list[np.ndarray] = []
        self._start: list[np.ndarray] = []
        for f in range(self.n_fields):
            col = self.records[:, f]
            order = np.argsort(col, kind="stable").astype(np.int32)
            counts = np.bincount(col, minlength=int(vocab_sizes[f]))
            self._order.append(order)
            self._start.append(np.concatenate(([0], np.cumsum(counts))).astype(np.int64))

    def posting(self, field: int, value: int) -> np.ndarray:
        s = self._start[field]
        return self._order[field][s[value] : s[value + 1]]

    def match_all(self, fields: np.ndarray, values: np.ndarray) -> np.ndarray:
        lists = sorted((self.posting(int(f), int(v)) for f, v in zip(fields, values)), key=len)
        res = lists[0]
        for nxt in lists[1:]:
            if res.size == 0:
                break
            res = np.intersect1d(res, nxt, assume_unique=True)
        return res

    def identify(self, fields: np.ndarray, values: np.ndarray) -> np.ndarray:
        out = np.full(fields.shape[0], -1, dtype=np.int64)
        for q in range(fields.shape[0]):
            res = self.match_all(fields[q], values[q])
            if res.size:
                out[q] = res.min()
        return out

    def index_bytes(self) -> int:
        return int(sum(a.nbytes for a in self._order) + sum(a.nbytes for a in self._start))

    def memory_bytes(self) -> int:
        """Posting lists plus the base table that ``reconstruct`` reads (as allocated)."""
        return self.index_bytes() + int(self.records.nbytes)


class CompositeIndexSet(_Base):
    """A hash index for every size-K field subset (what O(1) lookup for any K-of-F cue needs).

    ``C(F, K)`` indexes of ``M`` entries each; for "any subset at all" the count is ``2^F - 1``.
    Python dicts stand in for B-tree/hash secondary indexes (identical asymptotic memory for
    equality lookups).
    """

    name = "composite_indexes"
    label = "Composite index per field subset"

    def __init__(self, records: np.ndarray, subset_size: int) -> None:
        super().__init__(records)
        self.subset_size = subset_size
        self.subsets = list(itertools.combinations(range(self.n_fields), subset_size))
        self.indexes: dict[tuple[int, ...], dict[tuple[int, ...], int]] = {}
        for s in self.subsets:
            d: dict[tuple[int, ...], int] = {}
            for i, key in enumerate(map(tuple, self.records[:, list(s)].tolist())):
                d.setdefault(key, i)
            self.indexes[s] = d

    @property
    def n_indexes(self) -> int:
        return len(self.subsets)

    def identify(self, fields: np.ndarray, values: np.ndarray) -> np.ndarray:
        out = np.full(fields.shape[0], -1, dtype=np.int64)
        for q in range(fields.shape[0]):
            order = np.argsort(fields[q])
            subset = tuple(int(f) for f in fields[q][order])
            key = tuple(int(v) for v in values[q][order])
            out[q] = self.indexes[subset].get(key, -1)
        return out

    def index_bytes(self) -> int:
        """Python dict-of-tuples indexes as implemented (deep size)."""
        return deep_sizeof(self.indexes)

    def compact_bytes(self) -> int:
        """What the same indexes would cost as sorted arrays of (K x field-width key, int32 row id) -- the fair
        representation to compare with the numpy posting lists; the C(F,K)-vs-F scaling is representation-independent."""
        entry = self.subset_size * self.records.dtype.itemsize + 4
        return int(self.n_indexes * self.n_records * entry)

    def memory_bytes(self) -> int:
        """Indexes as implemented plus the base table (as allocated)."""
        return self.index_bytes() + int(self.records.nbytes)
