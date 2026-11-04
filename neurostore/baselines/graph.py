"""Pointer-chasing graph baselines for Workload D (exact, by construction)."""
from __future__ import annotations

import numpy as np

from ..common import deep_sizeof


class AdjacencyDictGraph:
    """``dict[(subject, relation)] -> object``: the textbook in-memory adjacency lookup."""

    name = "adjacency_dict"
    label = "Python dict adjacency (pointer chasing)"

    def __init__(self, triples: np.ndarray) -> None:
        self._d = {(int(s), int(r)): int(o) for s, r, o in np.asarray(triples).reshape(-1, 3)}

    def hop(self, s: int, r: int) -> int:
        return self._d.get((s, r), -1)

    def walk(self, start: int, relations: list[int]) -> list[int]:
        out, cur, get = [], start, self._d.get
        for r in relations:
            cur = get((cur, r), -1)
            out.append(cur)
        return out

    def memory_bytes(self) -> int:
        return deep_sizeof(self._d)


class DenseSuccessorTable:
    """``succ[subject, relation] = object`` as one ``int32`` array: batch pointer chasing."""

    name = "successor_table"
    label = "Dense successor table (numpy)"

    def __init__(self, triples: np.ndarray, n_entities: int, n_relations: int) -> None:
        t = np.asarray(triples).reshape(-1, 3)
        self.succ = np.full((n_entities, n_relations), -1, dtype=np.int32)
        self.succ[t[:, 0], t[:, 1]] = t[:, 2]

    def hop_batch(self, s: np.ndarray, r: np.ndarray) -> np.ndarray:
        return self.succ[s, r]

    def walk_batch(self, start: np.ndarray, relations: np.ndarray) -> np.ndarray:
        cur = np.asarray(start)
        out = np.empty(relations.shape, dtype=np.int64)
        for h in range(relations.shape[1]):
            cur = self.succ[cur, relations[:, h]]
            out[:, h] = cur
        return out

    def memory_bytes(self) -> int:
        return int(self.succ.nbytes)
