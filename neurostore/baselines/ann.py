"""Nearest-neighbour baselines for Workload A: exact brute force, FAISS and Qdrant.

``bruteforce_hamming`` is the most important entry in the whole comparison: it is *exact*
search, so it is the accuracy ceiling for every approximate or neural-inspired retriever on
the same data. A softmax associative memory with ``beta -> inf`` computes exactly this.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..common import GrowableRows, pack_bipolar, packed_distance_matrix, popcount_rows
from .base import IdBasedSystem, Unavailable, has_module


class BruteForceHammingSystem(IdBasedSystem):
    """Exact nearest neighbour by XOR + popcount over bit-packed keys."""

    name = "bruteforce_hamming"
    family = "ann"
    label = "Exact NN (packed Hamming scan)"

    def _build_index(self, keys: np.ndarray) -> None:
        packed = pack_bipolar(keys)
        self._k = GrowableRows(packed.shape[1], np.uint64, capacity=packed.shape[0])
        self._k.extend(packed)

    def _search(self, queries: np.ndarray) -> np.ndarray:
        return packed_distance_matrix(pack_bipolar(queries), self._k.view).argmin(axis=1).astype(np.int64)

    def _prepare_key(self, key: np.ndarray) -> np.ndarray:
        return pack_bipolar(key)

    def _search_one(self, native: np.ndarray) -> int:
        return int(popcount_rows(self._k.view ^ native).argmin())

    def _add_key(self, native: np.ndarray) -> None:
        self._k.append(native)

    def _index_bytes(self) -> int:
        return self._k.nbytes


class _FaissBinary(IdBasedSystem):
    family = "ann"
    efc = 200
    efs = 64
    hnsw_m = 32

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return (True, "") if has_module("faiss") else (False, "faiss-cpu not installed")

    def __init__(self) -> None:
        super().__init__()
        if not has_module("faiss"):
            raise Unavailable("faiss-cpu not installed")

    def _make_index(self, nbits: int) -> Any:
        raise NotImplementedError

    def _build_index(self, keys: np.ndarray) -> None:
        packed = pack_bipolar(keys)
        self._idx = self._make_index(packed.shape[1] * 64)
        self._idx.add(packed.view(np.uint8))

    def _search(self, queries: np.ndarray) -> np.ndarray:
        _, ids = self._idx.search(pack_bipolar(queries).view(np.uint8), 1)
        return ids[:, 0].astype(np.int64)

    def _prepare_key(self, key: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(pack_bipolar(key).view(np.uint8)[None, :])

    def _search_one(self, native: np.ndarray) -> int:
        return int(self._idx.search(native, 1)[1][0, 0])

    def _add_key(self, native: np.ndarray) -> None:
        self._idx.add(native)

    def _index_bytes(self) -> int:
        import faiss

        return int(len(faiss.serialize_index_binary(self._idx)))


class FaissBinaryFlatSystem(_FaissBinary):
    name = "faiss_binary_flat"
    label = "FAISS IndexBinaryFlat (exact)"

    def _make_index(self, nbits: int) -> Any:
        import faiss

        return faiss.IndexBinaryFlat(nbits)


class FaissBinaryHNSWSystem(_FaissBinary):
    name = "faiss_binary_hnsw"
    label = "FAISS IndexBinaryHNSW (approximate)"

    def _make_index(self, nbits: int) -> Any:
        import faiss

        idx = faiss.IndexBinaryHNSW(nbits, self.hnsw_m)
        idx.hnsw.efConstruction = self.efc
        idx.hnsw.efSearch = self.efs
        return idx


class QdrantSystem(IdBasedSystem):
    """Qdrant in embedded local mode (cosine similarity over float vectors).

    Local mode is a numpy brute-force scan behind a Python API: it measures the overhead of
    a vector-database client rather than HNSW search, hence the size caps.
    """

    name = "qdrant_local"
    family = "ann"
    label = "Qdrant (embedded local mode)"
    max_items = 20_000
    max_queries = 200

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return (True, "") if has_module("qdrant_client") else (False, "qdrant-client not installed")

    def __init__(self) -> None:
        super().__init__()
        if not has_module("qdrant_client"):
            raise Unavailable("qdrant-client not installed")
        self._client = None
        self._n = 0
        self._dim = 0

    def _build_index(self, keys: np.ndarray) -> None:
        from qdrant_client import QdrantClient, models

        self._client = QdrantClient(":memory:")
        self._dim = keys.shape[1]
        self._client.create_collection(
            "keys", vectors_config=models.VectorParams(size=self._dim, distance=models.Distance.COSINE)
        )
        self._client.upload_collection("keys", vectors=keys.astype(np.float32), ids=list(range(keys.shape[0])))
        self._n = keys.shape[0]

    def _search(self, queries: np.ndarray) -> np.ndarray:
        from qdrant_client import models

        reqs = [models.QueryRequest(query=q.astype(np.float32).tolist(), limit=1) for q in queries]
        resp = self._client.query_batch_points("keys", requests=reqs)
        return np.array([r.points[0].id if r.points else -1 for r in resp], dtype=np.int64)

    def _prepare_key(self, key: np.ndarray) -> list[float]:
        return key.astype(np.float32).tolist()

    def _search_one(self, native: list[float]) -> int:
        pts = self._client.query_points("keys", query=native, limit=1).points
        return int(pts[0].id) if pts else -1

    def _add_key(self, native: list[float]) -> None:
        from qdrant_client import models

        self._client.upsert("keys", points=[models.PointStruct(id=self._n, vector=native)])
        self._n += 1

    def _index_bytes(self) -> int:
        # lower bound: the float32 vectors; the client's internal bookkeeping is not exposed
        return int(self._n * self._dim * 4)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
