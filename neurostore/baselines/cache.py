"""Semantic prompt-cache systems for Workload C: FlyHash variants and their rivals.

Every system stores ``N`` unit-norm float embeddings (the cached prompts) and answers
"which cached prompt, if any, is this query a paraphrase of?" with a top-1 id and a score
(higher = more similar). A hit *decision* is a threshold on that score, calibrated by the
workload at a fixed false-hit rate, so systems with different score scales stay comparable.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..common import (
    deep_sizeof,
    make_rng,
    pack_bits,
    packed_distance_matrix,
    popcount_rows,
)
from ..primitives.flyhash import FlyHashEncoder
from .base import Unavailable, has_module


class CacheSystem:
    name = "cache"
    family = "other"  # neurostore | ann | lsh | kv
    label = ""
    max_items: int | None = None

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return True, ""

    def build(self, vectors: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batch top-1 -> ``(ids, scores)``; id ``-1`` means no candidate."""
        raise NotImplementedError

    def prepare_one(self, query: np.ndarray) -> Any:
        return np.ascontiguousarray(query, dtype=np.float32)

    def search_one(self, native: Any) -> tuple[int, float]:
        raise NotImplementedError

    def memory_bytes(self) -> int:
        """Bytes of the per-prompt index (excludes fixed, N-independent state)."""
        raise NotImplementedError

    def fixed_bytes(self) -> int:
        """Bytes of N-independent state (e.g. the FlyHash projection matrix)."""
        return 0

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------- classical
class ExactHashCache(CacheSystem):
    """Exact-match cache keyed by the embedding's bytes (what a string/ID cache degenerates to)."""

    name = "exact_hash"
    family = "kv"
    label = "Exact-match hash (dict)"

    def build(self, vectors: np.ndarray) -> None:
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        self._d = {v[i].tobytes(): i for i in range(v.shape[0])}

    def search(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        ids = np.array([self._d.get(q[i].tobytes(), -1) for i in range(q.shape[0])], dtype=np.int64)
        return ids, (ids >= 0).astype(np.float32)

    def prepare_one(self, query: np.ndarray) -> bytes:
        return np.ascontiguousarray(query, dtype=np.float32).tobytes()

    def search_one(self, native: bytes) -> tuple[int, float]:
        i = self._d.get(native, -1)
        return i, float(i >= 0)

    def memory_bytes(self) -> int:
        return deep_sizeof(self._d)


class BruteForceCosineCache(CacheSystem):
    """Exact float32 cosine (= dot product on unit vectors) scan with BLAS."""

    name = "bruteforce_cosine"
    family = "ann"
    label = "Exact cosine scan (float32 BLAS)"

    def build(self, vectors: np.ndarray) -> None:
        self._m = np.ascontiguousarray(vectors, dtype=np.float32)

    def search(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        ids = np.empty(q.shape[0], dtype=np.int64)
        scores = np.empty(q.shape[0], dtype=np.float32)
        step = max(1, (64 << 20) // (4 * self._m.shape[0]))
        for s in range(0, q.shape[0], step):
            sims = q[s : s + step] @ self._m.T
            j = sims.argmax(axis=1)
            ids[s : s + step] = j
            scores[s : s + step] = sims[np.arange(sims.shape[0]), j]
        return ids, scores

    def search_one(self, native: np.ndarray) -> tuple[int, float]:
        sims = self._m @ native
        j = int(sims.argmax())
        return j, float(sims[j])

    def memory_bytes(self) -> int:
        return int(self._m.nbytes)


class SimHashCache(CacheSystem):
    """Random-hyperplane LSH signatures (``bits`` per vector) with a Hamming scan."""

    family = "lsh"

    def __init__(self, bits: int = 2048, seed: int = 0) -> None:
        self.bits = int(bits)
        self.seed = seed
        self.name = f"simhash_{self.bits}"
        self.label = f"SimHash {self.bits}-bit (Hamming scan)"

    def _encode(self, x: np.ndarray) -> np.ndarray:
        return pack_bits((np.atleast_2d(x) @ self._planes) > 0)

    def build(self, vectors: np.ndarray) -> None:
        rng = make_rng(self.seed)
        self._planes = rng.standard_normal((vectors.shape[1], self.bits)).astype(np.float32)
        self._db = self._encode(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = packed_distance_matrix(self._encode(np.ascontiguousarray(queries, dtype=np.float32)), self._db)
        j = d.argmin(axis=1)
        return j.astype(np.int64), 1.0 - d[np.arange(d.shape[0]), j] / self.bits

    def search_one(self, native: np.ndarray) -> tuple[int, float]:
        d = popcount_rows(self._db ^ self._encode(native)[0])
        j = int(d.argmin())
        return j, 1.0 - float(d[j]) / self.bits

    def memory_bytes(self) -> int:
        return int(self._db.nbytes)

    def fixed_bytes(self) -> int:
        return int(self._planes.nbytes)


# ---------------------------------------------------------------------------- FAISS / Qdrant
class _FaissFloat(CacheSystem):
    family = "ann"
    hnsw_m = 32
    efc = 200
    efs = 64

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return (True, "") if has_module("faiss") else (False, "faiss-cpu not installed")

    def __init__(self) -> None:
        if not has_module("faiss"):
            raise Unavailable("faiss-cpu not installed")

    def _make(self, d: int) -> Any:
        raise NotImplementedError

    def build(self, vectors: np.ndarray) -> None:
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        self._idx = self._make(v.shape[1])
        self._idx.add(v)

    def search(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores, ids = self._idx.search(np.ascontiguousarray(queries, dtype=np.float32), 1)
        return ids[:, 0].astype(np.int64), scores[:, 0]

    def prepare_one(self, query: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(query, dtype=np.float32)[None, :]

    def search_one(self, native: np.ndarray) -> tuple[int, float]:
        s, i = self._idx.search(native, 1)
        return int(i[0, 0]), float(s[0, 0])

    def memory_bytes(self) -> int:
        import faiss

        return int(len(faiss.serialize_index(self._idx)))


class FaissFlatCache(_FaissFloat):
    name = "faiss_flat_ip"
    label = "FAISS IndexFlatIP (exact, SIMD)"

    def _make(self, d: int) -> Any:
        import faiss

        return faiss.IndexFlatIP(d)


class FaissHNSWCache(_FaissFloat):
    name = "faiss_hnsw"
    label = "FAISS IndexHNSWFlat (M=32, efSearch=64)"

    def _make(self, d: int) -> Any:
        import faiss

        idx = faiss.IndexHNSWFlat(d, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = self.efc
        idx.hnsw.efSearch = self.efs
        return idx


class QdrantCache(CacheSystem):
    """Qdrant embedded local mode (cosine). A numpy scan behind a client API; size-capped."""

    name = "qdrant_local"
    family = "ann"
    label = "Qdrant (embedded local mode)"
    max_items = 20_000

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return (True, "") if has_module("qdrant_client") else (False, "qdrant-client not installed")

    def __init__(self) -> None:
        if not has_module("qdrant_client"):
            raise Unavailable("qdrant-client not installed")
        self._client = None

    def build(self, vectors: np.ndarray) -> None:
        from qdrant_client import QdrantClient, models

        self._client = QdrantClient(":memory:")
        self._n, self._d = vectors.shape
        self._client.create_collection("c", vectors_config=models.VectorParams(size=self._d, distance=models.Distance.COSINE))
        self._client.upload_collection("c", vectors=np.ascontiguousarray(vectors, dtype=np.float32), ids=list(range(self._n)))

    def search(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from qdrant_client import models

        reqs = [models.QueryRequest(query=q.tolist(), limit=1) for q in np.asarray(queries, dtype=np.float32)]
        resp = self._client.query_batch_points("c", requests=reqs)
        ids = np.array([r.points[0].id if r.points else -1 for r in resp], dtype=np.int64)
        scores = np.array([r.points[0].score if r.points else 0.0 for r in resp], dtype=np.float32)
        return ids, scores

    def prepare_one(self, query: np.ndarray) -> list[float]:
        return np.asarray(query, dtype=np.float32).tolist()

    def search_one(self, native: list[float]) -> tuple[int, float]:
        pts = self._client.query_points("c", query=native, limit=1).points
        return (int(pts[0].id), float(pts[0].score)) if pts else (-1, 0.0)

    def memory_bytes(self) -> int:
        return int(self._n * self._d * 4)  # lower bound: the vectors alone

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


# ---------------------------------------------------------------------------- FlyHash (NeuroStore)
class _FlyHashBase(CacheSystem):
    family = "neurostore"

    def __init__(self, out_dim: int = 2048, k: int = 64, density: float = 0.1, seed: int = 0) -> None:
        self.out_dim, self.k, self.density, self.seed = out_dim, k, density, seed

    def _fit(self, vectors: np.ndarray) -> None:
        self.enc = FlyHashEncoder(vectors.shape[1], self.out_dim, self.k, self.density, seed=self.seed)

    def fixed_bytes(self) -> int:
        return self.enc.memory_bytes()


class FlyHashScanCache(_FlyHashBase):
    """FlyHash codes stored bit-packed; query = encode + AND/popcount scan (shared active bits)."""

    def __init__(self, out_dim: int = 2048, k: int = 64, **kw: Any) -> None:
        super().__init__(out_dim, k, **kw)
        self.name = f"flyhash_scan_{out_dim}_{k}"
        self.label = f"FlyHash m={out_dim}, k={k} (popcount scan)"

    def build(self, vectors: np.ndarray) -> None:
        self._fit(vectors)
        self._db = self.enc.encode(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ov = packed_distance_matrix(self.enc.encode(np.ascontiguousarray(queries, dtype=np.float32)), self._db, op="and")
        j = ov.argmax(axis=1)
        return j.astype(np.int64), ov[np.arange(ov.shape[0]), j] / self.k

    def search_one(self, native: np.ndarray) -> tuple[int, float]:
        ov = popcount_rows(self._db & self.enc.encode(native))
        j = int(ov.argmax())
        return j, float(ov[j]) / self.k

    def memory_bytes(self) -> int:
        return int(self._db.nbytes)


class FlyHashInvertedCache(_FlyHashBase):
    """FlyHash codes as an inverted index: one posting list of cached ids per active bit.

    Sparse codes are what make this possible: a query touches only its ``k`` posting lists
    (about ``N k / m`` ids each) instead of scanning all ``N`` codes.
    """

    def __init__(self, out_dim: int = 2048, k: int = 64, **kw: Any) -> None:
        super().__init__(out_dim, k, **kw)
        self.name = f"flyhash_inv_{out_dim}_{k}"
        self.label = f"FlyHash m={out_dim}, k={k} (inverted index)"

    def build(self, vectors: np.ndarray) -> None:
        self._fit(vectors)
        idx = self.enc.encode_indices(np.ascontiguousarray(vectors, dtype=np.float32)).astype(np.int64)
        n = idx.shape[0]
        flat = idx.ravel()
        items = np.repeat(np.arange(n, dtype=np.int32), self.k)
        order = np.argsort(flat, kind="stable")
        self._post = items[order]
        counts = np.bincount(flat, minlength=self.out_dim)
        self._ptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        self._n = n

    def _best(self, active: np.ndarray) -> tuple[int, float]:
        ptr, post = self._ptr, self._post
        cnt = np.bincount(np.concatenate([post[ptr[b] : ptr[b + 1]] for b in active.tolist()]), minlength=self._n)
        j = int(cnt.argmax())
        return j, float(cnt[j]) / self.k

    def search(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        act = self.enc.encode_indices(np.ascontiguousarray(queries, dtype=np.float32))
        res = [self._best(act[i]) for i in range(act.shape[0])]
        return np.array([r[0] for r in res], dtype=np.int64), np.array([r[1] for r in res], dtype=np.float32)

    def search_one(self, native: np.ndarray) -> tuple[int, float]:
        return self._best(self.enc.encode_indices(native)[0])

    def memory_bytes(self) -> int:
        return int(self._post.nbytes + self._ptr.nbytes)


class FlyHashBinaryHNSWCache(_FlyHashBase):
    """FlyHash codes indexed by FAISS binary HNSW (Hamming graph search over sparse codes)."""

    efc = 200
    efs = 64
    hnsw_m = 32

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return (True, "") if has_module("faiss") else (False, "faiss-cpu not installed")

    def __init__(self, out_dim: int = 2048, k: int = 64, **kw: Any) -> None:
        if not has_module("faiss"):
            raise Unavailable("faiss-cpu not installed")
        super().__init__(out_dim, k, **kw)
        self.name = f"flyhash_binhnsw_{out_dim}_{k}"
        self.label = f"FlyHash m={out_dim}, k={k} + FAISS binary HNSW"

    def build(self, vectors: np.ndarray) -> None:
        import faiss

        self._fit(vectors)
        codes = self.enc.encode(np.ascontiguousarray(vectors, dtype=np.float32))
        self._idx = faiss.IndexBinaryHNSW(codes.shape[1] * 64, self.hnsw_m)
        self._idx.hnsw.efConstruction = self.efc
        self._idx.hnsw.efSearch = self.efs
        self._idx.add(codes.view(np.uint8))

    def _score(self, hamming: np.ndarray) -> np.ndarray:
        return (self.k - hamming / 2.0) / self.k  # overlap / k for equal-k codes

    def search(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        codes = self.enc.encode(np.ascontiguousarray(queries, dtype=np.float32))
        d, ids = self._idx.search(codes.view(np.uint8), 1)
        return ids[:, 0].astype(np.int64), self._score(d[:, 0].astype(np.float32))

    def search_one(self, native: np.ndarray) -> tuple[int, float]:
        code = self.enc.encode(native)[None, :]  # native is a single 1-D query vector
        d, ids = self._idx.search(code.view(np.uint8), 1)
        return int(ids[0, 0]), float(self._score(d[0, 0:1].astype(np.float32))[0])

    def memory_bytes(self) -> int:
        import faiss

        return int(len(faiss.serialize_index_binary(self._idx)))
