"""Baseline systems the neural primitives are measured against.

Every optional dependency is detected at run time; a missing library or service is recorded
as *skipped* in the results (with the reason) rather than silently dropped or crashing a run.
"""
from __future__ import annotations

from .ann import BruteForceHammingSystem, FaissBinaryFlatSystem, FaissBinaryHNSWSystem, QdrantSystem
from .base import AssocSystem, IdBasedSystem, Unavailable, has_module
from .cache import (
    BruteForceCosineCache,
    CacheSystem,
    ExactHashCache,
    FaissFlatCache,
    FaissHNSWCache,
    FlyHashBinaryHNSWCache,
    FlyHashInvertedCache,
    FlyHashScanCache,
    QdrantCache,
    SimHashCache,
)
from .graph import AdjacencyDictGraph, DenseSuccessorTable
from .kv import DictKVSystem, RedisKVSystem, RocksKVSystem
from .lsh import BitSamplingLSHSystem, DatasketchLSHSystem
from .relational import CompositeIndexSet, FuzzyScan, InvertedIndexIntersect, LinearScan

# name -> (family, description, availability callable)
CATALOG: dict[str, tuple[str, str, object]] = {
    "python_dict": ("kv", "CPython dict, exact match", DictKVSystem.available),
    "rocksdb": ("kv", "Embedded RocksDB via rocksdict, exact match", RocksKVSystem.available),
    "redis": ("kv", "Real redis-server subprocess, exact match", RedisKVSystem.available),
    "bruteforce_hamming": ("ann", "Exact NN, packed XOR+popcount scan", BruteForceHammingSystem.available),
    "bruteforce_cosine": ("ann", "Exact NN, float32 BLAS scan", BruteForceCosineCache.available),
    "faiss_binary_flat": ("ann", "FAISS IndexBinaryFlat (exact)", FaissBinaryFlatSystem.available),
    "faiss_binary_hnsw": ("ann", "FAISS IndexBinaryHNSW", FaissBinaryHNSWSystem.available),
    "faiss_flat_ip": ("ann", "FAISS IndexFlatIP (exact, SIMD)", FaissFlatCache.available),
    "faiss_hnsw": ("ann", "FAISS IndexHNSWFlat", FaissHNSWCache.available),
    "qdrant_local": ("ann", "Qdrant embedded local mode", QdrantSystem.available),
    "bitsampling_lsh": ("lsh", "Bit-sampling LSH (Hamming), tunable k/L", lambda: (True, "")),
    "simhash": ("lsh", "Random-hyperplane signatures + Hamming scan", lambda: (True, "")),
    "datasketch_minhash_lsh": ("lsh", "datasketch MinHash-LSH", DatasketchLSHSystem.available),
}


def availability() -> dict[str, dict[str, object]]:
    """``{name: {family, description, available, reason}}`` for every catalogued baseline."""
    out = {}
    for name, (family, desc, check) in CATALOG.items():
        ok, why = check()  # type: ignore[operator]
        out[name] = {"family": family, "description": desc, "available": bool(ok), "reason": why}
    return out


__all__ = [n for n in dir() if not n.startswith("_")]
