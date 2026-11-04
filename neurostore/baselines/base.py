"""Interfaces shared by the baselines and the neural systems in Workload A.

Every system answers the same question -- "given a (possibly corrupted) key, return the
stored value" -- through the same small API, so that the harness can feed identical queries
to all of them and time the same operations.
"""
from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from ..common import GrowableRows, pack_bipolar, unpack_bits


class Unavailable(RuntimeError):
    """An optional dependency or service needed by a baseline is not present."""


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def bytes_to_bipolar(items: list[bytes | None], nbytes: int, dim: int) -> np.ndarray:
    """Decode returned packed values (``None`` = miss) to ``(Q, dim)`` int8 bipolar; misses -> 0."""
    zero = bytes(nbytes)
    buf = b"".join(x if x is not None else zero for x in items)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(len(items), nbytes)
    out = (np.unpackbits(arr, axis=1)[:, :dim].astype(np.int8)) * 2 - 1
    miss = np.fromiter((x is None for x in items), dtype=bool, count=len(items))
    out[miss] = 0
    return out


class AssocSystem:
    """A system under test for Workload A (key -> value recall from noisy keys).

    ``retrieve_values`` returns ``(Q, Dv)`` rows of the system's *actual output*; an all-zero
    row means "no answer", which can never equal a bipolar value. ``retrieve_ids`` (optional)
    exposes the identification decision for systems that have one.

    The ``prepare_*``/``*_one`` pairs exist so the latency harness can convert inputs into the
    system's native format *outside* the timed region and then time a single operation.
    """

    name = "system"
    family = "other"  # neurostore | kv | ann | lsh
    label = ""
    max_items: int | None = None  # skip the system when the store would be larger
    max_queries: int | None = None  # evaluate on at most this many queries (slow Python systems)

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return True, ""

    def build(self, keys: np.ndarray, values: np.ndarray) -> None:
        raise NotImplementedError

    def retrieve_values(self, queries: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def retrieve_ids(self, queries: np.ndarray) -> np.ndarray | None:
        return None

    def prepare_read(self, query: np.ndarray) -> Any:
        return query

    def read_one(self, prepared: Any) -> Any:
        raise NotImplementedError

    def prepare_write(self, key: np.ndarray, value: np.ndarray) -> Any:
        return (key, value)

    def write_one(self, prepared: Any) -> None:
        raise NotImplementedError

    def memory_bytes(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        pass


class IdBasedSystem(AssocSystem):
    """An index that maps a (noisy) key to the *index* of a stored item.

    Values live in a side array of bit-packed words, exactly like the payload column of a KV
    store, so memory comparisons are not skewed by an unpacked value matrix.
    Subclasses implement ``_build_index``, ``_search``, ``_search_one``, ``_prepare_key``,
    ``_add_key`` and ``_index_bytes``.
    """

    def __init__(self) -> None:
        self._vals: GrowableRows | None = None
        self.dim_v = 0

    # -- subclass hooks
    def _build_index(self, keys: np.ndarray) -> None:
        raise NotImplementedError

    def _search(self, queries: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _search_one(self, native: Any) -> int:
        raise NotImplementedError

    def _prepare_key(self, key: np.ndarray) -> Any:
        raise NotImplementedError

    def _add_key(self, native: Any) -> None:
        raise NotImplementedError

    def _index_bytes(self) -> int:
        raise NotImplementedError

    # -- AssocSystem API
    def build(self, keys: np.ndarray, values: np.ndarray) -> None:
        self.dim_v = values.shape[1]
        packed = pack_bipolar(values)
        self._vals = GrowableRows(packed.shape[1], np.uint64, capacity=packed.shape[0])
        self._vals.extend(packed)
        self._build_index(keys)

    def retrieve_ids(self, queries: np.ndarray) -> np.ndarray:
        return self._search(queries)

    def retrieve_values(self, queries: np.ndarray) -> np.ndarray:
        ids = self.retrieve_ids(queries)
        out = np.zeros((len(ids), self.dim_v), dtype=np.int8)
        ok = ids >= 0
        if ok.any():
            bits = unpack_bits(self._vals.view[ids[ok]], self.dim_v)
            out[ok] = bits.astype(np.int8) * 2 - 1
        return out

    def prepare_read(self, query: np.ndarray) -> Any:
        return self._prepare_key(query)

    def read_one(self, prepared: Any) -> Any:
        i = self._search_one(prepared)
        return self._vals.view[i] if i >= 0 else None

    def prepare_write(self, key: np.ndarray, value: np.ndarray) -> Any:
        return (self._prepare_key(key), pack_bipolar(value))

    def write_one(self, prepared: Any) -> None:
        key_native, value_words = prepared
        self._vals.append(value_words)
        self._add_key(key_native)

    def memory_bytes(self) -> int:
        return int(self._index_bytes() + self._vals.nbytes)
