"""Exact-match key-value stores: Python dict, RocksDB (embedded) and Redis (real server).

These are the "location-addressed" baselines: a lookup succeeds only if the key matches
*exactly*, so under any bit-flip noise they return nothing. They are included because the
evaluation spec asks for them -- and to measure what they are genuinely best at: latency and
memory per pair at 0% noise.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Any

import numpy as np

from ..common import deep_sizeof, pack_bipolar
from .base import AssocSystem, Unavailable, bytes_to_bipolar, has_module


def _key_bytes(x: np.ndarray) -> bytes:
    """Packed-bit bytes of one bipolar vector (the canonical exact-match key)."""
    return pack_bipolar(x).view(np.uint8).tobytes()


def _batch_bytes(x: np.ndarray) -> list[bytes]:
    packed = pack_bipolar(np.atleast_2d(x)).view(np.uint8)
    return [packed[i].tobytes() for i in range(packed.shape[0])]


class _KVBase(AssocSystem):
    """Shared logic: keys/values are packed-bit byte strings; subclasses supply get/put."""

    family = "kv"
    supports_ids = False

    def __init__(self) -> None:
        self.dim_v = 0
        self.nbytes_v = 0

    def _put_many(self, items: list[tuple[bytes, bytes]]) -> None:
        raise NotImplementedError

    def _put(self, key: bytes, value: bytes) -> None:
        raise NotImplementedError

    def _get(self, key: bytes) -> bytes | None:
        raise NotImplementedError

    def _get_many(self, keys: list[bytes]) -> list[bytes | None]:
        return [self._get(k) for k in keys]

    def build(self, keys: np.ndarray, values: np.ndarray) -> None:
        self.dim_v = values.shape[1]
        kb, vb = _batch_bytes(keys), _batch_bytes(values)
        self.nbytes_v = len(vb[0])
        self._put_many(list(zip(kb, vb)))

    def retrieve_values(self, queries: np.ndarray) -> np.ndarray:
        return bytes_to_bipolar(self._get_many(_batch_bytes(queries)), self.nbytes_v, self.dim_v)

    def prepare_read(self, query: np.ndarray) -> bytes:
        return _key_bytes(query)

    def read_one(self, prepared: bytes) -> Any:
        return self._get(prepared)

    def prepare_write(self, key: np.ndarray, value: np.ndarray) -> tuple[bytes, bytes]:
        return (_key_bytes(key), _key_bytes(value))

    def write_one(self, prepared: tuple[bytes, bytes]) -> None:
        self._put(*prepared)


class DictKVSystem(_KVBase):
    """CPython ``dict`` from packed key bytes to packed value bytes."""

    name = "python_dict"
    label = "Python dict (exact)"

    def __init__(self) -> None:
        super().__init__()
        self._d: dict[bytes, bytes] = {}

    def _put_many(self, items: list[tuple[bytes, bytes]]) -> None:
        self._d = dict(items)

    def _put(self, key: bytes, value: bytes) -> None:
        self._d[key] = value

    def _get(self, key: bytes) -> bytes | None:
        return self._d.get(key)

    def memory_bytes(self) -> int:
        return deep_sizeof(self._d)


class RocksKVSystem(_KVBase):
    """Embedded RocksDB through ``rocksdict`` (raw bytes mode).

    ``memory_bytes`` is the size of the persisted SST files after a flush -- an on-disk
    footprint, not resident memory, so it is not directly comparable with in-RAM structures.
    """

    name = "rocksdb"
    label = "RocksDB (exact, embedded)"

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return (True, "") if has_module("rocksdict") else (False, "rocksdict not installed")

    def __init__(self) -> None:
        super().__init__()
        if not has_module("rocksdict"):
            raise Unavailable("rocksdict not installed")
        from rocksdict import Options, Rdict

        self._dir = tempfile.mkdtemp(prefix="neurostore_rocks_")
        opts = Options(raw_mode=True)
        opts.create_if_missing(True)
        self._db = Rdict(self._dir, options=opts)

    def _put_many(self, items: list[tuple[bytes, bytes]]) -> None:
        for k, v in items:
            self._db[k] = v
        self._db.flush()

    def _put(self, key: bytes, value: bytes) -> None:
        self._db[key] = value

    def _get(self, key: bytes) -> bytes | None:
        return self._db.get(key)

    def memory_bytes(self) -> int:
        self._db.flush()
        return int(sum(os.path.getsize(os.path.join(self._dir, f)) for f in os.listdir(self._dir) if f.endswith(".sst")))

    def close(self) -> None:
        try:
            self._db.close()
        finally:
            shutil.rmtree(self._dir, ignore_errors=True)


class RedisKVSystem(_KVBase):
    """A real ``redis-server`` subprocess on a free localhost port (TCP round trip included).

    Requires the ``redis`` client package and a ``redis-server`` binary on ``PATH``.
    ``memory_bytes`` is the growth of ``INFO memory used_memory`` caused by the data.
    """

    name = "redis"
    label = "Redis (exact, localhost TCP)"

    @classmethod
    def available(cls) -> tuple[bool, str]:
        if not has_module("redis"):
            return False, "redis client not installed"
        if shutil.which("redis-server") is None:
            return False, "redis-server binary not found on PATH"
        return True, ""

    def __init__(self) -> None:
        super().__init__()
        ok, why = self.available()
        if not ok:
            raise Unavailable(why)
        import redis

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self._proc = subprocess.Popen(
            ["redis-server", "--port", str(port), "--bind", "127.0.0.1", "--save", "", "--appendonly", "no"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._r = redis.Redis(host="127.0.0.1", port=port)
        for _ in range(100):
            try:
                self._r.ping()
                break
            except redis.exceptions.ConnectionError:
                time.sleep(0.05)
        else:
            self.close()
            raise Unavailable("redis-server did not start")
        self._base_mem = int(self._r.info("memory")["used_memory"])

    def _put_many(self, items: list[tuple[bytes, bytes]]) -> None:
        for s in range(0, len(items), 5000):
            pipe = self._r.pipeline(transaction=False)
            for k, v in items[s : s + 5000]:
                pipe.set(k, v)
            pipe.execute()

    def _put(self, key: bytes, value: bytes) -> None:
        self._r.set(key, value)

    def _get(self, key: bytes) -> bytes | None:
        return self._r.get(key)

    def _get_many(self, keys: list[bytes]) -> list[bytes | None]:
        out: list[bytes | None] = []
        for s in range(0, len(keys), 5000):
            out.extend(self._r.mget(keys[s : s + 5000]))
        return out

    def memory_bytes(self) -> int:
        return int(self._r.info("memory")["used_memory"]) - self._base_mem

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        # Persistence is disabled (--save "" --appendonly no), so SIGTERM has nothing to flush and exits in
        # ~0.1 s; the client's SHUTDOWN command blocks for seconds waiting on the connection teardown.
        try:
            self._r.close()
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        self._proc = None
