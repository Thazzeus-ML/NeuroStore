import numpy as np
import pytest

from neurostore.common import make_rng
from neurostore.profiles import get_profile


@pytest.fixture
def rng() -> np.random.Generator:
    return make_rng(12345)


@pytest.fixture(scope="session")
def quick() -> dict:
    """The 'quick' profile: tiny sizes that still exercise every code path."""
    return get_profile("quick")


def have(module: str) -> bool:
    from neurostore.baselines.base import has_module

    return has_module(module)


requires_faiss = pytest.mark.skipif(not have("faiss"), reason="faiss-cpu not installed")
requires_qdrant = pytest.mark.skipif(not have("qdrant_client"), reason="qdrant-client not installed")
requires_datasketch = pytest.mark.skipif(not have("datasketch"), reason="datasketch not installed")
requires_rocksdict = pytest.mark.skipif(not have("rocksdict"), reason="rocksdict not installed")


def _redis_ok() -> bool:
    from neurostore.baselines import RedisKVSystem

    return RedisKVSystem.available()[0]


requires_redis = pytest.mark.skipif(not _redis_ok(), reason="redis client or redis-server missing")


@pytest.fixture(scope="session")
def quick_suite():
    """One full quick-profile run (all six workloads, no external services), shared by many tests."""
    from neurostore.runner import run_suite

    return run_suite("all", profile="quick", seed=1, include_external=False)
