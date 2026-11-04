from .base import SpikeStore
from .aos import AoSStore
from .soa import SoAStore
from .volley import VolleyStore
from .csr import CSRGraph
from .dictgraph import DictGraph

SPIKE_LAYOUTS = ("aos", "soa", "soa_event", "volley")
GRAPH_LAYOUTS = ("csr", "dict")


def make_store(name: str, k: int, **kw) -> SpikeStore:
    if name == "aos":
        return AoSStore(k, **kw)
    if name == "soa":
        return SoAStore(k, **kw)
    if name == "soa_event":
        return SoAStore(k, normalized=True, **kw)
    if name == "volley":
        return VolleyStore(k, **kw)
    raise ValueError(name)
