from .exact import ExactBackend
from .ann import IVFBackend
from .flyhash import FlyHashBackend
from .hopfield import HopfieldBackend
from .vsa import VSABackend

class ExactL2Backend(ExactBackend):
    """Euclidean exact kNN: the textbook reference next to the cosine ExactBackend."""
    name = "exact_l2"

    def __init__(self, beta=None):
        super().__init__(metric="l2", beta=beta)


class ExactL1Backend(ExactBackend):
    """Manhattan exact kNN (bounded-influence control next to the Euclidean store)."""
    name = "exact_l1"

    def __init__(self, beta=None):
        super().__init__(metric="l1", beta=beta)


BACKENDS = {"exact": ExactBackend, "exact_l2": ExactL2Backend, "exact_l1": ExactL1Backend, "ann": IVFBackend, "flyhash": FlyHashBackend, "hopfield": HopfieldBackend, "vsa": VSABackend}


def make_backend(name: str, **cfg):
    if name not in BACKENDS:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(BACKENDS)}")
    return BACKENDS[name](**cfg)
