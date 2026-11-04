from .exact import ExactBackend
from .ann import IVFBackend
from .flyhash import FlyHashBackend
from .hopfield import HopfieldBackend
from .vsa import VSABackend

BACKENDS = {"exact": ExactBackend, "ann": IVFBackend, "flyhash": FlyHashBackend, "hopfield": HopfieldBackend, "vsa": VSABackend}


def make_backend(name: str, **cfg):
    if name not in BACKENDS:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(BACKENDS)}")
    return BACKENDS[name](**cfg)
