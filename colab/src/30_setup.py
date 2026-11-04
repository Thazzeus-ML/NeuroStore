# %% id=setup
#@title Setup: JAX backend, devices, environment, provenance
import os, sys, json, time, gc, platform, subprocess, warnings, shutil, hashlib, datetime, importlib.metadata
from pathlib import Path
from functools import partial
from collections import Counter

if SIMULATE_CPU_DEVICES and int(SIMULATE_CPU_DEVICES) > 1:      # must happen before jax is imported
    _flag = f"--xla_force_host_platform_device_count={int(SIMULATE_CPU_DEVICES)}"
    if _flag not in os.environ.get("XLA_FLAGS", ""):
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _flag).strip()

import pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except Exception:                                              # plain Python
    display = print

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
except Exception as exc:
    raise RuntimeError("JAX could not be imported. See https://docs.jax.dev/en/latest/installation.html and install the "
                       "backend matching your Colab runtime, then restart the runtime.") from exc

HI = lax.Precision.HIGHEST        # explicit per-kernel precision; v2 sets NO global matmul precision
DEFAULT_PREC = lax.Precision.DEFAULT
DEVICES = jax.local_devices()
DEVICE = DEVICES[0]
BACKEND = DEVICE.platform
DEVICE_KIND = getattr(DEVICE, "device_kind", str(DEVICE))
IN_COLAB = "google.colab" in sys.modules or Path("/content").exists()
RUN_ID = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ver(pkg):
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return "n/a"


def _cmd(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def device_memory():
    """(bytes_in_use, peak_bytes_in_use, bytes_limit) of the benchmark device, None where the backend does not report."""
    try:
        s = DEVICE.memory_stats() or {}
    except Exception:
        s = {}
    return s.get("bytes_in_use"), s.get("peak_bytes_in_use"), s.get("bytes_limit")


def _host_ram():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


_lim = device_memory()[2]
MEM_LIMIT = _lim if _lim else (_host_ram() if BACKEND == "cpu" else None)       # bytes available to this benchmark
NMAX = P["nmax_cpu"] if BACKEND == "cpu" else P["nmax_acc"]
CONFIG = {"profile": PROFILE, "seed": SEED, "D": D, "beta": BETA, "profile_params": P,
          "run": {k: v for k, v in globals().items() if k.startswith("RUN_") and isinstance(v, bool)}}
CONFIG_HASH = hashlib.sha1(json.dumps(CONFIG, sort_keys=True, default=str).encode()).hexdigest()[:10]
OUTPUT_DIR = Path(OUTPUT_ROOT or ("/content/neurostore_results" if IN_COLAB else "./neurostore_results")) / RUN_ID
(OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

METADATA = {
    "notebook": "NeuroStore_Colab_Benchmarks_v2", "run_id": RUN_ID, "profile": PROFILE, "seed": SEED, "config_hash": CONFIG_HASH,
    "backend": BACKEND, "device_kind": DEVICE_KIND, "local_device_count": len(DEVICES), "devices": [str(x) for x in DEVICES[:16]],
    "jax": jax.__version__, "jaxlib": _ver("jaxlib"), "numpy": np.__version__, "pandas": pd.__version__,
    "matplotlib": _ver("matplotlib"), "scipy": _ver("scipy"), "python": sys.version.split()[0], "platform": platform.platform(),
    "in_colab": IN_COLAB, "nvidia_smi": _cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
    "tpu_environment": {k: v for k, v in os.environ.items() if k.startswith("TPU_") or k in ("COLAB_TPU_ADDR",)},
    "device_memory_limit_bytes": MEM_LIMIT, "default_matmul_precision_config": str(getattr(jax.config, "jax_default_matmul_precision", None)),
    "x64_enabled": bool(jax.config.jax_enable_x64), "largest_N_for_sweeps": NMAX,
}
META_ROW = {"run_id": RUN_ID, "profile": PROFILE, "seed_base": SEED, "config_hash": CONFIG_HASH, "backend": BACKEND,
            "device_kind": DEVICE_KIND, "jax_version": jax.__version__}
print(json.dumps(METADATA, indent=1, default=str))
if BACKEND == "cpu":
    warnings.warn("JAX is using the CPU. If you chose a GPU/TPU runtime, check the accelerator and the JAX backend installation.")
print(f"\nBenchmarks target ONE device ({DEVICE_KIND}) unless a section says otherwise; {len(DEVICES)} local device(s) visible.")
