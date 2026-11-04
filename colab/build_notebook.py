#!/usr/bin/env python3
"""Build colab/NeuroStore_Colab_Benchmarks_v2.ipynb from the percent-format parts in colab/src/."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nbtools

if __name__ == "__main__":
    cells = nbtools.build(HERE / "src", HERE / "NeuroStore_Colab_Benchmarks_v2.ipynb", "NeuroStore GPU/TPU benchmark lab (v2)")
    print(f"wrote {HERE / 'NeuroStore_Colab_Benchmarks_v2.ipynb'} with {len(cells)} cells "
          f"({sum(c['kind'] == 'code' for c in cells)} code, {sum(c['kind'] == 'markdown' for c in cells)} markdown)")
