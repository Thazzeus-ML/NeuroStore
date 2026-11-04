from __future__ import annotations
import numpy as np


def gather_ranges(indptr: np.ndarray, rows: np.ndarray):
    """Positions of all entries in CSR rows `rows` plus, for each entry, the index into `rows`.
    The sparse-accumulate kernel's gather step: vectorised, no python loop."""
    rows = np.asarray(rows, np.int64)
    starts = indptr[rows].astype(np.int64)
    lens = indptr[rows + 1].astype(np.int64) - starts
    tot = int(lens.sum())
    if tot == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    cum = np.cumsum(lens)
    offs = np.repeat(starts - (cum - lens), lens) + np.arange(tot)
    which = np.repeat(np.arange(len(rows)), lens)
    return offs, which
