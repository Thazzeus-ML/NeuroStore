"""Task metrics: regression (RMSE, MAE, NASA PHM08 score), retrieval (neighbor recall) and completion (NMSE)."""
from __future__ import annotations
import numpy as np


def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(y, float))))


def nasa_score(y, p):
    """PHM08 asymmetric score: late predictions (p > y) are penalised more than early ones. Lower is better."""
    d = np.asarray(p, float) - np.asarray(y, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def neighbor_recall(approx_idx, exact_idx):
    """Mean fraction of the exact top-k neighbors that the approximate search also returned."""
    a, e = np.asarray(approx_idx), np.asarray(exact_idx)
    k = e.shape[1]
    return float(np.mean([len(set(x.tolist()) & set(y.tolist())) / k for x, y in zip(a, e)]))


def completion_nmse(true, recon, missing):
    """Mean squared error on the missing entries, in units of the (unit) feature variance. Mean imputation scores ~1."""
    m = np.asarray(missing, bool)
    d = (np.asarray(recon, float) - np.asarray(true, float))[m]
    return float(np.mean(d ** 2)) if d.size else float("nan")
