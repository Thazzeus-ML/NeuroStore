"""Metrics shared by experiments and tests."""
from __future__ import annotations
import numpy as np


def set_f1(pred, true) -> float:
    p, t = set(map(int, pred)), set(map(int, true))
    if not p and not t:
        return 1.0
    inter = len(p & t)
    if inter == 0:
        return 0.0
    pr, rc = inter / len(p), inter / len(t)
    return 2 * pr * rc / (pr + rc)


def precision_at_m(pred_ranked, true_set, m: int) -> float:
    """Fraction of the top-m predicted that are in true_set (m = number to recover)."""
    if m <= 0:
        return 1.0
    top = list(map(int, pred_ranked[:m]))
    ts = set(map(int, true_set))
    return sum(1 for u in top if u in ts) / m


def recall_at_k(pred_ranked, true_set, k: int) -> float:
    ts = set(map(int, true_set))
    if not ts:
        return 1.0
    top = set(map(int, pred_ranked[:k]))
    return len(top & ts) / len(ts)


def auc(scores, labels) -> float:
    """Rank-based ROC AUC (ties averaged). labels: bool, True = positive."""
    scores = np.asarray(scores, float); labels = np.asarray(labels, bool)
    npos, nneg = labels.sum(), (~labels).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def tv_distance(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


def overlap(u, v, k: int) -> float:
    return len(np.intersect1d(u, v)) / k


def summarize(vals) -> dict:
    v = np.asarray(vals, float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v)),
            "values": [float(x) for x in v]}
