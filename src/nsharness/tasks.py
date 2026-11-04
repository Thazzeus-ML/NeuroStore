"""Evaluation tasks shared by every domain: regression metrics with group-bootstrap CIs, corruption robustness, completion, OOD."""
from __future__ import annotations
import numpy as np
from .core import Split
from .corruption import corrupt
from .metrics import rmse, mae, nasa_score, completion_nmse
from .stats import group_bootstrap, paired_group_bootstrap
from . import ood as _ood


def metric_bundle(y, P, groups, n_boot: int = 500, seed: int = 0, y_capped=None) -> dict:
    """P: (S, n) predictions of S seeds (or (n,)). RMSE / MAE / NASA score are averaged over seeds (RMSE as sqrt of the mean
    squared error); the CI resamples whole groups. `y_capped` (optional) adds the same metrics against capped labels."""
    P = np.atleast_2d(np.asarray(P, float))
    y = np.asarray(y, float)
    se = (P - y[None, :]) ** 2

    def rm(idx):
        return float(np.sqrt(se[:, idx].mean()))

    pt, lo, hi, _ = group_bootstrap(rm, groups, n_boot=n_boot, seed=seed)
    d = P - y[None, :]
    out = {"rmse": pt, "rmse_lo": lo, "rmse_hi": hi, "n": int(len(y)),
           "rmse_per_seed": [rmse(y, p) for p in P], "mae": float(np.mean(np.abs(d))),
           "nasa_sum": float(np.mean([nasa_score(y, p) for p in P])), "nasa_mean": float(np.mean([nasa_score(y, p) for p in P]) / len(y))}
    if y_capped is not None:
        yc = np.asarray(y_capped, float)
        out["rmse_capped"] = float(np.sqrt(((P - yc[None, :]) ** 2).mean()))
    return out


def paired_rmse_diff(y, PA, PB, groups, n_boot: int = 500, seed: int = 0) -> dict:
    """RMSE(A) - RMSE(B) with a paired group bootstrap (negative = A better). PA, PB: (S, n)."""
    PA, PB = np.atleast_2d(PA), np.atleast_2d(PB)
    y = np.asarray(y, float)
    sa, sb = (PA - y) ** 2, (PB - y) ** 2
    pt, lo, hi, _ = paired_group_bootstrap(lambda i: float(np.sqrt(sa[:, i].mean())), lambda i: float(np.sqrt(sb[:, i].mean())),
                                           groups, n_boot=n_boot, seed=seed)
    return {"diff": pt, "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)}


def predict_under_corruption(pipeline, Z, family: str, severity: float, seed: int, give_mask: bool = False):
    Zc, obs = corrupt(Z, family, severity, seed)
    return pipeline.predict(Zc, obs if give_mask else None)


def completion_eval(pipeline, Z_clean, severity: float, seed: int, give_mask: bool) -> dict:
    """Drop channels (dropout family), reconstruct the feature vector from the memory, and score the MISSING features only.
    Mean imputation (predict the training mean) scores 1.0 because features are z-scored."""
    Zc, obs = corrupt(Z_clean, "dropout", severity, seed)
    F_true = pipeline.features(Z_clean)
    fmask = pipeline.feat.feature_mask(obs)
    if fmask is None:
        raise ValueError("completion needs a feature mask: use a featurizer without PCA")
    R = pipeline.reconstruct(Zc, obs if give_mask else None)
    missing = ~fmask
    return {"nmse": completion_nmse(F_true, R, missing), "missing_frac": float(missing.mean())}


def recall_vs_exact(approx_pipeline, exact_pipeline, Z, k: int = 10) -> float:
    from .metrics import neighbor_recall
    ia, _ = approx_pipeline.neighbors(Z, k=k)
    ie, _ = exact_pipeline.neighbors(Z, k=k)
    return neighbor_recall(ia, ie)


def ood_eval(pipeline, id_split: Split, ood_sets: dict, n_boot: int = 300, seed: int = 0, max_n: int = 3000) -> dict:
    """AUROC (ID = positive) of every familiarity score against every OOD set (dict name -> Split). Large sets are subsampled."""
    rng = np.random.default_rng(seed)

    def sub(s: Split):
        if len(s) <= max_n:
            return s
        return s.take(np.sort(rng.choice(len(s), max_n, replace=False)))

    idsp = sub(id_split)
    fam_id = pipeline.familiarity(idsp.Z)
    out = {}
    for name, s in ood_sets.items():
        s = sub(s)
        out[name] = _ood.familiarity_auroc(fam_id, idsp.groups, pipeline.familiarity(s.Z), s.groups, n_boot=n_boot, seed=seed)
    return out
