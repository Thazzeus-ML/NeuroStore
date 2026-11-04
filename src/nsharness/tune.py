"""Hyper-parameter selection by group-held-out cross-validation on TRAINING data only (the official test set is never touched)."""
from __future__ import annotations
import itertools
import numpy as np
from .core import Split
from .pipeline import Pipeline


def make_group_folds(groups, n_folds: int, seed: int):
    """Assign whole groups (engines) to folds; returns a list of arrays of group ids."""
    u = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(len(u))
    return [u[perm[i::n_folds]] for i in range(n_folds)]


def sample_queries(split: Split, group_ids, per_group: int, rng, min_meta: tuple | None = None):
    """Indices of up to `per_group` random items for every group in group_ids (optionally requiring meta[key] >= value)."""
    out = []
    for g in group_ids:
        cand = np.where(split.groups == g)[0]
        if min_meta is not None:
            cand = cand[np.asarray(split.meta[min_meta[0]])[cand] >= min_meta[1]]
        if len(cand):
            out.append(rng.choice(cand, min(per_group, len(cand)), replace=False))
    return np.sort(np.concatenate(out)) if out else np.zeros(0, int)


def cv_plan(memory: Split, n_folds: int, per_group: int, seed: int, min_meta: tuple | None = None):
    """[(train_idx, query_idx)] with every held-out group removed from its fold's memory."""
    rng = np.random.default_rng(seed + 1)
    plan = []
    for gids in make_group_folds(memory.groups, n_folds, seed):
        held = np.isin(memory.groups, gids)
        plan.append((np.where(~held)[0], sample_queries(memory, gids, per_group, rng, min_meta)))
    return plan


def tune(backend: str, structural_grid: list, readout_grid: list, memory: Split, plan, feat_cfg: dict, seeds=(0,), use_mask=False,
         y_eval=None, progress=None):
    """For every structural config (backend constructor kwargs): fit one pipeline per fold and seed, search once with the largest k,
    then score every (k, beta) readout from the cached neighbours. Returns a list of dicts sorted by CV RMSE (best first).
    A backend whose readout is not top-k based (Hopfield full attention) is scored by calling readout directly for each beta."""
    y = memory.y if y_eval is None else y_eval
    kmax = max(k for k, _ in readout_grid)
    rows = []
    for scfg in structural_grid:
        se = {rb: [] for rb in readout_grid}
        for seed in seeds:
            errs = {rb: [] for rb in readout_grid}
            for tr, qi in plan:
                if len(qi) == 0:
                    continue
                pipe = Pipeline(backend, scfg, feat_cfg, k=kmax, use_mask=use_mask).fit(memory.take(tr), seed)
                F = pipe.features(memory.Z[qi])
                full_attn = bool(getattr(pipe.be, "cfg", {}).get("full_attention", False))
                if not full_attn:
                    idx, sim = pipe.be.search(F, kmax)
                for (k, beta) in readout_grid:
                    if full_attn:
                        pipe.be.cfg["beta"] = beta
                        p = pipe.be.readout(F, pipe.Y, k)
                    else:
                        p = pipe.be.readout_from(idx, sim, pipe.Y, k, beta)
                    errs[(k, beta)].append((p - y[qi]) ** 2)
            for rb in readout_grid:
                se[rb].append(np.concatenate(errs[rb]))
        for (k, beta) in readout_grid:
            allse = np.concatenate(se[(k, beta)])
            rows.append({"struct": scfg, "k": k, "beta": beta, "cv_rmse": float(np.sqrt(allse.mean())), "n_cv": int(len(se[(k, beta)][0]))})
        if progress:
            progress(scfg, min(r["cv_rmse"] for r in rows if r["struct"] == scfg))
    rows.sort(key=lambda r: r["cv_rmse"])
    return rows


def expand_grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*[axes[k] for k in keys])]
