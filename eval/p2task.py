"""The phase-2 masked-cue task applied to any trained NeuroStore (used by phases 4 and 6)."""
from __future__ import annotations
import itertools
import numpy as np
from neurostore.readout import select_assemblies, complete_pattern, temporal_readout
from eval.protocol import mask_cues, ConceptMap, ns_decode, f1_mean, prec_mean, recall_at_k_batch


def encode_stream(store, stream):
    return [store.enc.encode(stream.event(i)) for i in range(len(stream))]


def stored_labels(store, train_stream):
    n = store.log.n_stored
    codes = store.log.gather(np.arange(n))[0] if n else np.zeros((0, store.cfg.k), np.int32)
    eids = store.log.event_ids()
    return codes, [train_stream.concepts[int(e)] for e in eids]


def phase2_metrics(store, train_stream, va, te, seed, mf, grids, C):
    """-> dict(f1, prec, val_f1, val_prec) for the assembly vote at one mask level (params tuned on val)."""
    asm = store.asm
    codes, concepts = stored_labels(store, train_stream)
    cmap = ConceptMap(asm, codes, concepts, C)
    vc = [e[0] for e in encode_stream(store, va)]; tc = [e[0] for e in encode_stream(store, te)]
    vcu, vma = mask_cues(vc, mf, seed, 1); tcu, tma = mask_cues(tc, mf, seed, 2)
    vcos = [asm.vote(c) for c in vcu]; tcos = [asm.vote(c) for c in tcu]
    best = (-1, None)
    for rho, cm, thr in itertools.product(grids["grid_ns_rho"], grids["grid_ns_cos_min"], grids["grid_ns_map_thr"]):
        f = f1_mean([ns_decode(c, cmap, rho, cm, thr) for c in vcos], va.concepts)
        if f > best[0]: best = (f, (rho, cm, thr))
    rho, cm, thr = best[1]
    f1 = f1_mean([ns_decode(c, cmap, rho, cm, thr) for c in tcos], te.concepts)
    bp = (-1, None)
    for A_c in grids["grid_ns_complete_a"]:
        p = prec_mean([complete_pattern(asm, c, select_assemblies(c, rho, cm, A_c), q, len(m)) for c, q, m in zip(vcos, vcu, vma)], vma)
        if p > bp[0]: bp = (p, A_c)
    prec = prec_mean([complete_pattern(asm, c, select_assemblies(c, rho, cm, bp[1]), q, len(m)) for c, q, m in zip(tcos, tcu, tma)], tma)
    return {"f1": f1, "prec": prec, "val_f1": best[0], "val_prec": bp[0], "n_assemblies": asm.n_alive}


def temporal_recall(store, stream, hops=1, K=32):
    if store.lattice is None:
        return float("nan")
    e = encode_stream(store, stream)
    preds = [temporal_readout(store.lattice, c, s, hops=hops, hop_decay=store.cfg.hop_decay, keep=store.cfg.spread_keep, m=K)[0] for c, s in e[:-1]]
    return recall_at_k_batch(preds, [x[0] for x in e[1:]], K)
