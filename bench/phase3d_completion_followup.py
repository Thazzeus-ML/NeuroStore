"""Post-hoc follow-up to the FAILED Gate 2 (added after the verdict; does NOT change it).
Same masked-cue task, but the read path now also spreads activation along the learned
within-event synapses (cue -> lattice -> pair completion) and mixes it with the assembly
completion score. Mixing weight beta and number of assemblies are chosen on validation."""
import itertools
import numpy as np
from sim.generator import World, WorldConfig
from neurostore.config import Config
from neurostore.store import NeuroStore
from neurostore.readout import select_assemblies
from eval.baselines import JaccardNN
from eval.protocol import mask_cues, prec_mean
from eval.gates import load_gates, write_result
from eval.metrics import summarize

G = load_gates(); g = G["phase2"]; seeds = G["global"]["gate_seeds"]


def asm_vec(asm, cos, sel):
    sc = np.zeros(asm.N, np.float32)
    for a in sel:
        sc[asm.member_ids[a]] += cos[a] * asm.member_w[a]
    return sc


def top_m(score, cue, m):
    score = score.copy(); score[cue] = 0
    nz = int((score > 0).sum()); mm = min(m, nz)
    return np.argsort(-score, kind="stable")[:mm] if mm > 0 else np.zeros(0, np.int64)


def run(seed):
    w = World(WorldConfig(), seed)
    tr, va, te = w.stream("train", g["n_train"]), w.stream("val", g["n_val"]), w.stream("test", g["n_test"])
    st = NeuroStore(Config(seed=seed, homeo=False, gate=False, replay_budget=200)); st.ingest(tr); st.flush()
    asm, lat = st.asm, st.lattice
    codes = st.log.gather(np.arange(st.log.n_stored))[0]
    vc = [st.enc.encode(va.event(i))[0] for i in range(len(va))]; tc = [st.enc.encode(te.event(i))[0] for i in range(len(te))]
    nn = JaccardNN().fit(codes, 1024)
    out = {}
    for mf in g["masks"]:
        vcu, vma = mask_cues(vc, mf, seed, 1); tcu, tma = mask_cues(tc, mf, seed, 2)
        def feats(cues):
            cos = [asm.vote(c) for c in cues]
            sp = [lat.spread(c, hops=1, normalize=True, keep=0) for c in cues]
            return cos, sp
        vcos, vsp = feats(vcu); tcos, tsp = feats(tcu)
        def score(cos, sp, cues, mas, A, beta):
            comps = []
            for c, s, q, mk in zip(cos, sp, cues, mas):
                a = asm_vec(asm, c, select_assemblies(c, 0.7, 0.1, A))
                tot = a / max(a.max(), 1e-9) + (beta * s / max(s.max(), 1e-9) if beta else 0)
                comps.append(top_m(tot, q, len(mk)))
            return prec_mean(comps, mas)
        best = {0: (-1, None), "mix": (-1, None)}
        for A, beta in itertools.product(g["grid_ns_complete_a"], [0, 0.25, 0.5, 1, 2, 4]):
            p = score(vcos, vsp, vcu, vma, A, beta)
            key = 0 if beta == 0 else "mix"
            if p > best[key][0]: best[key] = (p, (A, beta))
        p_asm = score(tcos, tsp, tcu, tma, *best[0][1]); p_mix = score(tcos, tsp, tcu, tma, *best["mix"][1])
        # kNN reference (best k on val)
        vi, vs = nn.topk(vcu); ti, ts = nn.topk(tcu)
        def nnp(idx, sims, cues, mas, k):
            comps = []
            for ii, ss, q, mk in zip(idx, sims, cues, mas):
                sc = np.zeros(1024)
                for j, s_ in zip(ii[:k], ss[:k]): sc[codes[j]] += s_
                comps.append(top_m(sc, q, len(mk)))
            return prec_mean(comps, mas)
        bk = max(g["grid_nn_k"], key=lambda k: nnp(vi, vs, vcu, vma, k))
        out[str(mf)] = {"assembly_only": p_asm, "assembly_plus_lattice": p_mix, "knn": nnp(ti, ts, tcu, tma, bk),
                        "mix_params": best["mix"][1]}
    return out


per = [run(s) for s in seeds]
res = {}
for mf in g["masks"]:
    k = str(mf)
    a = np.array([p[k]["assembly_only"] for p in per]); b = np.array([p[k]["assembly_plus_lattice"] for p in per]); n = np.array([p[k]["knn"] for p in per])
    res[k] = {"assembly_only": summarize(a), "assembly_plus_lattice": summarize(b), "knn": summarize(n),
              "ratio_assembly_only": float(a.mean() / n.mean()), "ratio_plus_lattice": float(b.mean() / n.mean()),
              "required_ratio_gate2": g["approach_ratio"][k]}
    print(mf, f"asm={a.mean():.3f} +lattice={b.mean():.3f} knn={n.mean():.3f} ratios {a.mean()/n.mean():.3f} -> {b.mean()/n.mean():.3f} (need {g['approach_ratio'][k]})")
write_result("phase3_completion_followup", {"seeds": seeds, "masks": res, "per_seed": per})
