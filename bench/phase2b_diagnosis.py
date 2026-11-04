"""Diagnosis of the Gate-2 failure (mask 0.3, masked-unit recovery). Dev seeds only.
The sibling oracle peeks at the encoder's hidden feature->unit map; it is a DIAGNOSTIC, not a system."""
import json
import numpy as np
from sim.generator import World, WorldConfig
from neurostore.config import Config
from neurostore.store import NeuroStore
from neurostore.readout import select_assemblies, complete_pattern
from eval.protocol import ConceptMap, mask_cues, prec_mean
from eval.baselines import JaccardNN
def run(seed, mf=0.3, **kw):
    w=World(WorldConfig(),seed); tr=w.stream("train",6000); va=w.stream("val",1000)
    cfg=Config(seed=seed,homeo=False,gate=False,hebbian=False,replay_budget=200,hot_tau=0.0,**kw)
    s=NeuroStore(cfg); s.ingest(tr); s.flush(); asm=s.asm
    codes=s.log.gather(np.arange(s.log.n_stored))[0]
    vc=[s.enc.encode(va.event(i))[0] for i in range(len(va))]
    cues,masked=mask_cues(vc,mf,seed,1); ms=[len(m) for m in masked]
    cos=[asm.vote(c) for c in cues]
    res={}
    for A in (1,2,3):
        res[f"ns A={A}"]=prec_mean([complete_pattern(asm,c,select_assemblies(c,0.7,0.1,A),q,m) for c,q,m in zip(cos,cues,ms)],masked)
    # sharpened: weights cos^3
    def comp_sharp(c,q,m,p=3):
        sc=np.zeros(1024)
        sel=select_assemblies(c,0.5,0.1,3)
        for a in sel: sc[asm.member_ids[a]]+= (c[a]**p)*asm.member_w[a]
        sc[q]=0; top=np.argsort(-sc)[:m]; return top[sc[top]>0]
    res["ns sharp"]=prec_mean([comp_sharp(c,q,m) for c,q,m in zip(cos,cues,ms)],masked)
    # sibling oracle: boost partners (same raw feature) of visible cue units -- uses the encoder's hidden map, diagnosis only
    fm=s.enc.fmap; partner=[[] for _ in range(1024)]
    for f in range(fm.shape[0]):
        a,b=fm[f]; partner[a].append(b); partner[b].append(a)
    def comp_sib(c,q,m):
        sc=np.zeros(1024); sel=select_assemblies(c,0.7,0.1,2)
        for a in sel: sc[asm.member_ids[a]]+=c[a]*asm.member_w[a]
        sc/= max(sc.max(),1e-9)
        for u in q:
            for v in partner[u]: sc[v]+=0.5*sc[v]+0.3*(sc[v]>0.05)
        sc[q]=0; top=np.argsort(-sc)[:m]; return top[sc[top]>0]
    res["ns+sibling-oracle"]=prec_mean([comp_sib(c,q,m) for c,q,m in zip(cos,cues,ms)],masked)
    nn=JaccardNN().fit(codes,1024); idx,sims=nn.topk(cues)
    for k in (1,5,10):
        out=[]
        for ii,ss,q,m in zip(idx,sims,cues,ms):
            sc=np.zeros(1024)
            for j,sv in zip(ii[:k],ss[:k]): sc[codes[j]]+=sv
            sc[q]=0; top=np.argsort(-sc,kind="stable")[:m]; out.append(top[sc[top]>0])
        res[f"nn k={k}"]=prec_mean(out,masked)
    return res
out={}
for name,kw in {"default":{}, "member_tau_0.1":dict(asm_member_tau=0.1)}.items():
    r=[run(sd,**kw) for sd in (100,101,102)]
    out[name]={k:float(np.mean([x[k] for x in r])) for k in r[0]}
    print(name,{k:round(v,3) for k,v in out[name].items()})
from eval.gates import write_result
write_result("phase2_diagnosis",{"seeds":[100,101,102],"mask":0.3,"precision_at_m":out})
