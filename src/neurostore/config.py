from __future__ import annotations
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    # ---- code geometry
    F: int = 800              # raw feature id space
    N: int = 1024             # feature units
    k: int = 32               # active units per event (fixed)
    d: int = 2                # units each raw feature projects to
    seed: int = 0
    layout: str = "volley"    # spike-log physical layout
    # ---- homeostasis
    homeo: bool = True
    homeo_alpha: float = 0.02   # EMA rate of per-unit firing estimate (events)
    homeo_eta: float = 0.002    # threshold step per event
    theta_clip: float = 2.0
    # ---- prediction / surprise gate
    gate: bool = True
    gate_tau: float = 0.35      # events with surprise <= tau are logged as a count only
    strength_floor: float = 0.25
    pred_mode: str = "assembly"  # "assembly" (expect the previous event's best assembly to persist) | "lattice" (spread over synapses)
    pred_rho: float = 0.3       # expected set = predicted units with score >= rho * max
    # ---- consolidation
    consolidate: str = "batch"  # "batch" (hot replay) | "online" (learn every stored event immediately) | "off"
    consolidate_every: int = 200
    replay_budget: int = 150
    hot_tau: float = 0.3
    repeat_hot: int = 3         # an assembly matched >= this many times since last round marks the event hot
    merge_every: int = 2        # consolidation rounds between delta -> lattice merges
    # ---- synapse lattice
    hebbian: bool = True
    eta: float = 1.0
    lam: float = 0.03           # decay per epoch (epoch = one consolidation round)
    lag_window: float = 3.0
    tau_k: float = 2.0
    n_lags: int = 3
    within_event: bool = True   # also learn dt=0 co-firing edges inside one event
    prune_R: int = 128          # out-edges kept per pre unit at merge
    eps: float = 1e-3
    hops: int = 2
    hop_decay: float = 0.5
    spread_keep: int = 96
    # ---- assemblies
    assemblies: bool = True
    asm_max: int = 160
    asm_match_min: int = 5
    asm_cover_min: float = 0.2
    asm_member_tau: float = 0.2
    asm_gamma_min: float = 0.03
    asm_new_min: int = 10
    asm_prune_support: int = 3
    asm_prune_age: int = 2
    asm_merge_co: float = 0.5
    asm_claim: bool = False      # explaining-away: units claimed by another assembly are not credited
    asm_merge_min_n: int = 8
    asm_merge_cos: float = 0.7
    # ---- rates
    rate_window: float = 50.0
    rate_sub: int = 10

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)
