"""Assemblies: recurring sets of co-firing units standing for one concept.

Learning (during replay) is online competitive clustering with explaining-away:
  an event's units are matched to the best existing assembly (overlap >= asm_match_min and
  covering >= asm_cover_min of its members); that assembly's per-unit firing frequency C[a,u]
  (an EMA of P(u fires | assembly matched)) is updated; the matched members are removed and
  the residual is matched again (up to 3 explanations). A residual of >= asm_new_min units that
  nothing explains seeds a new assembly.
Members = units with C >= asm_member_tau (capped at 2k). Read side: inverted index
unit -> (assembly, weight) in CSR form; voting = sparse accumulate over the cue's units.
"""
from __future__ import annotations
import numpy as np
from .util import gather_ranges


class AssemblyIndex:
    def __init__(self, cfg):
        self.cfg = cfg
        A, N = cfg.asm_max, cfg.N
        self.A, self.N = A, N
        self.C = np.zeros((A, N), np.float32)
        self.M = np.zeros((A, N), bool)
        self.n = np.zeros(A, np.int64)
        self.alive = np.zeros(A, bool)
        self.born = np.zeros(A, np.int64)
        self.msize = np.zeros(A, np.int64)
        self.wmatch = np.zeros(A, np.int64)        # write-path matches since last round (hotness)
        self.co = np.zeros((A, A), np.float32)     # co-explanation counts (fragments of one concept co-fire)
        self.round = 0
        self._build_index()

    # ------------------------------------------------------------ learning
    def _refresh(self, a):
        row = self.C[a]
        m = row >= self.cfg.asm_member_tau
        cap = 2 * self.cfg.k
        if m.sum() > cap:
            thr = np.partition(row, -cap)[-cap]
            m = row >= max(thr, self.cfg.asm_member_tau)
        self.M[a] = m
        self.msize[a] = int(m.sum())

    def _credit(self, a, R):
        g = max(self.cfg.asm_gamma_min, 1.0 / (self.n[a] + 1))
        claimed = self.M[:, R].sum(0) - self.M[a, R]          # members of OTHER assemblies
        credit = R[(claimed == 0) | self.M[a, R]] if self.cfg.asm_claim else R
        self.C[a] *= (1.0 - g)
        self.C[a, credit] += g
        self.n[a] += 1
        self._refresh(a)

    def _new(self, R):
        free = np.flatnonzero(~self.alive)
        if len(free) == 0:
            return None
        a = int(free[0])
        self.C[a] = 0.0; self.C[a, R] = 1.0
        self.n[a] = 1; self.alive[a] = True; self.born[a] = self.round
        self._refresh(a)
        return a

    def learn(self, units) -> list:
        """One replayed event. Returns the assemblies that explained it (incl. a newly seeded one)."""
        cfg = self.cfg
        R = np.asarray(units, np.int64)
        out = []
        for _ in range(3):
            if len(R) < cfg.asm_match_min:
                break
            ov = np.where(self.alive, self.M[:, R].sum(1), 0)
            ms = np.maximum(self.msize, 1)
            ok = (ov >= cfg.asm_match_min) & (ov / ms >= cfg.asm_cover_min)
            if not ok.any():
                break
            a = int(np.where(ok, ov / np.sqrt(ms), -1.0).argmax())
            self._credit(a, R)
            out.append(a)
            R = R[~self.M[a, R]]
        if len(R) >= cfg.asm_new_min:
            a = self._new(R)
            if a is not None:
                out.append(a)
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                self.co[out[i], out[j]] += 1; self.co[out[j], out[i]] += 1
        return out

    def _kill(self, a):
        self.alive[a] = False; self.M[a] = False; self.C[a] = 0; self.n[a] = 0; self.msize[a] = 0
        self.co[a] = 0; self.co[:, a] = 0

    def maintain(self):
        """End of a consolidation round: prune unsupported assemblies, merge near-duplicates, reindex."""
        cfg = self.cfg
        self.round += 1
        for a in np.flatnonzero(self.alive & (self.round - self.born >= cfg.asm_prune_age) & (self.n < cfg.asm_prune_support)):
            self._kill(int(a))
        # merge fragments: (i) assemblies that explain the same events together (co-match ratio),
        # (ii) near-duplicate weight vectors (cosine). Merged weights = elementwise max.
        idx = np.flatnonzero(self.alive)
        if len(idx) > 1:
            W = np.where(self.M[idx], self.C[idx], 0.0)
            nr = np.linalg.norm(W, axis=1) + 1e-9
            cos = (W @ W.T) / np.outer(nr, nr)
            dead = set()
            for i in range(len(idx)):
                if i in dead:
                    continue
                for j in range(i + 1, len(idx)):
                    if j in dead:
                        continue
                    a, b = int(idx[i]), int(idx[j])
                    mn = min(self.n[a], self.n[b])
                    comatch = mn >= cfg.asm_merge_min_n and self.co[a, b] / mn >= cfg.asm_merge_co
                    if comatch or cos[i, j] >= cfg.asm_merge_cos:
                        self.C[a] = np.maximum(self.C[a], self.C[b])
                        self.n[a] = max(self.n[a], self.n[b])
                        self.co[a] += self.co[b]; self.co[:, a] += self.co[:, b]; self.co[a, a] = 0
                        self._refresh(a); self._kill(b); dead.add(j)
        self.wmatch[:] = 0
        self._build_index()

    # ------------------------------------------------------------ read side
    def _build_index(self):
        W = np.where(self.M, self.C, 0.0).astype(np.float32)
        self.W = W
        a_idx, u_idx = np.nonzero(W)                         # row-major over (a, u)
        order = np.argsort(u_idx, kind="stable")
        u_sorted, a_sorted = u_idx[order], a_idx[order]
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(u_sorted, minlength=self.N))]).astype(np.int32)
        self.aid = a_sorted.astype(np.int16)
        self.wt = W[a_sorted, u_sorted].astype(np.float32)
        self.norm = np.sqrt((W ** 2).sum(1)).astype(np.float32)
        self.member_ids = [np.flatnonzero(self.M[a]).astype(np.int32) for a in range(self.A)]
        self.member_w = [W[a, self.member_ids[a]] for a in range(self.A)]

    @property
    def n_alive(self) -> int:
        return int(self.alive.sum())

    def index_bytes(self) -> int:
        return int(self.indptr.nbytes + self.aid.nbytes + self.wt.nbytes + self.norm.nbytes)

    def vote(self, units, x=None) -> np.ndarray:
        """Population vote: cosine-like match of the cue against every assembly (sparse accumulate)."""
        units = np.asarray(units, np.int64)
        offs, which = gather_ranges(self.indptr, units)
        w = self.wt[offs] if x is None else self.wt[offs] * np.asarray(x, np.float32)[which]
        raw = np.bincount(self.aid[offs], weights=w, minlength=self.A)
        cn = np.sqrt(len(units) if x is None else float((np.asarray(x) ** 2).sum()))
        return (raw / (np.where(self.norm > 0, self.norm, 1.0) * max(cn, 1e-9))).astype(np.float32)

    def vote_dense(self, units, x=None) -> np.ndarray:
        """Reference implementation of vote (used by tests)."""
        v = np.zeros(self.N, np.float32); v[np.asarray(units)] = 1.0 if x is None else x
        raw = self.W @ v
        cn = np.sqrt((v ** 2).sum())
        return (raw / (np.where(self.norm > 0, self.norm, 1.0) * max(cn, 1e-9))).astype(np.float32)
