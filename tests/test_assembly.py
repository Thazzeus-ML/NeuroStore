import numpy as np
from hypothesis import given, strategies as st
from neurostore.config import Config
from neurostore.assembly import AssemblyIndex
from neurostore.util import gather_ranges
from neurostore.readout import select_assemblies, complete_pattern


@given(rows=st.lists(st.lists(st.integers(0, 40), max_size=6), min_size=1, max_size=15), q=st.lists(st.integers(0, 14), max_size=8))
def test_gather_ranges(rows, q):
    lens = [len(r) for r in rows]
    indptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    flat = np.array([x for r in rows for x in r], np.int64)
    q = np.array([x for x in q if x < len(rows)], np.int64)
    offs, which = gather_ranges(indptr, q)
    exp = [x for i in q for x in rows[i]]
    assert list(flat[offs]) == exp and list(which) == [j for j, i in enumerate(q) for _ in rows[i]]


def _two_concepts(seed=0, n=60):
    rng = np.random.default_rng(seed); cfg = Config(asm_max=20)
    A, B = np.arange(0, 40), np.arange(100, 140)
    asm = AssemblyIndex(cfg)
    for i in range(n):
        base = A if i % 2 == 0 else B
        asm.learn(np.sort(rng.choice(base, 32, replace=False)))
    asm.maintain()
    return asm, A, B, rng


def test_learns_two_separate_assemblies_and_votes_correctly():
    asm, A, B, rng = _two_concepts()
    assert asm.n_alive == 2
    for base, other in ((A, B), (B, A)):
        cue = np.sort(rng.choice(base, 8, replace=False))
        cos = asm.vote(cue)
        best = int(cos.argmax())
        assert set(asm.member_ids[best]) <= set(base) | set() or len(set(asm.member_ids[best]) & set(base)) > 30
        assert cos[best] > 3 * np.sort(cos)[-2]


def test_vote_matches_dense_reference():
    asm, A, B, rng = _two_concepts(1)
    for _ in range(10):
        cue = np.sort(rng.choice(1024, 12, replace=False))
        assert np.allclose(asm.vote(cue), asm.vote_dense(cue), atol=1e-5)
        x = rng.random(12).astype(np.float32)
        assert np.allclose(asm.vote(cue, x), asm.vote_dense(cue, x), atol=1e-5)


def test_completion_recovers_masked_members():
    asm, A, B, rng = _two_concepts(2)
    cue = np.sort(rng.choice(A, 10, replace=False)); cos = asm.vote(cue)
    sel = select_assemblies(cos, 0.7, 0.05)
    comp = complete_pattern(asm, cos, sel, cue, 10)
    assert len(comp) == 10 and set(comp) <= set(A) and not (set(comp) & set(cue))


def test_replaying_same_event_does_not_spawn_duplicates():
    cfg = Config(asm_max=20); asm = AssemblyIndex(cfg)
    u = np.arange(32)
    for _ in range(20):
        asm.learn(u)
    asm.maintain()
    assert asm.n_alive == 1


def test_unrelated_noise_events_get_pruned():
    cfg = Config(asm_max=60); asm = AssemblyIndex(cfg); rng = np.random.default_rng(0)
    for _ in range(40):
        asm.learn(np.sort(rng.choice(1024, 32, replace=False)))
    asm.maintain(); asm.maintain(); asm.maintain()
    assert asm.n_alive == 0      # singletons never recur -> pruned
