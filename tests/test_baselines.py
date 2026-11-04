import numpy as np
from eval.baselines import ExactMatch, JaccardNN
from eval.protocol import mask_cues


def test_jaccard_nn_matches_bruteforce_and_exact_match_semantics():
    rng = np.random.default_rng(0)
    codes = np.array([np.sort(rng.choice(200, 16, replace=False)) for _ in range(300)])
    nn = JaccardNN().fit(codes, 200)
    cues = [codes[i][:10] for i in (3, 50, 120)]
    idx, sims = nn.topk(cues, K=5)
    for q, ii, ss in zip(cues, idx, sims):
        js = np.array([len(np.intersect1d(q, c)) / len(np.union1d(q, c)) for c in codes])
        assert np.allclose(np.sort(js)[::-1][:5], ss, atol=1e-6)
    ex = ExactMatch().fit(codes, [(i % 5,) for i in range(300)], 200)
    pred, comp, hit = ex.query(codes[7], 4)
    assert hit and pred == (7 % 5,)
    assert not ex.query(codes[7][:10], 4)[2]          # a partial cue is a different code


def test_mask_cues_partition_and_determinism():
    codes = [np.arange(32) * 3 for _ in range(5)]
    c1, m1 = mask_cues(codes, 0.5, 7, 1); c2, m2 = mask_cues(codes, 0.5, 7, 1)
    for a, b, x, y in zip(c1, m1, c2, m2):
        assert np.array_equal(a, x) and np.array_equal(b, y)
        assert len(a) == 16 and len(b) == 16 and not set(a) & set(b) and set(a) | set(b) == set(codes[0])
