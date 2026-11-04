"""Fake C-MAPSS-format files (same 26-column layout, constant / low-resolution / informative sensors, optional 6 operating regimes).
Used by the tests and by the study runner's smoke mode so the full pipeline can be exercised without touching the real data."""
from __future__ import annotations
from pathlib import Path
import numpy as np

OPS = np.array([[0, 0, 100], [10, 0.25, 100], [20, 0.7, 100], [25, 0.62, 60], [35, 0.84, 100], [42, 0.84, 100]], float)


def fake_table(rng, n_eng, lens, regimes, shift=0.0):
    rows = []
    for u in range(1, n_eng + 1):
        L = int(lens[u - 1])
        t = np.arange(1, L + 1)
        reg = rng.integers(0, regimes, L)
        ops = OPS[reg] + rng.normal(0, 0.002, (L, 3)) if regimes > 1 else np.column_stack([rng.normal(0, 0.002, L), rng.normal(0, 0.0003, L), np.full(L, 100.0)])
        S = np.zeros((L, 21))
        for j in range(21):
            if j in (0, 4, 9):                         # constant sensors (s1, s5, s10)
                S[:, j] = 100.0 + j
            elif j == 5:                               # s6: two distinct values
                S[:, j] = 21.6 + 0.01 * (rng.random(L) > 0.5)
            elif j == 3:                               # s4: only 5 distinct values
                S[:, j] = rng.integers(0, 5, L) * 0.1 + 1.0
            else:
                off = 50.0 * reg if regimes > 1 else 0.0
                S[:, j] = j + shift + off + (t / L) ** 2 * (1 + 0.3 * j) + 0.05 * rng.normal(size=L)
        rows.append(np.column_stack([np.full(L, u), t, ops, S]))
    return np.concatenate(rows)


def write_fake(root, fd="FD001", n_train=7, n_test=5, regimes=1, seed=0, test_shift=0.0, life=(40, 70)):
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train = fake_table(rng, n_train, rng.integers(life[0], life[1], n_train), regimes)
    tl = rng.integers(10, life[1] - 10, n_test)                                  # truncated test engines (some shorter than the window)
    test = fake_table(rng, n_test, tl, regimes, shift=test_shift)
    rul = rng.integers(5, 140, n_test)
    for name, A in (("train", train), ("test", test)):
        np.savetxt(root / f"{name}_{fd}.txt", A, fmt="%.6f")
    np.savetxt(root / f"RUL_{fd}.txt", rul, fmt="%d")
    return train, test, rul


def write_fake_dataset(root, n_train=30, n_test=12, seed=0):
    """All four sub-datasets (FD002/FD004 with six regimes)."""
    for i, (fd, reg) in enumerate([("FD001", 1), ("FD002", 6), ("FD003", 1), ("FD004", 6)]):
        write_fake(root, fd, n_train=n_train, n_test=n_test, regimes=reg, seed=seed + i, life=(60, 120))
