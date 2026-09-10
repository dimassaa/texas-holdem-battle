"""Deterministic vectorized sampling without replacement."""
import numpy as np
from poker.equity import deal_permutations
from poker.rng import Rng


def test_sampler_returns_distinct_cards_per_row():
    rng = Rng(1)
    pool = np.arange(52, dtype=np.int32)
    out = deal_permutations(rng, pool, n=500, k=6)
    assert out.shape == (500, 6)
    assert all(len(set(row.tolist())) == 6 for row in out)


def test_sampler_never_uses_pool_exclusions():
    rng = Rng(2)
    pool = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    out = deal_permutations(rng, pool, n=100, k=3)
    assert set(out.ravel().tolist()).issubset({0, 1, 2, 3, 4, 5})


def test_sampler_is_deterministic():
    pool = np.arange(52, dtype=np.int32)
    a = deal_permutations(Rng(3), pool, 100, 5)
    b = deal_permutations(Rng(3), pool, 100, 5)
    assert np.array_equal(a, b)
