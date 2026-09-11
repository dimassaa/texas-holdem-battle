"""Preflop table: shape, determinism, monotonicity, benchmark spot-checks."""
import numpy as np
import pytest

from poker.config import Config
from poker.equity import build_preflop_table
from poker.rng import Rng


def test_table_shape_is_169x5():
    table = build_preflop_table(Rng(1), iterations=2000)
    assert table.shape == (169, 5)


def test_table_is_deterministic():
    a = build_preflop_table(Rng(2), iterations=3000)
    b = build_preflop_table(Rng(2), iterations=3000)
    assert np.array_equal(a, b)


def test_pocket_aces_are_top_equity_for_every_opponent_count():
    table = build_preflop_table(Rng(3), iterations=2000)
    from poker.equity import hand_type_index
    aa = hand_type_index(14, 14, 0)
    for opp in range(5):
        assert table[aa, opp] == table.max(axis=0)[opp]