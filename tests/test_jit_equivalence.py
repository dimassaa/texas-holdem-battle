"""numba fast path must be byte-identical to the pure-Python path."""
import numpy as np
import pytest

from poker.config import Config
from poker.equity import calc_equity, calc_equity_jit
from poker.hand_evaluator import hand_score, score_batch
from poker.rng import Rng

numba = pytest.importorskip("numba")

from poker.jit import score_batch_jit  # noqa: E402


def test_jit_and_python_scores_identical_over_random_hands():
    rng = Rng(77)
    hands = np.array([rng.generator.choice(52, 7, replace=False) for _ in range(5000)],
                     dtype=np.int32)
    assert np.array_equal(score_batch(hands), score_batch_jit(hands))


def test_jit_calc_equity_matches_python():
    from poker.card import card_id as H
    hand = np.array([H(11, 0), H(10, 1)])
    board = np.array([H(2, 0), H(9, 3), H(7, 1), H(5, 2)])
    a = calc_equity(hand, board, 3, Rng(1), 400)
    b = calc_equity_jit(hand, board, 3, Rng(1), 400)
    assert a == b