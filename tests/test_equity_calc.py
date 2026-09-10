"""Monte Carlo equity: benchmark sanity, tie handling, determinism, caching."""
import numpy as np
import pytest

from poker.card import card_id as H
from poker.config import Config
from poker.equity import calc_equity
from poker.hand_evaluator import hand_score
from poker.rng import Rng


def test_aa_preflop_equity_near_theoretical():
    acc = [calc_equity(np.array([H(14, 0), H(14, 1)]), np.empty(0, dtype=np.int32),
                       1, Rng(1000 + i), mc_iterations=20_000) for i in range(10)]
    mean = float(np.mean(acc))
    assert 0.83 < mean < 0.87   # known AA vs random ~ 85%


def test_ak_suited_vs_random_roughly_two_thirds():
    eq = calc_equity(np.array([H(14, 0), H(13, 0)]), np.empty(0, dtype=np.int32),
                     1, Rng(44), mc_iterations=30_000)
    assert 0.63 < eq < 0.71


def test_deterministic_given_seed():
    a = calc_equity(np.array([H(8, 0), H(7, 1)]), np.array([H(14, 2), H(9, 3), H(2, 0)]),
                    2, Rng(9), mc_iterations=2000)
    b = calc_equity(np.array([H(8, 0), H(7, 1)]), np.array([H(14, 2), H(9, 3), H(2, 0)]),
                    2, Rng(9), mc_iterations=2000)
    assert a == b


def test_tie_handling_is_fractional():
    # A 6-high straight-flush board plays itself: no hero hand can beat it, so
    # BOTH hero and every opponent tie with the board on every simulation.
    # Equity must be ~0.5, exercising the 0.5-tie branch for every opponent.
    board = np.array([H(2, 0), H(3, 0), H(4, 0), H(5, 0), H(6, 0)]).astype(np.int32)
    hero = np.array([H(12, 0), H(11, 0)], dtype=np.int32)   # Qc Jc cannot top it
    eq = calc_equity(hero, board, 1, Rng(1), mc_iterations=500)
    assert 0.45 < eq < 0.55


def test_dead_cards_are_excluded_from_the_random_pool():
    from poker.equity import _cards_left
    hand = np.array([H(8, 0), H(14, 1)])               # hero holds 8c, As
    board = np.array([H(2, 0), H(9, 3), H(7, 2)])      # board holds 2c, 9s, 7h
    pool = _cards_left(hand, board)
    used = set(hand.tolist() + board.tolist())
    assert len(pool) == 52 - 5
    assert not (set(pool.tolist()) & used)             # no known card can be dealt


def test_helper_pool_is_topologically_correct_for_draws():
    # every sampled opponent/runout card must come from the pool
    from poker.equity import _cards_left, deal_permutations
    hand = np.array([H(8, 0), H(14, 1)])
    board = np.array([H(2, 0), H(9, 3), H(7, 2)])
    pool = _cards_left(hand, board)
    drawn = deal_permutations(Rng(0), pool, 200, 5)
    assert set(drawn.ravel().tolist()).issubset(set(pool.tolist()))
