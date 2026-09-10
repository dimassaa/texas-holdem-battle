"""EquityProvider: cache correctness, pot odds, per-hand lifecycle."""
import numpy as np
from poker.card import card_id as H
from poker.config import Config
from poker.equity import EquityProvider
from poker.rng import Rng


def test_cache_skips_repeat_computation_with_same_result():
    prov = EquityProvider(rng=Rng(5), config=Config())
    hand = np.array([H(14, 0), H(13, 1)])
    board = np.array([H(2, 0), H(9, 3), H(7, 2)])
    a = prov.equity(hand, board, 2)
    b = prov.equity(hand, board, 2)
    assert a == b and prov._cache_hits > 0


def test_cache_key_changes_with_opponent_count():
    prov = EquityProvider(rng=Rng(6), config=Config())
    hand = np.array([H(10, 0), H(9, 1)])
    board = np.array([H(2, 0), H(3, 3), H(5, 2)])
    v1 = prov.equity(hand, board, 1)
    v2 = prov.equity(hand, board, 3)
    assert v1 != v2


def test_pot_odds_reflects_board_state():
    prov = EquityProvider(rng=Rng(7), config=Config(), pot=100, to_call=25)
    assert abs(prov.pot_odds() - 25 / (100 + 25)) < 1e-9