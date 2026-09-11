"""Stage-03 gate: Monte Carlo approximates the exact combinatorial answer."""
import itertools
import numpy as np
import pytest

from poker.card import card_id as H, rank_of, suit_of
from poker.equity import calc_equity
from poker.hand_evaluator import hand_score
from poker.rng import Rng


def exact_postflop_equity(hand, board, n_opp):
    """Reference: enumerate every remaining-board and every opponent hand.

    Only tractable for n_opp <= 2 and short runouts. Ties are fractional.
    """
    used = set(map(int, list(hand) + list(board)))
    remaining = [c for c in range(52) if c not in used]
    wins = ties = total = 0
    board_len = len(board)
    for runout in itertools.permutations(remaining, 5 - board_len):
        if n_opp == 1:
            opp_combos = itertools.combinations(
                [c for c in remaining if c not in runout], 2)
            for opp in opp_combos:
                total += 1
                hero = hand_score(np.array([*hand, *board, *runout], dtype=np.int32))
                foe = hand_score(np.array([*opp, *board, *runout], dtype=np.int32))
                if hero > foe: wins += 1
                elif hero == foe: ties += 1
        elif n_opp == 2:
            pool = [c for c in remaining if c not in runout]
            for opps in itertools.combinations(pool, 4):
                total += 1
                hero = hand_score(np.array([*hand, *board, *runout], dtype=np.int32))
                foe = max(
                    hand_score(np.array([*opps[:2], *board, *runout], dtype=np.int32)),
                    hand_score(np.array([*opps[2:], *board, *runout], dtype=np.int32)),
                )
                if hero > foe: wins += 1
                elif hero == foe: ties += 1
        else:
            raise ValueError("exact reference supports <= 2 opponents")
    return (wins + 0.5 * ties) / total


@pytest.mark.slow
def test_mc_matches_exact_on_flop():
    hand = np.array([H(11, 0), H(10, 1)])
    board = np.array([H(2, 0), H(9, 3), H(7, 2)])
    exact = exact_postflop_equity(hand, board, 1)
    mc = calc_equity(hand, board, 1, Rng(1), 50_000)
    assert abs(exact - mc) < 0.02