"""Stage 01 gate: seeded pipeline reproducibility + hand-strength ordering."""
import numpy as np
from poker.card import card_id as H, new_deck, shuffle_deck
from poker.hand_evaluator import hand_score
from poker.rng import Rng


def test_fully_seeded_hand_pipeline_is_reproducible():
    deck_a = shuffle_deck(Rng(2024), new_deck())
    deck_b = shuffle_deck(Rng(2024), new_deck())
    assert np.array_equal(deck_a, deck_b)
    board = deck_a[:5]
    assert np.all(board < 52)
    # hole + board draw from the same shuffled deck are all distinct
    assert len(set(board.tolist())) == 5


def test_hand_strength_ordering_holds_on_a_real_board():
    # A made hand (two pair, nines and sevens) outranks a bare straight draw
    # fragment on the *same* board + runout — exercises the monotonic score.
    board = np.array([H(9, 0), H(7, 1), H(3, 2), H(2, 0), H(14, 2)], dtype=np.int32)
    two_pair = hand_score(np.array([H(9, 1), H(7, 2), *board.tolist()], dtype=np.int32))
    one_pair = hand_score(np.array([H(9, 2), H(5, 3), *board.tolist()], dtype=np.int32))
    assert two_pair > one_pair
    # pocket rockets on this board (top pair + nut kicker) beat the same
    # board under a middling pocket pair
    with_aces = hand_score(np.array([H(14, 0), H(14, 1), *board.tolist()], dtype=np.int32))
    with_tens = hand_score(np.array([H(10, 0), H(10, 1), *board.tolist()], dtype=np.int32))
    assert with_aces > with_tens
