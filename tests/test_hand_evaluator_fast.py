"""Fast encoded scorer: monotonicity, best-of-7, batch, and agreement with the
reference evaluator across random hands."""
import numpy as np
import pytest

from poker.hand_evaluator import (
    evaluate_5_reference, hand_score, score_5, score_batch,
)
from poker.rng import Rng

B13 = 13 ** 5  # weight of the category digit


def encode_reference(cards):
    """Independent encoding straight from the reference oracle."""
    cat, kick = evaluate_5_reference(cards)
    score = cat * B13
    for i, k in enumerate(kick):
        score += k * 13 ** (4 - i)
    return score


class TestScore5:
    def test_score_matches_reference(self):
        rng = Rng(99)
        for _ in range(20_000):
            cards = rng.generator.choice(52, 5, replace=False)
            assert score_5(cards) == encode_reference(cards), cards

    def test_royal_flush_beats_all_lower_categories(self):
        from poker.card import card_id as H
        royal = [H(14, 0), H(13, 0), H(12, 0), H(11, 0), H(10, 0)]
        best_lower = [H(14, 1), H(14, 2), H(14, 3), H(13, 2), H(13, 3)]
        assert score_5(royal) > score_5(best_lower)

    def test_flush_and_high_card_scores_are_order_invariant(self):
        from poker.card import card_id as H
        # The same cards must score identically regardless of input order —
        # regression guard for the flush/high-card kicker ordering bug.
        flush = [H(14, 0), H(10, 0), H(8, 0), H(6, 0), H(3, 0)]
        assert score_5(flush) == score_5(flush[::-1])
        high = [H(14, 0), H(11, 1), H(8, 2), H(6, 0), H(2, 1)]
        assert score_5(high) == score_5(high[::-1])


class TestHandScore:
    def test_picks_best_five_of_seven(self):
        from poker.card import card_id as H
        # Board has a diamond flush; hero holds Qd for the nut flush over a
        # lower two pair / straight competitor within the same 7 cards.
        seven = [H(14, 0), H(13, 0), H(11, 0), H(9, 0), H(6, 0), H(2, 0), H(10, 1)]
        # Best 5 is the flush A-K-J-9-6 of diamonds.
        expected = encode_reference([H(14, 0), H(13, 0), H(11, 0), H(9, 0), H(6, 0)])
        assert hand_score(seven) == expected

    def test_hand_score_agrees_with_reference_best_of_21(self):
        import itertools
        from poker.card import card_id as H
        rng = Rng(7)
        for _ in range(5_000):
            seven = np.array(rng.generator.choice(52, 7, replace=False), dtype=np.int32)
            best = max(
                encode_reference(list(combo))
                for combo in itertools.combinations(seven.tolist(), 5)
            )
            assert hand_score(seven) == best


class TestScoreBatch:
    def test_batch_matches_scalar(self):
        rng = Rng(3)
        hands = np.array([rng.generator.choice(52, 7, replace=False) for _ in range(50)],
                         dtype=np.int32)
        scalar = np.array([hand_score(h) for h in hands])
        assert np.array_equal(score_batch(hands), scalar)
