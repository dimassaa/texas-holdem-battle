"""Reference evaluator: known hands and category ordering."""
import pytest

from poker.card import card_id
from poker.hand_evaluator import evaluate_5_reference

H = card_id  # readability alias


class TestKnownHands:
    def test_royal_flush(self):
        cards = [H(14, 0), H(13, 0), H(12, 0), H(11, 0), H(10, 0)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 8 and kick[0] == 14 - 2

    def test_straight_flush_nine_high(self):
        cards = [H(9, 1), H(8, 1), H(7, 1), H(6, 1), H(5, 1)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 8 and kick[0] == 9 - 2

    def test_wheel_straight_flush(self):
        # A-5-4-3-2 suited: the lowest straight, high card is the 5.
        cards = [H(14, 2), H(5, 2), H(4, 2), H(3, 2), H(2, 2)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 8 and kick[0] == 5 - 2

    def test_four_of_a_kind(self):
        cards = [H(9, 0), H(9, 1), H(9, 2), H(9, 3), H(4, 0)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 7 and kick == (9 - 2, 4 - 2, 0, 0, 0)

    def test_full_house(self):
        cards = [H(7, 0), H(7, 1), H(7, 2), H(3, 0), H(3, 1)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 6 and kick == (7 - 2, 3 - 2, 0, 0, 0)

    def test_flush_ranks_descending(self):
        cards = [H(14, 0), H(10, 0), H(8, 0), H(6, 0), H(3, 0)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 5 and kick == (12, 8, 6, 4, 1)

    def test_straight_ten_high(self):
        cards = [H(10, 0), H(9, 1), H(8, 2), H(7, 0), H(6, 1)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 4 and kick[0] == 10 - 2

    def test_wheel_straight(self):
        cards = [H(14, 0), H(2, 1), H(3, 2), H(4, 0), H(5, 1)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 4 and kick[0] == 5 - 2

    def test_three_of_a_kind(self):
        cards = [H(5, 0), H(5, 1), H(5, 2), H(12, 0), H(3, 1)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 3 and kick == (3, 10, 1, 0, 0)

    def test_two_pair(self):
        cards = [H(11, 0), H(11, 1), H(8, 2), H(8, 0), H(2, 1)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 2 and kick == (9, 6, 0, 0, 0)

    def test_one_pair(self):
        cards = [H(13, 0), H(13, 1), H(9, 2), H(7, 0), H(4, 1)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 1 and kick == (11, 7, 5, 2, 0)

    def test_high_card(self):
        cards = [H(14, 0), H(11, 1), H(8, 2), H(6, 0), H(2, 1)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 0 and kick == (12, 9, 6, 4, 0)

    def test_top_pair_kicker_order(self):
        a = evaluate_5_reference([H(9, 0), H(9, 1), H(14, 2), H(7, 0), H(4, 1)])
        b = evaluate_5_reference([H(9, 0), H(9, 1), H(13, 2), H(7, 0), H(4, 1)])
        # Same pair, bigger kicker wins.
        assert (a[1] > b[1])