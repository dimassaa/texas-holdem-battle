"""Card encoding and deck integrity invariants."""
import numpy as np
import pytest

from poker.card import (
    card_id, new_deck, rank_of, shuffle_deck, suit_of,
)
from poker.rng import Rng


class TestCardId:
    def test_encoding_roundtrip_across_all_52_cards(self):
        for rank in range(2, 15):
            for suit in range(4):
                cid = card_id(rank, suit)
                assert 0 <= cid < 52
                assert rank_of(cid) == rank
                assert suit_of(cid) == suit

    def test_encoding_is_bijective(self):
        ids = [card_id(rank, suit) for rank in range(2, 15) for suit in range(4)]
        assert sorted(ids) == list(range(52))

    def test_ace_is_14(self):
        assert rank_of(card_id(14, 0)) == 14


class TestDeck:
    def test_new_deck_is_full_and_unique(self):
        deck = new_deck()
        assert len(deck) == 52
        assert len(set(deck.tolist())) == 52

    def test_shuffle_preserves_card_multiset(self):
        deck = new_deck()
        shuffled = shuffle_deck(Rng(123), deck)
        assert sorted(shuffled.tolist()) == sorted(deck.tolist())

    def test_shuffle_does_not_mutate_input(self):
        deck = new_deck()
        original = deck.copy()
        shuffle_deck(Rng(1), deck)
        assert np.array_equal(deck, original)

    def test_shuffle_is_deterministic_per_seed(self):
        a = shuffle_deck(Rng(5), new_deck())
        b = shuffle_deck(Rng(5), new_deck())
        assert np.array_equal(a, b)
        c = shuffle_deck(Rng(6), new_deck())
        assert not np.array_equal(a, c)