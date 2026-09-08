"""Property tests: category distribution across random hands, plus a `slow`
exhaustive 5-card sweep. Guard against systematic scorer bias."""
import itertools

import numpy as np
import pytest

from poker.card import card_id
from poker.hand_evaluator import CATEGORY_NAMES, score_5
from poker.rng import Rng

# True combinatorial counts of each 5-card category from a 52-card deck.
EXPECTED_COUNTS = [
    1302540,   # 0 high card
    1098240,   # 1 one pair
    123552,    # 2 two pair
    54912,     # 3 three of a kind
    10200,     # 4 straight
    5108,      # 5 flush
    3744,      # 6 full house
    624,       # 7 four of a kind
    40,        # 8 straight flush
]


def test_random_sample_categories_are_consistent():
    rng = Rng(11)
    seen = np.zeros(9)
    for _ in range(100_000):
        cards = np.array(rng.generator.choice(52, 5, replace=False), dtype=np.int32)
        cat = score_5(cards) // 13 ** 5
        seen[cat] += 1
    assert np.all(seen > 0)  # every category appears in practice


@pytest.mark.slow
def test_exhaustive_5_card_category_histogram():
    counts = {n: 0 for n in range(9)}
    for combo in itertools.combinations(range(52), 5):
        cards = np.array(combo, dtype=np.int32)
        cat = score_5(cards) // 13 ** 5
        counts[cat] += 1
    for cat in range(9):
        assert counts[cat] == EXPECTED_COUNTS[cat], (
            f"category {cat} ({CATEGORY_NAMES[cat]}): "
            f"got {counts[cat]}, expected {EXPECTED_COUNTS[cat]}"
        )