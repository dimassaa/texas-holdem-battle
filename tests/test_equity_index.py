"""Canonical 169 start-hand classification."""
import numpy as np
from poker.card import rank_of, suit_of
from poker.equity import hand_type_index
from poker.rng import Rng


def test_all_169_types_are_covered_exactly_once():
    seen = set()
    for r_a in range(2, 15):
        for r_b in range(2, 15):
            for suited in (0, 1):
                if r_a == r_b and suited:
                    continue  # a pocket pair is neither suited nor offsuit
                seen.add(hand_type_index(r_a, r_b, suited))
    assert len(seen) == 169
    assert min(seen) == 0 and max(seen) == 168


def test_pocket_pair_maps_expected_cell():
    assert hand_type_index(14, 14, 0) == hand_type_index(14, 14, 1)
    assert hand_type_index(6, 6, 0) == hand_type_index(6, 6, 1)


def test_suited_and_offsuit_are_distinct():
    assert hand_type_index(9, 5, 1) != hand_type_index(9, 5, 0)


def test_rank_order_is_symmetric_and_canonical():
    # A2o and 2Ao are the same run-down hand type.
    assert hand_type_index(14, 2, 0) == hand_type_index(2, 14, 0)


def test_equity_table_shape_from_build_path():
    # sanity: the index composition supports a (169, 5) table.
    types = np.zeros(169, dtype=np.int32)
    for r_a in range(2, 15):
        for r_b in range(2, r_a + 1):
            for suited in (0, 1):
                if r_a == r_b and suited:
                    continue
                types[hand_type_index(r_a, r_b, suited)] += 1
    assert np.count_nonzero(types) == 169
