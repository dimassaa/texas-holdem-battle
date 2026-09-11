"""Strategy registry, positions, and range-combo accounting."""
import numpy as np
from poker.equity import hand_type_index
from poker.player import STRATEGIES, RANGES, combo_weight, relative_position, for_name
from poker.config import Config
from poker.game import GameState
from poker.player import Player


def test_registry_has_four_basic_strategies():
    assert {"Tight", "Loose", "Aggressive", "Passive"} <= set(STRATEGIES)
    assert for_name("Tight").name == "Tight"


def test_ranges_contract_shape():
    # Locked contract (stage04.md:19): RANGES[strategy] is a frozenset of
    # hand-type indices, one entry per registered strategy. Stage 05 consumes
    # exactly this — a shape break here is a contract break there.
    assert set(RANGES) == set(STRATEGIES)
    for name, rng in RANGES.items():
        assert isinstance(rng, frozenset)
        assert all(isinstance(i, int) and 0 <= i < 1820 for i in rng)
    # Tight = narrowest, Loose = widest (audited percentages: 14.5% vs 39.7%).
    assert len(RANGES["Tight"]) < len(RANGES["Loose"])


def test_combo_weights_sum_to_1326():
    total = sum(combo_weight(hand_type_index(r_a, r_b, s))
                for r_a in range(2, 15)
                for r_b in range(2, r_a + 1)
                for s in (0, 1) if not (r_a == r_b and s == 1))
    assert total == 1326


def test_relative_positions_are_deterministic():
    ps = [Player(name=f"p{i}", strategy="Tight", stack=200) for i in range(6)]
    state = GameState(players=ps, dealer_pos=0, config=Config())
    assert relative_position(state, 0) == "late"      # BTN
    assert relative_position(state, 1) == "blinds"    # SB
    assert relative_position(state, 2) == "blinds"    # BB
    assert relative_position(state, 3) == "early"     # UTG
    assert relative_position(state, 4) == "early"     # HJ
    assert relative_position(state, 5) == "middle"    # CO