"""GameState construction, query helpers, and pot-limit raise formula."""
import pytest

from poker.config import Config
from poker.game import GameState, max_raise_amount, starting_bets
from poker.player import Player


def make_players(n=6, stack=200):
    return [Player(name=f"p{i}", strategy="Tight", stack=stack) for i in range(n)]


def test_starting_bets_posts_blinds_and_sets_round_state():
    players = make_players()
    state = GameState(players=players, dealer_pos=3, config=Config())
    starting_bets(state)
    bb_idx = (state.dealer_pos + 1) % 6
    sb_idx = state.dealer_pos
    assert state.round_bets[sb_idx] == 1
    assert state.round_bets[bb_idx] == 2
    # preflop "pot" is 0 until the street closes; blinds live in round_bets so
    # adding them a second time at street finalization is impossible.
    assert state.pot == 0
    assert sum(state.round_bets.values()) == 3
    # each stack reduced by the posted blind
    assert players[sb_idx].stack == 199
    assert players[bb_idx].stack == 198


def test_pot_limit_max_raise_matches_doc_formula():
    players = make_players()
    state = GameState(players=players, dealer_pos=3, config=Config())
    starting_bets(state)
    # no further action yet: blinds total 3 in the round, UTG must call 2.
    # Doc formula: MaxRaise = pot + 2 * call_amount = (0 + 3) + 2*2 = 7.
    utg_idx = (state.dealer_pos + 2) % 6
    assert max_raise_amount(state, utg_idx) == 7
    assert state.to_call(utg_idx) == 2
    assert state.can_raise(utg_idx) is True


def test_aggro_cap_enforced_at_config_boundary():
    players = make_players()
    state = GameState(players=players, dealer_pos=3, config=Config())
    starting_bets(state)
    state.aggro_count = state.config.max_aggressions
    assert state.can_raise(3) is False
