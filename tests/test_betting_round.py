"""Betting-round loop: staging, capping, all-in recognition, reopen rule."""
import numpy as np

from poker.actions import ALLIN, BET, CALL, CHECK, FOLD, RAISE
from poker.config import Config
from poker.game import GameState, max_raise_amount, run_betting_round, starting_bets
from poker.player import Player

CFG = Config()


def players_n(stacks):
    return [Player(name=f"p{i}", strategy="X", stack=s) for i, s in enumerate(stacks)]


def test_blinds_paid_and_round_closes_when_all_fold():
    ps = players_n([200] * 6)
    st = GameState(players=ps, dealer_pos=0, config=CFG)
    starting_bets(st)
    # Everyone folds preflop; the street must close with the blinds moved into
    # `pot` exactly once (round_bets cleared for the next street).
    actions = {i: (FOLD, 0) for i in range(6)}
    run_betting_round(st, actions_override=actions)
    assert st.pot == 3
    assert st.round_bets == {}
    assert [p.stack for p in ps] == [199, 198, 200, 200, 200, 200]


def test_raise_all_fold_matches_hands():
    ps = players_n([200] * 6)
    st = GameState(players=ps, dealer_pos=0, config=CFG)
    starting_bets(st)
    # UTG = index 2 raises pot-limit max; SB/BB call it, everyone else folds.
    raise_to = max_raise_amount(st, 2)   # (0 + 3 blinds) + 2*2 = 7
    actions = {0: (CALL, 0), 1: (CALL, 0), 2: (RAISE, raise_to), 3: (FOLD, 0),
               4: (FOLD, 0), 5: (FOLD, 0)}
    run_betting_round(st, actions_override=actions)
    # three players each put the full 7 in; the street closed into the pot
    assert st.pot == 3 * raise_to
    assert st.round_bets == {}


def test_short_allin_does_not_force_extra_turn_on_matched_players():
    ps = players_n([200, 200, 200, 200, 6, 200])
    st = GameState(players=ps, dealer_pos=0, config=CFG)
    starting_bets(st)
    # p2 raises to 7, p3 calls; p4 (6 chips) goes all-in for a *short* raise
    # that stays below the 7 open wager, so it must NOT reopen p2/p3. If the
    # loop wrongly reopens them, `actions_override` has no entry for a second
    # visit -> KeyError fails the test.
    actions = {2: (RAISE, 7), 3: (CALL, 0), 4: (ALLIN, 0), 5: (FOLD, 0),
               0: (CALL, 0), 1: (CALL, 0)}
    run_betting_round(st, actions_override=actions)
    assert st.aggro_count == 1          # ALLIN is free under the aggro cap
    assert ps[4].all_in and ps[4].stack == 0
    assert st.round_bets == {}          # street closed into the pot
    assert st.pot == 7 + 7 + 6 + 7 + 7  # p2, p3, p4(short), p0, p1
