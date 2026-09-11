"""Aggressive: expand/raise ranges, c-bets, equity-raise logic."""
import numpy as np
from poker.actions import BET, CALL, CHECK, FOLD, RAISE
from poker.card import card_id as H
from poker.config import Config
from poker.game import GameState, starting_bets
from poker.player import AggressiveStrategy, Player
from poker.rng import Rng

CFG = Config()


class StubProvider:
    def __init__(self, eq, pot=100, to_call=0): self.eq, self.pot, self.to_call = eq, pot, to_call
    def equity(self, hand, board, n_opp): return self.eq
    def pot_odds(self): return self.to_call / (self.pot + self.to_call)


def make(strategy="Aggressive", hand=(14, 13), board=(), round_bets=None, pot=100, dealer=0):
    ps = [Player(name=f"p{i}", strategy=strategy, stack=200) for i in range(6)]
    st = GameState(players=ps, dealer_pos=dealer, config=CFG)
    st.board = np.array([H(*c) for c in board], dtype=np.int32) if board else np.empty(0, dtype=np.int32)
    st.round_idx = 1 if board else 0
    st.round_bets = dict(round_bets or {})
    st.config = CFG
    st.pot = pot
    for i, p in enumerate(ps):
        p.hole = np.array([H(hand[0], i % 4), H(hand[1], (i + 1) % 4)], dtype=np.int32)
    return st, st.players[3]


def test_aggressive_raises_ak_preflop():
    st, p = make(hand=(14, 13), board=(), round_bets={}, pot=0)
    st.pot = 3
    st.round_bets = {1: 1, 2: 2}
    prov = StubProvider(0.65, pot=3, to_call=2)
    a = AggressiveStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind in (RAISE, CALL)


def test_aggressive_cbets_as_preflop_aggressor_even_weakish_flop():
    st, p = make(hand=(9, 8), board=((13, 0), (2, 1), (5, 2)), round_bets={}, pot=100)
    st.preflop_raise_by = 3          # hero was the preflop aggressor
    st.pot = 100                     # no bet this street yet
    prov = StubProvider(0.2, pot=100, to_call=0)
    a = AggressiveStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind in (BET, RAISE)    # documented semi-bluff continuation bet


def test_aggressive_folds_weak_hand_to_big_bet_when_unraised():
    # not the aggressor, tiny equity, big bet -> fold (rarely-call bias)
    st, p = make(hand=(3, 2), board=((13, 0), (2, 1), (5, 2)), round_bets={0: 150}, pot=200)
    prov = StubProvider(0.15, pot=350, to_call=150)
    a = AggressiveStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind == FOLD


def test_aggressive_bb_reraise_meets_increment():
    # User-approved engine-legality TASK 4.5 fix: a re-raise from a seat with
    # chips already committed (the BB) must size high enough that gross ==
    # amount - own_contribution still clears the street increment. This is the
    # actual crash geometry the earlier to_call+bb formula missed: the open is
    # already at 8 (4bb), the BB has 2 committed, so a fixed 4bb re-raise of 8
    # yields gross 6 < last_full_raise 8 -> engine AssertionError. raise_size
    # must floor the wager at own + to_call + last_full_raise = 16.
    ps = [Player(name=f"p{i}", strategy="Aggressive", stack=200) for i in range(6)]
    st = GameState(players=ps, dealer_pos=0, config=CFG)
    st.board = np.empty(0, dtype=np.int32)
    st.round_idx = 0
    st.pot = 0
    st.round_bets = {0: 1, 1: 2, 2: 8}   # SB, BB, then UTG raises to 4bb
    st.last_full_raise = 8               # what the engine would be enforcing
    ps[1].hole = np.array([H(14, 1), H(14, 2)], dtype=np.int32)   # AA in the BB
    a = AggressiveStrategy().act(ps[1], st, 1, None, Rng(0))
    assert a.kind == RAISE
    assert a.amount == 16                                   # 2 + 6 + 8
    assert a.amount - st.round_bets[1] >= st.last_full_raise


def test_aggressive_caller_reraise_meets_increment():
    # HIGH-2 regression: a seat that already CALLED earlier in the round and
    # then faces a further re-raise (own=6, open=14) must still clear
    # last_full_raise. The pre-raise formula (own + to_call + bb) produced
    # gross = 10 < lfr 14 here -> engine AssertionError. raise_size floors
    # at own + to_call + last_full_raise which clears it from any geometry.
    ps = [Player(name=f"p{i}", strategy="Aggressive", stack=200) for i in range(6)]
    st = GameState(players=ps, dealer_pos=0, config=CFG)
    st.board = np.empty(0, dtype=np.int32)
    st.round_idx = 0
    st.pot = 0
    # UTG raised to 6, HJ called 6, CO re-raised to 14 (gross 14, lfr 14)
    st.round_bets = {0: 6, 1: 2, 2: 6, 3: 6, 4: 14}
    st.last_full_raise = 14
    ps[3].hole = np.array([H(14, 1), H(14, 2)], dtype=np.int32)   # AA in the HJ
    a = AggressiveStrategy().act(ps[3], st, 3, None, Rng(0))
    assert a.kind == RAISE
    assert a.amount - st.round_bets[3] >= st.last_full_raise


def test_aggressive_postflop_equity_raise_meets_increment():
    # Auditor-missed crash: postflop equity-raise sized as a 0.66-pot bet can
    # under-shoot the street increment when re-raising from a seat that called
    # earlier (own committed). raise_size must floor the wager at
    # own + to_call + last_full_raise so gross clears the increment.
    st, p = make(hand=(14, 14), board=((13, 0), (2, 1), (5, 2)),
                 round_bets={0: 100, 3: 50}, pot=100)
    st.last_full_raise = 100       # seat 0 raised gross 100 over a 50 bet
    prov = StubProvider(0.9, pot=200, to_call=50)
    a = AggressiveStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind == RAISE
    assert a.amount - st.round_bets[3] >= st.last_full_raise


def test_aggressive_short_stack_checks_instead_of_illegal_bet():
    # LOW-2 guard: a stack below the minimum bet cannot legally BET (engine:
    # assert paid >= min_bet). The strategy must check, not crash.
    st, p = make(hand=(14, 13), board=((13, 0), (2, 1), (5, 2)), round_bets={}, pot=100)
    st.preflop_raise_by = 3
    p.stack = 1
    prov = StubProvider(0.5, pot=100, to_call=0)
    a = AggressiveStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind == CHECK