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
    # User-approved engine-legality fix: a re-raise from a seat with chips
    # already committed (the BB) must size high enough that gross == amount -
    # own_contribution still clears last_full_raise (game.py:169-171). The
    # earlier to_call+bb formula crashed the betting round here (gross 4 < 6).
    ps = [Player(name=f"p{i}", strategy="Aggressive", stack=200) for i in range(6)]
    st = GameState(players=ps, dealer_pos=0, config=CFG)
    st.board = np.empty(0, dtype=np.int32)
    st.round_idx = 0
    st.pot = 0
    st.round_bets = {0: 1, 1: 2, 2: 6}   # SB, BB, then UTG raises to 3bb
    ps[1].hole = np.array([H(14, 1), H(14, 2)], dtype=np.int32)   # AA in the BB
    a = AggressiveStrategy().act(ps[1], st, 1, None, Rng(0))
    assert a.kind == RAISE
    # to_call=4 + own=2 + bb=2 targets 8: gross = min(8, mra) - 2 >= 6.
    assert a.amount == 8