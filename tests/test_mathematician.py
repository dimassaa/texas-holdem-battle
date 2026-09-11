"""Mathematician: pure pot-odds EV decision on every street."""
import numpy as np
from poker.actions import BET, CALL, FOLD, RAISE
from poker.card import card_id as H
from poker.config import Config
from poker.game import GameState
from poker.player import MathematicianStrategy, Player
from poker.rng import Rng

CFG = Config()


class StubProvider:
    def __init__(self, eq, pot=100, to_call=0): self.eq, self.pot, self.to_call = eq, pot, to_call
    def equity(self, hand, board, n_opp): return self.eq
    def pot_odds(self): return self.to_call / (self.pot + self.to_call) if self.pot else 0.0


def make(eq, to_call=0, pot=100, hand=(8, 5), board=(9, 3, 2)):
    ps = [Player(name=f"p{i}", strategy="Mathematician", stack=200) for i in range(6)]
    st = GameState(players=ps, dealer_pos=0, config=CFG)
    st.board = np.array([H(board[0], 0), H(board[1], 1), H(board[2], 2)], dtype=np.int32) if board else np.empty(0, dtype=np.int32)
    st.round_idx = 1 if board else 0
    st.round_bets = {}
    st.pot = pot
    for i, p in enumerate(ps): p.hole = np.array([H(hand[0], i % 4), H(hand[1], (i + 1) % 4)], dtype=np.int32)
    return st, st.players[3], StubProvider(eq, pot, to_call)


def test_math_calls_when_equity_exceeds_pot_odds():
    st, p, prov = make(eq=0.4, to_call=25, pot=100)   # pot odds .2
    a = MathematicianStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind in (CALL, RAISE)


def test_math_folds_when_under_pot_odds():
    st, p, prov = make(eq=0.15, to_call=150, pot=50)  # pot odds .75
    a = MathematicianStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind == FOLD


def test_math_raises_only_when_profitably_above_pot_odds():
    st, p, prov = make(eq=0.8, to_call=0, pot=100)    # no bet to call
    a = MathematicianStrategy().act(p, st, 3, prov, Rng(0))
    # strong equity + no opponent bet = value bet (50-75% pot)
    assert a.kind == BET and 0.5 * 100 <= a.amount <= 0.75 * 100