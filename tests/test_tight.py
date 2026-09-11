"""Tight strategy: range discipline and postflop value-or-fold."""
import numpy as np
from poker.actions import BET, CALL, CHECK, FOLD, RAISE
from poker.card import card_id as H
from poker.config import Config
from poker.equity import EquityProvider
from poker.game import GameState
from poker.player import Player, TightStrategy, required_equity
from poker.rng import Rng

CFG = Config()


def make(idx=0, stack=200, dealer=0, board=(), hand=(14, 13)):
    """State + player with the requested hole cards.

    `board` is a sequence of (rank, suit) tuples; `hand` is two ranks (suits
    0 and 1 are used so the two hole cards are always distinct).
    """
    ps = [Player(name=f"p{i}", strategy="Tight", stack=stack) for i in range(6)]
    st = GameState(players=ps, dealer_pos=dealer, config=CFG)
    st.board = (np.array([H(r, s) for r, s in board], dtype=np.int32)
                if board else np.empty(0, dtype=np.int32))
    st.round_idx = 1 if board else 0
    st.round_bets = {}
    st.pot = 100
    r1, r2 = hand
    ps[idx].hole = np.array([H(r1, 0), H(r2, 1)], dtype=np.int32)
    return st, ps[idx]


class StubProvider(EquityProvider):
    def __init__(self, eq, **kw):
        super().__init__(rng=Rng(1), config=CFG, **kw)
        self.eq = eq

    def equity(self, hand, board, n_opp):
        return self.eq


def test_tight_folds_weaker_hands_preflop_early():
    st, p = make(board=(), hand=(7, 5))
    prov = StubProvider(0.2, pot=0)
    a = TightStrategy().act(p, st, st.players.index(p), prov, Rng(0))
    assert a.kind == FOLD


def test_tight_plays_aces_preflop():
    st, p = make(board=(), hand=(14, 14))
    st.round_bets = {1: 1, 2: 2}          # blinds posted
    st.pot = 0
    prov = StubProvider(0.8, pot=3, to_call=2)
    a = TightStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind in (CALL, RAISE)


def test_tight_folds_flop_without_equity():
    st, p = make(board=((14, 0), (9, 1), (3, 2)), hand=(7, 5))
    st.round_bets = {0: 50}
    st.pot = 100
    prov = StubProvider(0.2, pot=150, to_call=50)
    a = TightStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind == FOLD


def test_tight_values_two_pair_plus():
    st, p = make(board=((14, 0), (9, 1), (3, 2)), hand=(14, 9))
    st.round_bets = {}
    st.pot = 100
    prov = StubProvider(0.8, pot=100, to_call=0)
    a = TightStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind in (BET, CHECK, RAISE)  # betting/raising with made two-pair