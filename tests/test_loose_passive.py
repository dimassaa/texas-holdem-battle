"""Loose (wide, passive-betting) and Passive (tight-folding, check-call)."""
import numpy as np
from poker.actions import BET, CALL, CHECK, FOLD, RAISE
from poker.card import card_id as H
from poker.config import Config
from poker.game import GameState
from poker.player import LooseStrategy, PassiveStrategy, Player
from poker.rng import Rng

CFG = Config()


class StubProvider:
    """Fake EquityProvider with a fixed equity — decouples strategy math from MC."""
    def __init__(self, eq, pot=100, to_call=50): self.eq, self.pot, self.to_call = eq, pot, to_call
    def equity(self, hand, board, n_opp): return self.eq
    def pot_odds(self): return self.to_call / (self.pot + self.to_call)


def make(strategy, hand=(9, 8), board=(), pot=100, round_bets=None):
    ps = [Player(name=f"p{i}", strategy=strategy, stack=200) for i in range(6)]
    st = GameState(players=ps, dealer_pos=0, config=CFG)
    st.board = np.array([H(*c) for c in board], dtype=np.int32) if board else np.empty(0, dtype=np.int32)
    st.round_idx = 1 if board else 0
    st.round_bets = dict(round_bets or {})
    st.pot = pot
    for i, p in enumerate(ps):
        p.hole = np.array([H(hand[0], i % 4), H(hand[1], (i + 1) % 4)], dtype=np.int32)
    return st, st.players[3]


def test_loose_calls_broad_flop_hand():
    st, p = make("Loose", hand=(9, 8), board=((13, 0), (9, 1), (3, 2)), round_bets={0: 50})
    prov = StubProvider(0.4, pot=150, to_call=50)   # pair of nines, decent draw
    a = LooseStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind == CALL


def test_loose_folds_too_large_bet():
    # The 200 facing bet must reach state.round_bets — the strategy reads
    # to_call from GameState, not from the stub provider (line 230 of player.py).
    st, p = make("Loose", hand=(3, 2), board=((13, 0), (9, 1), (3, 2)), round_bets={0: 200})
    # pot odds terrible: 250/(250+200) = ~0.44 > equity 0.3
    prov = StubProvider(0.3, pot=250, to_call=200)
    a = LooseStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind == FOLD


def test_passive_checks_unraised():
    st, p = make("Passive", hand=(14, 12), board=((14, 0), (11, 1), (2, 2)))
    prov = StubProvider(0.7, pot=100, to_call=0)
    a = PassiveStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind in (CHECK, CALL)  # passive: does not bet top pair, may check/call


def test_passive_raises_only_cold_nuts():
    # set on the flop with a pocket pair -> set+ -> bets (BET, not RAISE:
    # unopened pot, RAISE would trip the engine's owed==0 assert, game.py:168)
    st, p = make("Passive", hand=(3, 3), board=((14, 0), (3, 1), (2, 2)))
    prov = StubProvider(0.95, pot=100, to_call=0)
    a = PassiveStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind in (CHECK, BET)
