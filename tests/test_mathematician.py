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


def make(eq, to_call=0, pot=100, hand=(8, 5), board=(9, 3, 2),
         round_bets=None, last_full_raise=0, stack=200):
    ps = [Player(name=f"p{i}", strategy="Mathematician", stack=stack) for i in range(6)]
    st = GameState(players=ps, dealer_pos=0, config=CFG)
    st.board = np.array([H(board[0], 0), H(board[1], 1), H(board[2], 2)], dtype=np.int32) if board else np.empty(0, dtype=np.int32)
    st.round_idx = 1 if board else 0
    st.round_bets = dict(round_bets or {})
    st.pot = pot
    st.last_full_raise = last_full_raise
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


def test_math_value_bets_unopposed_pot():
    st, p, prov = make(eq=0.8, to_call=0, pot=100)    # no bet to call
    a = MathematicianStrategy().act(p, st, 3, prov, Rng(0))
    # strong equity + no opponent bet = value bet (50-75% pot)
    assert a.kind == BET and 0.5 * 100 <= a.amount <= 0.75 * 100


def test_math_raises_when_equity_surplus_clears_raise_gate():
    # Stake a real bet this round (round_bets={0:50}) so state.to_call(idx) > 0
    # and the engine's RAISE gate (owed > 0, game.py:189) is genuinely
    # reachable — with an empty round tray the raise branch can never fire and
    # the test would silently assert the BET path instead. Seat 0's open of 50
    # is a full raise (last_full_raise=50); equity 0.9 beats pot odds (1/3) by
    # well over the 1.15x surplus, so the strategy must raise not call.
    st, p, prov = make(eq=0.9, to_call=50, pot=100, round_bets={0: 50}, last_full_raise=50)
    a = MathematicianStrategy().act(p, st, 3, prov, Rng(0))
    own = st.round_bets.get(3, 0)
    assert a.kind == RAISE
    assert a.amount >= own + st.to_call(3)                       # covers the bet raised
    assert a.amount <= own + p.stack                             # never above the stack
    gross = a.amount - own
    assert gross >= st.last_full_raise or gross == p.stack       # engine legality (game.py:192)


def test_math_short_stack_raise_caps_at_allin():
    # CRITICAL regression (Fix 1): a short stack (10) raising a large open bet
    # must emit a total wager capped at own + stack — the all-in raise the
    # engine accepts via `gross == p.stack` (game.py:192). Before the cap,
    # raise_size returned max(desired, own + to_call + last_full_raise) = 100,
    # so gross 100 > stack 10 crashed the engine's assert gross <= p.stack
    # (game.py:191); the cap turns that same geometry into an all-in of 10.
    st, p, prov = make(eq=0.9, to_call=50, pot=100, round_bets={0: 50},
                       last_full_raise=50, stack=10)
    a = MathematicianStrategy().act(p, st, 3, prov, Rng(0))
    own = st.round_bets.get(3, 0)
    assert a.kind == RAISE
    assert a.amount == own + p.stack                            # 0 + 10 = all-in cap