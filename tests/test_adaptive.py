"""Adaptive: windowed opponent stats drive dynamic range/cbet thresholds."""
from collections import deque
import numpy as np
from poker.actions import BET, CALL, CHECK, FOLD, RAISE
from poker.card import card_id as H
from poker.config import Config
from poker.game import GameState
from poker.player import AdaptiveStrategy, Player
from poker.rng import Rng

CFG = Config()


class StubProvider:
    def __init__(self, eq, pot=100, to_call=0): self.eq, self.pot, self.to_call = eq, pot, to_call
    def equity(self, hand, board, n_opp): return self.eq
    def pot_odds(self): return self.to_call / (self.pot + self.to_call) if self.pot else 0.0


def fresh_brain(memory=None, adjust_every=None):
    return AdaptiveStrategy(memory=memory, adjust_every=adjust_every)


def make_postflop(hand=(14, 13), board=((13, 0), (2, 1), (5, 2)), pot=100):
    """6-max postflop state with the hero (seat 0) as the preflop aggressor.

    `round_bets` stays empty so to_call(0) == 0 and hero faces a free check/bet
    decision as the c-bettor; the engine legality guards (stack >= min_bet,
    can_raise) are all satisfiable with the defaults below.
    """
    ps = [Player(name=f"p{i}", strategy="Adaptive", stack=200) for i in range(6)]
    st = GameState(players=ps, dealer_pos=0, config=CFG)
    st.board = np.array([H(*c) for c in board], dtype=np.int32) if board else np.empty(0, dtype=np.int32)
    st.round_idx = 1 if board else 0
    st.round_bets = {}
    st.config = CFG
    st.pot = pot
    st.preflop_raise_by = 0      # hero raised preflop -> owns the c-bet decision
    for i, p in enumerate(ps):
        p.hole = np.array([H(hand[0], i % 4), H(hand[1], (i + 1) % 4)], dtype=np.int32)
    return st, st.players[0]


def test_stats_accumulate_over_public_actions():
    b = fresh_brain()
    # one hand: seat 5 raised, seat 3 folded to it (fold-to-raise evidence)
    b.observe({"seat": 5, "faces_raise": 1, "folds_to_raise": 1})
    st = b.stats()[5]
    assert st["hands"] == 1
    assert st["fold_to_raise"] == 1.0     # single observed instance


def test_window_is_capped():
    b = fresh_brain(memory=10)
    for i in range(20):
        b.observe({"seat": 1, "faces_raise": 1, "folds_to_raise": 1})
    assert b.stats()[1]["hands"] == 10   # rolling window, not cumulative

def test_loose_table_tightens_preflop_range():
    b = fresh_brain()
    # many calling opponents, no raises/folds -> loose table, steals unprofitable
    b.window_dump([{"seat": s, "calls": 5, "raises": 0} for s in range(1, 6)])
    assert b.play_share < 0.20


def test_tight_fold_heavy_table_expands_raise_range():
    b = fresh_brain()
    b.window_dump([{"seat": s, "folds": 9, "plays": 10} for s in range(1, 6)])
    assert b.raise_share > 0.25


def test_cbet_scales_inversely_with_opponent_fold_rate():
    """Regression pinning the §5.6 c-bet gate `equity >= max(0.45, 0.60 - 0.25*fold_bias)`.

    equity 0.5 sits between the two thresholds the rule produces — 0.45 on a
    fold-heavy table, 0.60 on a calling one — so the SAME hand+board+equity
    flips the decision purely on the opponents' observed fold-to-raise.
    `_fold_bias` reads the live window (`_aggregate_window(self._window)`), so
    window_dump-then-act on one brain must reflect each dump.
    """
    b = fresh_brain()
    st, hero = make_postflop()
    prov = StubProvider(0.5, pot=100, to_call=0)
    # Fold-heavy opponents (fold_to_raise == faces_raise == 1.0): threshold
    # drops to 0.45 -> the 0.50-equity c-bet fires.
    b.window_dump([{"seat": s, "faces_raise": 10, "folds_to_raise": 10} for s in range(1, 6)])
    assert b.act(hero, st, 0, prov, Rng(0)).kind == BET
    # Never-folding callers (fold_to_raise 0.0): threshold rises to 0.60 ->
    # the same 0.50-equity semi-bluff dries up and checks.
    b.window_dump([{"seat": s, "faces_raise": 10, "folds_to_raise": 0} for s in range(1, 6)])
    assert b.act(hero, st, 0, prov, Rng(0)).kind == CHECK


def test_ranges_refresh_on_adjust_every_boundary():
    """Regression pinning periodic range refresh: it fires at EVERY boundary.

    White-box on the brain's cached ranges: `_refresh_ranges_if_due` must not
    touch the cache off-boundary and must rewrite it on each hit of
    hands_seen % adjust_every == 0 — proving it re-adapts, not freezes.
    """
    b = fresh_brain(adjust_every=5)
    base = type(b)._play_range              # neutral cold-start cache, per direct-init
    # hands_seen 1..4 are not boundaries: a fold-heavy window would widen the
    # cache if the refresh ran, so an unchanged cache proves the gate gates.
    for _ in range(4):
        b.observe({"seat": 1, "faces_raise": 10, "folds_to_raise": 10})
    b._refresh_ranges_if_due()
    assert b._play_range == base and b._steal_scale == 1.0
    # Crossing hands_seen=5 fires once and re-reads the live fold-heavy window:
    # ranges widen and steal sizing grows (cold-start steals are too small).
    b.observe({"seat": 2, "faces_raise": 10, "folds_to_raise": 10})
    b._refresh_ranges_if_due()
    widened = b._play_range
    assert b._steal_scale == 1.4 and widened != base
    # hands_seen=6 is NOT a boundary: after swapping the live window to a
    # loose-calling table (which WOULD tighten the cache if the refresh ran),
    # the fold-heavy cache must survive untouched.
    b.observe({"seat": 1, "faces_raise": 10, "folds_to_raise": 10})
    b.window_dump([{"seat": s, "calls": 5, "raises": 0} for s in range(1, 6)])
    b._refresh_ranges_if_due()
    assert b._play_range == widened and b._steal_scale == 1.4
    # Crossing hands_seen=10 fires AGAIN (repeated, not one-shot): the refresh
    # re-reads the swapped window and now tightens the cache below the base.
    for _ in range(4):
        b.observe({"seat": 1, "faces_raise": 10, "folds_to_raise": 0})
    b._refresh_ranges_if_due()
    assert b._steal_scale == 0.7
    assert b._play_range != widened and b._play_range != base
