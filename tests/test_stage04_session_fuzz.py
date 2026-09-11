"""Stage-04 gate: multi-hand fuzz with the four strategies keeps chips legal.

This is the first test that exercises the production branch of the engine
(`actions_override is None`): real strategies drive every betting round, so any
engine bug that only surfaces with generated actions (as opposed to scripted
Stage-02 harnesses) is caught here, pre-Stage-06.
"""
import numpy as np

from poker.config import Config
from poker.game import run_hand
from poker.player import Player, for_name
from poker.rng import Rng

CFG = Config(
    strategy_combo=("Tight", "Loose", "Aggressive", "Passive", "Passive", "Tight"),
)


def test_20_hands_conserve_and_are_legal():
    players = [Player(name=f"p{i}", strategy=CFG.strategy_combo[i], stack=200)
               for i in range(6)]
    for p in players:
        p.brain = for_name(p.strategy)
    initial = sum(p.stack for p in players)
    dealer = 0
    for h in range(20):
        res = run_hand(players, dealer, Rng(h * 7 + 1), CFG)
        dealer = (dealer + 1) % 6
        assert sum(p.stack for p in players) == initial          # conservation
        assert all(p.stack >= 0 for p in players)                 # no overdraft
        assert len(set(np.concatenate([p.hole for p in players]).tolist() + res.board.tolist())) == 12 + len(res.board)  # unique cards
    # all actions were recorded (audit trail exists for Stage 06)
    extra = Rng(300)
    final = run_hand(players, dealer, extra, CFG)
    assert len(res.actions) > 0 and len(final.actions) > 0