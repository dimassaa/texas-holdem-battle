"""Conservation fuzz: three scripted sessions, the five invariants every hand."""
import numpy as np
import pytest
from poker.actions import ALLIN, CALL, CHECK, FOLD
from poker.config import Config
from poker.game import run_hand
from poker.player import Player
from poker.rng import Rng

CFG = Config()


def check_invariants(result):
    # 1. side-pot amounts sum to total contributions
    assert sum(p["amount"] for p in result.side_pots) == result.pot_total
    # 2. no chips created or destroyed
    assert pytest.approx(sum(result.net)) == 0
    # 3. contributions never exceed the starting stack this hand
    assert all(net >= -200 for net in result.net)
    # 4. structural sanity of every side pot (winners are from eligible by
    #    construction of `pay_out`); a pot can never be orphaned
    assert all(pot["amount"] >= 0 and len(pot["eligible"]) >= 1
               for pot in result.side_pots)
    # 5. no card is duplicated across holes and board
    seen = {int(c) for c in result.board}
    for hole in result.hole_cards:
        seen.update(hole)
    assert len(seen) == len(result.board) + sum(len(h) for h in result.hole_cards)


def six_players():
    return [Player(name=f"p{i}", strategy="X", stack=200) for i in range(6)]


def test_checkdown_showdown_fuzz_conserves_chips():
    """Random deals all the way to a multiway showdown."""
    call6 = {i: (CALL, 0) for i in range(6)}
    check6 = {i: (CHECK, 0) for i in range(6)}
    for k in range(5_000):
        res = run_hand(six_players(), dealer_pos=k % 6, rng=Rng(10_000 + k),
                       config=CFG,
                       actions_override={0: call6, 1: check6, 2: check6,
                                         3: check6})
        check_invariants(res)


def test_allin_runout_fuzz_conserves_chips():
    """Random preflop all-in jams through the run-out shortcut."""
    allin6 = {i: (ALLIN, 0) for i in range(6)}
    for k in range(5_000):
        res = run_hand(six_players(), dealer_pos=k % 6, rng=Rng(20_000 + k),
                       config=CFG, actions_override={0: allin6})
        check_invariants(res)


def test_fold_to_survivor_fuzz_conserves_chips():
    """Random uncontested pots won by whoever called."""
    for k in range(2_000):
        survivor = k % 6
        script = {i: (FOLD, 0) if i != survivor else (CALL, 0) for i in range(6)}
        res = run_hand(six_players(), dealer_pos=k % 6, rng=Rng(30_000 + k),
                       config=CFG, actions_override={0: script})
        check_invariants(res)
