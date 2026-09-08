"""End-to-end hand: run-out shortcut, showdown split, conservation."""
import numpy as np
import pytest
from poker.actions import ALLIN, CALL, CHECK, FOLD
from poker.config import Config
from poker.game import run_hand
from poker.player import Player
from poker.rng import Rng

CFG = Config()


def six_nplayers():
    return [Player(name=f"p{i}", strategy="X", stack=200) for i in range(6)]


def test_fold_to_big_blind_conserves_chips():
    players = six_nplayers()
    before = sum(p.stack for p in players)
    # Everyone folds preflop; the BB collects the blinds uncontested (no board).
    st = run_hand(players, dealer_pos=0, rng=Rng(1), config=CFG,
                  actions_override={0: {i: (FOLD, 0) for i in range(6)}})
    assert sum(p.stack for p in players) == before
    assert len(st.board) == 0
    assert st.pot_total == 3
    assert st.net == [-1, 1, 0, 0, 0, 0]   # SB paid 1, BB won 3 (net +1)
    assert pytest.approx(sum(st.net)) == 0


def test_all_in_runout_deals_remaining_board_without_betting():
    players = six_nplayers()
    before = sum(p.stack for p in players)
    # Every stack goes in preflop; the run-out shortcut must deal the rest of
    # the board with no betting and a showdown.
    st = run_hand(players, dealer_pos=0, rng=Rng(2), config=CFG,
                  actions_override={0: {i: (ALLIN, 0) for i in range(6)}})
    assert len(st.board) == 5
    assert sum(p.stack for p in players) == before
    assert st.pot_total == 6 * 200
    assert pytest.approx(sum(st.net)) == 0


def test_multiway_check_down_reaches_showdown_and_splits():
    players = six_nplayers()
    before = sum(p.stack for p in players)
    call6 = {i: (CALL, 0) for i in range(6)}
    check6 = {i: (CHECK, 0) for i in range(6)}
    st = run_hand(players, dealer_pos=0, rng=Rng(3), config=CFG,
                  actions_override={0: call6, 1: check6, 2: check6, 3: check6})
    assert len(st.board) == 5
    assert any(a["kind"] == "showdown" for a in st.actions)
    assert sum(p.stack for p in players) == before
    assert st.pot_total == 12      # 6 players put in 2 each
    assert pytest.approx(sum(st.net)) == 0
