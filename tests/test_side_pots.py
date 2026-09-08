"""Side-pot construction and payout against the conservation invariants."""
import numpy as np
import pytest
from poker.config import Config
from poker.game import build_side_pots, pay_out
from poker.player import Player
from poker.card import card_id as H

CFG = Config()


def ps(*contributed):
    out = []
    for i, c in enumerate(contributed):
        out.append(Player(name=f"p{i}", strategy="X", stack=max(0, 200 - c),
                          contributed=c))
    return out


def test_side_pots_layered_correctly():
    # contributions (200, 100, 50, 200, 200, 0) with nobody folded:
    #   L50:  250  eligible (0,1,2,3,4)
    #   L100: 200  eligible (0,1,3,4)
    #   L200: 300  eligible (0,3,4)
    players = ps(200, 100, 50, 200, 200, 0)
    pots = build_side_pots(players)
    assert sum(p.amount for p in pots) == 750
    assert [p.amount for p in pots] == [250, 200, 300]
    assert {tuple(sorted(p.eligible)) for p in pots} == {
        (0, 1, 2, 3, 4), (0, 1, 3, 4), (0, 3, 4),
    }


def test_folded_players_forfeit_eligibility_and_dead_chips_roll_down():
    # p5 put 200 in but folded; the five live players each put 100. p5's
    # unmatched top 100 chips are dead money that rolls into the live main pot.
    players = ps(100, 100, 100, 100, 100, 200)
    players[5].folded = True
    pots = build_side_pots(players)
    assert sum(p.amount for p in pots) == 700   # conservation holds
    assert len(pots) == 1
    assert pots[0].amount == 700
    assert sorted(pots[0].eligible) == [0, 1, 2, 3, 4]


def test_payout_distributes_pots_with_fractional_ties():
    from poker.game import GameState
    players = ps(200, 100, 50, 200, 200, 0)
    state = GameState(players=players, dealer_pos=0, config=CFG)
    # p0, p3 and p4 tie for the best hand on every layer they share.
    scores = {0: 10, 1: 1, 2: 1, 3: 10, 4: 10, 5: 0}
    pay_out(state, scores)
    after = [p.stack for p in players]
    # chips never appear nor disappear at the table level
    assert pytest.approx(sum(after)) == 1200
    # each tied winner should net the same from the pot they share
    tie_net = {i: after[i] - players[i].contributed for i in (0, 3, 4)}
    assert pytest.approx(tie_net[0]) == tie_net[3] == tie_net[4]


def test_build_side_pots_conserves_total():
    from poker.rng import Rng
    rng = Rng(0)
    for _ in range(200):
        cont = sorted(rng.integers(0, 201, size=6).tolist())
        players = ps(*cont)
        for p in players:
            p.folded = bool(rng.integers(0, 2))
        pots = build_side_pots(players)
        assert sum(p.amount for p in pots) == sum(cont)
