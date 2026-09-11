"""Shared decision math: pot odds, required equity, bet sizing."""
import numpy as np
from poker.actions import BET, RAISE
from poker.config import Config
from poker.equity import EquityProvider
from poker.game import GameState
from poker.player import Player, bet_size, required_equity
from poker.rng import Rng

CFG = Config()


def state_and_provider(pot=100, to_call=25):
    ps = [Player(name=f"p{i}", strategy="X", stack=200) for i in range(3)]
    st = GameState(players=ps, dealer_pos=0, config=CFG)
    st.pot = pot
    st.round_bets = {0: 25, 1: 50, 2: 50}
    prov = EquityProvider(rng=Rng(1), config=CFG, pot=pot + sum(st.round_bets.values()),
                          to_call=to_call)
    return st, prov


def test_required_equity_is_max_of_pot_odds_and_hud_floor():
    _, prov = state_and_provider()
    # 1 opponent: required = max(pot_odds, tight floor)
    req = required_equity(prov, n_opp=1, str_floor=0.6, style_factor=1.0)
    pot_odds = prov.pot_odds()
    assert req == max(pot_odds, 0.6)


def test_required_equity_multiway_uses_style_factor_not_floor():
    _, prov = state_and_provider(to_call=100)
    req = required_equity(prov, n_opp=3, str_floor=0.5, style_factor=1.15)
    assert abs(req - prov.pot_odds() * 1.15) < 1e-9


def test_bet_size_floor_and_cap():
    from poker.game import max_raise_amount
    st, _ = state_and_provider()
    size = bet_size(st, 0, fraction=0.5, min_bet=CFG.min_bet)
    assert size >= CFG.min_bet
    assert size <= max_raise_amount(st, 0)
