"""Heads-up engine regression: first-actor ordering for the 2-handed table.

Post-elimination play runs with fewer than six seats; the first-actor rule is
pinned here so the explicit heads-up formulation cannot silently revert to a
6-max-only assumption.
"""


def test_heads_up_first_actor_ordering():
    # dealer(SB) acts first preflop, then BB; postflop BB leads.
    from poker.config import Config
    from poker.game import GameState, _setting_first_actor
    from poker.player import Player

    ps = [Player(name="SB", strategy="Tight", stack=200, rank=0),
          Player(name="BB", strategy="Tight", stack=200, rank=1)]
    st = GameState(players=ps, dealer_pos=0, config=Config())
    assert _setting_first_actor(st) == 0     # preflop: button(SB) acts first
    st.round_idx = 1
    assert _setting_first_actor(st) == 1     # postflop: BB leads