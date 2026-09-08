"""The hand & betting engine.

Owns GameState, the betting-round loop, side-pot construction, showdown, and
the run_hand driver. All pot math lives here; strategies only emit Actions.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from poker.config import Config
from poker.player import Player


@dataclass
class GameState:
    players: List[Player]
    dealer_pos: int
    config: Config
    board: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    round_idx: int = 0          # 0=preflop, 1=flop, 2=turn, 3=river
    pot: int = 0                # chips already committed before this round
    round_bets: Dict[int, int] = field(default_factory=dict)
    aggro_count: int = 0        # count of bet/raise events this round
    history: List[dict] = field(default_factory=list)
    preflop_raise_by: int = -1  # seat that last raised on round 0 (Stage-04 c-bet signal)

    def to_call(self, player_idx: int) -> int:
        """Chips the player must wager to match the current round's highest bet."""
        max_round = max(self.round_bets.values(), default=0)
        return max(0, max_round - self.round_bets.get(player_idx, 0))

    def num_opponents(self, player_idx: int) -> int:
        """Number of non-folded players excluding the given player."""
        active = [i for i, p in enumerate(self.players) if not p.folded]
        return len(active) - 1

    def can_raise(self, player_idx: int) -> bool:
        """Raises are bounded by the 4-aggression cap and the stack."""
        return self.aggro_count < self.config.max_aggressions


def starting_bets(state: GameState) -> None:
    """Post the small and big blinds into the preflop round.

    The button posts SB, the next player posts BB (blinds are on real seats;
    eliminated seats are skipped at the session layer, so `players` here are
    exactly the live ones). Bets are always capped by stack.

    The blinds live in `round_bets`, and `pot` stays at 0 until the street
    closes: `run_betting_round` moves the whole street into `pot` exactly once
    at finalization, which prevents the blinds from being counted twice.
    """
    sb_idx = state.dealer_pos
    bb_idx = (state.dealer_pos + 1) % len(state.players)
    state.round_bets = {}
    for idx, amt in ((sb_idx, state.config.sb), (bb_idx, state.config.bb)):
        paid = min(state.players[idx].stack, amt)
        state.players[idx].stack -= paid
        state.players[idx].contributed += paid
        state.round_bets[idx] = paid
    state.pot = 0


def max_raise_amount(state: GameState, player_idx: int) -> int:
    """Pot-limit maximum total wager this action (doc rule 2.2).

    MaxRaise = (prior streets' pot + all current street bets) + 2 * to_call.
    `state.pot` holds only previous streets; the current street lives in
    `round_bets`. The formula equals: call, then re-raise by the new pot.
    """
    call = state.to_call(player_idx)
    return state.pot + sum(state.round_bets.values()) + 2 * call
