"""Player model and strategy decision layer.

Strategies are pure functions of observable state. The engine supplies an
EquityProvider (Stage 03) so strategies never re-implement hand evaluation —
this keeps every strategy's decisions reproducible and its stochastic inputs
(Monte Carlo equity) bounded to the provider.
"""

from dataclasses import dataclass, field

import numpy as np

from poker.actions import Action, ALLIN, BET, CALL, CHECK, FOLD, RAISE
from poker.config import Config
from poker.equity import EquityProvider, hand_type_index


@dataclass
class Player:
    name: str
    strategy: str
    stack: int
    hole: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    folded: bool = False
    all_in: bool = False
    contributed: int = 0   # total chips committed across the whole hand
    stats: dict = field(default_factory=dict)
    rank: int = 0          # seat index, set by the simulator when seating


class Strategy:
    name = "base"

    def act(self, player: Player, state, idx: int, provider: EquityProvider, rng) -> Action:
        raise NotImplementedError


def for_name(name: str) -> Strategy:
    return STRATEGIES[name]()


def relative_position(state, idx: int) -> str:
    """Map a seat to a 6-max zone by distance clock-wise from the button.

    Button = late; +1/+2 = blinds; +3/+4 = early; +5 = middle/cutoff. The
    mapping is modular so it degrades gracefully as seats are eliminated.
    """
    n = len(state.players)
    dist = (idx - state.dealer_pos) % n
    if dist == 0:
        return "late"
    if dist in (1, 2):
        return "blinds"
    if dist in (3, 4):
        return "early"
    return "middle"


def combo_weight(hand_type: int) -> int:
    """Distinct-combo count for a 169-class: pairs 6, offsuit 12, suited 4.

    Used only for auditing a range's share of the 1326-combo universe so
    range definitions can be validated against the doc's approximate bands.
    """
    # A hand type is a pair when its two ranks are equal; recover from index
    # by scanning (169 is small and this is test/audit-only code).
    for hi in range(2, 15):
        for lo in range(2, hi + 1):
            for s in (0, 1):
                if hand_type_index(hi, lo, s) != hand_type:
                    continue
                if hi == lo:
                    return 6
                return 4 if s == 1 else 12
    raise AssertionError(f"unreachable hand type {hand_type}")


def _type_set(pairs=(), suited=(), offsuit=()):
    """Build a frozenset of 169-class indices from rank-lists.

    `pairs` = tuple of pair ranks; `suited`/`offsuit` = tuples of (hi, lo).
    Keeps range definitions as readable constants rather than magic indices.
    """
    out = set()
    for r in pairs:
        out.add(hand_type_index(r, r, 0))
    for hi, lo in suited:
        a, b = max(hi, lo), min(hi, lo)
        out.add(hand_type_index(a, b, 1))
    for hi, lo in offsuit:
        a, b = max(hi, lo), min(hi, lo)
        out.add(hand_type_index(a, b, 0))
    return frozenset(out)


def ranked_types(table, fraction: float) -> frozenset:
    """Return the hand types whose combined combo-weight ≈ `fraction` of 1326,
    taken greedily by descending preflop equity vs 1 opponent (deterministic
    given the Stage-03 table). Used for Aggressive/Passive top-X% ranges."""
    order = np.argsort(table[:, 0])[::-1]
    out, weight = set(), 0.0
    target = 1326 * fraction
    for idx in order:
        out.add(int(idx))
        weight += combo_weight(int(idx))
        if weight >= target:
            break
    return frozenset(out)


# Preflop ranges (exact lists; percentages audited in Task 4.6).
TIGHT_RANGE = _type_set(
    pairs=(14, 13, 12, 11, 10, 9, 8, 7),           # 77+
    suited=((14, 11), (14, 12), (14, 13), (13, 12)),  # AJs, AQs, AKs, KQs
    offsuit=((14, 13), (14, 12)),                   # AKo, AQo
)
TIGHT_PREMIUM = _type_set(
    pairs=(14, 13, 12, 11),  # JJ+
    suited=((14, 13),),      # AKs
    offsuit=((14, 13),),     # AKo
)
LOOSE_RANGE = _type_set(
    pairs=tuple(range(2, 15)),
    suited=((14, lo) for lo in range(2, 15)),       # A2s+
    offsuit=((14, 12), (14, 11), (14, 10), (13, 12), (13, 11), (12, 11)),  # KTo+ QJo+
)
LOOSE_RAISE = None  # filled from the equity table in visited ranges

STRATEGIES = {}


def _register(cls):
    STRATEGIES[cls.name] = cls
    return cls
