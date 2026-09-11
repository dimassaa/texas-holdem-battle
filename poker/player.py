"""Player model and strategy decision layer.

Strategies are pure functions of observable state. The engine supplies an
EquityProvider (Stage 03) so strategies never re-implement hand evaluation —
this keeps every strategy's decisions reproducible and its stochastic inputs
(Monte Carlo equity) bounded to the provider.
"""

import os

from dataclasses import dataclass, field

import numpy as np

from poker.actions import Action, BET, CALL, CHECK, FOLD, RAISE
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
    # brain is a Strategy instance set at seating; forward reference spares a
    # circular import (Strategy is defined later in this module).
    brain: "Strategy | None" = None


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
# Widened from the plan's literal (was 6.6%) to Table-1's 12-15% band:
# 14.5% of combos measured by the audit. Keeps TIGHT_PREMIUM mandatory (JJ+/AK).
TIGHT_RANGE = _type_set(
    pairs=(14, 13, 12, 11, 10, 9, 8, 7, 6, 5),     # 55+
    suited=((14, 9), (14, 10), (14, 11), (14, 12), (14, 13),  # A9s-AKs
            (13, 12), (13, 11),                     # KQs, KJs
            (12, 11), (12, 10),                     # QJs, QTs
            (11, 10), (10, 9), (9, 8)),             # JTs, T9s, 98s
    offsuit=((14, 13), (14, 12), (14, 11), (14, 10),  # AKo-ATo
             (13, 12), (13, 11), (12, 11)),         # KQo, KJo, QJo
)
TIGHT_PREMIUM = _type_set(
    pairs=(14, 13, 12, 11),  # JJ+
    suited=((14, 13),),      # AKs
    offsuit=((14, 13),),     # AKo
)
# Widened from the plan's literal (was 14.9%) to Table-1's ~40% band:
# 39.7% of combos measured by the audit. Superset of both LooseStrategy's
# preferred (TIGHT_PREMIUM) raise hands and its re-raise set (top-10%),
# so no wired hand falls into the range gate.
LOOSE_RANGE = _type_set(
    pairs=tuple(range(2, 15)),                                      # 22+
    suited=(*((h, lo) for h in range(12, 15) for lo in range(2, h)),   # Q2s+ K2s+ A2s+
            *((lo, lo + 2) for lo in range(2, 13)),                 # 2-apart suited (42s..J9s, KJs, AQs)
            *((lo, lo + 3) for lo in range(2, 12)),                 # 3-apart suited (53s..J8s, Q9s, KTs)
            (9, 8), (8, 7), (10, 9), (11, 10)),                     # 87s 98s T9s JTs
    offsuit=(*((h, lo) for h in range(10, 15) for lo in range(2, h)
               if h - lo in (1, 2, 3, 4)),),                        # T9o..KJo step gaps
)

STRATEGIES = {}


def _register(cls):
    STRATEGIES[cls.name] = cls
    return cls


def required_equity(provider, n_opp, str_floor, style_factor=1.0, floor_only_heads_up=True):
    """Minimum equity a strategy requires to continue.

    Multiway (approved decision): pot odds, scaled by the strategy's calling
    style factor — the pot odds already encode opponents + bet size.
    Heads-up: at least the street's Table-2 threshold floor (floors keep the
    doc's quoted behavior vs a single random opponent).
    """
    if n_opp > 1:
        return provider.pot_odds() * style_factor
    return max(provider.pot_odds(), str_floor)


def pot_total(state) -> int:
    """Effective pot this actor faces = prior streets + this round's bets."""
    return state.pot + sum(state.round_bets.values())


def bet_size(state, idx, fraction, min_bet) -> int:
    """Round a fractional-pot bet into a legal, capped chip amount."""
    # Floor the raw wager at 1 chip so a tiny-pot bet never rounds to zero.
    raw = max(1.0, fraction * pot_total(state))
    size = int(round(raw))
    from poker.game import max_raise_amount  # local import breaks the game<->player cycle
    return min(max_raise_amount(state, idx), max(min_bet, size))


def raise_size(state, idx, desired) -> int:
    """Total wager for a raise that the engine will always accept as legal.

    The engine asserts gross = amount - round_bets[actor] >= last_full_raise
    (game.py). A fixed 3/4bb target under-shoots that increment from any seat
    with chips already committed this round (the BB re-raising, or anyone who
    called and then pops again), causing an engine AssertionError — the crash
    the Task-4.4/4.5 legality fixes only partially covered. Flooring the wager
    at own + to_call + last_full_raise makes gross exactly to_call + increment,
    which is legal and still a genuine raise from any seat geometry.
    """
    own = state.round_bets.get(idx, 0)
    legal_min = own + state.to_call(idx) + state.last_full_raise
    return max(int(desired), legal_min)


@_register
class TightStrategy(Strategy):
    """Plays only premium starting hands; posts flop only with strong equity
    or a made two-pair+ hand; value-bets very strong hands."""

    name = "Tight"
    _FLOORS = {1: 0.60, 2: 0.70, 3: 0.75}   # street -> heads-up call floor

    def _preflop_action(self, player, state, idx, provider):
        from poker.equity import start_hand_type
        pos = relative_position(state, idx)
        htype = start_hand_type(player.hole)
        action_range = TIGHT_RANGE if pos != "early" else TIGHT_PREMIUM
        if htype not in action_range:
            return Action(FOLD, 0)
        open_bet = max(state.round_bets.values(), default=0)
        to_call = state.to_call(idx)
        if htype in TIGHT_PREMIUM or (to_call == 0):
            # premium or unopened pot: raise to a normal size when allowed
            if state.can_raise(idx) and open_bet == 0:
                amount = max(state.config.bb, 3 * state.config.bb)
                return Action(RAISE, amount)
            if to_call > 0 and to_call <= player.stack:
                return Action(CALL, to_call)
            return Action(CHECK, 0) if to_call == 0 else Action(FOLD, 0)
        return Action(CALL, to_call) if to_call <= player.stack else Action(FOLD, 0)

    def act(self, player, state, idx, provider, rng):
        if state.round_idx == 0:
            return self._preflop_action(player, state, idx, provider)
        equity = provider.equity(player.hole, state.board, state.num_opponents(idx))
        req = required_equity(provider, state.num_opponents(idx),
                              self._FLOORS[state.round_idx], style_factor=1.15)
        if equity >= req:
            to_call = state.to_call(idx)
            if to_call == 0:
                # A stack below the minimum bet cannot make a legal BET
                # (engine: assert paid >= min_bet) — check instead.
                if equity >= 0.75 and state.can_raise(idx) and player.stack >= state.config.min_bet:
                    return Action(BET, bet_size(state, idx, 0.6, state.config.min_bet))
                return Action(CHECK, 0)
            return Action(CALL, min(player.stack, to_call))
        return Action(FOLD, 0)


@_register
class LooseStrategy(Strategy):
    """Wide range, many calls, minimal raising; bet/raise only with made hands."""

    name = "Loose"
    _FLOORS = {1: 0.35, 2: 0.40, 3: 0.50}

    def _preflop_action(self, player, state, idx, provider):
        from poker.equity import start_hand_type
        if start_hand_type(player.hole) not in LOOSE_RANGE:
            return Action(FOLD, 0)
        to_call = state.to_call(idx)
        raise_ok = start_hand_type(player.hole) in self._top10 if hasattr(self, "_top10") else False
        # Engine rejects RAISE with owed==0 (game.py:168) — only re-raise when a net
        # bet is outstanding, mirroring PassiveStrategy._preflop_action. The 4bb
        # target is only the desired size: raise_size floors it at the legal
        # minimum from a seat that already has chips committed (e.g. the BB).
        if raise_ok and state.can_raise(idx) and state.to_call(idx) > 0:
            return Action(RAISE, raise_size(state, idx, max(state.config.bb, 4 * state.config.bb)))
        return Action(CALL, to_call) if to_call <= player.stack else Action(FOLD, 0)

    def act(self, player, state, idx, provider, rng):
        if state.round_idx == 0:
            return self._preflop_action(player, state, idx, provider)
        eq = provider.equity(player.hole, state.board, state.num_opponents(idx))
        req = required_equity(provider, state.num_opponents(idx),
                              self._FLOORS[state.round_idx], style_factor=0.9)
        if state.to_call(idx) == 0:
            # loose bets/raises only with a made pair or better: use equity 0.5
            # as a proxy for a hand that can win at showdown vs one opponent.
            if eq >= 0.5 and state.can_raise(idx) and player.stack >= state.config.min_bet:
                return Action(BET, bet_size(state, idx, 0.5, state.config.min_bet))
            return Action(CHECK, 0)
        if eq >= req:
            return Action(CALL, min(player.stack, state.to_call(idx)))
        return Action(FOLD, 0)


@_register
class PassiveStrategy(Strategy):
    """Tight-folding preflop, check/call postflop, raises only set+."""

    name = "Passive"
    _FLOORS = {1: 0.40, 2: 0.45, 3: 0.55}
    _play_range = TIGHT_RANGE   # wired from ranked_types by Task 4.6 when the table exists
    _raise_range = _type_set(pairs=(14, 13, 12), suited=((14, 13),), offsuit=((14, 13),))

    def _preflop_action(self, player, state, idx, provider):
        from poker.equity import start_hand_type
        htype = start_hand_type(player.hole)
        if htype not in self._play_range:
            return Action(FOLD, 0)
        to_call = state.to_call(idx)
        # Only re-raise an existing open bet — engine rejects RAISE with owed==0
        # (game.py:168). to_call>0 ⇔ owed>0; open_bet alone stays true for the BB
        # whose own blind already covers it. raise_size keeps the re-raise legal
        # from committed seats (BB) where the fixed 4bb target under-shoots.
        if htype in self._raise_range and state.can_raise(idx) and state.to_call(idx) > 0:
            return Action(RAISE, raise_size(state, idx, max(state.config.bb, 4 * state.config.bb)))
        return Action(CALL, to_call) if to_call <= player.stack else Action(FOLD, 0)

    def act(self, player, state, idx, provider, rng):
        if state.round_idx == 0:
            return self._preflop_action(player, state, idx, provider)
        eq = provider.equity(player.hole, state.board, state.num_opponents(idx))
        req = required_equity(provider, state.num_opponents(idx),
                              self._FLOORS[state.round_idx], style_factor=1.0)
        if state.to_call(idx) == 0:
            # passive bets almost never; bet (not RAISE) an unopened pot with
            # set+: RAISE would trip owed==0 assert in game.py:168.
            if eq >= 0.95 and state.can_raise(idx) and player.stack >= state.config.min_bet:
                return Action(BET, bet_size(state, idx, 0.5, state.config.min_bet))
            return Action(CHECK, 0)
        if eq >= req * 0.95:   # documented slight looseness of passive callers
            return Action(CALL, min(player.stack, state.to_call(idx)))
        return Action(FOLD, 0)


@_register
class AggressiveStrategy(Strategy):
    """Frequent aggressive actions, continuation bets, equity-raise postflop.

    The c-bet-with-air and in-position steal are deliberate SEMI-BLUFFS —
    documented relaxation of the project's no-bluff limitation (approved
    decision) so this strategy stays distinct from Passive. They never bloat
    the action log; they are plain BET/RAISE records analyzed in Stage 06.
    """

    name = "Aggressive"
    _FLOORS = {1: 0.45, 2: 0.50, 3: 0.55}
    # Conservative class-level defaults so preflop paths work even when the
    # Stage-03 table file is absent; Task 4.6 re-wires these from ranked_types.
    _play_range = _type_set(
        pairs=(14, 13, 12, 11, 10, 9, 8),
        suited=((14, 13), (13, 12), (12, 11), (11, 10), (10, 9), (9, 8), (8, 7)),
        offsuit=((14, 13), (13, 12), (12, 11), (11, 10), (10, 9), (9, 8)))
    _raise_range = _type_set(pairs=(14, 13, 12), suited=((14, 13),), offsuit=((14, 13),))
    _steal_range = _type_set(suited=((12, 11), (11, 10), (10, 9)), offsuit=((12, 11), (11, 10)))

    def _preflop_action(self, player, state, idx, provider):
        from poker.equity import start_hand_type
        htype = start_hand_type(player.hole)
        in_range = htype in self._play_range
        if not in_range:
            return Action(FOLD, 0)
        to_call = state.to_call(idx)
        pos = relative_position(state, idx)
        steal = pos == "late" and htype in self._steal_range
        raise_now = (htype in self._raise_range or steal) and state.can_raise(idx)
        # Engine-legality (user-approved, mirrors Task 4.4): RAISE requires net
        # to_call>0 (game.py:168, owed==0 crash). raise_size floors the wager at
        # own + to_call + last_full_raise, which clears the increment assert
        # (game.py:185) from ANY committed seat — the BB, or a caller popping
        # again over a re-raise (the earlier to_call+own+bb formula only cleared
        # it for the blinds).
        if raise_now and state.to_call(idx) > 0:
            base = max(state.config.bb, (3 if not steal else 4) * state.config.bb)
            return Action(RAISE, raise_size(state, idx, base))
        return Action(CALL, to_call) if to_call <= player.stack else Action(FOLD, 0)

    def act(self, player, state, idx, provider, rng):
        if state.round_idx == 0:
            return self._preflop_action(player, state, idx, provider)
        equity = provider.equity(player.hole, state.board, state.num_opponents(idx))
        to_call = state.to_call(idx)
        is_aggressor = getattr(state, "preflop_raise_by", None) == idx
        if to_call == 0:
            if is_aggressor and state.can_raise(idx) and player.stack >= state.config.min_bet:
                # continuation bet — fires on every unbet flop/turn/river, even
                # with air (documented semi-bluff, plan §5.3).
                return Action(BET, bet_size(state, idx, 0.66, state.config.min_bet))
            if equity >= 0.5 and state.can_raise(idx) and player.stack >= state.config.min_bet:
                return Action(BET, bet_size(state, idx, 0.6, state.config.min_bet))
            return Action(CHECK, 0)
        # facing a bet: aggressive prefers raising strong equity over calling
        req = required_equity(provider, state.num_opponents(idx),
                              self._FLOORS[state.round_idx], style_factor=1.05)
        if equity >= req * 1.2 and state.can_raise(idx):
            # Equity-raise: floor the pot-fraction target so gross clears the
            # street's increment even when re-raising from a seat that called
            # earlier (a 0.66-pot raise can otherwise crash the engine assert).
            return Action(RAISE, raise_size(state, idx,
                                            bet_size(state, idx, 0.66, state.config.min_bet)))
        if equity >= req:
            return Action(CALL, min(player.stack, to_call))
        return Action(FOLD, 0)


def _wire_ranges_from_table(path: str) -> None:
    """Attach equity-ranked ranges to Aggressive/Passive/Loose from the saved
    preflop table (Stage 03). Deterministic; runs once at import time."""
    table = np.load(path)
    top20 = ranked_types(table, 0.20)
    top25 = ranked_types(table, 0.25)
    top30 = ranked_types(table, 0.30)
    top10 = ranked_types(table, 0.10)
    AggressiveStrategy._play_range = top30
    AggressiveStrategy._raise_range = top20
    PassiveStrategy._play_range = top25
    PassiveStrategy._raise_range = _type_set(
        pairs=(14, 13, 12), suited=((14, 13),), offsuit=((14, 13),))
    AggressiveStrategy._steal_range = top20 | _type_set(
        suited=((12, 11), (11, 10), (10, 9), (9, 8), (8, 7), (7, 6), (6, 5)))
    LooseStrategy._top10 = top10


if os.path.exists(Config().preflop_table_path):
    _wire_ranges_from_table(Config().preflop_table_path)


# Locked Stage-04 contract (stage04.md:19): RANGES[<strategy>] maps each
# strategy's PLAY range to a frozenset of hand-type indices. Defined after
# _wire_ranges_from_table so Aggressive/Passive expose their table-wired
# top-X% sets (not the conservative pre-wiring defaults), while Tight/Loose
# keep their audited literal constants. Stage-05 (adaptive, mathematician)
# strategies consume this dict instead of reaching into private attrs.
RANGES = {
    "Tight": TIGHT_RANGE,
    "Loose": LOOSE_RANGE,
    "Aggressive": AggressiveStrategy._play_range,
    "Passive": PassiveStrategy._play_range,
}
