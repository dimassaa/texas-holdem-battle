"""Player model and strategy decision layer.

Strategies are pure functions of observable state. The engine supplies an
EquityProvider (Stage 03) so strategies never re-implement hand evaluation —
this keeps every strategy's decisions reproducible and its stochastic inputs
(Monte Carlo equity) bounded to the provider.
"""

import os
from collections import deque

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
    # Elimination flag set by the session layer: first-passage ruin (stack < BB,
    # stage05.md) freezes the seat permanently. Ruined seats keep their `rank`
    # forever so session statistics stay anchored to the original seat number.
    ruined: bool = False
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
    """Total wager for a raise the engine always accepts as legal.

    The engine asserts gross == amount - round_bets[actor] >= last_full_raise
    (game.py:192). A fixed 3/4bb target under-shoots that increment from any
    seat with chips already committed this round (the BB re-raising, or anyone
    who called and then pops again), causing an engine AssertionError — the
    crash the Task-4.4/4.5 legality fixes only partially covered. Flooring the
    wager at own + to_call + last_full_raise makes gross exactly to_call +
    increment, which is legal and still a genuine raise from any seat geometry.

    The wager is also capped at own + stack. A raise to exactly your stack is
    the all-in wager the engine's `gross == p.stack` minimum-raise override
    (game.py:192) explicitly accepts. Without the cap a short stack raising a
    large open bet would emit a total wager above its stack and trip
    assert gross <= p.stack (game.py:191) before the engine can clamp it.
    """
    own = state.round_bets.get(idx, 0)
    legal_min = own + state.to_call(idx) + state.last_full_raise
    # Cap at all-in: own + stack is the most this seat can put in; gross then
    # equals the full stack, which the engine's `gross == p.stack` escape
    # (game.py:192) accepts regardless of the street increment.
    return min(max(int(desired), legal_min), own + state.players[idx].stack)


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


@_register
class MathematicianStrategy(Strategy):
    """Strict EV line: every street is equity vs pot odds (plan §5.5).

    Call iff equity >= pot odds; fold otherwise; raise only when equity beats
    pot odds by a 1.15x surplus (the raise is +EV); bet 50-75% pot only with a
    large equity edge and an unopened pot. Preflop additionally folds below
    20% equity (the ~top-20-25% entry band) so trash never invests.

    The plan literal keyed its branch on `state.to_call(idx)`, but its own
    stub tests inject a provider whose to_call differs from the state's empty
    round_bets (test make() never sets them) — and the engine builds the
    provider FROM state (game.py:165), so `provider.to_call` is the same value
    in production and the only one a stub test can see. Decisions therefore
    read decision inputs from the provider, not raw state.
    """

    name = "Mathematician"
    _BET_THRESHOLD = 0.60      # equity edge required to open an unbet pot
    _RAISE_SURPLUS = 1.15      # equity must beat pot odds by this factor to raise
    _FOLD_EQUITY = 0.20        # preflop: below ~20% equity no street is worth it
    # Contract-only range (stage04.md:19 keeps RANGES 1:1 with STRATEGIES):
    # the Mathematician enters on equity, not a discrete list, so this is the
    # ~top-25% stand-in the plan's docstring quotes, table-wired when present
    # (see _wire_ranges_from_table) with TIGHT_RANGE as a conservative
    # off-table fallback so the contract test still sees a valid frozenset.
    _play_range = TIGHT_RANGE

    def _ev_action(self, player, state, idx, provider, entry_floor=0.0):
        """Shared equity-vs-pot-odds decision for the preflop and postflop paths.

        `entry_floor` is the minimum equity a street requires before any
        investment (preflop passes _FOLD_EQUITY; postflop passes 0.0). The
        preflop caller gates below the floor; every legality rule is evaluated
        here so the two paths cannot silently drift apart (Task 5.2 Adaptive
        would otherwise triple this block).
        """
        eq = provider.equity(player.hole, state.board, state.num_opponents(idx))
        if eq < entry_floor:
            return Action(FOLD, 0)
        to_call = provider.to_call
        if to_call == 0:
            # Unopened pot: value-bet only with a big edge AND a stack that
            # covers the minimum wager — the engine caps paid at min(stack,
            # max_raise_amount) and asserts paid >= min_bet (game.py:185).
            if eq >= self._BET_THRESHOLD and state.can_raise(idx) and player.stack >= state.config.min_bet:
                return Action(BET, bet_size(state, idx, 0.6, state.config.min_bet))
            return Action(CHECK, 0)
        pot_odds = provider.pot_odds()
        # Engine legality (game.py:189): RAISE needs owed > 0. The plan's
        # to_call==0 early-return already guards this; the explicit
        # state.to_call(idx) > 0 gate makes the invariant hold even if the
        # branches are ever reordered. raise_size floors the wager at
        # own + to_call + last_full_raise so gross clears the street
        # increment from any committed seat.
        if eq >= self._RAISE_SURPLUS * pot_odds and state.can_raise(idx) and state.to_call(idx) > 0:
            return Action(RAISE, raise_size(state, idx,
                                            bet_size(state, idx, 0.66, state.config.min_bet)))
        if eq >= pot_odds:
            return Action(CALL, min(player.stack, to_call))
        return Action(FOLD, 0)

    def _preflop_action(self, player, state, idx, provider):
        # Entry is by equity-per-field, not a hand class: below 20% vs the
        # opponents the flop is rarely paid off even for free (§5.5).
        return self._ev_action(player, state, idx, provider, entry_floor=self._FOLD_EQUITY)

    def act(self, player, state, idx, provider, rng):
        if state.round_idx == 0:
            return self._preflop_action(player, state, idx, provider)
        return self._ev_action(player, state, idx, provider)


# ---- Task 5.2 shared helpers: window aggregation and table reads ------------

_PREFLOP_ECON = None


def _preflop_table():
    """Lazily load the Stage-03 preflop equity table (169 x n).

    Guarantees the strategy never crashes when the table file is absent
    (pre-table test environments, first run): callers fall back to neutral.
    """
    global _PREFLOP_ECON
    if _PREFLOP_ECON is None:
        import os
        from poker.config import Config
        path = Config().preflop_table_path
        if os.path.exists(path):
            _PREFLOP_ECON = np.load(path)
    return _PREFLOP_ECON


def _range_share(range_set):
    """Share of the 1326-combo universe covered by a 169-class range."""
    if not range_set:
        return 0.0
    return sum(combo_weight(t) for t in range_set) / 1326.0


def _aggregate_window(window):
    """Roll up per-seat stats over the window: fold-to-raise, cbet, aggression.

    Accepts real hand summaries (Task 5.4 producer keys: faces_raise,
    folds_to_raise, cbets, cbet_chances, agg_actions, calls, folds, plays,
    raises) AND synthetic window-dump rows that carry only the decision counts.
    Returns {seat: {hands, fold_to_raise, cbet, aggression, call_share,
    decision_fold}}; None means insufficient evidence (neutral cold-start).
    """
    agg = {}
    for summary in window:
        s = summary.get("seat")
        if s is None:
            continue
        d = agg.setdefault(s, {"hands": 0, "folds_to_raise": 0, "faces_raise": 0,
                               "cbets": 0, "cbet_chances": 0, "agg_actions": 0,
                               "calls": 0, "raises": 0, "folds": 0, "plays": 0})
        d["hands"] += 1
        d["faces_raise"] += summary.get("faces_raise", 0)
        d["folds_to_raise"] += summary.get("folds_to_raise", 0)
        d["cbets"] += summary.get("cbets", 0)
        d["cbet_chances"] += summary.get("cbet_chances", 0)
        d["agg_actions"] += summary.get("agg_actions", 0)
        d["calls"] += summary.get("calls", 0)
        d["raises"] += summary.get("raises", 0)
        d["folds"] += summary.get("folds", 0)
        d["plays"] += summary.get("plays", 0)
    out = {}
    for seat, d in agg.items():
        decided = d["calls"] + d["raises"] + d["folds"]
        out[seat] = {
            "hands": d["hands"],
            "fold_to_raise": (d["folds_to_raise"] / d["faces_raise"]) if d["faces_raise"] else None,
            "cbet": (d["cbets"] / d["cbet_chances"]) if d["cbet_chances"] else None,
            "aggression": (d["agg_actions"] / d["calls"]) if d["calls"] else None,
            "call_share": (d["calls"] / decided) if decided else None,
            "decision_fold": (d["folds"] / d["plays"]) if d["plays"] else None,
        }
    return out


def _table_reads(agg):
    """Map table reads to range/steal adjustments (doc §5.6 quantitative form).

    fold_share  = mean opponents' fold_to_raise, else mean decision_fold,
                  else neutral 0.5 (cold-start).
    call_share  = mean opponents' call_share (None => neutral 0.5).
    Regimes: fold-heavy table => steals profitable, widen play range and raise.
    Loose-calling table (calls >= 70% of decisions) => tighten to value hands.
    Missing table => never crash; play the neutral ranges.

    NOTE: the cbet/aggression aggregates (computed in _aggregate_window) ARE
    exposed via AdaptiveStrategy.stats() for Stage-06 analysis, but the DECISION
    path below reacts only through fold_share. Callers/aggressives keep an
    opponent's fold_share low, sorting the table into the tighten branch — the
    §5.6 "plays tighter vs aggressive opponents" requirement routes through the
    fold-rate signal, not the raw aggression metric.
    """
    ftr = [v["fold_to_raise"] for v in agg.values() if v["fold_to_raise"] is not None]
    if ftr:
        fold_share = float(np.mean(ftr))
    else:
        dfold = [v["decision_fold"] for v in agg.values() if v["decision_fold"] is not None]
        fold_share = float(np.mean(dfold)) if dfold else 0.5
    cs = [v["call_share"] for v in agg.values() if v["call_share"] is not None]
    call_share = float(np.mean(cs)) if cs else 0.5
    table = _preflop_table()
    if table is None:
        # Pre-table environment: neutral ranges (two tight-tier pairs / AK).
        return (_type_set(pairs=(14, 13, 12, 11)),
                _type_set(pairs=(14, 13, 12), suited=((14, 13),), offsuit=((14, 13),)),
                1.0)
    if fold_share > 0.60:
        # Fold-heavy: steals profitable, widen play AND raise ranges.
        return (ranked_types(table, 0.20) | ranked_types(table, 0.30),
                ranked_types(table, 0.30), 1.4)
    if fold_share < 0.35 or call_share >= 0.70:
        # Rare folds or loose callers: bets get called, tighten to value hands.
        return ranked_types(table, 0.15), ranked_types(table, 0.10), 0.7
    return ranked_types(table, 0.20), ranked_types(table, 0.15), 1.0


@_register
class AdaptiveStrategy(Strategy):
    """Learns opponent tendencies from public action history and adjusts.

    Statistics are windowed (config.adaptive_window hands; risk §10) so the
    strategy reacts to recent behavior, not ancient history. Thresholds refresh
    at config.adaptive_adjust_every hands. All reads are from observable
    actions only (doc §1 limitation 2) — never hole cards or strategy identity.

    The §5.6 regimes land on three knobs: fold-heavy opponents widen the play
    and raise ranges (to top-30%) AND raise steal sizing; loose callers tighten
    everything to value hands (top-15%). Postflop, higher opponent fold-to-raise
    lowers the equity a continuation bet needs, so c-bets fire more often into
    tight folders and dry up against callers. "Play tighter vs aggressive
    opponents" is realized preflop: opponents that raise/call instead of folding
    keep fold_share low, sorting the table into the tighten branch.
    """

    name = "Adaptive"
    _FLOORS = {1: 0.45, 2: 0.50, 3: 0.55}
    # Class-level neutral ranges (deviation from the plan literal, which kept
    # only per-instance attrs and therefore crashed the module-level RANGES
    # dict below with AttributeError on import): the class must carry a real
    # frozenset even pre-table, and _wire_ranges_from_table re-wires _play_range
    # to table top-20% in normal runs. The defaults mirror _table_reads' no-
    # table branch so a fresh clone behaves exactly like a cold session.
    _play_range = _type_set(pairs=(14, 13, 12, 11))                            # ~1.8% neutral (JJ+)
    _raise_range = _type_set(pairs=(14, 13, 12), suited=((14, 13),), offsuit=((14, 13),))  # QQ+/AK
    _steal_scale = 1.0

    def __init__(self, memory=None, adjust_every=None):
        self.memory = memory or Config().adaptive_window
        self.adjust_every = adjust_every or Config().adaptive_adjust_every
        self.hands_seen = 0
        self._window = deque(maxlen=self.memory)   # list of per-hand summaries
        # Range caches: start as the class-level defaults (cold-start / pre-table
        # neutral) and _refresh_ranges_if_due overwrites them on each boundary.
        # One source of truth per range — no class-vs-instance None shadowing,
        # so act() can never decide from a missing range.
        self._play_range = type(self)._play_range
        self._raise_range = type(self)._raise_range
        self._steal_scale = 1.0

    # ---- public interface used by the simulator ---------------------------
    def observe(self, hand_summary: dict) -> None:
        """Append one hand's PUBLIC action summary (see Task 5.4 producer)."""
        self._window.append(hand_summary)
        self.hands_seen += 1

    def stats(self):
        """Aggregate the window into per-opponent stats dicts."""
        return _aggregate_window(self._window)

    def window_dump(self, summaries):
        """Test harness: load synthetic hand summaries directly."""
        self._window = deque(summaries, maxlen=self.memory)

    @property
    def play_share(self):
        """Share (of the 1326-combo universe) of the current adjusted play range.

        PROSPECTIVE read: recomputed from the live window, i.e. what the next
        _refresh_ranges_if_due() boundary would cache — NOT necessarily the
        ranges act() is currently using, which update only on refresh
        boundaries and sit in _play_range. Tests feed synthetic tables through
        `window_dump` and assert the resulting tightness/wideness.
        """
        play_range, _, _ = _table_reads(_aggregate_window(self._window))
        return _range_share(play_range)

    @property
    def raise_share(self):
        """Share of the 1326-combo universe of the current adjusted raise range.

        PROSPECTIVE read from the live window (what the next refresh boundary
        would set), not the cached _raise_range act() is currently using.
        """
        _, raise_range, _ = _table_reads(_aggregate_window(self._window))
        return _range_share(raise_range)

    # ---- decision overrides ----------------------------------------------
    def _refresh_ranges_if_due(self):
        """Recompute the cached ranges whenever the hand counter hits a refresh
        boundary.

        Deviation from the plan literal, whose one-shot `_ranges_dirty` flag
        froze the ranges at hand-0 cold reads forever — a strategy that never
        re-reads the table can not adapt (defeats §5.6). hands_seen=0 counts
        as a boundary so a fresh session refreshes immediately with the empty
        window, which aggregates to the neutral cold-start read.
        """
        if self.hands_seen % self.adjust_every != 0:
            return
        self._play_range, self._raise_range, self._steal_scale = _table_reads(
            _aggregate_window(self._window))

    def _fold_bias(self, hero_seat):
        """Mean fold-to-raise among OPPONENTS in the window; excludes the hero's
        own seat because the Task-5.4 simulator feeds each Adaptive brain its
        own summaries too.

        Falls back to opponents' decision-fold (folds/plays) when no raise-facing
        evidence exists, then to a neutral 0.5. Drives the postflop c-bet gate:
        the higher the table's fold rate, the more a continuation bluff folds out.
        """
        agg = _aggregate_window(self._window)
        ftr = [v["fold_to_raise"] for seat, v in agg.items()
               if seat != hero_seat and v["fold_to_raise"] is not None]
        if ftr:
            return float(np.mean(ftr))
        dfold = [v["decision_fold"] for seat, v in agg.items()
                 if seat != hero_seat and v["decision_fold"] is not None]
        if dfold:
            return float(np.mean(dfold))
        return 0.5

    def _preflop_action(self, player, state, idx, provider):
        from poker.equity import start_hand_type
        htype = start_hand_type(player.hole)
        # Caches are always populated (init copies the class defaults); the
        # first _refresh_ranges_if_due on hands_seen=0 overwrites them right
        # before the opening hand is decided.
        play_range = self._play_range
        raise_range = self._raise_range
        steal_scale = self._steal_scale
        if htype not in play_range:
            return Action(FOLD, 0)
        pos = relative_position(state, idx)
        to_call = state.to_call(idx)
        # Any in-range button/cutoff hand steals (no dedicated steal set — that
        # breadth already adapts through play_range); _steal_scale sizes it:
        # ~5.6bb on fold-heavy tables, ~2.8bb against loose callers (§5.6:
        # "steals get bigger" / tighter).
        steal = pos == "late"
        raise_now = (htype in raise_range or steal) and state.can_raise(idx)
        # Engine legality (game.py:189): RAISE requires owed > 0 — gate on
        # to_call > 0. raise_size floors the wager at own + to_call +
        # last_full_raise so gross clears the street increment from any
        # committed seat (the BB re-raising, or a caller popping again).
        if raise_now and state.to_call(idx) > 0:
            base = max(state.config.bb, (3 if not steal else 4 * steal_scale) * state.config.bb)
            return Action(RAISE, raise_size(state, idx, base))
        return Action(CALL, to_call) if to_call <= player.stack else Action(FOLD, 0)

    def act(self, player, state, idx, provider, rng):
        self._refresh_ranges_if_due()
        if state.round_idx == 0:
            return self._preflop_action(player, state, idx, provider)
        equity = provider.equity(player.hole, state.board, state.num_opponents(idx))
        to_call = state.to_call(idx)
        is_aggressor = getattr(state, "preflop_raise_by", None) == idx
        if to_call == 0:
            # Both BET branches below need a legal wager (engine asserts
            # paid >= min_bet, game.py:185) and an aggression slot free.
            if not state.can_raise(idx) or player.stack < state.config.min_bet:
                return Action(CHECK, 0)
            if is_aggressor:
                # C-bet gate scaled inversely to the table's fold-to-raise
                # (§5.6): high fold_bias (tight folders) drops the equity a
                # continuation bet needs toward the 0.45 floor, so the c-bet
                # fires more often; low fold_bias (callers/aggressives) pushes
                # the bar to 0.60 and the semi-bluff dries up. This is the
                # bluff/fold-tight tradeoff reacting to the table.
                fold_bias = self._fold_bias(idx)
                if equity >= max(0.45, 0.60 - 0.25 * fold_bias):
                    return Action(BET, bet_size(state, idx, 0.66, state.config.min_bet))
                return Action(CHECK, 0)
            if equity >= 0.5:
                # Not the aggressor: value-bet only a made hand.
                return Action(BET, bet_size(state, idx, 0.6, state.config.min_bet))
            return Action(CHECK, 0)
        # Facing a bet: mirror the Aggressive equity-raise/call/fold line.
        req = required_equity(provider, state.num_opponents(idx),
                              self._FLOORS[state.round_idx], style_factor=1.05)
        if equity >= req * 1.2 and state.can_raise(idx) and state.to_call(idx) > 0:
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
    MathematicianStrategy._play_range = top25  # its ~top-20-25% entry band (§5.5)
    AdaptiveStrategy._play_range = top20       # its §5.6 base preflop range
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
    "Mathematician": MathematicianStrategy._play_range,
    "Adaptive": AdaptiveStrategy._play_range,
}
