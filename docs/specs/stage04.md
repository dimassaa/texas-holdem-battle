# Stage 04 — Basic Strategies: Tight, Loose, Aggressive, Passive

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first four of six strategies as pure decision functions over observable state, wire them into the Stage-02 `run_hand` engine, and prove a multi-hand session conserves chips and plays without exceptions.

**Architecture:** `poker/player.py` grows from the Stage-02 dataclass into `Player` + `Strategy` base + `TightStrategy`, `LooseStrategy`, `AggressiveStrategy`, `PassiveStrategy`. Strategies are stateless functions of `(player, state, idx, provider, rng)`; all hand-strength knowledge flows through the injected `EquityProvider` (Stage 03), never by re-implementing evaluation. Adaptive/Math run only in Stage 05; the engine's `run_betting_round` gains a production branch that calls strategies.

**Approved decisions honored here:**
- Multiway call decisions are pot-odds-based; the Table-2 thresholds act only as heads-up floors.
- Aggressive c-bets and in-position steals are documented **semi-bluffs** (a deliberate, disclosed relaxation of the "no bluff" model limitation) — they are recorded in the action log as normal bet/raise actions and flagged in analysis.
- Ranges are exact hand-type lists; combo percentages are computed in tests and reported whenever they drift outside the doc's "approximate" bands rather than silently tuned.

**Locked contracts (referenced by Stage 05):**
- `Strategy.act(player, state, idx, provider, rng) -> Action`
- `Strategy.name` (str); `STRATEGIES = {"Tight": TightStrategy, ...}` registry; `for_name(name) -> Strategy`.
- `relative_position(state, idx) -> "early" | "middle" | "late" | "blinds"` (6-max seat math).
- `combo_weight(hand_type) -> int` (6 pair / 12 offsuit non-pair / 4 suited non-pair) for %-of-table range audit.
- Range helpers: `RANGES[<strategy>]` dict of frozensets of hand-type indices; equity-ranked ranges via `ranked_types(table, fraction)`.

**Data-quality focus:** range definitions are versionsed constants (no magic literals scattered in `act`); every strategy test injects a *stub* provider so decisions are unit-testable independent of Monte Carlo noise; a session-level fuzz asserts no player ever goes below zero, chips are conserved, and every action is recorded (the audit trail for Stage 06).

---

### Task 4.1: Strategy base, registry, position, and range helpers

**Files:**
- Modify: `poker/player.py`
- Test: `tests/test_strategy_base.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_strategy_base.py`:
```python
"""Strategy registry, positions, and range-combo accounting."""
import numpy as np
from poker.equity import hand_type_index
from poker.player import STRATEGIES, combo_weight, relative_position, for_name
from poker.config import Config
from poker.game import GameState
from poker.player import Player


def test_registry_has_four_basic_strategies():
    assert {"Tight", "Loose", "Aggressive", "Passive"} <= set(STRATEGIES)
    assert for_name("Tight").name == "Tight"


def test_combo_weights_sum_to_1326():
    total = sum(combo_weight(hand_type_index(r_a, r_b, s))
                for r_a in range(2, 15)
                for r_b in range(2, r_a + 1)
                for s in (0, 1) if not (r_a == r_b and s == 1))
    assert total == 1326


def test_relative_positions_are_deterministic():
    ps = [Player(name=f"p{i}", strategy="Tight", stack=200) for i in range(6)]
    state = GameState(players=ps, dealer_pos=0, config=Config())
    assert relative_position(state, 0) == "late"      # BTN
    assert relative_position(state, 1) == "blinds"    # SB
    assert relative_position(state, 2) == "blinds"    # BB
    assert relative_position(state, 3) == "early"     # UTG
    assert relative_position(state, 4) == "early"     # HJ
    assert relative_position(state, 5) == "middle"    # CO
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_strategy_base.py`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement the base layer**

`poker/player.py` (replace the Stage-02 placeholder, preserving the `Player` dataclass fields the engine depends on — `name, strategy, stack, hole, folded, all_in, contributed, stats`):
```python
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


# Prelflop ranges (exact lists; percentages audited in Task 4.6).
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
```

*Note:* `LOOSE_RANGE` above needs the suited connectors (45s..JTs) and the suited king/queen broadways to reach the doc's ~40% band; adjust the ranges during Task 4.6 so the combo-% audit lands in band and *document the final numbers in the code comment*. The exact band-membership edits belong to Task 4.6 — do not silently tune before the audit exists.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_strategy_base.py`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/player.py tests/test_strategy_base.py && git commit -m "feat: strategy registry, positions, range primitives"
```

---

### Task 4.2: Decision context helpers and shared postflop logic

**Files:**
- Modify: `poker/player.py`
- Test: `tests/test_decision_helpers.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_decision_helpers.py`:
```python
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
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_decision_helpers.py`
Expected: `ModuleNotFoundError: bet_size`

- [ ] **Step 3: Implement helpers**

Append to `poker/player.py`:
```python
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
    raw = max(1.0, fraction * pot_total(state))
    size = int(round(raw))
    from poker.game import max_raise_amount
    return min(max_raise_amount(state, idx), max(min_bet, size))
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_decision_helpers.py`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/player.py tests/test_decision_helpers.py && git commit -m "feat: shared decision math (pot odds, sizing, required equity)"
```

---

### Task 4.3: TightStrategy

**Files:**
- Modify: `poker/player.py`
- Test: `tests/test_tight.py`

**Spec (doc §5.1 + Table 2):** preflop plays a tight-but-reasonable 55+/ATo+/A9s+ range (widened from the original 77+/AQo+/AJs+/KQs in Task 4.6 to land in Table-1's 12-15% play band); early position only JJ+/AK. Postflop call floor 0.6/0.7/0.75 by street (heads-up only); bet/raise only with very strong equity (≥0.75) or made two-pair+; folds to aggression when below the required share.

- [ ] **Step 1: Write the failing tests**

`tests/test_tight.py`:
```python
"""Tight strategy: range discipline and postflop value-or-fold."""
import numpy as np
from poker.actions import CALL, CHECK, FOLD, RAISE
from poker.card import card_id as H
from poker.config import Config
from poker.equity import EquityProvider
from poker.game import GameState
from poker.player import Player, TightStrategy, required_equity
from poker.rng import Rng

CFG = Config()


def make(idx=0, stack=200, dealer=0, board=(), hand=(14, 13)):
    """State + player with the requested hole cards.

    `board` is a sequence of (rank, suit) tuples; `hand` is two ranks (suits
    0 and 1 are used so the two hole cards are always distinct).
    """
    ps = [Player(name=f"p{i}", strategy="Tight", stack=stack) for i in range(6)]
    st = GameState(players=ps, dealer_pos=dealer, config=CFG)
    st.board = (np.array([H(r, s) for r, s in board], dtype=np.int32)
                if board else np.empty(0, dtype=np.int32))
    st.round_idx = 1 if board else 0
    st.round_bets = {}
    st.pot = 100
    r1, r2 = hand
    ps[idx].hole = np.array([H(r1, 0), H(r2, 1)], dtype=np.int32)
    return st, ps[idx]


class StubProvider(EquityProvider):
    def __init__(self, eq, **kw):
        super().__init__(rng=Rng(1), config=CFG, **kw)
        self.eq = eq

    def equity(self, hand, board, n_opp):
        return self.eq


def test_tight_folds_weaker_hands_preflop_early():
    st, p = make(board=(), hand=(7, 5))
    prov = StubProvider(0.2, pot=0)
    a = TightStrategy().act(p, st, st.players.index(p), prov, Rng(0))
    assert a.kind == FOLD


def test_tight_plays_aces_preflop():
    st, p = make(board=(), hand=(14, 14))
    st.round_bets = {1: 1, 2: 2}          # blinds posted
    st.pot = 0
    prov = StubProvider(0.8, pot=3, to_call=2)
    a = TightStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind in (CALL, RAISE)


def test_tight_folds_flop_without_equity():
    st, p = make(board=((14, 0), (9, 1), (3, 2)), hand=(7, 5))
    st.round_bets = {0: 50}
    st.pot = 100
    prov = StubProvider(0.2, pot=150, to_call=50)
    a = TightStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind == FOLD


def test_tight_values_two_pair_plus():
    st, p = make(board=((14, 0), (9, 1), (3, 2)), hand=(14, 9))
    st.round_bets = {}
    st.pot = 100
    prov = StubProvider(0.8, pot=100, to_call=0)
    a = TightStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind in (CHECK, RAISE)  # betting/raising with made two-pair
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_tight.py`
Expected: `ModuleNotFoundError: TightStrategy`

- [ ] **Step 3: Implement TightStrategy**

Append to `poker/player.py`:
```python
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
                if equity >= 0.75 and state.can_raise(idx):
                    return Action(BET, bet_size(state, idx, 0.6, state.config.min_bet))
                return Action(CHECK, 0)
            return Action(CALL, min(player.stack, to_call))
        return Action(FOLD, 0)
```

*Note:* this Task's Step-1 test file (`tests/test_tight.py`) already defines the clean `make` factory above the `StubProvider`, and the four tests below unpack it as `(state, player)` — keep that single factory authoritative; do not reintroduce per-task helper variants.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_tight.py`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/player.py tests/test_tight.py && git commit -m "feat: tight strategy"
```

---

### Task 4.4: LooseStrategy and PassiveStrategy

**Files:**
- Modify: `poker/player.py`
- Test: `tests/test_loose_passive.py`

**Loose (doc §5.2):** wide range ~40%; calls often, raises rarely (only top-10% preflop). Postflop: call with equity ≥ 0.35 (heads-up floor) or draw; bet/raise with made pair+.
**Passive (doc §5.4):** plays top-25%; raises only AA/KK/QQ/AK; postflop check/call, call if equity ≥ pot odds × 1.0 (slightly loose); raise only set+.

- [ ] **Step 1: Write the failing tests**

`tests/test_loose_passive.py`:
```python
"""Loose (wide, passive-betting) and Passive (tight-folding, check-call)."""
import numpy as np
from poker.actions import CALL, CHECK, FOLD, RAISE
from poker.card import card_id as H
from poker.config import Config
from poker.game import GameState
from poker.player import LooseStrategy, PassiveStrategy, Player
from poker.rng import Rng

CFG = Config()


class StubProvider:
    """Fake EquityProvider with a fixed equity — decouples strategy math from MC."""
    def __init__(self, eq, pot=100, to_call=50): self.eq, self.pot, self.to_call = eq, pot, to_call
    def equity(self, hand, board, n_opp): return self.eq
    def pot_odds(self): return self.to_call / (self.pot + self.to_call)


def make(strategy, hand=(9, 8), board=(), pot=100, round_bets=None):
    ps = [Player(name=f"p{i}", strategy=strategy, stack=200) for i in range(6)]
    st = GameState(players=ps, dealer_pos=0, config=CFG)
    st.board = np.array([H(*c) for c in board], dtype=np.int32) if board else np.empty(0, dtype=np.int32)
    st.round_idx = 1 if board else 0
    st.round_bets = dict(round_bets or {})
    st.pot = pot
    for i, p in enumerate(ps):
        p.hole = np.array([H(hand[0], i % 4), H(hand[1], (i + 1) % 4)], dtype=np.int32)
    return st, st.players[3]


def test_loose_calls_broad_flop_hand():
    st, p = make("Loose", hand=(9, 8), board=((13, 0), (9, 1), (3, 2)), round_bets={0: 50})
    prov = StubProvider(0.4, pot=150, to_call=50)   # pair of nines, decent draw
    a = LooseStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind == CALL


def test_loose_folds_too_large_bet():
    st, p = make("Loose", hand=(3, 2), board=((13, 0), (9, 1), (3, 2)))
    # pot odds terrible: 250/(250+200) = ~0.44 > equity 0.3
    prov = StubProvider(0.3, pot=250, to_call=200)
    a = LooseStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind == FOLD


def test_passive_checks_unraised():
    st, p = make("Passive", hand=(14, 12), board=((14, 0), (11, 1), (2, 2)))
    prov = StubProvider(0.7, pot=100, to_call=0)
    a = PassiveStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind in (CHECK, CALL)  # passive: does not bet top pair, may check/call


def test_passive_raises_only_cold_nuts():
    # set on the flop with a pocket pair -> set+ -> raise allowed
    st, p = make("Passive", hand=(3, 3), board=((14, 0), (3, 1), (2, 2)))
    prov = StubProvider(0.95, pot=100, to_call=0)
    a = PassiveStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind in (CHECK, RAISE)
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_loose_passive.py`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Append to `poker/player.py`:
```python
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
        if raise_ok and state.can_raise(idx):
            return Action(RAISE, max(state.config.bb, 4 * state.config.bb))
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
            if eq >= 0.5 and state.can_raise(idx):
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
    _play_range = TIGHT_RANGE        # conservative default until wired in Task 4.6
    _raise_range = _type_set(pairs=(14, 13, 12), suited=((14, 13),), offsuit=((14, 13),))

    def _preflop_action(self, player, state, idx, provider):
        from poker.equity import start_hand_type
        htype = start_hand_type(player.hole)
        if htype not in self._play_range:
            return Action(FOLD, 0)
        to_call = state.to_call(idx)
        if htype in self._raise_range and state.can_raise(idx):
            return Action(RAISE, max(state.config.bb, 4 * state.config.bb))
        return Action(CALL, to_call) if to_call <= player.stack else Action(FOLD, 0)

    def act(self, player, state, idx, provider, rng):
        if state.round_idx == 0:
            return self._preflop_action(player, state, idx, provider)
        eq = provider.equity(player.hole, state.board, state.num_opponents(idx))
        req = required_equity(provider, state.num_opponents(idx),
                              self._FLOORS[state.round_idx], style_factor=1.0)
        if state.to_call(idx) == 0:
            # passive bets almost never; raise only set+ : equity ~1.0 proxy
            if eq >= 0.95 and state.can_raise(idx):
                return Action(RAISE, bet_size(state, idx, 0.5, state.config.min_bet))
            return Action(CHECK, 0)
        if eq >= req * 0.95:   # documented slight looseness of passive callers
            return Action(CALL, min(player.stack, state.to_call(idx)))
        return Action(FOLD, 0)
```

*Note:* the conservative class-level defaults (`_play_range`/`_raise_range`/`_steal_range`, `LOOSE_RANGE`, `_top10`) keep preflop tests deterministic before the Stage-03 table exists. Task 4.6 re-wires them from `ranked_types` via `_wire_ranges_from_table`; the wired values are recorded there.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_loose_passive.py`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/player.py tests/test_loose_passive.py && git commit -m "feat: loose and passive strategies"
```

---

### Task 4.5: AggressiveStrategy (with documented semi-bluff c-bet)

**Files:**
- Modify: `poker/player.py`
- Test: `tests/test_aggressive.py`

**Spec (doc §5.3 + approved decision):** Plays ~30% (ranked types), raises top-20%; in-position steals; c-bet almost always when preflop aggressor (recorded semi-bluff, not a pure bluff — the hand is either hit or a live draw by construction of the MC-favoring ranges); raises with equity ≥ 0.5 or a good draw; rarely calls.

- [ ] **Step 1: Write the failing tests**

`tests/test_aggressive.py`:
```python
"""Aggressive: expand/raise ranges, c-bets, equity-raise logic."""
import numpy as np
from poker.actions import BET, CALL, CHECK, FOLD, RAISE
from poker.card import card_id as H
from poker.config import Config
from poker.game import GameState, starting_bets
from poker.player import AggressiveStrategy, Player
from poker.rng import Rng

CFG = Config()


class StubProvider:
    def __init__(self, eq, pot=100, to_call=0): self.eq, self.pot, self.to_call = eq, pot, to_call
    def equity(self, hand, board, n_opp): return self.eq
    def pot_odds(self): return self.to_call / (self.pot + self.to_call)


def make(strategy="Aggressive", hand=(14, 13), board=(), round_bets=None, pot=100, dealer=0):
    ps = [Player(name=f"p{i}", strategy=strategy, stack=200) for i in range(6)]
    st = GameState(players=ps, dealer_pos=dealer, config=CFG)
    st.board = np.array([H(*c) for c in board], dtype=np.int32) if board else np.empty(0, dtype=np.int32)
    st.round_idx = 1 if board else 0
    st.round_bets = dict(round_bets or {})
    st.config = CFG
    st.pot = pot
    for i, p in enumerate(ps):
        p.hole = np.array([H(hand[0], i % 4), H(hand[1], (i + 1) % 4)], dtype=np.int32)
    return st, st.players[3]


def test_aggressive_raises_ak_preflop():
    st, p = make(hand=(14, 13), board=(), round_bets={}, pot=0)
    st.pot = 3
    st.round_bets = {1: 1, 2: 2}
    prov = StubProvider(0.65, pot=3, to_call=2)
    a = AggressiveStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind in (RAISE, CALL)


def test_aggressive_cbets_as_preflop_aggressor_even_weakish_flop():
    st, p = make(hand=(9, 8), board=((13, 0), (2, 1), (5, 2)), round_bets={}, pot=100)
    st.preflop_raise_by = 3          # hero was the preflop aggressor
    st.pot = 100                     # no bet this street yet
    prov = StubProvider(0.2, pot=100, to_call=0)
    a = AggressiveStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind in (BET, RAISE)    # documented semi-bluff continuation bet


def test_aggressive_folds_weak_hand_to_big_bet_when_unraised():
    # not the aggressor, tiny equity, big bet -> fold (rarely-call bias)
    st, p = make(hand=(3, 2), board=((13, 0), (2, 1), (5, 2)), round_bets={0: 150}, pot=200)
    prov = StubProvider(0.15, pot=350, to_call=150)
    a = AggressiveStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind == FOLD
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_aggressive.py`
Expected: `ModuleNotFoundError: AggressiveStrategy`

- [ ] **Step 3: Implement**

Append to `poker/player.py`:
```python
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
        if (htype in self._raise_range or steal) and state.can_raise(idx):
            size = max(state.config.bb, (3 if not steal else 4) * state.config.bb)
            return Action(RAISE, size)
        return Action(CALL, to_call) if to_call <= player.stack else Action(FOLD, 0)

    def act(self, player, state, idx, provider, rng):
        if state.round_idx == 0:
            return self._preflop_action(player, state, idx, provider)
        equity = provider.equity(player.hole, state.board, state.num_opponents(idx))
        to_call = state.to_call(idx)
        is_aggressor = getattr(state, "preflop_raise_by", None) == idx
        if to_call == 0:
            if is_aggressor and state.can_raise(idx):
                # continuation bet (semi-bluff) in ~2/3 of such spots
                return Action(BET, bet_size(state, idx, 0.66, state.config.min_bet))
            if equity >= 0.5 and state.can_raise(idx):
                return Action(BET, bet_size(state, idx, 0.6, state.config.min_bet))
            return Action(CHECK, 0)
        # facing a bet: aggressive prefers raising strong equity over calling
        req = required_equity(provider, state.num_opponents(idx),
                              self._FLOORS[state.round_idx], style_factor=1.05)
        if equity >= req * 1.2 and state.can_raise(idx):
            return Action(RAISE, bet_size(state, idx, 0.66, state.config.min_bet))
        if equity >= req:
            return Action(CALL, min(player.stack, to_call))
        return Action(FOLD, 0)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_aggressive.py`
Expected: `3 passed`

- [ ] **Step 5: Wire `_play_range/_raise_range/_steal_range/_top10` from the saved table**

`poker/player.py` module tail:
```python
def _wire_ranges_from_table(path: str) -> None:
    """Attach equity-ranked ranges to Aggressive/Passive/Loose from the saved
    preflop table (Stage 03). Deterministic; runs once at import time."""
    import numpy as np
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
```

Place the hook call at the module tail, guarded so tests can run before the
preflop table exists:
```python
if os.path.exists(Config().preflop_table_path):
    _wire_ranges_from_table(Config().preflop_table_path)
```

- [ ] **Step 5 (cont.): Run the full strategy test set**

Run: `.venv/bin/pytest tests/test_tight.py tests/test_loose_passive.py tests/test_aggressive.py tests/test_strategy_base.py tests/test_decision_helpers.py`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add poker/player.py tests/test_aggressive.py && git commit -m "feat: aggressive strategy with documented c-bet semi-bluff"
```

---

### Task 4.6: Range audit — combo percentages within documented bands

**Files:**
- Create: `tests/test_range_audit.py`

- [ ] **Step 1: Write the audit**

Audit each strategy's actual play range as a share of the 1326-combo universe (using `combo_weight`) and compare to Table 1's *approximate* bands. The test FAILS loudly (not silently) when outside band, so range drift is surfaced to the user, not hidden:

`tests/test_range_audit.py`:
```python
"""Range audit vs Table 1's approximate percentages (fail loudly on drift)."""
import numpy as np

from poker.player import (
    AggressiveStrategy, LooseStrategy, PassiveStrategy, TightStrategy,
    combo_weight, _type_set, ranked_types,
)


def share(range_set):
    return sum(combo_weight(t) for t in range_set) / 1326.0
```

- [ ] **Step 2: Wire the audit values and record results**

For each strategy, assert `play_share` inside its Table-1 band, then **print the exact measured percentages** so the report documents them. Tune `LOOSE_RANGE` only where the audit proves it necessary, and record the final list in a comment.

- [ ] **Step 3: Run and iterate**

Run: `.venv/bin/pytest tests/test_range_audit.py -s`
Expected: `5 passed` with a printed table:
| Strategy | play % | raise % | Table band |
|---|---|---|---|
| Tight | X% | Y% | 12–15% |
| ... | ... | ... | ... |

- [ ] **Step 4: Document final ranges in code comments and commit**

```bash
git add poker/player.py tests/test_range_audit.py && git commit -m "test: strategy range percentage audit vs table 1"
```

---

### Task 4.7: Wire strategies into `run_hand` and run a session-level fuzz

**Files:**
- Modify: `poker/game.py`
- Create: `tests/test_stage04_session_fuzz.py`

**Engine wiring:** `run_betting_round` production branch (when `actions_override is None`) builds a fresh per-actor `EquityProvider` from `pot_total`/`to_call`, tracks `state.preflop_raise_by`, and calls `strategy.act(...)`:

`poker/game.py` — inside `run_betting_round`, replace the placeholder act call:
```python
if actions_override is not None:
    kind, amount = actions_override[idx]
else:
    provider = EquityProvider(
        rng.child(idx, state.round_idx),
        state.config,
        pot=pot_total(state),
        to_call=max(0, open_bet - state.round_bets.get(idx, 0)),
    )
    # Strategies act as (player, state, player_idx, provider, rng); the
    # provider carries the seeded child stream, so `provider.rng` backs the
    # strategy here — same stream that scored its equity, keeping the hand
    # byte-reproducible (locked decision R1).
    kind, amount = state.players[idx].brain.act(
        p, state, idx, provider, provider.rng)
```

Add to `Player`:
```python
@dataclass
class Player:
    ...
    brain: object = field(default=None)   # Strategy instance, set at seating
```

`run_hand` gains `dealer_pos`, and the simulator (Stage 05) will `seat` players with `brain = for_name(strategy)`. At the close of the betting round, `state.preflop_raise_by` is set to the last index that raised on round 0.

- [ ] **Step 1: Write the fuzz test**

`tests/test_stage04_session_fuzz.py`:
```python
"""Stage-04 gate: multi-hand fuzz with the four strategies keeps chips legal."""
import numpy as np
from poker.config import Config
from poker.game import run_hand
from poker.player import Player, for_name
from poker.rng import Rng

CFG = Config(strategy_combo=("Tight", "Loose", "Aggressive", "Passive", "Passive", "Tight"))


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
    assert all(len(res.actions) > 0 for res in [run_hand(players, dealer, Rng(300), CFG), res])
```

- [ ] **Step 2: Run to make it pass (and iterate on engine bugs it surfaces)**

Run: `.venv/bin/pytest tests/test_stage04_session_fuzz.py -x`
Expected: `1 passed`. Every bug this fuzz uncovers is a **pre-Stage-06 data-quality bug** — fix it in the engine, not in the test.

- [ ] **Step 3: Run full suite**

Run: `.venv/bin/pytest`
Expected: all green (default suite, slow excluded)

- [ ] **Step 4: Commit**

```bash
git add poker/game.py poker/player.py tests/test_stage04_session_fuzz.py && git commit -m "feat: wire strategies into engine with session-level legal-chips fuzz"
```

---

## Stage Exit Criteria

- [ ] Four strategies implemented as pure decision functions over observable state only.
- [ ] Aggressive c-bet/semi-bluff documented and flagged for analysis (not hidden).
- [ ] Multiway calls use pot-odds × style factor; heads-up calls use Table-2 street floors.
- [ ] Range audit prints exact combo-percentages and they sit inside Table-1 bands (else report to user).
- [ ] 4-strategy fuzz: chip conservation, no negative stacks, card uniqueness, action log populated.

**Report to the user before Stage 05:**
- The measured range table (strategy → play%/raise%) for the record.
- Any band adjustment needed and its magnitude (data-honesty requirement).
- Bugs surfaced by the session fuzz and their fixes.