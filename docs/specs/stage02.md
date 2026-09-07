# Stage 02 — Betting Engine: Pot-Limit, All-In, Side Pots, Hand Cycle

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full deal → bet → board → showdown → payout hand cycle with pot-limit rules, the 4-aggression cap, and correct all-in/side-pot handling, validated by scripted scenarios and conservation invariants.

**Architecture:** `poker/game.py` owns `GameState`, the betting-round loop, side-pot construction, and `run_hand`. `poker/actions.py` holds Action kinds. Stack updates and the "ruin = stack < BB" first-passage check (approved decision: seat frozen, game continues with fewer players) live at the session layer in Stage 05, but the per-hand stack mutation happens here.

**Why this stage is the data-quality keystone:** any pot-splitting error propagates into every EV/BB/ruin metric. Therefore every betting round is tested for explicit conservation invariants, not just expected pot sizes.

**Locked contracts (referenced later):**
- `actions.py`: constants `FOLD`, `CHECK`, `CALL`, `BET`, `RAISE`, `ALLIN` (strings) and `Action = namedtuple("Action", "kind amount")`. `amount` is only meaningful for `BET`/`RAISE`/`ALLIN`/`CALL` (call amount).
- `game.py`:
  - `GameState` dataclass with fields: `players`, `board`, `round_idx`, `pot`, `round_bets`, `dealer_pos`, `config`, `history`, and computed helpers `to_call(player_idx)`, `num_opponents(player_idx)`, `can_raise(player_idx)`, `aggro_count`.
  - `run_betting_round(state) -> None` — loops until round complete; raises capped by `config.max_aggressions`; an all-in shove does not count toward the cap; a short all-in raise does not reopen betting to players who already acted.
  - `build_side_pots(players) -> list[SidePot]`, `SidePot = (amount, eligible: tuple[player_idx,...])` — level widths count all contributors, eligibility excludes folded players, and dead chips (folded money above every live contribution) roll into the live pot beneath them.
  - `run_hand(players, dealer_pos, rng, config, actions_override=None) -> HandResult` — full hand; `HandResult` carries `board`, `actions`, `side_pots`, `stacks_before`, `stacks_after`, `net`, `pot_total`, `ruined` (ruin flags are filled in by the Stage-05 session layer).

**Data-quality invariants enforced by tests, each hand:**
1. Sum of side-pot amounts == sum of all players' total contributed.
2. Sum of net stack changes == 0 (no chips created or destroyed).
3. No player's contributions ever exceed their starting stack for the hand.
4. Every side-pot's eligibility matches the players who can win it (all covered by `eligible`; winner only from eligible).
5. Deck used in a hand has no card duplication (two hole cards × players + board are all distinct).

---

### Task 2.1: Actions

**Files:**
- Create: `poker/actions.py`
- Test: `tests/test_actions.py`

- [ ] **Step 1: Write the failing test**

`tests/test_actions.py`:
```python
"""Action vocabulary is a stable constant set."""
from poker.actions import (
    ALLIN, BET, CALL, CHECK, FOLD, RAISE, ALL_ACTION_KINDS, Action,
)


def test_action_is_namedtuple_with_kind_and_amount():
    a = Action(BET, 10)
    assert a.kind == BET and a.amount == 10


def test_all_kinds_are_distinct():
    assert len(set(ALL_ACTION_KINDS)) == 6
    assert FOLD not in (CHECK, CALL, BET, RAISE, ALLIN)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_actions.py`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`poker/actions.py`:
```python
"""Action vocabulary.

A stable, documented set of string constants so strategy code and logging agree
on spellings and no typo can corrupt a hand's recorded history.
"""

from collections import namedtuple

FOLD = "fold"
CHECK = "check"
CALL = "call"
BET = "bet"
RAISE = "raise"
ALLIN = "all-in"

ALL_ACTION_KINDS = (FOLD, CHECK, CALL, BET, RAISE, ALLIN)

# amount: size of the chip wager for BET/RAISE/CALL/ALLIN, else 0.
Action = namedtuple("Action", "kind amount")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_actions.py`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/actions.py tests/test_actions.py && git commit -m "feat: stable action vocabulary"
```

---

### Task 2.2: GameState and pot-limit raise math

**Files:**
- Create: `poker/game.py` (state part only here)
- Test: `tests/test_game_state.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_game_state.py`:
```python
"""GameState construction, query helpers, and pot-limit raise formula."""
import pytest

from poker.config import Config
from poker.game import GameState, max_raise_amount, starting_bets
from poker.player import Player


def make_players(n=6, stack=200):
    return [Player(name=f"p{i}", strategy="Tight", stack=stack) for i in range(n)]


def test_starting_bets_posts_blinds_and_sets_round_state():
    players = make_players()
    state = GameState(players=players, dealer_pos=3, config=Config())
    starting_bets(state)
    bb_idx = (state.dealer_pos + 1) % 6
    sb_idx = state.dealer_pos
    assert state.round_bets[sb_idx] == 1
    assert state.round_bets[bb_idx] == 2
    # preflop "pot" is 0 until the street closes; blinds live in round_bets so
    # adding them a second time at street finalization is impossible.
    assert state.pot == 0
    assert sum(state.round_bets.values()) == 3
    # each stack reduced by the posted blind
    assert players[sb_idx].stack == 199
    assert players[bb_idx].stack == 198


def test_pot_limit_max_raise_matches_doc_formula():
    players = make_players()
    state = GameState(players=players, dealer_pos=3, config=Config())
    starting_bets(state)
    # no further action yet: blinds total 3 in the round, UTG must call 2.
    # Doc formula: MaxRaise = pot + 2 * call_amount = (0 + 3) + 2*2 = 7.
    utg_idx = (state.dealer_pos + 2) % 6
    assert max_raise_amount(state, utg_idx) == 7
    assert state.to_call(utg_idx) == 2
    assert state.can_raise(utg_idx) is True


def test_aggro_cap_enforced_at_config_boundary():
    players = make_players()
    state = GameState(players=players, dealer_pos=3, config=Config())
    starting_bets(state)
    state.aggro_count = state.config.max_aggressions
    assert state.can_raise(3) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_game_state.py`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`poker/game.py` (state + raise math only in this task; the round loop is Task 2.3):
```python
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
```

- [ ] **Step 4: Wire up `poker/player.py` placeholder now**

`poker/player.py` (minimal — conventional strategies replace it wholesale in Stage 04):
```python
"""Player model used by the engine.

Stage 02 needs only the mutable chip state (stack, contributed, folded,
all_in); full strategy classes are implemented in Stage 04/05. The dataclass
is intentionally behavior-free to keep engine tests strategy-agnostic.
"""

from dataclasses import dataclass, field

import numpy as np


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
```

Also add to `Player`:
```python
    rank: int = 0          # seat index, set by the simulator when seating
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_game_state.py`
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add poker/game.py poker/player.py tests/test_game_state.py && git commit -m "feat: game state and pot-limit raise formula"
```

---

### Task 2.3: Betting-round loop with all-in, short-raise reopen and aggro cap

**Files:**
- Modify: `poker/game.py`
- Test: `tests/test_betting_round.py`

**Rules implemented here (from the technical document + approved decisions):**
- Action order preflop starts UTG (left of BB); postflop starts left of button (SB). In full-ring terms `first_to_act` = `(dealer_pos + sb_step) % n`.
- A round ends when all non-folded players have matched the highest wager (or are all-in).
- `MAX 4 aggressions/round`: `BET`/`RAISE` count toward `aggro_count`; an `ALLIN` does not.
- A raise below the previous full raise (short all-in) does **not** reopen action to players who already faced the larger bet.
- A player who has already matched the current wager may `CHECK` (if no wager) or `CALL`/`RAISE`; otherwise must `FOLD`, `CALL`, or `RAISE` (may `ALLIN` if insufficient).
- Pot-limit max raise uses `max_raise_amount`. `CALL` is capped at the player's stack; if the capped call is < the required amount the player is all-in with the call.

- [ ] **Step 1: Write the failing tests**

`tests/test_betting_round.py`:
```python
"""Betting-round loop: staging, capping, all-in recognition, reopen rule."""
import numpy as np

from poker.actions import BET, CALL, CHECK, FOLD, RAISE
from poker.config import Config
from poker.game import GameState, max_raise_amount, run_betting_round, starting_bets
from poker.player import Player

CFG = Config()


def players_n(stacks):
    return [Player(name=f"p{i}", strategy="X", stack=s) for i, s in enumerate(stacks)]


def test_blinds_paid_and_round_closes_when_all_fold():
    ps = players_n([200] * 6)
    st = GameState(players=ps, dealer_pos=0, config=CFG)
    starting_bets(st)
    # Everyone folds preflop; the street must close with the blinds moved into
    # `pot` exactly once (round_bets cleared for the next street).
    actions = {i: (FOLD, 0) for i in range(6)}
    run_betting_round(st, actions_override=actions)
    assert st.pot == 3
    assert st.round_bets == {}
    assert [p.stack for p in ps] == [199, 198, 200, 200, 200, 200]


def test_raise_all_fold_matches_hands():
    ps = players_n([200] * 6)
    st = GameState(players=ps, dealer_pos=0, config=CFG)
    starting_bets(st)
    # UTG = index 2 raises pot-limit max; SB/BB call it, everyone else folds.
    raise_to = max_raise_amount(st, 2)   # (0 + 3 blinds) + 2*2 = 7
    actions = {0: (CALL, 0), 1: (CALL, 0), 2: (RAISE, raise_to), 3: (FOLD, 0),
               4: (FOLD, 0), 5: (FOLD, 0)}
    run_betting_round(st, actions_override=actions)
    # three players each put the full 7 in; the street closed into the pot
    assert st.pot == 3 * raise_to
    assert st.round_bets == {}


def test_short_allin_does_not_force_extra_turn_on_matched_players():
    ps = players_n([200, 200, 200, 200, 6, 200])
    st = GameState(players=ps, dealer_pos=0, config=CFG)
    starting_bets(st)
    # p2 raises to 7, p3 calls; p4 (6 chips) goes all-in for a *short* raise
    # that stays below the 7 open wager, so it must NOT reopen p2/p3. If the
    # loop wrongly reopens them, `actions_override` has no entry for a second
    # visit -> KeyError fails the test.
    actions = {2: (RAISE, 7), 3: (CALL, 0), 4: (ALLIN, 0), 5: (FOLD, 0),
               0: (CALL, 0), 1: (CALL, 0)}
    run_betting_round(st, actions_override=actions)
    assert st.aggro_count == 1          # ALLIN is free under the aggro cap
    assert ps[4].all_in and ps[4].stack == 0
    assert st.round_bets == {}          # street closed into the pot
    assert st.pot == 7 + 7 + 6 + 7 + 7  # p2, p3, p4(short), p0, p1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_betting_round.py`
Expected: `ModuleNotFoundError: run_betting_round`

- [ ] **Step 3: Implement the betting-round loop**

Append to `poker/game.py` exactly this code (single authoritative formulation):
```python
def _setting_first_actor(state: "GameState") -> int:
    """Index of the first actor for this street.

    Preflop: UTG (left of the BB). Postflop: the SB seat (left of the button).
    Eliminated seats are removed upstream at the session layer, so live indices
    are contiguous; blinds still sit on real seats.
    """
    if state.round_idx == 0:
        return (state.dealer_pos + 2) % len(state.players)
    return (state.dealer_pos + 1) % len(state.players)


def _round_finished(state: "GameState") -> bool:
    """Every non-folded, non-all-in player has matched the highest open wager."""
    open_bet = max(state.round_bets.values(), default=0)
    for idx, p in enumerate(state.players):
        if p.folded or p.all_in:
            continue
        if state.round_bets.get(idx, 0) < open_bet:
            return False
    return True


def run_betting_round(state: "GameState", actions_override=None) -> None:
    """Execute one betting street with the full rule set.

    Reopen rule (doc requirement): a short all-in raise does not reopen players
    who already matched the larger bet. The loop walks seats forever forward;
    a seat gets a new turn only when the open wager increased *after* its last
    turn (i.e. a full raise happened behind it) — that is exactly when the round
    is legally still open for it. When the walk returns to a seat whose
    last-acting open bet equals the current one AND everyone has matched, the
    street closes. This avoids lap counters entirely and matches real poker:
    a matched player with nothing to do is never asked to "act" again.

    `actions_override` is the Stage-02 test harness: {player_idx: (kind, amount)}
    consumed when set. Production passes None and strategies act via
    `Player.brain.act(...)` (wired in Stage 04).
    """
    n = len(state.players)
    idx = _setting_first_actor(state)
    last_full_raise = state.config.bb   # next bet/raise increment must reach this
    acted_at = {}                       # seat -> open_bet when it last acted
    safety = 0
    while True:
        safety += 1
        if safety > n * 8:
            raise RuntimeError("betting round did not converge")
        open_bet = max(state.round_bets.values(), default=0)
        # Street closed when either:
        #  (a) one player is left to collect an uncontested pot, or
        #  (b) nothing is left that could act (every seat folded or all-in), or
        #  (c) action is back at a seat that already acted under this exact
        #      open wager and everyone has matched it.
        live = [p for p in state.players if not p.folded]
        if len(live) <= 1:
            break
        if _round_finished(state) and (open_bet == acted_at.get(idx)
                                       or all(p.folded or p.all_in
                                              for p in state.players)):
            break
        p = state.players[idx]
        if p.folded or p.all_in:
            idx = (idx + 1) % n
            continue

        if actions_override is not None:
            kind, amount = actions_override[idx]
        else:
            kind, amount = p.act(state, idx)   # resolved in Stage 04

        owed = max(0, open_bet - state.round_bets.get(idx, 0))
        if kind == FOLD:
            p.folded = True
        elif kind == CHECK:
            # legal only when the player owes nothing
            assert owed == 0, "CHECK with a bet outstanding"
        elif kind in (CALL, ALLIN):
            # CALL is capped by the stack; the cap turns it into an all-in call
            paid = min(p.stack, owed if kind == CALL else p.stack)
            bet_kind = kind
        elif kind == BET:
            assert owed == 0, "BET into a live bet"
            paid = min(amount, p.stack, max_raise_amount(state, idx))
            assert paid >= state.config.min_bet, "bet below minimum"
            state.aggro_count += 1
            bet_kind = "BET"
        elif kind == RAISE:
            assert owed > 0, "RAISE with no bet to raise"
            gross = min(amount, max_raise_amount(state, idx)) - state.round_bets.get(idx, 0)
            assert gross <= p.stack, "raise above stack"
            assert gross >= last_full_raise or gross == p.stack, "raise below minimum increment"
            paid = gross
            state.aggro_count += 1
            bet_kind = "RAISE"
        else:
            raise ValueError(f"unknown action kind: {kind!r}")

        if kind in (CALL, ALLIN, BET, RAISE):
            paid = min(paid, p.stack)
            p.stack -= paid
            p.contributed += paid
            state.round_bets[idx] = state.round_bets.get(idx, 0) + paid
            if p.stack == 0:
                p.all_in = True
            if bet_kind in ("BET", "RAISE"):
                # A full raise (or an all-in that reaches it) resets the minimum
                # increment; a short all-in raise keeps it and does not reopen.
                if paid >= last_full_raise:
                    last_full_raise = paid
            state.history.append({"round": state.round_idx, "pos": idx,
                                  "kind": kind, "amount": paid})
        acted_at[idx] = open_bet if kind in (FOLD, CHECK) else max(state.round_bets.values())
        idx = (idx + 1) % n
    # Finalize: move the whole street into the pot exactly once.
    state.pot += sum(state.round_bets.values())
    state.round_bets = {}
```

- [ ] **Step 4: Run the betting test file**

Run: `.venv/bin/pytest tests/test_betting_round.py tests/test_game_state.py tests/test_actions.py`
Expected: all green

- [ ] **Step 5: Commit**

```bash
git add poker/game.py tests/test_betting_round.py && git commit -m "feat: betting round loop with all-in and reopen rules"
```

**Engineer note:** the loop intentionally gives a seat a new turn only when the
open wager grew since its last turn. That single rule delivers the short-raise
"no reopen" semantics, the min-raise increment, and street termination without
lap counters — reason about it before changing it.

---

### Task 2.4: Side pots and showdown payouts

**Files:**
- Modify: `poker/game.py`
- Test: `tests/test_side_pots.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_side_pots.py`:
```python
"""Side-pot construction and payout against the conservation invariants."""
import numpy as np
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_side_pots.py`
Expected: `ModuleNotFoundError: build_side_pots`

- [ ] **Step 3: Implement side-pot builder and showdown payout**

Append to `poker/game.py`:
```python
@dataclass
class SidePot:
    """A layer of the pot plus the player indices eligible to win it."""
    amount: int
    eligible: tuple[int, ...]


def build_side_pots(players) -> list[SidePot]:
    """Layered side-pot construction from per-player total contributions.

    Level widths follow ALL contributors (folded money included); eligibility
    is restricted to players who have NOT folded — a folded hand forfeits every
    layer it stands in. When no live player covers a layer (everyone there
    folded), that width is dead money: it rolls into the live pot beneath it,
    so the amounts always sum to total contributions (conservation invariant).
    """
    contrib = [p.contributed for p in players]
    levels = sorted(set(c for c in contrib if c > 0))
    pots = []
    prev = 0
    dead = 0   # chips forfeited by folded players above every live contribution
    for level in levels:
        width = level - prev
        layer = width * len([c for c in contrib if c >= level])
        eligible = tuple(i for i, c in enumerate(contrib)
                         if c >= level and not players[i].folded)
        if eligible:
            pots.append(SidePot(amount=layer + dead, eligible=eligible))
            dead = 0
        else:
            dead += layer
        prev = level
    if dead:
        pots[-1] = SidePot(amount=pots[-1].amount + dead, eligible=pots[-1].eligible)
    return pots or [SidePot(0, tuple())]


def _winner_indices(scores, eligible) -> tuple[int, ...]:
    """Indices among `eligible` holding the highest score (ties included)."""
    best = max((scores[i] for i in eligible))
    return tuple(i for i in eligible if scores[i] == best)


def pay_out(state: GameState, scores) -> None:
    """Distribute every side pot to the best hand among its eligible players.

    `scores` maps player index -> encoded hand score (higher better). Ties
    split equally. Mutates players' stacks by the net win. Conservation is
    enforced by construction (amounts sum to total contributions).
    """
    pots = build_side_pots(state.players)
    for pot in pots:
        winners = _winner_indices(scores, pot.eligible)
        share = pot.amount / len(winners)
        for w in winners:
            state.players[w].stack += share
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_side_pots.py`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/game.py tests/test_side_pots.py && git commit -m "feat: layered side pots with conservation guarantees"
```

---

### Task 2.5: `run_hand` — full hand cycle

**Files:**
- Modify: `poker/game.py`
- Test: `tests/test_run_hand.py`

**Cycle (doc §6.2):** shuffle → deal → preflop → if ≥2 contestants flop/round → turn/round → river/round → run-out shortcut when no further betting possible → showdown → stack updates.

- [ ] **Step 1: Write the failing tests**

`tests/test_run_hand.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_run_hand.py`
Expected: `ModuleNotFoundError: run_hand`

- [ ] **Step 3: Implement `HandResult`, `deal_hole`, and `run_hand`**

Append to `poker/game.py` exactly this code (single authoritative formulation):
```python
@dataclass
class HandResult:
    """Immutable record of one complete hand, consumed by the recorder."""
    dealer_pos: int
    rng_seed: int
    board: np.ndarray
    stacks_before: list
    stacks_after: list
    hole_cards: list
    actions: list
    side_pots: list
    pot_total: int
    net: list
    ruined: list = field(default_factory=list)   # filled in by the Stage-05 session layer


def deal_hole(state: GameState, deck, cursor=0) -> int:
    """Deal two unique hole cards to each live player from `deck` (top-down)."""
    for p in state.players:
        p.hole = deck[cursor:cursor + 2]
        cursor += 2
    return cursor


def run_hand(players, dealer_pos, rng, config, actions_override=None):
    """Play one complete hand (doc §6.2) and return its HandResult.

    Street sequence: preflop -> flop -> turn -> river, each street betting
    after its board cards are dealt. The run-out shortcut skips betting on any
    street where at most one player can act (all others folded or all-in) and
    deals the remaining board through the river. `pay_out` distributes layered
    side pots; a lone survivor collects the uncontested pot. Chips are
    conserved by construction (stacks only move via blinds/bets/payouts).

    `actions_override` maps a street index to a `run_betting_round`-style
    {player_idx: (kind, amount)} dict (Stage-02/03 test harness). A street that
    still requires betting but is absent from the map raises a KeyError — the
    harness must script every betting street explicitly.
    """
    deck = shuffle_deck(rng, new_deck())
    before = [p.stack for p in players]
    for p in players:
        p.folded = False
        p.all_in = False
        p.contributed = 0

    state = GameState(players=players, dealer_pos=dealer_pos, config=config)
    board_index = deal_hole(state, deck)
    actions_log = []

    for street in range(4):
        state.round_idx = street
        # Preflop opens with the blinds; later streets open with an empty tray.
        if street == 0:
            starting_bets(state)
        else:
            state.round_bets = {}

        non_folded = [i for i, p in enumerate(players) if not p.folded]
        if len(non_folded) <= 1:
            break                       # everyone else folded: uncontested pot
        actable = [i for i, p in enumerate(players) if not p.folded and not p.all_in]

        # Deal the street's board cards (none on the preflop).
        if street == 1:
            state.board = np.concatenate([state.board, deck[board_index:board_index + 3]])
            board_index += 3
        elif street in (2, 3):
            state.board = np.concatenate([state.board, deck[board_index:board_index + 1]])
            board_index += 1

        if len(actable) <= 1:
            continue                    # run-out shortcut: no betting possible

        street_script = actions_override.get(street) if actions_override else None
        run_betting_round(state, street_script)
        actions_log.extend(state.history)
        state.history = []

    survivors = [i for i, p in enumerate(players) if not p.folded]
    if len(survivors) == 1:
        # Uncontested pot: the lone survivor takes everything already in.
        winner = survivors[0]
        players[winner].stack += state.pot + sum(state.round_bets.values())
        actions_log.append({"round": state.round_idx, "pos": list(winner),
                            "kind": "showdown", "amount": state.pot,
                            "winners": (winner,), "board": state.board.tolist()})
    else:
        # Showdown: top up an all-in board through the river, then score hands.
        while len(state.board) < 5:
            state.board = np.concatenate([state.board, deck[board_index:board_index + 1]])
            board_index += 1
        scores = {i: hand_score(np.concatenate([p.hole, state.board]))
                  for i, p in enumerate(players) if not p.folded}
        pay_out(state, scores)
        best = max(scores.values())
        winners = tuple(i for i, s in scores.items() if s == best)
        actions_log.append({"round": state.round_idx, "pos": list(winners),
                            "kind": "showdown", "amount": state.pot,
                            "winners": winners, "board": state.board.tolist(),
                            "scores": {i: int(s) for i, s in scores.items()}})

    return HandResult(
        dealer_pos=dealer_pos,
        rng_seed=getattr(rng, "seed", None),
        board=state.board,
        stacks_before=before,
        stacks_after=[p.stack for p in players],
        hole_cards=[p.hole.tolist() for p in players],
        actions=actions_log,
        side_pots=[{"amount": p.amount, "eligible": list(p.eligible)}
                   for p in build_side_pots(players)],
        pot_total=sum(p.contributed for p in players),
        net=[p.stack - b for p, b in zip(players, before)],
    )
```

**Engineer note:** the run-out shortcut and the showdown top-up both hand out board
cards strictly once each (collectively 0/3/4/5 cards: the "0" case is an
uncontested preflop pot). `HandResult.pot_total` is reconstructed from
contributions, so it is exact even when `pay_out` split fractional shares.

---

### Task 2.6: Data-quality invariant sweep over random scripted sessions

**Files:**
- Create: `tests/test_stage02_invariants.py`

- [ ] **Step 1: Write the invariant fuzz**

`tests/test_stage02_invariants.py`:
```python
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
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/pytest tests/test_stage02_invariants.py`
Expected: `3 passed` (12k hands total, no invariant violated)

- [ ] **Step 3: Commit**

```bash
git add tests/test_stage02_invariants.py && git commit -m "test: betting-engine conservation invariants fuzz"
```

---

## Stage Exit Criteria

- [ ] Betting rounds respect pot-limit max-raise formula, 4-aggression cap, min-raise, and short-all-in non-reopen.
- [ ] Side pots built and paid out with layered eligibility; conservation exact over fuzz.
- [ ] `run_hand` produces correct boards (0/3/4/5), run-out shortcut, showdown splits.
- [ ] Chip conservation invariant holds over 10k+ scripted hands.

**Report to the user before Stage 03:**
- All five conservation invariants hold (paste the fuzz counts).
- Any place where the pot-limit doc formula required interpretation under all-in caps (should be none — formula is cap-immune since pays are min(stack, owed)).