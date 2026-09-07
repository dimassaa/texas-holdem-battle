# Stage 05 — Mathematician & Adaptive, Simulator, Recorder, Reproducibility

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the six-strategy roster (Mathematician, Adaptive), build the session simulator that consumes Stage-02/03/04 pieces, persist every hand as a structured raw record, and guarantee byte-for-byte reproducibility (approved decision) plus the first-passage ruin model (approved decision: seat frozen, game continues).

**Architecture:**
- `poker/player.py` gains `MathematicianStrategy`, `AdaptiveStrategy`.
- `poker/recorder.py` — raw JSONL hand log + session metadata (the audit trail).
- `poker/simulator.py` — `run_session` (seat, hand loop, elimination/ruin, button advance, adaptive-brain updates) and `compute_session_stats`.
- `poker/game.py` — minor engine change to support 2–6 handed games (heads-up ordering) for post-elimination play.

**Approved decisions:**
- **Ruin:** first-passage — a seat is ruined the first time its stack < BB; the seat is frozen (removed from play), the game continues with the remaining players; if fewer than 2 remain the session truncates and that is recorded.
- **Reproducibility:** session seed → per-hand seed (`subseed`), all randomness flows through `Rng`; two runs with equal seeds produce identical raw logs, byte for byte.
- **Adaptive:** stats tracked per original seat over the last `adaptive_window` hands; thresholds recomputed every `adaptive_adjust_every` hands.

**Data-quality focus of this stage:** the raw log becomes the single source of truth for all Stage-06 analysis. Its schema is pinned by tests; the session-level invariants (chip conservation across the whole session, monotone hand ids, seed trail) are asserted *inside the recorder*, so corrupted output fails fast instead of contaminating metrics.

**Locked contracts (for Stage 06):**
- `recorder.HandRecord` dataclass; `recorder.JsonlHandLog(path)` writes one JSON line per hand and `session_meta.json`/`session_stats.json` at close.
- `simulator.run_session(strategy_combo, num_hands, seed, config, log_dir, recorder_factory=None) -> dict` returning `{"meta":..., "stats":..., "log_path":..., "num_hands_run":...}`.
- `simulator.compute_session_stats(records, meta) -> dict` ($7.1 §metrics: BB/100, winrate, ruin flag/hand, aggression frequencies, positional EV).
- `Player.rank` = original seat index; eliminated seats keep their rank forever (stat anchoring).

---

### Task 5.1: MathematicianStrategy

**Files:**
- Modify: `poker/player.py`
- Test: `tests/test_mathematician.py`

**Spec (doc §5.5 + resolved scaling):** all streets decide strictly on equity vs pot odds: call if `equity ≥ pot_odds`, fold otherwise; raise only when equity ≥ 1.15 × pot odds and the raise is +EV (pot-limit sized); bet 50–75% pot when equity > 0.6 and unopposed.

- [ ] **Step 1: Write the failing tests**

`tests/test_mathematician.py`:
```python
"""Mathematician: pure pot-odds EV decision on every street."""
import numpy as np
from poker.actions import BET, CALL, FOLD, RAISE
from poker.card import card_id as H
from poker.config import Config
from poker.game import GameState
from poker.player import MathematicianStrategy, Player
from poker.rng import Rng

CFG = Config()


class StubProvider:
    def __init__(self, eq, pot=100, to_call=0): self.eq, self.pot, self.to_call = eq, pot, to_call
    def equity(self, hand, board, n_opp): return self.eq
    def pot_odds(self): return self.to_call / (self.pot + self.to_call) if self.pot else 0.0


def make(eq, to_call=0, pot=100, hand=(8, 5), board=(9, 3, 2)):
    ps = [Player(name=f"p{i}", strategy="Mathematician", stack=200) for i in range(6)]
    st = GameState(players=ps, dealer_pos=0, config=CFG)
    st.board = np.array([H(board[0], 0), H(board[1], 1), H(board[2], 2)], dtype=np.int32) if board else np.empty(0, dtype=np.int32)
    st.round_idx = 1 if board else 0
    st.round_bets = {}
    st.pot = pot
    for i, p in enumerate(ps): p.hole = np.array([H(hand[0], i % 4), H(hand[1], (i + 1) % 4)], dtype=np.int32)
    return st, st.players[3], StubProvider(eq, pot, to_call)


def test_math_calls_when_equity_exceeds_pot_odds():
    st, p, prov = make(eq=0.4, to_call=25, pot=100)   # pot odds .2
    a = MathematicianStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind in (CALL, RAISE)


def test_math_folds_when_under_pot_odds():
    st, p, prov = make(eq=0.15, to_call=150, pot=50)  # pot odds .75
    a = MathematicianStrategy().act(p, st, 3, prov, Rng(0))
    assert a.kind == FOLD


def test_math_raises_only_when_profitably_above_pot_odds():
    st, p, prov = make(eq=0.8, to_call=0, pot=100)    # no bet to call
    a = MathematicianStrategy().act(p, st, 3, prov, Rng(0))
    # strong equity + no opponent bet = value bet (50-75% pot)
    assert a.kind == BET and 0.5 * 100 <= a.amount <= 0.75 * 100
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_mathematician.py`
Expected: `ModuleNotFoundError: MathematicianStrategy`

- [ ] **Step 3: Implement**

Append to `poker/player.py`:
```python
@_register
class MathematicianStrategy(Strategy):
    """Strict EV comparison: call iff equity >= pot odds; raise only when the
    surplus over pot odds (1.15x) pays for the raise; bet 50-75% of pot when
    the equity edge is large and the pot is unopened this street."""

    name = "Mathematician"
    _BET_THRESHOLD = 0.60
    _RAISE_SURPLUS = 1.15   # equity must beat pot odds by this factor to raise

    def _preflop_action(self, player, state, idx, provider):
        # Preflop uses the same EV rule; the preflop table (via provider) gives
        # equity vs n opponents. Enters with top ~20-25% by equity (playable),
        # then EV-decides vs the pot odds offered.
        from poker.equity import start_hand_type
        htype = start_hand_type(player.hole)
        eq = provider.equity(player.hole, state.board, state.num_opponents(idx))
        if eq < 0.20:                       # equity too low to invest any street
            return Action(FOLD, 0)
        to_call = state.to_call(idx)
        if to_call == 0:
            if eq >= self._BET_THRESHOLD and state.can_raise(idx):
                return Action(BET, bet_size(state, idx, 0.6, state.config.min_bet))
            return Action(CHECK, 0)
        pot_odds = provider.pot_odds()
        if eq >= self._RAISE_SURPLUS * pot_odds and state.can_raise(idx):
            return Action(RAISE, bet_size(state, idx, 0.66, state.config.min_bet))
        if eq >= pot_odds:
            return Action(CALL, min(player.stack, to_call))
        return Action(FOLD, 0)

    def act(self, player, state, idx, provider, rng):
        if state.round_idx == 0:
            return self._preflop_action(player, state, idx, provider)
        eq = provider.equity(player.hole, state.board, state.num_opponents(idx))
        to_call = state.to_call(idx)
        pot_odds = provider.pot_odds()
        if to_call == 0:
            if eq >= self._BET_THRESHOLD and state.can_raise(idx):
                return Action(BET, bet_size(state, idx, 0.6, state.config.min_bet))
            return Action(CHECK, 0)
        if eq >= self._RAISE_SURPLUS * pot_odds and state.can_raise(idx):
            return Action(RAISE, bet_size(state, idx, 0.66, state.config.min_bet))
        if eq >= pot_odds:
            return Action(CALL, min(player.stack, to_call))
        return Action(FOLD, 0)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_mathematician.py`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/player.py tests/test_mathematician.py && git commit -m "feat: mathematician strategy (strict EV decision)"
```

---

### Task 5.2: AdaptiveStrategy — opponent stats and dynamic ranges

**Files:**
- Modify: `poker/player.py`
- Test: `tests/test_adaptive.py`

**Spec (doc §5.6):** track per-opponent Fold-to-Raise %, Flop C-bet %, and overall aggression (raises/bets ÷ calls). Base preflop range top-20%; table full of tight folders → raise range expands to 30% and steals get bigger; table of loose callers → tighten to 15%. Postflop: c-bet more often vs opponents who fold to bets; play tighter vs aggressive opponents. Recomputed every `adaptive_adjust_every` hands over the last `adaptive_window` hands.

**Data-quality note:** adaptive never sees hole cards or strategy names — only observable actions (same information rule as every other strategy, doc §1 limitation 2). Its `stats` field is updated by the simulator from each completed hand's *public* action log only.

- [ ] **Step 1: Write the failing tests**

`tests/test_adaptive.py`:
```python
"""Adaptive: windowed opponent stats drive dynamic range/cbet thresholds."""
from collections import deque
import numpy as np
from poker.actions import BET, CALL, CHECK, FOLD, RAISE
from poker.card import card_id as H
from poker.config import Config
from poker.game import GameState
from poker.player import AdaptiveStrategy, Player
from poker.rng import Rng

CFG = Config()


class StubProvider:
    def __init__(self, eq, pot=100, to_call=0): self.eq, self.pot, self.to_call = eq, pot, to_call
    def equity(self, hand, board, n_opp): return self.eq
    def pot_odds(self): return self.to_call / (self.pot + self.to_call) if self.pot else 0.0


def fresh_brain(memory=None, adjust_every=None):
    return AdaptiveStrategy(memory=memory, adjust_every=adjust_every)


def test_stats_accumulate_over_public_actions():
    b = fresh_brain()
    # one hand: seat 5 raised, seat 3 folded to it (fold-to-raise evidence)
    b.observe({"seat": 5, "faces_raise": 1, "folds_to_raise": 1})
    st = b.stats()[5]
    assert st["hands"] == 1
    assert st["fold_to_raise"] == 1.0     # single observed instance


def test_window_is_capped():
    b = fresh_brain(memory=10)
    for i in range(20):
        b.observe({"seat": 1, "faces_raise": 1, "folds_to_raise": 1})
    assert b.stats()[1]["hands"] == 10   # rolling window, not cumulative

def test_loose_table_tightens_preflop_range():
    b = fresh_brain()
    # many calling opponents, no raises/folds -> loose table, steals unprofitable
    b.window_dump([{"seat": s, "calls": 5, "raises": 0} for s in range(1, 6)])
    assert b.play_share < 0.20


def test_tight_fold_heavy_table_expands_raise_range():
    b = fresh_brain()
    b.window_dump([{"seat": s, "folds": 9, "plays": 10} for s in range(1, 6)])
    assert b.raise_share > 0.25
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_adaptive.py`
Expected: `ModuleNotFoundError: AdaptiveStrategy`

- [ ] **Step 3: Implement**

Append to `poker/player.py`:
```python
@_register
class AdaptiveStrategy(Strategy):
    """Learns opponent tendencies from public action history and adjusts.

    Statistics are windowed (config.adaptive_window hands; risk §10) so the
    strategy reacts to recent behavior, not ancient history. Thresholds refresh
    at config.adaptive_adjust_every hands. All reads are from observable
    actions only (doc §1 limitation 2) — never hole cards or strategy identity.
    """

    name = "Adaptive"

    def __init__(self, memory=None, adjust_every=None):
        self.memory = memory or Config().adaptive_window
        self.adjust_every = adjust_every or Config().adaptive_adjust_every
        self.hands_seen = 0
        self._window = deque(maxlen=self.memory)   # list of per-hand summaries
        self._ranges_dirty = True
        self._play_range = None
        self._raise_range = None
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
        """Share (of the 1326-combo universe) of the CURRENT adjusted play range.

        Recomputed from the live window so tests can feed synthetic tables
        through `window_dump` and assert the resulting tightness/wideness.
        """
        play_range, _, _ = _table_reads(_aggregate_window(self._window))
        return _range_share(play_range)

    @property
    def raise_share(self):
        """Share of the 1326-combo universe of the current adjusted raise range."""
        _, raise_range, _ = _table_reads(_aggregate_window(self._window))
        return _range_share(raise_range)

    # ---- decision overrides ----------------------------------------------
    def _refresh_ranges_if_due(self):
        if self._ranges_dirty and self.hands_seen % self.adjust_every == 0:
            agg = _aggregate_window(self._window)
            self._play_range, self._raise_range, self._steal_scale = _table_reads(agg)
            self._ranges_dirty = False

    def act(self, player, state, idx, provider, rng):
        self._refresh_ranges_if_due()
        ...  # same shape as AggressiveStrategy, using self._play_range / self._raise_range
             # and scaling c-bet aggressiveness inversely with opponents' fold-to-bet
```

**Shared helpers (add to `poker/player.py`):**
```python
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
```

*Note:* `_preflop_table()` lazily loads the Stage-03 artifact once; when the
table file is absent (pre-table test environments or a fresh clone) the
strategy plays neutral ranges instead of crashing.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_adaptive.py`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/player.py tests/test_adaptive.py && git commit -m "feat: adaptive strategy with windowed opponent modelling"
```

---

### Task 5.3: Raw hand recorder (`recorder.py`)

**Files:**
- Create: `poker/recorder.py`
- Test: `tests/test_recorder.py`

**Why raw:** Stage-06 analysis and any future re-analysis read only from these files; nothing is recomputed from memory. Schema is pinned by tests so a renamed field or type change is a test failure, not a silent schema drift.

- [ ] **Step 1: Write the failing tests**

`tests/test_recorder.py`:
```python
"""Raw JSONL hand log: schema, round-trip, and integrity of session files."""
import json
import os
import tempfile

import numpy as np
import pytest

from poker.recorder import (
    HandRecord, JsonlHandLog, hand_record_from_json,
)
from poker.card import card_id as H


def sample_record():
    return HandRecord(
        hand_id=7,
        rng_seed=1234,
        dealer_pos=2,
        round_idx_start=0,
        board=np.array([H(2, 0), H(9, 1), H(3, 2)], dtype=np.int32),
        stacks_before=[200] * 6,
        stacks_after=[212, 190, 205, 180, 211, 202],
        hole=[np.array([H(14, 0), H(14, 1)])] + [np.array([H(r, 0), H(r, 1)]) for r in (13, 12, 11, 10, 9)],
        actions=[{"round": 0, "pos": 3, "kind": "raise", "amount": 14}],
        side_pots=[{"amount": 42, "eligible": [0, 1, 3]}],
        pot_total=42,
        net=[12.0, -10.0, 5.0, -20.0, 11.0, 2.0],
        ruined=[],
        seats=[0, 1, 2, 3, 4, 5],
        showdown=True,
    )


def test_record_roundtrips_through_json():
    import json
    as_json = json.dumps(hand_record_from_json(sample_record().to_dict()))
    rec2 = HandRecord.from_dict(json.loads(as_json))
    assert rec2.hand_id == 7 and rec2.board.tolist() == sample_record().board.tolist()
    assert rec2.net == sample_record().net


def test_jsonl_log_writes_and_reloads(tmp_path):
    p = tmp_path / "hands.jsonl"
    log = JsonlHandLog(p)
    log.write(sample_record())
    log.write(sample_record())
    log.close()
    rows = [json.loads(line) for line in open(p)]
    assert len(rows) == 2 and rows[0]["hand_id"] == 7


def test_session_meta_and_stats_are_persisted(tmp_path):
    p = tmp_path / "session"
    with JsonlHandLog(p / "hands.jsonl") as log:
        log.write(sample_record())
        log.write_meta({"seed": 1, "combo": ["Tight"] * 6})
        log.write_stats({"bb100": {"0": 1.2}})
    assert json.load(open(p / "session_meta.json"))["seed"] == 1
    assert json.load(open(p / "session_stats.json"))["bb100"]["0"] == 1.2
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_recorder.py`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`poker/recorder.py`:
```python
"""Raw session persistence.

Single source of truth for all downstream analysis. Files:
  <log_dir>/hands.jsonl        one JSON object per hand (action-by-action audit)
  <log_dir>/session_meta.json  config + seeds + timing (provenance)
  <log_dir>/session_stats.json aggregated per-session numbers

Writing is append-only JSONL: a partially-written session is still parseable
up to the last complete line, so a crashed run loses only the trailing hand.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np


def _ser(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, tuple):
        return list(o)
    return o


@dataclass
class HandRecord:
    hand_id: int
    rng_seed: int
    dealer_pos: int
    round_idx_start: int
    board: np.ndarray
    stacks_before: list
    stacks_after: list
    hole: list
    actions: list
    side_pots: list
    pot_total: int
    net: list
    ruined: list
    seats: list        # seat rank per entry position, so post-elimination
                       # records (with fewer entries) still map to seats
    showdown: bool

    def to_dict(self):
        d = asdict(self)
        d["board"] = _ser(self.board)
        d["hole"] = [_ser(h) for h in self.hole]
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["board"] = np.array(d.pop("board"), dtype=np.int32)
        d["hole"] = [np.array(h, dtype=np.int32) for h in d.pop("hole")]
        return cls(**d)


def hand_record_from_json(d):
    """Deserialize a raw dict into a numpy-bearing HandRecord."""
    return HandRecord.from_dict(d)


class JsonlHandLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self.count = 0

    def write(self, record: HandRecord) -> None:
        self._fh.write(json.dumps(record.to_dict()) + "\n")
        self.count += 1

    def write_meta(self, meta: dict) -> None:
        meta_path = self.path.with_name("session_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(_ser(meta), f, indent=2)

    def write_stats(self, stats: dict) -> None:
        stats_path = self.path.with_name("session_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(_ser(stats), f, indent=2)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_recorder.py`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/recorder.py tests/test_recorder.py && git commit -m "feat: raw jsonl hand recorder with pinned schema"
```

---

### Task 5.4: Simulator — `run_session` with ruin and adaptive feeding

**Files:**
- Modify: `poker/game.py` (heads-up + variable-handed support)
- Create: `poker/simulator.py`
- Test: `tests/test_simulator.py`

- [ ] **Step 1: Engine support for 2–6 handed games**

Update `game._setting_first_actor(state)`:
```python
def _setting_first_actor(state: GameState) -> int:
    n = len(state.players)
    if state.round_idx == 0:
        # preflop: left of BB normally; heads-up the SB (button) acts first.
        return (state.dealer_pos + 2) % n if n >= 3 else state.dealer_pos
    # postflop: left of button every time — heads-up that IS the BB (non-button).
    return (state.dealer_pos + 1) % n
```
Update `starting_bets` docstring to note heads-up blind roles. Add a regression test:
```python
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
```
(`tests/test_engine_short_handed.py`)

- [ ] **Step 2: Write the simulator tests**

`tests/test_simulator.py`:
```python
"""Session simulator: ruin, seeds, adaptive feeding, conservation at session level."""
import json
import os
from pathlib import Path

from poker.config import Config
from poker.simulator import run_session

CFG = Config(num_hands=30)


def test_session_runs_and_persists(tmp_path):
    out = run_session(
        ("Tight", "Loose", "Aggressive", "Passive", "Mathematician", "Adaptive"),
        num_hands=30, seed=7, config=CFG, log_dir=str(tmp_path),
    )
    assert out["num_hands_run"] == 30
    log = Path(tmp_path) / "hands.jsonl"
    assert log.exists() and log.stat().st_size > 0
    meta = json.load(open(Path(tmp_path) / "session_meta.json"))
    assert meta["seed"] == 7 and meta["combo"] == list(CFG.strategy_combo)


def test_ruin_freezes_seat_and_session_continues(tmp_path):
    # Deterministic bust: seat 5 starts at stack 0, so it can neither post a
    # blind nor win anything — it ends hand 0 at 0 (< BB) and is frozen.
    combo = ("Aggressive", "Aggressive", "Aggressive", "Passive", "Passive", "Passive")
    stacks = [200] * 5 + [0]
    out = run_session(combo, num_hands=100, seed=3, config=Config(),
                      log_dir=str(tmp_path), initial_stacks=stacks)
    assert out["stats"]["ruined"]["5"] == 0                   # busted on hand 0
    assert out["num_hands_run"] == 100                        # session continued
    assert out["num_ruined"] >= 1                             # seat 5 busted for sure
    # the frozen seat's final stack never went negative anywhere in the log
    rows = [json.loads(l) for l in open(Path(tmp_path) / "hands.jsonl")]
    assert all(min(r["stacks_after"]) >= 0 for r in rows)
    # seat 5 played its bust hand, then vanishes from every later record
    assert 5 in rows[0]["seats"]
    assert all(5 not in r["seats"] for r in rows[1:])


def test_two_identical_seeds_give_identical_logs(tmp_path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a = run_session(("Tight",) * 6, 20, 42, Config(), str(a_dir))
    b = run_session(("Tight",) * 6, 20, 42, Config(), str(b_dir))
    fa = open(a_dir / "hands.jsonl").read()
    fb = open(b_dir / "hands.jsonl").read()
    assert fa == fb          # byte-for-byte reproducibility (approved decision)
    assert a["stats"]["bb100"] == b["stats"]["bb100"]
```

- [ ] **Step 3: Run to verify fail**

Run: `.venv/bin/pytest tests/test_simulator.py`
Expected: `ModuleNotFoundError: run_session`

- [ ] **Step 4: Implement `poker/simulator.py`**

```python
"""Session manager: seats the declared strategy combo, plays num_hands hands,
freezes first-passage-ruined seats, feeds adaptive brains from PUBLIC hand
summaries, and writes the raw session artifacts (task 5.3 format)."""

import json
import time
from pathlib import Path

import numpy as np

from poker.config import Config
from poker.game import run_hand
from poker.player import Player, for_name, start_hand_type  # noqa: F401
from poker.recorder import HandRecord, JsonlHandLog
from poker.rng import Rng, subseed


def _hand_summary(player, hand_id, actions, dealer_pos):
    """Build the PUBLIC per-hand summary the adaptive brain observes.

    Only actions a real opponent could have seen: who raised, who faced a
    raise and folded, who c-bet, plus aggregate aggression/calls/folds. Never
    hole cards, never showdown results (doc §1 limitation 2). Keys match
    `_aggregate_window` exactly so real and synthetic windows merge.
    """
    seat = player.rank
    aggro = ("bet", "raise", "all-in")   # all-in wagers are aggression
    preflop_raises = [a for a in actions if a["round"] == 0 and a["kind"] in aggro]
    mine = [a for a in actions if a["pos"] == seat]
    summary = {"seat": seat, "hand_id": hand_id}
    summary["faces_raise"] = len(preflop_raises)
    summary["folds_to_raise"] = int(any(a["kind"] == "fold" and a["round"] == 0 and a["pos"] == seat for a in actions))
    summary["cbets"] = int(any(a["kind"] in aggro and a["round"] == 1 and a["pos"] == seat for a in actions))
    # c-bet chance exists only when hero was the preflop aggressor (raised/bet).
    summary["cbet_chances"] = int(any(a["kind"] in aggro and a["round"] == 0 and a["pos"] == seat for a in actions))
    summary["agg_actions"] = sum(1 for a in mine if a["kind"] in aggro)
    summary["raises"] = summary["agg_actions"]
    summary["calls"] = sum(1 for a in mine if a["kind"] == "call")
    summary["folds"] = int(any(a["kind"] == "fold" for a in mine))
    summary["plays"] = int(any(a["kind"] in ("call", "raise", "bet", "check", "all-in") for a in mine))
    return summary
```

`run_session` (production path):
```python
def run_session(strategy_combo, num_hands, seed, config, log_dir,
                recorder_factory=None, initial_stacks=None):
    """Run a full session; returns the provenance/stats dict and persists logs.

    `initial_stacks`, when given, is a per-seat stack list matching
    `strategy_combo` (used by the ruin test to seat a guaranteed-bust player);
    defaults to `config.initial_stack` for every seat.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    recorder = (recorder_factory or JsonlHandLog)(log_dir / "hands.jsonl")
    started = time.time()

    stacks = initial_stacks or [config.initial_stack] * len(strategy_combo)
    active = [Player(name=f"seat{i}", strategy=name, stack=stacks[i],
                     rank=i, brain=for_name(name))
              for i, name in enumerate(strategy_combo)]
    dealer = int(subseed(seed, 9000) % len(active))
    records = []
    hand = 0
    num_ruined_total = 0
    while hand < num_hands and len(active) >= 2:
        hand_seed = subseed(seed, hand)
        played = list(active)                     # pre-ruin seating for this hand
        result = run_hand(played, dealer, Rng(hand_seed), config)
        hand_dealer = dealer                      # used by this hand's record

        ruined = _mark_ruined(active, config)     # first-passage stack < bb
        if ruined:
            num_ruined_total += len(ruined)
            active = [p for p in active if not p.ruined]   # freeze seats

        record = _to_record(hand, hand_seed, result, played, hand_dealer, ruined)
        recorder.write(record)
        records.append(record)
        _feed_adaptives(played, result, hand, hand_dealer)
        dealer = (dealer + 1) % len(active)   # button advances; modulo tolerates
        hand += 1                              # eliminations below

    finished = time.time()
    meta = {
        "combo": list(strategy_combo),
        "num_hands": num_hands,
        "num_players": len(strategy_combo),   # seats keep their original ranks
        "seed": seed,
        "config": config.as_dict(),
        "started": started,
        "finished": finished,
        "elapsed": finished - started,
    }
    stats = compute_session_stats([r.to_dict() for r in records], meta)
    recorder.write_meta(meta)
    recorder.write_stats(stats)
    recorder.close()
    return {"meta": meta, "stats": stats, "log_path": str(log_dir / "hands.jsonl"),
            "num_hands_run": hand, "num_ruined": num_ruined_total}
```

Supporting helpers (add to `poker/player.py` / `poker/simulator.py`):
```python
def _mark_ruined(players, config):
    """First-passage ruin: flag every live seat whose stack fell below BB.

    Returns the ruined ranks (already marked `p.ruined = True` so the caller
    can freeze them) or an empty list when nobody busted this hand.
    """
    ruined = [p for p in players if not p.ruined and p.stack < config.bb]
    for p in ruined:
        p.ruined = True
    return [p.rank for p in ruined]


def _feed_adaptives(players, result, hand_id, dealer_pos):
    """Feeds each Adaptive brain a PUBLIC summary of this hand (doc §1.2)."""
    for p in players:
        if isinstance(p.brain, AdaptiveStrategy):
            p.brain.observe(_hand_summary(p, hand_id, result.actions, dealer_pos))


def _to_record(hand, hand_seed, result, played, dealer_pos, ruined):
    """Serialize one hand into the pinned HandRecord schema.

    `played` is the pre-ruin seating so the record covers exactly the seats
    that took part; `ruined` marks the seats that busted on THIS hand.
    """
    return HandRecord(
        hand_id=hand,
        rng_seed=hand_seed,
        dealer_pos=dealer_pos,
        round_idx_start=0,
        board=result.board,
        stacks_before=result.stacks_before,
        stacks_after=result.stacks_after,
        hole=[p.hole for p in played],
        actions=result.actions,
        side_pots=result.side_pots,
        pot_total=result.pot_total,
        net=[float(x) for x in result.net],
        ruined=list(ruined),
        seats=[p.rank for p in played],
        showdown=any(a["kind"] == "showdown" for a in result.actions),
    )
```

Add the `Player.ruined` field (default False) and the `Player.brain` field
(default None, set to `for_name(name)` per seat in the simulator). The session's
stats are computed by the single `compute_session_stats` implementation (Task
5.5) from the raw records, so the live run and any later recompute-from-log are
the same function — the reproducibility test holds by construction.

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/test_simulator.py tests/test_engine_short_handed.py`
Expected: `3 + 1 passed`

- [ ] **Step 6: Commit**

```bash
git add poker/game.py poker/player.py poker/simulator.py tests/test_simulator.py tests/test_engine_short_handed.py && git commit -m "feat: session simulator with ruin handling and adaptive feeding"
```

---

### Task 5.5: Session statistics (`compute_session_stats`)

**Files:**
- Modify: `poker/simulator.py`
- Test: `tests/test_session_stats.py`

Metrics per §7.1: **BB/100** (net BB × 100 / hands played by that seat, before ruin freeze), **winrate** (hands won to showdown or by fold ÷ hands dealt), **ruin probability over the run** (ruined hands ÷ hands dealt), **final stack** per seat, **positional EV** (net BB averaged by `relative_position` at hand start), **aggression frequencies** (bet/raise/call/fold counts).

Best-of-effort: per-seat play stops at ruin (frozen), so BB/100 is computed over the *live* hands only; document this denominator in the stats dict.

- [ ] **Step 1: Write the failing tests**

`tests/test_session_stats.py`:
```python
"""Aggregation of a recorded session into §7.1 metrics."""
import json
import math
from pathlib import Path

from poker.config import Config
from poker.simulator import compute_session_stats, run_session

CFG = Config(num_hands=60)


def test_computed_metrics_are_well_formed(tmp_path):
    out = run_session(("Tight", "Loose", "Aggressive", "Passive", "Mathematician", "Adaptive"),
                      num_hands=60, seed=5, config=CFG, log_dir=str(tmp_path))
    st = out["stats"]
    assert set(st) >= {"bb100", "winrate", "ruined", "final_stacks", "positional_ev", "aggression"}
    for bb in st["bb100"].values():
        assert bb is None or math.isfinite(bb)   # None = seat never played a live hand
    assert len(st["bb100"]) == CFG.num_players


def test_recompute_from_log_matches_session(tmp_path):
    out = run_session(("Tight",) * 6, 40, 9, Config(num_hands=40), str(tmp_path))
    rows = [json.loads(l) for l in open(Path(tmp_path) / "hands.jsonl")]
    meta = json.load(open(Path(tmp_path) / "session_meta.json"))
    recomputed = compute_session_stats(rows, meta)
    assert recomputed["bb100"] == out["stats"]["bb100"]
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_session_stats.py`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`poker/simulator.py`:
```python
def _pos_of(dealer_pos, rank, n):
    """Same zone mapping as `relative_position`: button=late, +1/+2 blinds,
    +3/+4 early, else middle. Computed from a raw record (no live state)."""
    dist = (rank - dealer_pos) % n
    if dist == 0:
        return "late"
    if dist in (1, 2):
        return "blinds"
    if dist in (3, 4):
        return "early"
    return "middle"


def compute_session_stats(records, meta) -> dict:
    """§7.1 metrics from the raw JSONL records + meta.

    This is the SINGLE stats implementation: `run_session` returns exactly this,
    so recomputing the same metrics from the persisted log is trivially equal.
    BB/100 denominator = hands the seat was live for (before first-passage
    ruin); seats with no live hands report None, never 0.0.
    """
    n = meta["num_players"]
    bb = meta["config"]["bb"]
    seats = range(n)
    net = {s: 0.0 for s in seats}
    live_hands = {s: 0 for s in seats}
    won_hands = {s: 0 for s in seats}
    ruin_hand = {s: None for s in seats}
    last_stack = {s: meta["config"]["initial_stack"] for s in seats}
    aggression = {s: {"bet": 0, "raise": 0, "call": 0, "fold": 0, "check": 0} for s in seats}
    pos_net = {s: {p: 0.0 for p in ("early", "middle", "late", "blinds")} for s in seats}
    pos_hands = {s: {p: 0 for p in ("early", "middle", "late", "blinds")} for s in seats}

    for hand_no, rec in enumerate(records):
        # map position-in-record -> seat rank; post-elimination records are
        # shorter, so zip against the record's own seats field.
        ruined_in_hand = set(rec.get("ruined", []))
        for pos, rank in enumerate(rec["seats"]):
            last_stack[rank] = rec["stacks_after"][pos]
            # A seat's bust hand DOES count as a live hand — it played and lost
            # it; first-passage ruin freezes the seat only from the NEXT hand.
            live_hands[rank] += 1
            net[rank] += rec["net"][pos]
            if rec["net"][pos] > 0:
                won_hands[rank] += 1
            zone = _pos_of(rec["dealer_pos"], rank, n)
            pos_net[rank][zone] += rec["net"][pos] / bb
            pos_hands[rank][zone] += 1
            for a in rec["actions"]:
                if a.get("pos") == rank:
                    # an all-in wager is aggression consuming a stack, so fold
                    # it into the bet counter rather than dropping it silent.
                    k = "bet" if a["kind"] == "all-in" else a["kind"]
                    if k in aggression[rank]:
                        aggression[rank][k] += 1
            if rank in ruined_in_hand:
                ruin_hand[rank] = hand_no     # first hand the seat busted on

    bb100 = {}
    winrate = {}
    for s in seats:
        bb100[str(s)] = (net[s] / bb / live_hands[s] * 100) if live_hands[s] else None
        winrate[str(s)] = (won_hands[s] / live_hands[s]) if live_hands[s] else None
    positional_ev = {
        str(s): {z: (pos_net[s][z] / pos_hands[s][z]) if pos_hands[s][z] else None
                 for z in ("early", "middle", "late", "blinds")} for s in seats}
    return {
        "bb100": bb100,
        "winrate": winrate,
        "ruined": {str(s): ruin_hand[s] for s in seats},
        "final_stacks": {str(s): last_stack[s] for s in seats},
        "positional_ev": positional_ev,
        "aggression": {str(s): aggression[s] for s in seats},
    }
```

*(Use `meta["config"]["bb"]` for BB scaling; re-derive `config` from meta in the recompute path.)*

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_session_stats.py`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/simulator.py tests/test_session_stats.py && git commit -m "feat: session statistics aggregation"
```

---

### Task 5.6: Stage gate — full-suite reproducibility and an end-to-end mini campaign

**Files:**
- Create: `tests/test_stage05_gate.py`

- [ ] **Step 1: Write the gate**

`tests/test_stage05_gate.py`:
```python
"""Stage-05 gate: an end-to-end mini campaign is reproducible and self-consistent."""
import hashlib
import json
from pathlib import Path

from poker.config import Config
from poker.simulator import run_session

COMBOS = [
    ("Tight",) * 6,
    ("Loose",) * 6,
    ("Tight", "Tight", "Tight", "Aggressive", "Aggressive", "Aggressive"),
    ("Tight", "Loose", "Aggressive", "Passive", "Mathematician", "Adaptive"),
]


def test_mini_campaign_is_reproducible(tmp_path):
    for i, combo in enumerate(COMBOS):
        run_session(combo, 25, 101 + i, Config(num_hands=25),
                    str(tmp_path / f"c{i}"))
    logs = sorted((tmp_path / "c0").glob("*.json*"))
    assert logs

    # rerun c0 and byte-compare
    run_session(COMBOS[0], 25, 101, Config(num_hands=25), str(tmp_path / "c0b"))
    a = (Path(tmp_path) / "c0" / "hands.jsonl").read_text()
    b = (Path(tmp_path) / "c0b" / "hands.jsonl").read_text()
    assert a == b
    assert hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(b.encode()).hexdigest()
```

- [ ] **Step 2: Run the gate**

Run: `.venv/bin/pytest tests/test_stage05_gate.py -q`
Expected: `1 passed`

- [ ] **Step 3: Measure a session's wall clock**

Run:
```bash
.venv/bin/python - <<'PY'
import time
from poker.config import Config
from poker.simulator import run_session
CFG = Config(num_hands=1000)
t0 = time.perf_counter()
out = run_session(CFG.strategy_combo, 1000, 42, CFG, "output/logs/smoke")
dt = time.perf_counter() - t0
print(f"{dt:.1f}s for 1000 hands ({dt:.3f}s/hand); ruined={out['num_ruined']}")
PY
```
Expected: a sane number you record in the report (this extrapolates the Stage-06 experiment cost at 10k–100k hands/combo).

- [ ] **Step 4: Commit**

```bash
git add tests/test_stage05_gate.py && git commit -m "test: stage 05 reproducibility gate and campaign smoke"
```

---

## Stage Exit Criteria

- [ ] Mathematician & Adaptive implemented and unit-tested (adaptive proven to widen/tighten ranges from reads).
- [ ] First-passage ruin freezes seats; session continues to <2 players or run-out.
- [ ] Raw per-hand JSONL + meta + stats persisted with pinned schema (recorder tested).
- [ ] Two identical runs → byte-identical logs; campaign gate green.
- [ ] Session wall-clock measured and recorded.
- [ ] All six strategies registered; registry count = 6.

**Report to the user before Stage 06:**
- Measured hands/second — final data for sizing the Stage-06 experiment matrix.
- The exact ruin behavior observed in the smoke run (which matchup busts first, and how many hands to first ruin).
- Any engine fixes forced by short-handed play.