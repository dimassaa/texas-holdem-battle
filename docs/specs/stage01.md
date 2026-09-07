# Stage 01 — Project Foundation: Cards, Deck, Hand Evaluator, RNG, Config

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the repository skeleton plus the deterministic-RNG and card/hand-evaluator primitives that every later stage builds on, with full unit-test coverage and a data-quality invariant suite.

**Architecture:** Flat `poker/` package mirroring the modules named in the technical document (`card.py`, `hand_evaluator.py`, `config.py`) plus a dedicated `rng.py` deterministic-stream harness required by the reproducibility decision. Cards are `int` 0..51 for numpy/numba speed. Hand evaluation produces a single monotonic `int` score (higher = better) that is payload for all equity and showdown comparisons downstream.

**Tech Stack:** Python 3.10+, numpy, pytest. (numba/pandas/matplotlib arrive in later stages.)

**Locked contracts (referenced by every later stage — do NOT rename):**
- `rng.Rng(seed)` wraps a `np.random.Generator`; `Rng.integers/random/shuffle`; `Rng.child(*labels)` spawns an independent derived stream; `subseed(root, *labels) -> int`.
- `card.card_id(rank, suit) -> id` with `id = (rank-2)*4 + suit`, ranks 2..14 (Ace=14), suits 0..3; `card.rank_of(id)`, `card.suit_of(id)`, `card.new_deck()`, `card.shuffle_deck(rng, deck)`.
- `hand_evaluator.evaluate_5_reference(cards) -> (category, kickers)` — verbose, for tests only; `hand_evaluator.score_5(cards5) -> int` — monotonic encoded score; `hand_evaluator.hand_score(cards7) -> int` — best of 21 takes; `hand_evaluator.score_batch(hands(n,7)) -> np.ndarray(n)`.
- `config.Config` frozen dataclass: 6 players, SB=1, BB=2, initial stack 200, min bet 2, max 4 aggressions/round, MC 400 iters, seed 42, adaptive window 100 / adjust every 20, log dir `output/logs`.

**Data-quality focus of this stage:** ensure the *randomness substrate* (seeded, hierarchical, byte-deterministic) and the *hand-strength substrate* (verified against an independent reference implementation) are correct before any betting or strategy logic depends on them. Every function gets a property test, not just a happy-path test.

---

### Task 1.1: Repository scaffold, dependencies, pytest

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `pytest.ini`
- Create: `.gitignore`
- Create: `poker/__init__.py`
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1: Create the dependency files**

`requirements.txt`:
```text
numpy>=1.26
```

`requirements-dev.txt`:
```text
-r requirements.txt
pytest>=8.0
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
addopts = -q -m "not slow"
markers =
    slow: long-running validation tests, skip by default
```

`.gitignore`:
```text
.venv/
__pycache__/
*.pyc
.pytest_cache/
output/
data/
```

`poker/__init__.py`:
```python
"""Poker engine: card primitives, hand evaluation, equity, strategies, simulation."""
```

`tests/__init__.py`:
```python
# Marks the tests directory as a package so pytest can import `poker` cleanly.
```

- [ ] **Step 2: Create and activate the virtual environment**

Run:
```bash
cd /mnt/d/opencode/new13
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -c "import numpy, pytest; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Initialize the repository**

Run:
```bash
cd /mnt/d/opencode/new13
git init
```
Expected: `Initialized empty Git repository in ...`

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "chore: scaffold project skeleton, deps, pytest config"
```

---

### Task 1.2: Deterministic RNG harness (`rng.py`)

**Files:**
- Create: `poker/rng.py`
- Test: `tests/test_rng.py`

**Why this exists:** full deterministic reproducibility is a hard data-quality requirement (approved decision). Every random consume — deck shuffle, opponent cards, run-out community cards, Monte Carlo sampling — must draw from a seeded `np.random.Generator`, and sessions must be able to spawn independent per-hand/per-evaluation streams so changing one component never perturbs another.

- [ ] **Step 1: Write the failing tests**

`tests/test_rng.py`:
```python
"""Deterministic-stream guarantees: reproducibility, independence, hierarchy."""
import numpy as np
import pytest

from poker.rng import Rng, subseed


class TestRng:
    def test_same_seed_same_stream(self):
        a = Rng(7).random(5)
        b = Rng(7).random(5)
        assert np.array_equal(a, b)

    def test_different_seed_different_stream(self):
        a = Rng(7).random(5)
        b = Rng(8).random(5)
        assert not np.array_equal(a, b)

    def test_integers_and_shuffle_are_deterministic(self):
        r1 = Rng(1)
        r2 = Rng(1)
        assert r1.integers(0, 100) == r2.integers(0, 100)
        d1 = np.arange(52)
        d2 = np.arange(52)
        r1.shuffle(d1)
        r2.shuffle(d2)
        assert np.array_equal(d1, d2)

    def test_child_uses_independent_stream(self):
        parent = Rng(10)
        c1 = parent.child(1)
        c2 = parent.child(2)
        assert not np.array_equal(c1.random(10), c2.random(10))
        # parent stream position must not be affected by spawning children
        before = parent.random(3)
        parent.child(3).random(10)
        after = parent.random(3)
        assert np.array_equal(before, after)


class TestSubseed:
    def test_subseed_is_deterministic_and_label_sensitive(self):
        assert subseed(5, 1, 2) == subseed(5, 1, 2)
        assert subseed(5, 1, 2) != subseed(5, 2, 1)

    def test_subseed_output_is_a_uint32(self):
        s = subseed(42, 9)
        assert isinstance(s, int)
        assert 0 <= s < 2**32
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_rng.py`
Expected: `ModuleNotFoundError: No module named 'poker.rng'` (or collection errors)

- [ ] **Step 3: Write the implementation**

`poker/rng.py`:
```python
"""Deterministic random-number streams.

Reproducibility is a hard requirement for this project's data quality: a run
configured with a seed must be byte-for-byte repeatable. Every random consume
(decks, opponent cards, run-outs, Monte Carlo draws) goes through an Rng,
and sessions/hands/evaluations each get an independent stream via child() so
no component's randomness perturbs another's.
"""

import numpy as np
from numpy.random import Generator, SeedSequence


def subseed(root_seed: int, *labels: int) -> int:
    """Derive a child seed from a root seed plus integer labels.

    SeedSequence mixes all inputs, so distinct label tuples give distinct,
    uncorrelated seeds regardless of root. This is how hierarchical streams
    are made independent yet reproducible.
    """
    ss = SeedSequence([root_seed, *labels])
    return int(ss.generate_state(1)[0])


class Rng:
    """Thin facade over a numpy Generator with hierarchical child spawning."""

    def __init__(self, seed: int):
        self.seed = seed
        self.generator: Generator = np.random.default_rng(seed)

    def integers(self, low: int, high: int | None = None, size=None) -> np.ndarray | int:
        return self.generator.integers(low, high, size=size)

    def random(self, size=None) -> np.ndarray | float:
        return self.generator.random(size)

    def shuffle(self, array) -> None:
        self.generator.shuffle(array)

    def child(self, *labels: int) -> "Rng":
        """Spawn an independent child stream keyed by labels."""
        return Rng(subseed(self.seed, *labels))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_rng.py`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/rng.py tests/test_rng.py && git commit -m "feat: deterministic hierarchical rng harness"
```

---

### Task 1.3: Card and deck primitives (`card.py`)

**Files:**
- Create: `poker/card.py`
- Test: `tests/test_card.py`

**Why integers:** every downstream hot path (Monte Carlo sampling, batch hand evaluation, numba JIT) operates on numpy int arrays; object tuples would block vectorization. The 0..51 encoding is documented once here and used everywhere.

- [ ] **Step 1: Write the failing tests**

`tests/test_card.py`:
```python
"""Card encoding and deck integrity invariants."""
import numpy as np
import pytest

from poker.card import (
    card_id, new_deck, rank_of, shuffle_deck, suit_of,
)
from poker.rng import Rng


class TestCardId:
    def test_encoding_roundtrip_across_all_52_cards(self):
        for rank in range(2, 15):
            for suit in range(4):
                cid = card_id(rank, suit)
                assert 0 <= cid < 52
                assert rank_of(cid) == rank
                assert suit_of(cid) == suit

    def test_encoding_is_bijective(self):
        ids = [card_id(rank, suit) for rank in range(2, 15) for suit in range(4)]
        assert sorted(ids) == list(range(52))

    def test_ace_is_14(self):
        assert rank_of(card_id(14, 0)) == 14


class TestDeck:
    def test_new_deck_is_full_and_unique(self):
        deck = new_deck()
        assert len(deck) == 52
        assert len(set(deck.tolist())) == 52

    def test_shuffle_preserves_card_multiset(self):
        deck = new_deck()
        shuffled = shuffle_deck(Rng(123), deck)
        assert sorted(shuffled.tolist()) == sorted(deck.tolist())

    def test_shuffle_does_not_mutate_input(self):
        deck = new_deck()
        original = deck.copy()
        shuffle_deck(Rng(1), deck)
        assert np.array_equal(deck, original)

    def test_shuffle_is_deterministic_per_seed(self):
        a = shuffle_deck(Rng(5), new_deck())
        b = shuffle_deck(Rng(5), new_deck())
        assert np.array_equal(a, b)
        c = shuffle_deck(Rng(6), new_deck())
        assert not np.array_equal(a, c)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_card.py`
Expected: `ModuleNotFoundError: No module named 'poker.card'`

- [ ] **Step 3: Write the implementation**

`poker/card.py`:
```python
"""Card and deck primitives.

Cards are encoded as integers 0..51 (id = (rank-2)*4 + suit, rank 2..14,
suit 0..3). Integers, not tuples, so every downstream function stays numpy/
numba friendly. These id <-> (rank, suit) maps are the single source of truth.
"""

import numpy as np

SUITS = (0, 1, 2, 3)
RANKS = tuple(range(2, 15))
_ACE_RANK = 14

_RANK_OF = np.arange(52, dtype=np.int8) // 4 + 2   # canonical rank per id
_SUIT_OF = np.arange(52, dtype=np.int8) % 4        # canonical suit per id


def card_id(rank: int, suit: int) -> int:
    """Return the 0..51 id for a (rank, suit) pair."""
    return (rank - 2) * 4 + suit


def rank_of(card: np.int32 | int) -> int:
    """Return the poker rank (2..14) of a card id."""
    return int(_RANK_OF[card])


def suit_of(card: np.int32 | int) -> int:
    """Return the suit (0..3) of a card id."""
    return int(_SUIT_OF[card])


def new_deck() -> np.ndarray:
    """Return an ordered 52-card deck as int32 (id 0..51)."""
    return np.arange(52, dtype=np.int32)


def shuffle_deck(rng, deck: np.ndarray) -> np.ndarray:
    """Shuffle a copy of `deck` in place using the supplied Rng stream.

    Uses the Generator's own Fisher-Yates, which is deterministic for a fixed
    seed and consumes random numbers from that stream only.
    """
    shuffled = deck.copy()
    rng.shuffle(shuffled)
    return shuffled
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_card.py`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/card.py tests/test_card.py && git commit -m "feat: integer card encoding and deterministic deck shuffle"
```

---

### Task 1.4: Reference 5-card hand evaluator (`evaluate_5_reference`)

**Files:**
- Create: `poker/hand_evaluator.py` (reference part only in this task)
- Test: `tests/test_hand_evaluator_reference.py`

**Why a separate reference:** the fast encoded `score_5` is the production path, but an independent, readable implementation prevents a subtle off-by-one in scoring logic from silently contaminating every downstream metric (equity, showdowns, BB/100). The two are forced to agree in Task 1.6.

- [ ] **Step 1: Write the failing tests**

`tests/test_hand_evaluator_reference.py`:
```python
"""Reference evaluator: known hands and category ordering."""
import pytest

from poker.card import card_id
from poker.hand_evaluator import evaluate_5_reference

H = card_id  # readability alias


class TestKnownHands:
    def test_royal_flush(self):
        cards = [H(14, 0), H(13, 0), H(12, 0), H(11, 0), H(10, 0)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 8 and kick[0] == 14 - 2

    def test_straight_flush_nine_high(self):
        cards = [H(9, 1), H(8, 1), H(7, 1), H(6, 1), H(5, 1)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 8 and kick[0] == 9 - 2

    def test_wheel_straight_flush(self):
        # A-5-4-3-2 suited: the lowest straight, high card is the 5.
        cards = [H(14, 2), H(5, 2), H(4, 2), H(3, 2), H(2, 2)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 8 and kick[0] == 5 - 2

    def test_four_of_a_kind(self):
        cards = [H(9, 0), H(9, 1), H(9, 2), H(9, 3), H(4, 0)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 7 and kick == (9 - 2, 4 - 2, 0, 0, 0)

    def test_full_house(self):
        cards = [H(7, 0), H(7, 1), H(7, 2), H(3, 0), H(3, 1)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 6 and kick == (7 - 2, 3 - 2, 0, 0, 0)

    def test_flush_ranks_descending(self):
        cards = [H(14, 0), H(10, 0), H(8, 0), H(6, 0), H(3, 0)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 5 and kick == (12, 8, 6, 4, 1)

    def test_straight_ten_high(self):
        cards = [H(10, 0), H(9, 1), H(8, 2), H(7, 0), H(6, 1)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 4 and kick[0] == 10 - 2

    def test_wheel_straight(self):
        cards = [H(14, 0), H(2, 1), H(3, 2), H(4, 0), H(5, 1)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 4 and kick[0] == 5 - 2

    def test_three_of_a_kind(self):
        cards = [H(5, 0), H(5, 1), H(5, 2), H(12, 0), H(3, 1)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 3 and kick == (3, 10, 1, 0, 0)

    def test_two_pair(self):
        cards = [H(11, 0), H(11, 1), H(8, 2), H(8, 0), H(2, 1)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 2 and kick == (9, 6, 0, 0, 0)

    def test_one_pair(self):
        cards = [H(13, 0), H(13, 1), H(9, 2), H(7, 0), H(4, 1)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 1 and kick == (11, 7, 5, 2, 0)

    def test_high_card(self):
        cards = [H(14, 0), H(11, 1), H(8, 2), H(6, 0), H(2, 1)]
        cat, kick = evaluate_5_reference(cards)
        assert cat == 0 and kick == (12, 9, 6, 4, 0)

    def test_top_pair_kicker_order(self):
        a = evaluate_5_reference([H(9, 0), H(9, 1), H(14, 2), H(7, 0), H(4, 1)])
        b = evaluate_5_reference([H(9, 0), H(9, 1), H(13, 2), H(7, 0), H(4, 1)])
        # Same pair, bigger kicker wins.
        assert (a[1] > b[1])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_hand_evaluator_reference.py`
Expected: `ModuleNotFoundError: No module named 'poker.hand_evaluator'`

- [ ] **Step 3: Write the implementation**

`poker/hand_evaluator.py` — reference part (the fast scoring functions are added in Task 1.5):
```python
"""Five- and seven-card hand evaluation.

Two implementations:
  * evaluate_5_reference  — verbose, human-readable; used ONLY in tests as an
    independent oracle so bugs in the fast path can't silently poison results.
  * score_5 / hand_score / score_batch — fast encoded scores used in
    production and in Monte Carlo equity. Implemented in Task 1.5.

Encoded scores are a single monotonic int: strictly larger = strictly better.
Encoding: score = category * 13**5 + k0*13**4 + ... + k4, where k_i are
normalized kickers (rank - 2, i.e. 0..12) ordered most-significant first.
Base 13 suffices because normalized kickers never exceed 12. The maximum value
8*13**5 + 12*(13**4 + ...) is ~ 3.19 million, safely inside int32.
"""

from poker.card import rank_of, suit_of

# Categories per the technical document (0 = high card .. 8 = straight flush).
CATEGORY_NAMES = (
    "high card", "one pair", "two pair", "three of a kind", "straight",
    "flush", "full house", "four of a kind", "straight flush",
)


def _straight_high(rank_list):
    """Return the high rank of a straight made of `rank_list`, else None.

    Handles the wheel (A-2-3-4-5), whose effective high card is the 5.
    """
    uniq = sorted(set(rank_list), reverse=True)
    if len(uniq) != 5:
        return None
    if uniq[0] - uniq[4] == 4:
        return uniq[0]
    if uniq[0] == 14 and uniq[1] == 5 and uniq[4] == 2:
        return 5
    return None


def _kind_groups(rank_list):
    """Rank counts as ((count, rank), ...) sorted by count then rank (desc)."""
    from collections import Counter
    counts = Counter(rank_list)
    return sorted(((n, r) for r, n in counts.items()), reverse=True)


def evaluate_5_reference(cards):
    """Return (category, kickers) of a 5-card hand.

    kickers is a 5-tuple of normalized ranks (rank - 2, so 0..12), most
    significant first, zero-padded — purely so it is directly comparable to the
    encoded score of Task 1.5.
    """
    ranks = [rank_of(c) for c in cards]
    is_flush = all(suit_of(c) == suit_of(cards[0]) for c in cards)
    straight_high = _straight_high(ranks)
    groups = _kind_groups(ranks)

    if straight_high is not None and is_flush:
        return (8, (straight_high - 2, 0, 0, 0, 0))
    if groups[0][0] == 4:                     # four of a kind
        quad, kick = groups[0][1], groups[1][1]
        return (7, (quad - 2, kick - 2, 0, 0, 0))
    if groups[0][0] == 3 and groups[1][0] == 2:  # full house
        trips, pair = groups[0][1], groups[1][1]
        return (6, (trips - 2, pair - 2, 0, 0, 0))
    if is_flush:
        return (5, tuple(r - 2 for r in ranks))  # 5 kickers, already 5 long
    if straight_high is not None:
        return (4, (straight_high - 2, 0, 0, 0, 0))
    if groups[0][0] == 3:                     # three of a kind
        trips = groups[0][1]
        kickers = [r for r in ranks if r != trips]
        return (3, (trips - 2,) + tuple(sorted((r - 2 for r in kickers), reverse=True)) + (0, 0))
    if groups[0][0] == 2 and groups[1][0] == 2:  # two pair
        hi_pair, lo_pair = groups[0][1], groups[1][1]
        kick = next(r for r in ranks if r not in (hi_pair, lo_pair))
        return (2, (hi_pair - 2, lo_pair - 2, kick - 2, 0, 0))
    if groups[0][0] == 2:                     # one pair
        pair = groups[0][1]
        kickers = [r for r in ranks if r != pair]
        return (1, (pair - 2,) + tuple(sorted((r - 2 for r in kickers), reverse=True)) + (0,))
    return (0, tuple(r - 2 for r in ranks))   # high card, already 5 long
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_hand_evaluator_reference.py`
Expected: `13 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/hand_evaluator.py tests/test_hand_evaluator_reference.py && git commit -m "feat: reference 5-card hand evaluator oracle"
```

---

### Task 1.5: Fast encoded scorer (`score_5`, `hand_score`, `score_batch`)

**Files:**
- Modify: `poker/hand_evaluator.py` (append fast path)
- Create: `tests/test_hand_evaluator_fast.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_hand_evaluator_fast.py`:
```python
"""Fast encoded scorer: monotonicity, best-of-7, batch, and agreement with the
reference evaluator across random hands."""
import numpy as np
import pytest

from poker.hand_evaluator import (
    evaluate_5_reference, hand_score, score_5, score_batch,
)
from poker.rng import Rng

B13 = 13 ** 5  # weight of the category digit


def encode_reference(cards):
    """Independent encoding straight from the reference oracle."""
    cat, kick = evaluate_5_reference(cards)
    score = cat * B13
    for i, k in enumerate(kick):
        score += k * 13 ** (4 - i)
    return score


class TestScore5:
    def test_score_matches_reference(self):
        rng = Rng(99)
        for _ in range(20_000):
            cards = rng.generator.choice(52, 5, replace=False)
            assert score_5(cards) == encode_reference(cards), cards

    def test_royal_flush_beats_all_lower_categories(self):
        from poker.card import card_id as H
        royal = [H(14, 0), H(13, 0), H(12, 0), H(11, 0), H(10, 0)]
        best_lower = [H(14, 1), H(14, 2), H(14, 3), H(13, 2), H(13, 3)]
        assert score_5(royal) > score_5(best_lower)


class TestHandScore:
    def test_picks_best_five_of_seven(self):
        from poker.card import card_id as H
        # Board has a diamond flush; hero holds Qd for the nut flush over a
        # lower two pair / straight competitor within the same 7 cards.
        seven = [H(14, 0), H(13, 0), H(11, 0), H(9, 0), H(6, 0), H(2, 0), H(10, 1)]
        # Best 5 is the flush A-K-J-9-6 of diamonds.
        expected = encode_reference([H(14, 0), H(13, 0), H(11, 0), H(9, 0), H(6, 0)])
        assert hand_score(seven) == expected

    def test_hand_score_agrees_with_reference_best_of_21(self):
        import itertools
        from poker.card import card_id as H
        rng = Rng(7)
        for _ in range(5_000):
            seven = np.array(rng.generator.choice(52, 7, replace=False), dtype=np.int32)
            best = max(
                encode_reference(list(combo))
                for combo in itertools.combinations(seven.tolist(), 5)
            )
            assert hand_score(seven) == best


class TestScoreBatch:
    def test_batch_matches_scalar(self):
        rng = Rng(3)
        hands = np.array([rng.generator.choice(52, 7, replace=False) for _ in range(50)],
                         dtype=np.int32)
        scalar = np.array([hand_score(h) for h in hands])
        assert np.array_equal(score_batch(hands), scalar)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_hand_evaluator_fast.py`
Expected: `AttributeError` (functions not defined yet)

- [ ] **Step 3: Write the implementation**

Append to `poker/hand_evaluator.py`:
```python
import numpy as np

_CAT_WEIGHT = 13 ** 5   # weight of the category digit in the encoded score
_POW = np.array([13**4, 13**3, 13**2, 13, 1], dtype=np.int64)  # kicker weights


def score_5(cards) -> int:
    """Return the monotonic encoded score of an exactly-5-card hand.

    Mirrors evaluate_5_reference's category logic but emits a single int. It
    is written in plain Python (indexing, not numpy) so it is numba-jittable
    in Stage 03; correctness is pinned to the reference by tests.
    """
    ranks = sorted((card.rank_of(c) for c in cards), reverse=True)
    suits = [card.suit_of(c) for c in cards]
    is_flush = all(s == suits[0] for s in suits)

    uniq = sorted(set(ranks), reverse=True)
    straight_high = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_high = uniq[0]
        elif uniq[0] == 14 and uniq[1] == 5 and uniq[4] == 2:
            straight_high = 5

    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    groups = sorted(((n, r) for r, n in counts.items()), reverse=True)

    if is_flush and straight_high is not None:
        kick = (straight_high - 2, 0, 0, 0, 0)
        cat = 8
    elif groups[0][0] == 4:
        kick = (groups[0][1] - 2, groups[1][1] - 2, 0, 0, 0)
        cat = 7
    elif groups[0][0] == 3 and groups[1][0] == 2:
        kick = (groups[0][1] - 2, groups[1][1] - 2, 0, 0, 0)
        cat = 6
    elif is_flush:
        kick = tuple(r - 2 for r in ranks)
        cat = 5
    elif straight_high is not None:
        kick = (straight_high - 2, 0, 0, 0, 0)
        cat = 4
    elif groups[0][0] == 3:
        trip_r = groups[0][1]
        rest = sorted((r for r in ranks if r != trip_r), reverse=True)
        kick = (trip_r - 2,) + tuple(r - 2 for r in rest) + (0, 0)
        cat = 3
    elif groups[0][0] == 2 and groups[1][0] == 2:
        hi, lo = groups[0][1], groups[1][1]
        third = next(r for r in ranks if r not in (hi, lo))
        kick = (hi - 2, lo - 2, third - 2, 0, 0)
        cat = 2
    elif groups[0][0] == 2:
        pair_r = groups[0][1]
        rest = sorted((r for r in ranks if r != pair_r), reverse=True)
        kick = (pair_r - 2,) + tuple(r - 2 for r in rest) + (0,)
        cat = 1
    else:
        kick = tuple(r - 2 for r in ranks)
        cat = 0

    # Assemble: category digit + five normalized kicker digits (base 13).
    return cat * _CAT_WEIGHT + int(np.dot(np.asarray(kick), _POW))


def hand_score(cards7) -> int:
    """Best-5-of-7 score: max score_5 over all C(7,5)=21 combinations."""
    best = -1
    for i in range(7):
        for j in range(i + 1, 7):
            five = np.delete(cards7, [i, j])
            s = score_5(five)
            if s > best:
                best = s
    return best


def score_batch(hands: np.ndarray) -> np.ndarray:
    """Vectorized hand_score over an (N, 7) int array -> (N,) encoded scores."""
    out = np.empty(len(hands), dtype=np.int64)
    for i, seven in enumerate(hands):
        out[i] = hand_score(seven)
    return out
```

Make sure module-level `import numpy as np` and a `card` module alias are placed at the top of the file (add `from poker import card` import). The file must start with all imports. **Adjustment:** put `from poker import card` and `import numpy as np` at the top of `hand_evaluator.py`, then reference `card.rank_of` etc. or rename local aliases accordingly. The module already has a `from poker.card import rank_of, suit_of` — the fast path uses `card.rank_of`; keep both by importing the module too: `from poker import card`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_hand_evaluator_fast.py`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/hand_evaluator.py tests/test_hand_evaluator_fast.py && git commit -m "feat: monotonic encoded hand scorer and best-of-7 batch path"
```

---

### Task 1.6: Cross-validation & category distribution property test

**Files:**
- Create: `tests/test_hand_evaluator_crossval.py`

**Why this exists:** an exhaustive category-count test is the strongest possible guard against a systematically wrong scorer. It is expensive, so it is marked `slow` and not part of the default run; a fast sampled variant runs always.

- [ ] **Step 1: Write the tests**

`tests/test_hand_evaluator_crossval.py`:
```python
"""Property tests: category distribution across random hands, plus a `slow`
exhaustive 5-card sweep. Guard against systematic scorer bias."""
import itertools

import numpy as np
import pytest

from poker.card import card_id
from poker.hand_evaluator import CATEGORY_NAMES, score_5
from poker.rng import Rng

# True combinatorial counts of each 5-card category from a 52-card deck.
EXPECTED_COUNTS = [
    1302540,   # 0 high card
    1098240,   # 1 one pair
    123552,    # 2 two pair
    54912,     # 3 three of a kind
    10200,     # 4 straight
    5108,      # 5 flush
    3744,      # 6 full house
    624,       # 7 four of a kind
    40,        # 8 straight flush
]


def test_random_sample_categories_are_consistent():
    rng = Rng(11)
    seen = np.zeros(9)
    for _ in range(100_000):
        cards = np.array(rng.generator.choice(52, 5, replace=False), dtype=np.int32)
        cat = score_5(cards) // 13 ** 5
        seen[cat] += 1
    assert np.all(seen > 0)  # every category appears in practice


@pytest.mark.slow
def test_exhaustive_5_card_category_histogram():
    counts = {n: 0 for n in range(9)}
    for combo in itertools.combinations(range(52), 5):
        cards = np.array(combo, dtype=np.int32)
        cat = score_5(cards) // 13 ** 5
        counts[cat] += 1
    for cat in range(9):
        assert counts[cat] == EXPECTED_COUNTS[cat], (
            f"category {cat} ({CATEGORY_NAMES[cat]}): "
            f"got {counts[cat]}, expected {EXPECTED_COUNTS[cat]}"
        )
```

- [ ] **Step 2: Run the fast test**

Run: `.venv/bin/pytest tests/test_hand_evaluator_crossval.py`
Expected: `1 passed, 1 skipped`

- [ ] **Step 3: Run the slow exhaustive test once**

Run: `MOVENV=.venv/bin pytest -m slow -q tests/test_hand_evaluator_crossval.py` (or `.venv/bin/pytest -m slow tests/test_hand_evaluator_crossval.py`)
Expected: `1 passed` with the exact histogram matching `EXPECTED_COUNTS`. This is the moment any scorer bias would surface.

- [ ] **Step 4: Commit**

```bash
git add tests/test_hand_evaluator_crossval.py && git commit -m "test: hand-category distribution cross-validation"
```

---

### Task 1.7: Central configuration (`config.py`)

**Files:**
- Create: `poker/config.py`
- Test: `tests/test_config.py`

**Data-quality role:** every experiment and replay records its exact `Config` dump in metadata, so results are self-describing and auditable.

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
"""Config is immutable and serializable for experiment metadata."""
import pytest

from poker.config import Config


def test_defaults_match_rules_document():
    c = Config()
    assert c.num_players == 6
    assert c.sb == 1 and c.bb == 2
    assert c.initial_stack == 200
    assert c.min_bet == c.bb
    assert c.max_aggressions == 4  # bet + 3 raises per round
    assert len(c.strategy_combo) == 6


def test_config_is_frozen():
    c = Config()
    with pytest.raises(Exception):
        c.num_players = 5


def test_config_serializes_to_plain_dict():
    d = Config().as_dict()
    assert d["bb"] == 2 and d["mc_iterations"] == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`poker/config.py`:
```python
"""Central configuration.

One source of truth so every experiment and replay declares its exact rules as
metadata (data-quality requirement: results must be self-describing). Frozen so
a running session cannot mutate its own provenance.
"""

from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class Config:
    num_players: int = 6
    sb: int = 1
    bb: int = 2
    initial_stack: int = 200
    min_bet: int = 2            # rule 2.2: minimum bet = big blind
    max_aggressions: int = 4    # rule 2.1: bet + up to 3 raises per round
    mc_iterations: int = 400
    preflop_table_path: str = "data/preflop_equity.npy"
    num_hands: int = 50_000
    strategy_combo: Tuple[str, ...] = (
        "Tight", "Loose", "Aggressive", "Passive", "Mathematician", "Adaptive",
    )
    seed: int = 42
    log_dir: str = "output/logs"
    adaptive_window: int = 100     # roll history kept for adaptive reads (risk §10)
    adaptive_adjust_every: int = 20  # how often adaptive recomputes thresholds

    def as_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/config.py tests/test_config.py && git commit -m "feat: frozen central config with metadata serialization"
```

---

### Task 1.8: Stage gate — full suite and integration smoke

**Files:**
- Create: `tests/test_stage01_integration.py`

- [ ] **Step 1: Write the integration smoke test**

`tests/test_stage01_integration.py`:
```python
"""Stage 01 gate: seeded pipeline reproducibility + hand-strength ordering."""
import numpy as np
from poker.card import card_id as H, new_deck, shuffle_deck
from poker.hand_evaluator import hand_score
from poker.rng import Rng


def test_fully_seeded_hand_pipeline_is_reproducible():
    deck_a = shuffle_deck(Rng(2024), new_deck())
    deck_b = shuffle_deck(Rng(2024), new_deck())
    assert np.array_equal(deck_a, deck_b)
    board = deck_a[:5]
    assert np.all(board < 52)
    # hole + board draw from the same shuffled deck are all distinct
    assert len(set(board.tolist())) == 5


def test_hand_strength_ordering_holds_on_a_real_board():
    # A made hand (two pair, nines and sevens) outranks a bare straight draw
    # fragment on the *same* board + runout — exercises the monotonic score.
    board = np.array([H(9, 0), H(7, 1), H(3, 2), H(2, 0), H(14, 2)], dtype=np.int32)
    two_pair = hand_score(np.array([H(9, 1), H(7, 2), *board.tolist()], dtype=np.int32))
    one_pair = hand_score(np.array([H(9, 2), H(5, 3), *board.tolist()], dtype=np.int32))
    assert two_pair > one_pair
    # pocket rockets on this board (top pair + nut kicker) beat the same
    # board under a middling pocket pair
    with_aces = hand_score(np.array([H(14, 0), H(14, 1), *board.tolist()], dtype=np.int32))
    with_tens = hand_score(np.array([H(10, 0), H(10, 1), *board.tolist()], dtype=np.int32))
    assert with_aces > with_tens
```

- [ ] **Step 2: Run the smoke test**

Run: `.venv/bin/pytest tests/test_stage01_integration.py`
Expected: `2 passed`

- [ ] **Step 3: Run the full stage suite**

Run: `.venv/bin/pytest`
Expected: all tests in the package pass (default suite; slow excluded)

- [ ] **Step 4: Update module docstring note if needed and commit**

```bash
git add tests/test_stage01_integration.py && git commit -m "test: stage 01 integration gate"
```

---

## Stage Exit Criteria

- [ ] `pytest` default suite green: RNG determinism/independence, card encoding/bijection, reference+fast evaluator agreement (25k+ random cross-checks), batch/scalar agreement, config frozen.
- [ ] Slow exhaustive category histogram verified once (`-m slow`).
- [ ] Repo initialized, three proceeding stages can import `poker.card`, `poker.hand_evaluator`, `poker.config`, `poker.rng`.

**Report to the user before Stage 02:**
- Any deviation from the locked contracts above (none expected).
- Measured speed of the pure-Python `hand_score` — this is the baseline Stage 03 must beat with numba, and it sizes the Monte Carlo profile.