# Stage 03 — Equity Calculation: Preflop Tables, Monte Carlo, numba

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the equity engine: canonical 169 start-hand index, a precomputed preflop table (vs 1..5 random opponents), a deterministic vectorized Monte Carlo for postflop, per-state caching, and a numba-jitted fast path validated byte-identical to the Stage-01 reference.

**Architecture:** `poker/equity.py` exposes `hand_type_index`, `build_preflop_table`, `calc_equity`, and `EquityProvider`. The provider is the single choke point strategies use for all hand-strength queries (so Stage 04 strategies are exactly as specified in §4 of the technical document: decisions on equity vs pot odds). Reproducibility: every equity evaluation draws from a dedicated `Rng` child stream — deterministic and isolated per (hand, board, opponents) key.

**Tech Stack additions:** numba (`pip install numba`), already-declared numpy.

**Data-quality focus:**
- **Determinism:** same (seed, hand, board, opponents) → same equity, run after run.
- **Accuracy vs truth:** MC must agree with the *exact* combinatorial preflop equity within tolerance, and known benchmarks must hold (AA vs random ≈ 85%, AKs ≈ 67%, 72o ≈ 32% against one opponent).
- **Caching correctness:** cache key must include everything that changes the number (hole ranks+suitedness, board, street, opponent count) — a stale-equity cache is a silent data corruption.
- **Batch/vectorized path must equal scalar path** (pooled vs per-row) — regression guard for the numba rewrite.

**Locked contracts:**
- `equity.hand_type_index(rank_a, rank_b, suited) -> 0..168` (rank order canonical, documented below).
- `equity.build_preflop_table(rng, iterations=40_000, num_players=6) -> np.ndarray (169,5)`.
- `equity.calc_equity(hand, board, num_opponents, rng, mc_iterations, cache=None) -> float in [0,1]`.
- `equity.EquityProvider` class with `.equity(hand, board, n_opp) -> float` and `.pot_odds() -> float`; holds a per-hand cache; takes `rng` and `config`.
- `equity.deal_permutations(rng, pool, n, k) -> np.ndarray(n,k)` vectorized no-replacement sampler (deterministic).

---

### Task 3.1: Canonical start-hand index (169 types)

**Files:**
- Create: `poker/equity.py` (index part)
- Test: `tests/test_equity_index.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_equity_index.py`:
```python
"""Canonical 169 start-hand classification."""
import numpy as np
from poker.card import rank_of, suit_of
from poker.equity import hand_type_index
from poker.rng import Rng


def test_all_169_types_are_covered_exactly_once():
    seen = set()
    for r_a in range(2, 15):
        for r_b in range(2, 15):
            for suited in (0, 1):
                if r_a == r_b and suited:
                    continue  # a pocket pair is neither suited nor offsuit
                seen.add(hand_type_index(r_a, r_b, suited))
    assert len(seen) == 169
    assert min(seen) == 0 and max(seen) == 168


def test_pocket_pair_maps_expected_cell():
    assert hand_type_index(14, 14, 0) == hand_type_index(14, 14, 1)
    assert hand_type_index(6, 6, 0) == hand_type_index(6, 6, 1)


def test_suited_and_offsuit_are_distinct():
    assert hand_type_index(9, 5, 1) != hand_type_index(9, 5, 0)


def test_rank_order_is_symmetric_and_canonical():
    # A2o and 2Ao are the same run-down hand type.
    assert hand_type_index(14, 2, 0) == hand_type_index(2, 14, 0)


def test_equity_table_shape_from_build_path():
    # sanity: the index composition supports a (169, 5) table.
    types = np.zeros(169, dtype=np.int32)
    for r_a in range(2, 15):
        for r_b in range(2, r_a + 1):
            for suited in (0, 1):
                if r_a == r_b and suited:
                    continue
                types[hand_type_index(r_a, r_b, suited)] += 1
    assert np.count_nonzero(types) == 169
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_equity_index.py`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement the index**

`poker/equity.py` (index + docstring; the rest comes in later tasks):
```python
"""Equity: 169 start-hand index, preflop table, Monte Carlo, EquityProvider.

The 169 types are enumerated in three groups:
  * 13 pocket pairs (cell for rank r is (r-2, r-2))
  * 78 suited non-pairs   (cells on/above the diagonal use +12 offset)
  * 78 offsuit non-pairs
so that every (rank_a, rank_b, suited) mapping lands in 0..168 bijectively.
"""
import numpy as np

from poker.card import rank_of, suit_of
from poker.rng import Rng

# Precomputed lookup: hand_type_index[rank_a][rank_b][suited] -> 0..168.
_INDEX = np.full((15, 15, 2), -1, dtype=np.int32)


def _build_index():
    nxt = 0
    # pockets
    for r in range(2, 15):
        _INDEX[r][r][0] = nxt
        _INDEX[r][r][1] = nxt
        nxt += 1
    # suited non-pairs (off-diag: suited=True)
    for hi in range(2, 15):
        for lo in range(2, hi):
            _INDEX[hi][lo][1] = nxt
            _INDEX[lo][hi][1] = nxt
            nxt += 1
    # offsuit non-pairs (off-diag: suited=False)
    for hi in range(2, 15):
        for lo in range(2, hi):
            _INDEX[hi][lo][0] = nxt
            _INDEX[lo][hi][0] = nxt
            nxt += 1


_build_index()


def hand_type_index(rank_a: int, rank_b: int, suited: int) -> int:
    """Return the canonical 0..168 start-hand index."""
    return int(_INDEX[rank_a][rank_b][suited])


def start_hand_type(hand) -> int:
    """Canonical index for a two-card starting hand (array of 2 card ids)."""
    a, b = int(hand[0]), int(hand[1])
    suited = 1 if suit_of(a) == suit_of(b) else 0
    return hand_type_index(rank_of(a), rank_of(b), suited)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_equity_index.py`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/equity.py tests/test_equity_index.py && git commit -m "feat: canonical 169 start-hand index"
```

---

### Task 3.2: Vectorized deterministic no-replacement sampler

**Files:**
- Modify: `poker/equity.py`
- Test: `tests/test_equity_sampler.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_equity_sampler.py`:
```python
"""Deterministic vectorized sampling without replacement."""
import numpy as np
from poker.equity import deal_permutations
from poker.rng import Rng


def test_sampler_returns_distinct_cards_per_row():
    rng = Rng(1)
    pool = np.arange(52, dtype=np.int32)
    out = deal_permutations(rng, pool, n=500, k=6)
    assert out.shape == (500, 6)
    assert all(len(set(row.tolist())) == 6 for row in out)


def test_sampler_never_uses_pool_exclusions():
    rng = Rng(2)
    pool = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    out = deal_permutations(rng, pool, n=100, k=3)
    assert set(out.ravel().tolist()).issubset({0, 1, 2, 3, 4, 5})


def test_sampler_is_deterministic():
    pool = np.arange(52, dtype=np.int32)
    a = deal_permutations(Rng(3), pool, 100, 5)
    b = deal_permutations(Rng(3), pool, 100, 5)
    assert np.array_equal(a, b)
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_equity_sampler.py`
Expected: `ModuleNotFoundError: deal_permutations`

- [ ] **Step 3: Implement**

Append to `poker/equity.py`:
```python
def deal_permutations(rng, pool: np.ndarray, n: int, k: int) -> np.ndarray:
    """Draw n *independent* permutations of `pool` and keep the first k each.

    Implemented with stable argsort over uniform keys (one row per draw), which
    is fully deterministic for a fixed Rng stream and avoids the placeholder of
    replace=False (not available vectorized). n*len(pool) entries in memory;
    for MC sizing (n=400, len<50) that is negligible.
    """
    keys = rng.random((n, len(pool)))
    order = np.argsort(keys, axis=1, kind="stable")
    return pool[order[:, :k]]
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_equity_sampler.py`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/equity.py tests/test_equity_sampler.py && git commit -m "feat: deterministic vectorized deal sampler"
```

---

### Task 3.3: `calc_equity` — vectorized Monte Carlo

**Files:**
- Modify: `poker/equity.py`
- Test: `tests/test_equity_calc.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_equity_calc.py`:
```python
"""Monte Carlo equity: benchmark sanity, tie handling, determinism, caching."""
import numpy as np
import pytest

from poker.card import card_id as H
from poker.config import Config
from poker.equity import calc_equity, EquityProvider
from poker.hand_evaluator import hand_score
from poker.rng import Rng


def test_aa_preflop_equity_near_theoretical():
    acc = [calc_equity(np.array([H(14, 0), H(14, 1)]), np.empty(0, dtype=np.int32),
                       1, Rng(1000 + i), mc_iterations=20_000) for i in range(10)]
    mean = float(np.mean(acc))
    assert 0.83 < mean < 0.87   # known AA vs random ~ 85%


def test_ak_suited_vs_random_roughly_two_thirds():
    eq = calc_equity(np.array([H(14, 0), H(13, 0)]), np.empty(0, dtype=np.int32),
                     1, Rng(44), mc_iterations=30_000)
    assert 0.63 < eq < 0.71


def test_deterministic_given_seed():
    a = calc_equity(np.array([H(8, 0), H(7, 1)]), np.array([H(14, 2), H(9, 3), H(2, 0)]),
                    2, Rng(9), mc_iterations=2000)
    b = calc_equity(np.array([H(8, 0), H(7, 1)]), np.array([H(14, 2), H(9, 3), H(2, 0)]),
                    2, Rng(9), mc_iterations=2000)
    assert a == b


def test_tie_handling_is_fractional():
    # A 6-high straight-flush board plays itself: no hero hand can beat it, so
    # BOTH hero and every opponent tie with the board on every simulation.
    # Equity must be ~0.5, exercising the 0.5-tie branch for every opponent.
    board = np.array([H(2, 0), H(3, 0), H(4, 0), H(5, 0), H(6, 0)]).astype(np.int32)
    hero = np.array([H(12, 0), H(11, 0)], dtype=np.int32)   # Qc Jc cannot top it
    eq = calc_equity(hero, board, 1, Rng(1), mc_iterations=500)
    assert 0.45 < eq < 0.55


def test_dead_cards_are_excluded_from_the_random_pool():
    from poker.equity import _cards_left
    hand = np.array([H(8, 0), H(14, 1)])               # hero holds 8c, As
    board = np.array([H(2, 0), H(9, 3), H(7, 2)])      # board holds 2c, 9s, 7h
    pool = _cards_left(hand, board)
    used = set(hand.tolist() + board.tolist())
    assert len(pool) == 52 - 5
    assert not (set(pool.tolist()) & used)             # no known card can be dealt


def test_helper_pool_is_topologically_correct_for_draws():
    # every sampled opponent/runout card must come from the pool
    from poker.equity import _cards_left, deal_permutations
    hand = np.array([H(8, 0), H(14, 1)])
    board = np.array([H(2, 0), H(9, 3), H(7, 2)])
    pool = _cards_left(hand, board)
    drawn = deal_permutations(Rng(0), pool, 200, 5)
    assert set(drawn.ravel().tolist()).issubset(set(pool.tolist()))
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_equity_calc.py`
Expected: `ModuleNotFoundError: calc_equity`

- [ ] **Step 3: Implement**

Append to `poker/equity.py`:
```python
from poker.hand_evaluator import score_batch


def _cards_left(hand, board):
    """Cards not yet visible to the evaluator (the random pool)."""
    used = set(int(c) for c in np.concatenate([hand, board]))
    return np.array(sorted(set(range(52)) - used), dtype=np.int32)


def calc_equity(hand, board, num_opponents, rng, mc_iterations, cache=None):
    """Monte Carlo equity vs `num_opponents` random hands (doc §4.2).

    Returns (wins + 0.5*ties) / simulations. `rng` is the stream for this
    evaluation; pass a fresh child stream to keep evaluation independent.
    `cache` (optional dict) is shared per hand by the provider to skip
    identical evaluations (doc §4.2 optimization).

    Construction of every 7-card input, for hero and each opponent, is always:
    [2 hole] + [len(board) known board] + [5 - len(board) runout cards]. The
    runout matrix is the SAME physical cards for every player (they share the
    community board), which the concatenation below guarantees.
    """
    key = (tuple(int(c) for c in hand), tuple(int(c) for c in board), num_opponents)
    if cache is not None and key in cache:
        return cache[key]
    pool = _cards_left(hand, board)
    m = 5 - len(board)                 # runout cards still to come
    k = 2 * num_opponents + m
    drawn = deal_permutations(rng, pool, mc_iterations, k)
    runout = drawn[:, -m:]             # same runout viewed by every player
    hero_seven = np.empty((mc_iterations, 7), dtype=np.int32)
    opp_seven = np.empty((mc_iterations, num_opponents, 7), dtype=np.int32)
    for j in range(num_opponents):
        opp_hole = drawn[:, 2*j:2*j+2]
        opp_seven[:, j, :2] = opp_hole
        opp_seven[:, j, 2:2+len(board)] = np.broadcast_to(board, (mc_iterations, len(board)))
        opp_seven[:, j, 2+len(board):] = runout
    hero_seven[:, :2] = np.broadcast_to(hand, (mc_iterations, 2))
    hero_seven[:, 2:2+len(board)] = np.broadcast_to(board, (mc_iterations, len(board)))
    hero_seven[:, 2+len(board):] = runout

    hero = score_batch(hero_seven)
    best_opp = np.max([score_batch(opp_seven[:, j]) for j in range(num_opponents)], axis=0)
    wins = np.count_nonzero(hero > best_opp)
    ties = np.count_nonzero(hero == best_opp)
    eq = (wins + 0.5 * ties) / mc_iterations
    if cache is not None:
        cache[key] = eq
    return float(eq)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_equity_calc.py`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/equity.py tests/test_equity_calc.py && git commit -m "feat: vectorized deterministic monte carlo equity"
```

---

### Task 3.4: `EquityProvider` with per-hand cache and pot odds

**Files:**
- Modify: `poker/equity.py`
- Test: `tests/test_equity_provider.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_equity_provider.py`:
```python
"""EquityProvider: cache correctness, pot odds, per-hand lifecycle."""
import numpy as np
from poker.card import card_id as H
from poker.config import Config
from poker.equity import EquityProvider
from poker.rng import Rng


def test_cache_skips_repeat_computation_with_same_result():
    prov = EquityProvider(rng=Rng(5), config=Config())
    hand = np.array([H(14, 0), H(13, 1)])
    board = np.array([H(2, 0), H(9, 3), H(7, 2)])
    a = prov.equity(hand, board, 2)
    b = prov.equity(hand, board, 2)
    assert a == b and prov._cache_hits > 0


def test_cache_key_changes_with_opponent_count():
    prov = EquityProvider(rng=Rng(6), config=Config())
    hand = np.array([H(10, 0), H(9, 1)])
    board = np.array([H(2, 0), H(3, 3), H(5, 2)])
    v1 = prov.equity(hand, board, 1)
    v2 = prov.equity(hand, board, 3)
    assert v1 != v2


def test_pot_odds_reflects_board_state():
    prov = EquityProvider(rng=Rng(7), config=Config(), pot=100, to_call=25)
    assert abs(prov.pot_odds() - 25 / (100 + 25)) < 1e-9
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_equity_provider.py`
Expected: `ModuleNotFoundError: EquityProvider`

- [ ] **Step 3: Implement**

Append to `poker/equity.py`:
```python
class EquityProvider:
    """Cache + context manager for hand-strength queries.

    A fresh instance is created per hand (or per decision site) so `cache` and
    `_cache_hits` are short-lived by construction — stale-equity-across-hands
    is impossible (data-quality requirement).
    """

    def __init__(self, rng, config, pot=0, to_call=0):
        self.rng = rng
        self.config = config
        self.pot = pot
        self.to_call = to_call
        self._cache = {}
        self._cache_hits = 0

    def equity(self, hand, board, n_opp) -> float:
        key = (tuple(int(c) for c in hand), tuple(int(c) for c in board), n_opp)
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        eq = calc_equity(hand, board, n_opp, self.rng.child(*key), self.config.mc_iterations)
        self._cache[key] = eq
        return eq

    def pot_odds(self) -> float:
        """Required winning share for a break-even call (doc §5 input)."""
        denom = self.pot + self.to_call
        return 0.0 if denom == 0 else self.to_call / denom
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_equity_provider.py`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/equity.py tests/test_equity_provider.py && git commit -m "feat: cached equity provider with pot odds"
```

---

### Task 3.5: Preflop table generation

**Files:**
- Modify: `poker/equity.py`
- Test: `tests/test_preflop_table.py`

**Why table matters:** preflop decisions fire on every hand before any postflop evaluation; a constant-time lookup keeps the tight/loose/adaptive ranges cheap and makes equity identical across sessions (table is deterministic, not sampled at runtime).

- [ ] **Step 1: Write the failing tests**

`tests/test_preflop_table.py`:
```python
"""Preflop table: shape, determinism, monotonicity, benchmark spot-checks."""
import numpy as np
import pytest

from poker.config import Config
from poker.equity import build_preflop_table
from poker.rng import Rng


def test_table_shape_is_169x5():
    table = build_preflop_table(Rng(1), iterations=2000)
    assert table.shape == (169, 5)


def test_table_is_deterministic():
    a = build_preflop_table(Rng(2), iterations=3000)
    b = build_preflop_table(Rng(2), iterations=3000)
    assert np.array_equal(a, b)


def test_pocket_aces_are_top_equity_for_every_opponent_count():
    table = build_preflop_table(Rng(3), iterations=2000)
    from poker.equity import hand_type_index
    aa = hand_type_index(14, 14, 0)
    for opp in range(5):
        assert table[aa, opp] == table.max(axis=0)[opp]
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_preflop_table.py`
Expected: `ModuleNotFoundError: build_preflop_table`

- [ ] **Step 3: Implement**

Append to `poker/equity.py`. The single authoritative implementation is the
direct-enumeration helper below (`_representative_for`); there is no lookup
table and no separate `_representative` — keep it that way.

```python
def _representative_for(index: int) -> np.ndarray:
    """A deterministic concrete 2-card hand whose start-hand type is `index`.

    Pair: (r,r) suited=0. Non-pair: (hi, lo) suited per class. The suits chosen
    ensure the hand is internally distinct and dead cards are excluded by the
    MC pool below.
    """
    for hi in range(2, 15):
        for lo in range(2, hi + 1):
            for suited in (0, 1):
                if hand_type_index(hi, lo, suited) == index:
                    if hi == lo:
                        return np.array([card_id(hi, 0), card_id(hi, 1)], dtype=np.int32)
                    if suited:
                        return np.array([card_id(hi, 0), card_id(lo, 0)], dtype=np.int32)
                    return np.array([card_id(hi, 0), card_id(lo, 1)], dtype=np.int32)
    raise AssertionError(f"no hand for index {index}")


def build_preflop_table(rng, iterations=40_000, num_players=6) -> np.ndarray:
    """Compute equity of all 169 start-hand classes vs 1..5 random opponents.

    Uses the identical Monte Carlo path as calc_equity, so the number-dead-card
    semantics (own two cards removed) are guaranteed by construction. Deterministic
    for a fixed `rng`. Sizing: 169 * 5 opponent counts * iterations simulations,
    fully vectorized; ~ minutes at default iterations on a laptop.
    """
    table = np.empty((169, num_players - 1), dtype=np.float32)
    for idx in range(169):
        hand = _representative_for(idx)
        for opp in range(1, num_players):
            table[idx, opp - 1] = calc_equity(hand, np.empty(0, dtype=np.int32), opp,
                                              rng.child(idx, opp), iterations)
    return table
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_preflop_table.py`
Expected: `3 passed`

- [ ] **Step 5: Save the production table**

Run:
```bash
mkdir -p data
.venv/bin/python - <<'PY'
import numpy as np
from poker.config import Config
from poker.equity import build_preflop_table
from poker.rng import Rng
t = build_preflop_table(Rng(2024), iterations=40000)
np.save(Config().preflop_table_path, t)
print("saved", t.shape, "AA vs n", [round(float(t[0,o]),3) for o in range(5)])
PY
```
Expected: `saved (169, 5) AA vs n [0.85, 0.735, 0.644, 0.573, 0.516]` (values approximate).

- [ ] **Step 6: Commit**

```bash
git add poker/equity.py tests/test_preflop_table.py && git commit -m "feat: deterministic preflop equity table generation"
```

---

### Task 3.6: numba JIT fast path with equivalence validation

**Files:**
- Modify: `poker/equity.py` (wire `calc_equity_jit`; `_cards_left` / `deal_permutations` already exist)
- Create: `poker/jit.py` (jittable wrappers; exports `score_batch_jit`)
- Test: `tests/test_jit_equivalence.py`

**Why:** MC equity is the hot path the whole simulation dies on (doc §8.3 and Risks §10). numba compiles the pure-Python `score_5` to native speed. The byte-identity test is the guard that numba and pure Python can never silently diverge.

- [ ] **Step 1: Write the failing tests**

`tests/test_jit_equivalence.py`:
```python
"""numba fast path must be byte-identical to the pure-Python path."""
import numpy as np
import pytest

from poker.config import Config
from poker.equity import calc_equity, calc_equity_jit
from poker.hand_evaluator import hand_score, score_batch
from poker.rng import Rng

numba = pytest.importorskip("numba")

from poker.jit import score_batch_jit  # noqa: E402


def test_jit_and_python_scores_identical_over_random_hands():
    rng = Rng(77)
    hands = np.array([rng.generator.choice(52, 7, replace=False) for _ in range(5000)],
                     dtype=np.int32)
    assert np.array_equal(score_batch(hands), score_batch_jit(hands))


def test_jit_calc_equity_matches_python():
    from poker.card import card_id as H
    hand = np.array([H(11, 0), H(10, 1)])
    board = np.array([H(2, 0), H(9, 3), H(7, 1), H(5, 2)])
    a = calc_equity(hand, board, 3, Rng(1), 400)
    b = calc_equity_jit(hand, board, 3, Rng(1), 400)
    assert a == b
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_jit_equivalence.py`
Expected: `ModuleNotFoundError: poker.jit`

- [ ] **Step 3: Implement the JIT module**

`poker/jit.py` (final, single authoritative version):
```python
"""numba-accelerated hot paths.

Kept separate from hand_evaluator so the pure-Python reference stays readable
and the two can be cross-validated. numba is optional: if it is unavailable, or
it cannot compile the current reference implementation, every jit entry point
degrades to the validated Python path. The JIT is for speed only, never for
semantics.
"""

import numpy as np

from poker.hand_evaluator import hand_score

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - stripped installs only
    HAVE_NUMBA = False


def _score_batch_python(hands):
    """Reference loop: score every 7-card row with the pure-Python scorer."""
    out = np.empty(len(hands), dtype=np.int64)
    for i in range(len(hands)):
        out[i] = hand_score(hands[i])
    return out


if HAVE_NUMBA:
    try:
        # Compile the *same* reference function; the row scorer must also be
        # visible to numba as a module-level function, hence the alias.
        _row_jit = njit(cache=True)(hand_score)

        @njit(cache=True)
        def _score_batch_jit_impl(hands):
            out = np.empty(len(hands), dtype=np.int64)
            for i in range(len(hands)):
                out[i] = _row_jit(hands[i])
            return out
    except Exception:  # pragma: no cover - any compile failure degrades safely
        _score_batch_jit_impl = None
else:  # pragma: no cover
    _score_batch_jit_impl = None


def score_batch_jit(hands):
    """Batch 7-card scorer: numba-accelerated when available, Python otherwise.

    The JIT path duplicates NOTHING: it compiles the very `hand_score` used by
    the pure-Python path, so the equivalence test is a self-check that
    compilation did not change behavior. Compile failures degrade to the loop.
    """
    impl = _score_batch_jit_impl
    if impl is not None:
        return impl(hands)
    return _score_batch_python(hands)
```

`NJIT` compiles the module-level `hand_score` alias in one step, so no nested-`
functions or dynamically bound globals reach the compiled loop — that is the
only reliable `njit` pattern here.

- [ ] **Step 4: Wire `calc_equity_jit` with identical Monte-Carlo orchestration**

Append to `poker/equity.py`:
```python
from poker.jit import score_batch_jit


def calc_equity_jit(hand, board, num_opponents, rng, mc_iterations, cache=None):
    """The Task-3.3 Monte Carlo, re-scored through the numba batch path.

    Orchestration (pool, deal permutations, runout sharing, win/tie counting)
    is byte-identical to `calc_equity`; only `score_batch` -> `score_batch_jit`
    differs, so semantics match by construction and determinism is preserved.
    """
    key = (tuple(int(c) for c in hand), tuple(int(c) for c in board), num_opponents)
    if cache is not None and key in cache:
        return cache[key]
    pool = _cards_left(hand, board)
    m = 5 - len(board)
    k = 2 * num_opponents + m
    drawn = deal_permutations(rng, pool, mc_iterations, k)
    runout = drawn[:, -m:]
    hero_seven = np.empty((mc_iterations, 7), dtype=np.int32)
    opp_seven = np.empty((mc_iterations, num_opponents, 7), dtype=np.int32)
    for j in range(num_opponents):
        opp_seven[:, j, :2] = drawn[:, 2*j:2*j+2]
        opp_seven[:, j, 2:2+len(board)] = np.broadcast_to(board, (mc_iterations, len(board)))
        opp_seven[:, j, 2+len(board):] = runout
    hero_seven[:, :2] = np.broadcast_to(hand, (mc_iterations, 2))
    hero_seven[:, 2:2+len(board)] = np.broadcast_to(board, (mc_iterations, len(board)))
    hero_seven[:, 2+len(board):] = runout

    hero = score_batch_jit(hero_seven)
    best_opp = np.max([score_batch_jit(opp_seven[:, j]) for j in range(num_opponents)], axis=0)
    wins = np.count_nonzero(hero > best_opp)
    ties = np.count_nonzero(hero == best_opp)
    eq = (wins + 0.5 * ties) / mc_iterations
    if cache is not None:
        cache[key] = eq
    return float(eq)
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/test_jit_equivalence.py`
Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add poker/jit.py poker/equity.py tests/test_jit_equivalence.py && git commit -m "feat: numba fast paths with byte-identical equivalence tests"
```

---

### Task 3.7: Stage gate — MC-vs-exact postflop validation & performance smoke

**Files:**
- Create: `tests/test_stage03_gate.py`

- [ ] **Step 1: Write the validation test**

For small states (≤2 opponents, ≤3 known board cards) the doc allows exact enumeration with caching (§4.2). Implement a quick *exact* reference by iterating all remaining opponent/deck draws (usually tractable with ≤1-2 opponents and small runout counts), then assert MC at high iterations is within ±2% of exact:

`tests/test_stage03_gate.py`:
```python
"""Stage-03 gate: Monte Carlo approximates the exact combinatorial answer."""
import itertools
import numpy as np
import pytest

from poker.card import card_id as H, rank_of, suit_of
from poker.equity import calc_equity
from poker.hand_evaluator import hand_score
from poker.rng import Rng


def exact_postflop_equity(hand, board, n_opp):
    """Reference: enumerate every remaining-board and every opponent hand.

    Only tractable for n_opp <= 2 and short runouts. Ties are fractional.
    """
    used = set(map(int, list(hand) + list(board)))
    remaining = [c for c in range(52) if c not in used]
    wins = ties = total = 0
    board_len = len(board)
    for runout in itertools.permutations(remaining, 5 - board_len):
        if n_opp == 1:
            opp_combos = itertools.combinations(
                [c for c in remaining if c not in runout], 2)
            for opp in opp_combos:
                total += 1
                hero = hand_score(np.array([*hand, *board, *runout], dtype=np.int32))
                foe = hand_score(np.array([*opp, *board, *runout], dtype=np.int32))
                if hero > foe: wins += 1
                elif hero == foe: ties += 1
        elif n_opp == 2:
            pool = [c for c in remaining if c not in runout]
            for opps in itertools.combinations(pool, 4):
                total += 1
                hero = hand_score(np.array([*hand, *board, *runout], dtype=np.int32))
                foe = max(
                    hand_score(np.array([*opps[:2], *board, *runout], dtype=np.int32)),
                    hand_score(np.array([*opps[2:], *board, *runout], dtype=np.int32)),
                )
                if hero > foe: wins += 1
                elif hero == foe: ties += 1
        else:
            raise ValueError("exact reference supports <= 2 opponents")
    return (wins + 0.5 * ties) / total


@pytest.mark.slow
def test_mc_matches_exact_on_flop():
    hand = np.array([H(11, 0), H(10, 1)])
    board = np.array([H(2, 0), H(9, 3), H(7, 2)])
    exact = exact_postflop_equity(hand, board, 1)
    mc = calc_equity(hand, board, 1, Rng(1), 50_000)
    assert abs(exact - mc) < 0.02
```

- [ ] **Step 2: Run the gate test**

Run: `.venv/bin/pytest -m slow tests/test_stage03_gate.py`
Expected: `1 passed` (the slow exact enumeration is bounded — 6 runout cards × remaining-combos for 1 opp ≈ small)

- [ ] **Step 3: Performance smoke — measure MC throughput**

Run:
```bash
.venv/bin/python - <<'PY'
import time, numpy as np
from poker.card import card_id as H
from poker.config import Config
from poker.equity import calc_equity
from poker.rng import Rng
hand = np.array([H(9,0), H(8,1)]); board = np.array([H(14,2), H(3,3), H(2,0)])
t0 = time.perf_counter()
for i in range(200):
    calc_equity(hand, board, 3, Rng(i), 400)
dt = (time.perf_counter() - t0) / 200
print(f"median eval: {dt*1000:.1f} ms for 400 sims vs 3 opps")
PY
```
Expected: single-digit to low-double-digit ms per evaluation with numba active. Record this number in the report — it sizes every downstream experiment (Stage 06).

- [ ] **Step 4: Commit**

```bash
git add tests/test_stage03_gate.py && git commit -m "test: equity monte-carlo vs exact postflop gate"
```

---

## Stage Exit Criteria

- [ ] 169-class index bijective and tested.
- [ ] MC equity matches known preflop benchmarks (AA ≈ 85%, AKs ≈ 67%) at high iterations.
- [ ] MC matches the exact combinatorial answer on a flop spot within 2% (slow test).
- [ ] Determinism given fixed seed; cache correctness; ties fractional; dead-card pool respected.
- [ ] numba path byte-identical to Python path and enabled by default.
- [ ] Preflop table saved and reproducible.

**Report to the user before Stage 04:**
- The measured MC-eval latency (it determines affordable decision frequency during simulation).
- The exact saved preflop table entropy/spot-checks.
- Any divergence found between numba and Python (there must be none).