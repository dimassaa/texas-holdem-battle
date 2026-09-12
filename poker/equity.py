"""Equity: 169 start-hand index, preflop table, Monte Carlo, EquityProvider.

The 169 types are enumerated in three groups:
  * 13 pocket pairs (cell for rank r is (r-2, r-2))
  * 78 suited non-pairs   (cells on/above the diagonal use +12 offset)
  * 78 offsuit non-pairs
so that every (rank_a, rank_b, suited) mapping lands in 0..168 bijectively.
"""
import numpy as np

from poker.card import card_id, rank_of, suit_of
from poker.hand_evaluator import score_batch
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


def deal_permutations(rng, pool: np.ndarray, n: int, k: int) -> np.ndarray:
    """Draw n independent permutations of `pool` and keep the first k each.

    Implemented with stable argsort over uniform keys (one row per draw), which
    is fully deterministic for a fixed Rng stream and avoids the placeholder of
    replace=False (not available vectorized). n*len(pool) entries in memory;
    for MC sizing (n=400, len<50) that is negligible.
    """
    keys = rng.random((n, len(pool)))
    order = np.argsort(keys, axis=1, kind="stable")
    return pool[order[:, :k]]


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
    # When m == 0 the board is already complete; -0 == 0 in Python, so
    # drawn[:, -0:] would return the full array instead of an empty slice.
    runout = drawn[:, -m:] if m else np.empty((mc_iterations, 0), dtype=np.int32)
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
        # Rng.child labels must be flat ints (SeedSequence rejects nested
        # sequences), so the (hand, board, n_opp) key is flattened. Hand is
        # always 2 cards and board length is fixed per game state, making the
        # flattening injective: each state gets its own deterministic stream.
        #
        # calc_equity_jit routes the rollout scoring through the numba batch
        # scorer (poker/jit.py) instead of the pure-Python score_batch. Their
        # outputs are byte-identical -- pinned by test_jit_equivalence and the
        # 200k-hand validation -- and the rng stream is consumed identically,
        # so the provider's numbers (and therefore hands.jsonl bytes) are
        # unchanged; only runtime drops by ~250x on MC-heavy sessions (doc
        # §8.1's promised numpy+numba+multiprocessing stack).
        eq = calc_equity_jit(hand, board, n_opp,
                             self.rng.child(*key[0], *key[1], key[2]),
                             self.config.mc_iterations)
        self._cache[key] = eq
        return eq

    def pot_odds(self) -> float:
        """Required winning share for a break-even call (doc §5 input)."""
        denom = self.pot + self.to_call
        return 0.0 if denom == 0 else self.to_call / denom


from poker.jit import score_batch_jit  # noqa: E402  (module-level, after class)


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
    m = 5 - len(board)                 # runout cards still to come
    k = 2 * num_opponents + m
    drawn = deal_permutations(rng, pool, mc_iterations, k)
    # When m == 0 the board is already complete; -0 == 0 in Python, so
    # drawn[:, -0:] would return the full array instead of an empty slice.
    runout = drawn[:, -m:] if m else np.empty((mc_iterations, 0), dtype=np.int32)
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

    hero = score_batch_jit(hero_seven)
    best_opp = np.max([score_batch_jit(opp_seven[:, j]) for j in range(num_opponents)], axis=0)
    wins = np.count_nonzero(hero > best_opp)
    ties = np.count_nonzero(hero == best_opp)
    eq = (wins + 0.5 * ties) / mc_iterations
    if cache is not None:
        cache[key] = eq
    return float(eq)


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

    Uses the identical Monte Carlo path as calc_equity (via calc_equity_jit,
    the byte-identical numba fast path), so the number-dead-card semantics
    (own two cards removed) are guaranteed by construction. Deterministic
    for a fixed `rng`. Sizing: 169 * 5 opponent counts * iterations simulations,
    fully vectorized; ~ minutes at default iterations on a laptop with numba.
    """
    table = np.empty((169, num_players - 1), dtype=np.float32)
    for idx in range(169):
        hand = _representative_for(idx)
        for opp in range(1, num_players):
            table[idx, opp - 1] = calc_equity_jit(hand, np.empty(0, dtype=np.int32), opp,
                                                  rng.child(idx, opp), iterations)
    return table
