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
