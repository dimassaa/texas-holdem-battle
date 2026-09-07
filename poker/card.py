"""Card and deck primitives.

Cards are encoded as integers 0..51 (id = (rank-2)*4 + suit, rank 2..14,
suit 0..3). Integers, not tuples, so every downstream function stays numpy/
numba friendly. These id <-> (rank, suit) maps are the single source of truth.
"""

import numpy as np

SUITS = (0, 1, 2, 3)
RANKS = tuple(range(2, 15))
_ACe_RANK = 14

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