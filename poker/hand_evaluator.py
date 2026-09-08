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

import numpy as np

from poker import card
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


# ---------------------------------------------------------------------------
# Fast encoded scorer (Task 1.5)
# ---------------------------------------------------------------------------

_CAT_WEIGHT = 13 ** 5   # weight of the category digit in the encoded score
_POW = np.array([13**4, 13**3, 13**2, 13, 1], dtype=np.int64)  # kicker weights


def score_5(cards) -> int:
    """Return the monotonic encoded score of an exactly-5-card hand.

    Mirrors evaluate_5_reference's category logic but emits a single int. It
    is written in plain Python (indexing, not numpy) so it is numba-jittable
    in Stage 03; correctness is pinned to the reference by tests.
    """
    # Input-order ranks (reference evaluator uses these for flush / high card).
    ranks_input = [card.rank_of(c) for c in cards]
    # Sorted ranks (used for group detection and most kicker orderings).
    ranks = sorted(ranks_input, reverse=True)
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
        # Reference returns flush kickers in input order, not sorted.
        kick = tuple(r - 2 for r in ranks_input)
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
        # Reference returns high-card kickers in input order, not sorted.
        kick = tuple(r - 2 for r in ranks_input)
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