"""numba fast path must be byte-identical to the pure-Python path."""
import numpy as np

from poker.equity import calc_equity, calc_equity_jit
from poker.hand_evaluator import score_batch
from poker.jit import HAVE_NUMBA, score_batch_jit
from poker.rng import Rng

_CAT_BASE = 13 ** 5


def test_numba_is_active_by_default():
    # Stage-03 exit criterion: the compiled scorer is the default path, so the
    # equivalence tests below exercise compiled code rather than degenerating to
    # Python-vs-Python. A numba-less environment must fail loudly here, not
    # silently skip the suite.
    assert HAVE_NUMBA is True


def test_jit_and_python_scores_identical_over_random_hands():
    rng = Rng(77)
    hands = np.array(
        [rng.generator.choice(52, 7, replace=False) for _ in range(5000)],
        dtype=np.int32,
    )
    assert np.array_equal(score_batch(hands), score_batch_jit(hands))


def test_crafted_hands_cover_every_scorer_branch():
    # Hand-picked 7-card sets, one per scorer branch: straights (incl. wheel),
    # straight flush (run and wheel), one-suit wheel flush that is NOT a
    # straight flush, three pairs (third pair becomes the two-pair kicker),
    # full house from a pair and from two trips, quad, trips, pair, high card.
    crafted = np.array(
        [
            # royal flush (straight flush from a real run)
            [51, 47, 43, 39, 35, 0, 5],
            # wheel straight flush (A-2-3-4-5)
            [50, 2, 6, 10, 14, 20, 25],
            # A-high heart flush, no straight flush (7 replaces the 5)
            [51, 3, 7, 11, 23, 28, 37],
            # plain straight 6..T, no flush
            [16, 21, 27, 30, 32, 1, 50],
            # wheel straight A-2-3-4-5, no flush
            [50, 3, 4, 10, 13, 23, 24],
            # three pairs -> two pair with third pair as kicker
            [51, 10, 8, 5, 7, 2, 0],
            # full house, trip + pair
            [50, 51, 49, 44, 46, 43, 1],
            # full house, two trips
            [50, 51, 49, 44, 46, 45, 43],
            # four of a kind + top kicker
            [30, 31, 29, 28, 51, 46, 1],
            # three of a kind
            [26, 27, 25, 48, 46, 43, 1],
            # one pair
            [22, 23, 49, 44, 42, 31, 1],
            # high card
            [51, 46, 41, 36, 30, 27, 21],
        ],
        dtype=np.int32,
    )
    reference = score_batch(crafted)
    assert np.array_equal(reference, score_batch_jit(crafted))
    # The table is a branch map: every category 0..8 must appear, or the table
    # rots silently and rare-branch coverage regresses.
    assert len(set(reference // _CAT_BASE)) == 9


def test_python_fallback_matches_score_batch(monkeypatch):
    # Force the compiled dispatcher away to exercise the graceful-degradation
    # path on numba installs too, pinning the fallback to the reference.
    import poker.jit as jit_mod

    rng = Rng(78)
    hands = np.array(
        [rng.generator.choice(52, 7, replace=False) for _ in range(100)],
        dtype=np.int32,
    )
    monkeypatch.setattr(jit_mod, "_score_batch_jit_impl", None)
    assert np.array_equal(score_batch_jit(hands), score_batch(hands))


def test_jit_calc_equity_matches_python():
    from poker.card import card_id as H

    hand = np.array([H(11, 0), H(10, 1)])
    board = np.array([H(2, 0), H(9, 3), H(7, 1), H(5, 2)])
    a = calc_equity(hand, board, 3, Rng(1), 400)
    b = calc_equity_jit(hand, board, 3, Rng(1), 400)
    assert a == b