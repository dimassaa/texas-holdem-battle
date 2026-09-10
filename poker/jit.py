"""numba-accelerated hot paths (Option A: native mask-based 7-card scorer).

Kept separate from hand_evaluator so the pure-Python reference stays readable
and the two can be cross-validated. The controller-approved plan (stage03,
Task 3.6) does NOT compile hand_score/score_5 into nopython mode (impossible:
score_5 calls the plain-Python helper set, uses generator expressions and
np.delete). Instead this module implements a from-scratch, mask-based scorer
whose output is byte-identical to hand_evaluator.score_batch on every input,
while using completely different internals (suit rank masks, bit tricks).

numba is optional: a stripped install without numba degrades to the Python
loop. When numba IS present the fast path must activate; a compile failure
here is detected at import time and surfaces via the byte-identity and
performance gates, never silently disabled. The JIT is for speed only, never
for semantics.
"""

import numpy as np

from poker.hand_evaluator import hand_score

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - stripped installs only
    HAVE_NUMBA = False

# Encoding weights shared with hand_evaluator.score_5:
# score = category*13**5 + k0*13**4 + k1*13**3 + k2*13**2 + k3*13 + k4, with
# kicker digits rank-2 (0..12).  Constants are module-level ints, which numba
# freezes into the compiled code as constants (no Python-listed globals).
_CAT_W = 13 ** 5
_POW4 = 13 ** 4
_POW3 = 13 ** 3
_POW2 = 13 ** 2
_POW1 = 13

if HAVE_NUMBA:
    @njit(cache=True)
    def _top_bit(m):
        """Index of the highest set bit of `m`, or -1.

        numba 0.67 does not expose int.bit_length(), so the highest set bit is
        resolved with an explicit high-to-low scan over the 13 rank bits.
        """
        for b in range(12, -1, -1):
            if (m >> b) & 1:
                return b
        return -1

    @njit(cache=True)
    def _bit_count(m):
        """Number of set bits of `m` (Kernighan: one iter per set bit).

        numba 0.67 does not expose int.bit_count(); a 7-card hand's suit has at
        most 7 cards, so this loop is at most 7 iterations.
        """
        c = 0
        while m:
            m &= m - 1
            c += 1
        return c

    @njit(cache=True)
    def _top_fold(mask, n):
        """Fold the top n set-bit indices of `mask` into base-13 digits.

        Bits are rank-2 (0..12); scanning high index first yields the encoded
        kicker digit sequence of n distinct ranks, MSB first. Used for every
        "top-k non-anchor ranks" component of the score.
        """
        v = 0
        got = 0
        for b in range(12, -1, -1):
            if (mask >> b) & 1:
                v = v * 13 + b
                got += 1
                if got == n:
                    break
        return v

    @njit(cache=True)
    def _straight_digit(m):
        """Highest straight in rank mask `m` as a score digit, or -1.

        `m & (m>>1) & ... & (m>>4)` sets bit i iff bits i..i+4 are all present,
        i.e. i is the LOWEST bit of a 5-run, whose top rank-digit is i+4.
        The wheel (A-2-3-4-5; bits 12,0,1,2,3) is not a contiguous run in the
        bit mask, so it is checked explicitly and maps to digit 3 (rank 5).
        A real run (digit >= 4) always dominates the wheel.
        """
        run = m & (m >> 1) & (m >> 2) & (m >> 3) & (m >> 4)
        if run:
            return _top_bit(run) + 4
        if (m & 0x100F) == 0x100F:
            return 3
        return -1

    @njit(cache=True)
    def _score_7(hand):
        """Encoded best-5-of-7 score; byte-identical to hand_score(hand).

        Internals differ from score_5 on purpose: each suit is a 13-bit rank
        mask (bit r-2), and categories/kickers are resolved with bit tricks and
        a single low-to-high rank scan. Zero heap allocations per hand (only
        scalars), so the batch loop stays allocation-free.
        """
        # Build the four suit masks. Card id = (rank-2)*4 + suit means the
        # bit index is id>>2 and the suit id is id&3.
        m0 = 0
        m1 = 0
        m2 = 0
        m3 = 0
        for i in range(7):
            cid = hand[i]
            b = cid >> 2
            s = cid & 3
            if s == 0:
                m0 |= 1 << b
            elif s == 1:
                m1 |= 1 << b
            elif s == 2:
                m2 |= 1 << b
            else:
                m3 |= 1 << b

        allmask = m0 | m1 | m2 | m3

        # Straight flush is the highest 5-run inside one suit; dominating.
        sf = _straight_digit(m0)
        d = _straight_digit(m1)
        if d > sf:
            sf = d
        d = _straight_digit(m2)
        if d > sf:
            sf = d
        d = _straight_digit(m3)
        if d > sf:
            sf = d
        if sf >= 0:
            return _CAT_W * 8 + sf * _POW4

        # Plain straight: 5 consecutive ranks anywhere in the 7 cards.
        straight = _straight_digit(allmask)

        # Rank counts, scanning high-to-low; each rank's digit `b` doubles as
        # its encoded kicker value. Capture the single possible quad, up to two
        # trips, and up to two pairs (all a 7-card rank multiset admits).
        quad = -1
        t1 = -1
        t2 = -1
        p1 = -1
        p2 = -1
        for b in range(12, -1, -1):
            c = ((m0 >> b) & 1) + ((m1 >> b) & 1) + ((m2 >> b) & 1) + ((m3 >> b) & 1)
            if c >= 4:
                quad = b
            elif c == 3:
                if t1 < 0:
                    t1 = b
                elif t2 < 0:
                    t2 = b
            elif c == 2:
                # Guard each slot so a THIRD pair cannot overwrite p2; the
                # third pair is handled purely as the two-pair kicker below.
                if p1 < 0:
                    p1 = b
                elif p2 < 0:
                    p2 = b

        # Four of a kind: the highest remaining rank is the single kicker.
        if quad >= 0:
            kick = _top_fold(allmask ^ (1 << quad), 1)
            return _CAT_W * 7 + quad * _POW4 + kick * _POW3

        # Full house: trips plus the second trip (borrowing 2 of its cards) or
        # the higher real pair.
        if t1 >= 0 and (t2 >= 0 or p1 >= 0):
            pair = t2 if t2 >= 0 else p1
            return _CAT_W * 6 + t1 * _POW4 + pair * _POW3

        # Flush: best suit's top five ranks folded as base-13 digits. The
        # numeric comparison is lexicographic on the folded rank sequence.
        flush = -1
        if _bit_count(m0) >= 5:
            flush = _top_fold(m0, 5)
        if _bit_count(m1) >= 5:
            v = _top_fold(m1, 5)
            if v > flush:
                flush = v
        if _bit_count(m2) >= 5:
            v = _top_fold(m2, 5)
            if v > flush:
                flush = v
        if _bit_count(m3) >= 5:
            v = _top_fold(m3, 5)
            if v > flush:
                flush = v
        if flush >= 0:
            return _CAT_W * 5 + flush

        if straight >= 0:
            return _CAT_W * 4 + straight * _POW4

        # Three of a kind: trip rank plus the top two remaining ranks.
        if t1 >= 0:
            kick = _top_fold(allmask ^ (1 << t1), 2)
            return _CAT_W * 3 + t1 * _POW4 + kick * _POW2

        # Two pair: two highest pairs plus the top remaining rank as kicker
        # (which may itself be the third pair when three pairs are present).
        if p1 >= 0 and p2 >= 0:
            kick = _top_fold(allmask ^ (1 << p1) ^ (1 << p2), 1)
            return _CAT_W * 2 + p1 * _POW4 + p2 * _POW3 + kick * _POW2

        # One pair: pair rank plus the top three remaining ranks.
        if p1 >= 0:
            kick = _top_fold(allmask ^ (1 << p1), 3)
            return _CAT_W * 1 + p1 * _POW4 + kick * _POW1

        # High card: top five distinct ranks, folded base-13 (no category).
        return _top_fold(allmask, 5)

    @njit(cache=True)
    def _score_batch_jit_impl(hands):
        """Batch loop over all rows; the ONLY Python<->numba boundary.

        Each row is handled by _score_7 with only scalar state, so this loop
        never allocates. Output dtype and values mirror score_batch exactly.
        """
        out = np.empty(hands.shape[0], dtype=np.int64)
        for i in range(hands.shape[0]):
            out[i] = _score_7(hands[i])
        return out

    # Eager warm-up: forces compilation of the whole chain now (cold cache) so
    # a scorer that cannot compile fails at import time instead of degrading.
    _score_batch_jit_impl(np.zeros((1, 7), dtype=np.int32))
else:  # pragma: no cover - stripped installs only
    _score_batch_jit_impl = None


def _score_batch_python(hands):
    """Reference loop: score every 7-card row with the pure-Python scorer."""
    out = np.empty(len(hands), dtype=np.int64)
    for i in range(len(hands)):
        out[i] = hand_score(hands[i])
    return out


def score_batch_jit(hands):
    """Batch 7-card scorer: numba-accelerated when available, Python otherwise.

    Output is byte-identical to hand_evaluator.score_batch (pinned by
    tests/test_jit_equivalence.py and the 200k-hand ad-hoc validation).
    """
    impl = _score_batch_jit_impl
    if impl is not None:
        return impl(hands)
    return _score_batch_python(hands)