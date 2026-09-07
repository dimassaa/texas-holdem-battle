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
        # spawning children must not perturb the parent stream: an identical
        # parent that spawned/consumed a child draws the same values as one
        # that never spawned.
        p1 = Rng(10)
        p2 = Rng(10)
        p1.child(3).random(10)
        assert np.array_equal(p1.random(3), p2.random(3))


class TestSubseed:
    def test_subseed_is_deterministic_and_label_sensitive(self):
        assert subseed(5, 1, 2) == subseed(5, 1, 2)
        assert subseed(5, 1, 2) != subseed(5, 2, 1)

    def test_subseed_output_is_a_uint32(self):
        s = subseed(42, 9)
        assert isinstance(s, int)
        assert 0 <= s < 2**32
