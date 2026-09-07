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
