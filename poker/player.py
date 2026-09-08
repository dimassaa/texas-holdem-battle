"""Player model used by the engine.

Stage 02 needs only the mutable chip state (stack, contributed, folded,
all_in); full strategy classes are implemented in Stage 04/05. The dataclass
is intentionally behavior-free to keep engine tests strategy-agnostic.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Player:
    name: str
    strategy: str
    stack: int
    hole: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    folded: bool = False
    all_in: bool = False
    contributed: int = 0   # total chips committed across the whole hand
    stats: dict = field(default_factory=dict)
    rank: int = 0          # seat index, set by the simulator when seating
