"""Central configuration.

One source of truth so every experiment and replay declares its exact rules as
metadata (data-quality requirement: results must be self-describing). Frozen so
a running session cannot mutate its own provenance.
"""

from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class Config:
    num_players: int = 6
    sb: int = 1
    bb: int = 2
    initial_stack: int = 200
    min_bet: int = 2            # rule 2.2: minimum bet = big blind
    max_aggressions: int = 4    # rule 2.1: bet + up to 3 raises per round
    mc_iterations: int = 400
    preflop_table_path: str = "data/preflop_equity.npy"
    num_hands: int = 50_000
    strategy_combo: Tuple[str, ...] = (
        "Tight", "Loose", "Aggressive", "Passive", "Mathematician", "Adaptive",
    )
    seed: int = 42
    log_dir: str = "output/logs"
    adaptive_window: int = 100     # roll history kept for adaptive reads (risk §10)
    adaptive_adjust_every: int = 20  # how often adaptive recomputes thresholds

    def as_dict(self) -> dict:
        return asdict(self)
