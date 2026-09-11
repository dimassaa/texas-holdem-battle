"""Range audit vs Table 1's approximate percentages (fail loudly on drift).

Every basic strategy's playable range must stay inside Table-1's band. The
band check is by combo-share (weighted for pairs/suited/offsuit asymmetry),
not by raw hand-type count: a hand type is 1-6 combos, so type counts lie.
"""
import pytest

from poker.player import (
    AggressiveStrategy,
    LooseStrategy,
    PassiveStrategy,
    TightStrategy,
    TIGHT_PREMIUM,
    TIGHT_RANGE,
    LOOSE_RANGE,
    combo_weight,
)


def share(range_set):
    return sum(combo_weight(t) for t in range_set) / 1326.0


# Table-1 "approximate ranges". Each window keeps the band quoted in the plan's
# audit table (Tight 12-15%, Loose 40%, Aggressive 30%, Passive 25%) but with
# ~1pp spring so the check fails on real range drift, not on pen-rounding.
# Endpoints measured against ranges as wired in player.py (Task 4.5, 4.6).
BANDS = {
    "Tight": (0.12, 0.15),
    "Loose": (0.38, 0.42),
    "Aggressive": (0.28, 0.32),
    "Passive": (0.23, 0.27),
}


def play_share(strategy):
    return {
        "Tight": TIGHT_RANGE,
        "Loose": LOOSE_RANGE,
        "Aggressive": AggressiveStrategy._play_range,
        "Passive": PassiveStrategy._play_range,
    }[strategy]


def raise_share(strategy):
    return {
        "Tight": TIGHT_PREMIUM,
        "Loose": LooseStrategy._top10,
        "Aggressive": AggressiveStrategy._raise_range,
        "Passive": PassiveStrategy._raise_range,
    }[strategy]


@pytest.fixture(scope="module", autouse=True)
def print_measured_table():
    """Print the measured table once (captured unless -s), so the stage report
    documents exact play % and raise % next to Table-1 bands."""
    print("| Strategy | play % | raise % | Table band |")
    for strategy in BANDS:
        low, high = BANDS[strategy]
        print(
            f"| {strategy} | {share(play_share(strategy)) * 100:.1f}% | "
            f"{share(raise_share(strategy)) * 100:.1f}% | "
            f"{low * 100:.0f}-{high * 100:.0f}% |"
        )


@pytest.mark.parametrize("strategy", sorted(BANDS))
def test_play_share_in_table1_band(strategy):
    low, high = BANDS[strategy]
    actual = share(play_share(strategy))
    assert low <= actual <= high, (
        f"{strategy} play share {actual:.1%} outside Table-1 band "
        f"[{low:.0%}, {high:.0%}]"
    )


def test_raise_range_is_subset_of_play_range():
    """Raise sets (premium/top-X%) must never include hands the strategy would
    fold preflop; otherwise raising is self-contradictory."""
    for strategy in BANDS:
        assert raise_share(strategy) <= play_share(strategy), (
            f"{strategy} raises hands outside its playable range"
        )