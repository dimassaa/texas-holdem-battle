"""Stage-05 gate: an end-to-end mini campaign is reproducible and self-consistent."""
import hashlib
import json
from pathlib import Path

from poker.config import Config
from poker.simulator import run_session

COMBOS = [
    ("Tight",) * 6,
    ("Loose",) * 6,
    ("Tight", "Tight", "Tight", "Aggressive", "Aggressive", "Aggressive"),
    ("Tight", "Loose", "Aggressive", "Passive", "Mathematician", "Adaptive"),
]


def test_mini_campaign_is_reproducible(tmp_path):
    for i, combo in enumerate(COMBOS):
        run_session(combo, 25, 101 + i, Config(num_hands=25),
                    str(tmp_path / f"c{i}"))
    logs = sorted((tmp_path / "c0").glob("*.json*"))
    assert logs

    # rerun c0 and byte-compare
    run_session(COMBOS[0], 25, 101, Config(num_hands=25), str(tmp_path / "c0b"))
    a = (Path(tmp_path) / "c0" / "hands.jsonl").read_text()
    b = (Path(tmp_path) / "c0b" / "hands.jsonl").read_text()
    assert a == b
    assert hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(b.encode()).hexdigest()