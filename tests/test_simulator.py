"""Session simulator: ruin, seeds, adaptive feeding, conservation at session level."""
import json
import os
from pathlib import Path

from poker.config import Config
from poker.simulator import run_session

CFG = Config(num_hands=30)


def test_session_runs_and_persists(tmp_path):
    out = run_session(
        ("Tight", "Loose", "Aggressive", "Passive", "Mathematician", "Adaptive"),
        num_hands=30, seed=7, config=CFG, log_dir=str(tmp_path),
    )
    assert out["num_hands_run"] == 30
    log = Path(tmp_path) / "hands.jsonl"
    assert log.exists() and log.stat().st_size > 0
    meta = json.load(open(Path(tmp_path) / "session_meta.json"))
    assert meta["seed"] == 7 and meta["combo"] == list(CFG.strategy_combo)


def test_ruin_freezes_seat_and_session_continues(tmp_path):
    # Deterministic bust: seat 5 starts at stack 0, so it can neither post a
    # blind nor win anything — it ends hand 0 at 0 (< BB) and is frozen.
    combo = ("Aggressive", "Aggressive", "Aggressive", "Passive", "Passive", "Passive")
    stacks = [200] * 5 + [0]
    out = run_session(combo, num_hands=100, seed=3, config=Config(),
                      log_dir=str(tmp_path), initial_stacks=stacks)
    assert out["stats"]["ruined"]["5"] == 0                   # busted on hand 0
    assert out["num_hands_run"] == 100                        # session continued
    assert out["num_ruined"] >= 1                             # seat 5 busted for sure
    # the frozen seat's final stack never went negative anywhere in the log
    rows = [json.loads(l) for l in open(Path(tmp_path) / "hands.jsonl")]
    assert all(min(r["stacks_after"]) >= 0 for r in rows)
    # seat 5 played its bust hand, then vanishes from every later record
    assert 5 in rows[0]["seats"]
    assert all(5 not in r["seats"] for r in rows[1:])


def test_two_identical_seeds_give_identical_logs(tmp_path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a = run_session(("Tight",) * 6, 20, 42, Config(), str(a_dir))
    b = run_session(("Tight",) * 6, 20, 42, Config(), str(b_dir))
    fa = open(a_dir / "hands.jsonl").read()
    fb = open(b_dir / "hands.jsonl").read()
    assert fa == fb          # byte-for-byte reproducibility (approved decision)
    assert a["stats"]["bb100"] == b["stats"]["bb100"]