"""Aggregation of a recorded session into §7.1 metrics."""
import json
import math
from pathlib import Path

from poker.config import Config
from poker.simulator import compute_session_stats, run_session

CFG = Config(num_hands=60)


def test_computed_metrics_are_well_formed(tmp_path):
    out = run_session(("Tight", "Loose", "Aggressive", "Passive", "Mathematician", "Adaptive"),
                      num_hands=60, seed=5, config=CFG, log_dir=str(tmp_path))
    st = out["stats"]
    assert set(st) >= {"bb100", "winrate", "ruined", "final_stacks", "positional_ev", "aggression"}
    for bb in st["bb100"].values():
        assert bb is None or math.isfinite(bb)   # None = seat never played a live hand
    assert len(st["bb100"]) == CFG.num_players


def test_recompute_from_log_matches_session(tmp_path):
    out = run_session(("Tight",) * 6, 40, 9, Config(num_hands=40), str(tmp_path))
    rows = [json.loads(l) for l in open(Path(tmp_path) / "hands.jsonl")]
    meta = json.load(open(Path(tmp_path) / "session_meta.json"))
    recomputed = compute_session_stats(rows, meta)
    assert recomputed["bb100"] == out["stats"]["bb100"]