"""Campaign-scale invariants: schema, determinism across workers, recompute."""
import json
from pathlib import Path

from poker.experiments import run_campaign

COMBOS = ("six_tight", "six_loose", "benchmark_mix")
SEEDS = (1, 2)
NUM = 25

HAND_KEYS = {"hand_id", "rng_seed", "dealer_pos", "round_idx_start", "board",
             "stacks_before", "stacks_after", "hole", "actions", "side_pots",
             "pot_total", "net", "ruined", "seats", "showdown"}


def test_campaign_scale_quality_gate(tmp_path):
    """1) schema, 3) conservation, 4) recompute at campaign scale (3x2x25)."""
    m = run_campaign(combos=COMBOS, seeds=SEEDS, num_hands=NUM,
                     log_root=str(tmp_path / "camp"))
    total = 0
    for s in m["sessions"]:
        rows = [json.loads(line) for line in open(s["raw_log"])]
        for rec in rows:
            total += 1
            assert HAND_KEYS <= set(rec)
            for a in rec["actions"]:
                assert set(("round", "pos", "kind", "amount")) <= set(a)
            assert abs(sum(float(x) for x in rec["net"])) < 1e-9   # conservation
        meta = json.load(open(s["meta_log"]))
        from poker.simulator import compute_session_stats
        session_bb100 = json.load(open(s["stats_log"]))["bb100"]
        assert compute_session_stats(rows, meta)["bb100"] == session_bb100
    assert total == len(COMBOS) * len(SEEDS) * NUM

    # 2) cross-worker determinism: the same (combo, seed) is byte-identical
    # whether run with workers=1 or workers=2 (parallelism never mutates a
    # session's internal rng stream).
    a = run_campaign(combos=COMBOS, seeds=SEEDS, num_hands=NUM,
                     log_root=str(tmp_path / "solo"))
    b = run_campaign(combos=COMBOS, seeds=SEEDS, num_hands=NUM,
                     log_root=str(tmp_path / "duo"), workers=2)
    for combo, seed in [(c, s) for c in COMBOS for s in SEEDS]:
        rel = Path(combo) / f"seed{seed}" / f"n{NUM}"
        fa = (tmp_path / "solo" / rel / "hands.jsonl").read_bytes()
        fb = (tmp_path / "duo" / rel / "hands.jsonl").read_bytes()
        assert fa == fb