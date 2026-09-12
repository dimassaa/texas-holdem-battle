"""Campaign runner: combo definitions, parallel execution, manifest integrity."""
import json
from pathlib import Path

from poker.experiments import CAMPAIGN, run_combo_seeds, session_output_path

CFG_KWS = dict(num_hands=20)


def test_campaign_enumerates_all_documented_combos(tmp_path):
    assert set(CAMPAIGN) >= {
        "six_tight", "six_loose", "six_aggressive", "six_passive",
        "six_mathematician", "six_adaptive", "benchmark_mix",
        "tight3_agg3", "agg2_passive4", "tight3_loose3",
    }
    for name, combo in CAMPAIGN.items():
        assert len(combo) == 6


def test_run_combo_seeds_writes_manifest(tmp_path):
    manifest = run_combo_seeds("six_tight", seeds=(1, 2), num_hands=20, log_root=str(tmp_path))
    assert len(manifest["sessions"]) == 2
    for s in manifest["sessions"]:
        assert Path(s["raw_log"]).exists()
        assert s["combo"] == "six_tight"
    assert manifest["num_hands"] == 20
    # reproducibility across seeds is preserved per seed
    assert manifest["sessions"][0]["seed"] == 1