"""Analysis datasets: aggregation, recompute-check, and CI formatting."""
import json
from pathlib import Path

import numpy as np

from poker.analysis import (
    build_summary_sessions_df, build_summary_combo_df, load_all_records,
)
from poker.experiments import run_combo_seeds

CFG = dict(num_hands=15)


def test_summary_sessions_recomputed_from_raw(tmp_path):
    manifest = run_combo_seeds("six_tight", seeds=(1, 2), num_hands=15, log_root=str(tmp_path))
    df = build_summary_sessions_df(manifest)
    assert len(df) == 2
    # spot-check: BB/100 for seat 0 recomputed independently from raw log
    s0 = manifest["sessions"][0]
    rows = [json.loads(l) for l in open(s0["raw_log"])]
    meta = json.load(open(s0["meta_log"]))
    from poker.simulator import compute_session_stats
    st = compute_session_stats(rows, meta)
    assert np.isclose(float(df.iloc[0]["bb100_0"]), st["bb100"]["0"])


def test_summary_combo_has_ci(tmp_path):
    manifest = run_combo_seeds("six_loose", seeds=(1, 2, 3), num_hands=15, log_root=str(tmp_path))
    combo = build_summary_combo_df(manifest)
    assert "bb100_0_mean" in combo.columns and "bb100_0_ci" in combo.columns
    assert (combo["bb100_0_ci"] > 0).all()