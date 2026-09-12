"""Post-processing: raw logs -> analysis datasets + plots (doc §7.2).

Every aggregation here is a pure function of persisted artifacts; re-running
analysis on the same manifest reproduces identical figures (determinism).
"""

import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from poker.config import Config
from poker.simulator import compute_session_stats

SEATS = ["sb", "bb", "utg", "hj", "co", "btn"]   # human-readable per full ring


def load_all_records(manifest) -> pd.DataFrame:
    """One row per session-carried hand summary (lightweight fields only)."""
    frames = []
    for s in manifest["sessions"]:
        with open(s["raw_log"]) as f:
            for line in f:
                r = json.loads(line)
                frames.append({
                    "combo": s["combo"], "seed": s["seed"],
                    "hand_id": r["hand_id"], "dealer_pos": r["dealer_pos"],
                    "net": r["net"], "stacks_after": r["stacks_after"],
                    "ruined": r["ruined"], "showdown": r["showdown"],
                })
    return pd.DataFrame(frames)


def build_summary_sessions_df(manifest) -> pd.DataFrame:
    """Per-session per-seat headline metrics (BB/100, winrate, ruin)."""
    records = []
    for s in manifest["sessions"]:
        rows = [json.loads(l) for l in open(s["raw_log"])]
        meta = json.load(open(s["meta_log"]))
        st = compute_session_stats(rows, meta)
        base = {"combo": s["combo"], "seed": s["seed"]}
        for i, seat in enumerate(SEATS):
            base[f"bb100_{i}"] = st["bb100"][str(i)]
            base[f"winrate_{i}"] = st["winrate"][str(i)]
            base[f"ruin_hand_{i}"] = st["ruined"][str(i)]
        records.append(base)
    return pd.DataFrame(records)


def build_summary_combo_df(manifest) -> pd.DataFrame:
    """Combo-level aggregate: mean + 95% CI across seeds, per §7.1."""
    sessions = build_summary_sessions_df(manifest)
    groups = sessions.groupby("combo")
    rows = []
    for combo, g in groups:
        row = {"combo": combo}
        n = len(g)
        for i in range(6):
            col = f"bb100_{i}"
            vals = g[col].dropna()
            mean = float(vals.mean())
            sd = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
            row[f"{col}_mean"] = mean
            row[f"{col}_ci"] = 2 * sd / (n ** 0.5) if n else 0.0
            row[f"first_ruin_hand_{i}_med"] = float(g[f"ruin_hand_{i}"].dropna().median())
        rows.append(row)
    return pd.DataFrame(rows)