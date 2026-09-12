"""Post-processing: raw logs -> analysis datasets + plots (doc §7.2).

Every aggregation here is a pure function of persisted artifacts; re-running
analysis on the same manifest reproduces identical figures (determinism).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Headless-safe plotting backend. matplotlib pins the backend at first pyplot
# import, and CI boxes run without a display, so this MUST execute before any
# pyplot import: the Agg backend is what keeps every builder deterministic.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from poker.simulator import compute_session_stats

SEATS = ["sb", "bb", "utg", "hj", "co", "btn"]   # human-readable per full ring

# Rank-display names for heatmap axes: ranks run 2..14 (14 = Ace in
# poker.card.RANKS); 10 renders as the conventional "T" to keep labels compact.
RANK_LABELS = ["T" if r == 10 else str(r) for r in range(2, 15)]

# Pure combos are the ONLY source of per-strategy heatmap data: six_<strategy>
# seats a table full of that strategy, so every dealt hand is attributable to
# it (approved decision — mixed/benchmark combos never feed a heatmap).
STRATEGIES = ("tight", "loose", "aggressive", "passive", "mathematician", "adaptive")


def load_all_records(manifest) -> pd.DataFrame:
    """One row per session-carried hand summary (lightweight fields only).

    `hole` holds each played seat's two card ids as plain ints: the raw JSON
    round-trips per-card numpy ints, and normalizing keeps the DataFrame and
    any CSV/JSON serialization clean. `bb` records the session's big-blind
    size so the heatmap can express every cell in BB units (its unit-
    normalization contract) without a second pass over the raw logs.
    """
    frames = []
    for s in manifest["sessions"]:
        meta = json.load(open(s["meta_log"]))
        bb = float(meta["config"]["bb"])
        with open(s["raw_log"]) as f:
            for line in f:
                r = json.loads(line)
                frames.append({
                    "combo": s["combo"], "seed": s["seed"],
                    "hand_id": r["hand_id"], "dealer_pos": r["dealer_pos"],
                    "hole": [[int(a), int(b)] for a, b in r["hole"]],
                    "net": r["net"], "stacks_after": r["stacks_after"],
                    "ruined": r["ruined"], "showdown": r["showdown"],
                    "bb": bb,
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


# ---------------------------------------------------------------------------
# Visualization builders (doc §7.2).
#
# Every builder returns a (Figure, Axes) pair and only paints — persistence to
# PNG and figure cleanup happen in the single render_all_plots controller, so
# callers never need to know the savefig/close protocol. All of them run on
# the Agg backend at a fixed 150 dpi with no random draws, so re-rendering
# the same manifest is byte-for-byte deterministic.
# ---------------------------------------------------------------------------

def heatmap_of(hands_light, strategy):
    """13x13 start-hand profitability for ONE pure-combo strategy.

    Each cell holds the unit-normalized mean net BB PER DEALT hand of that
    starting-hole type. Folded hands stay in the denominator — a player has
    been "dealt" the hole even when he folds — because the comparability
    contract is mean-per-dealt-hand, not mean-per-played-hand. Because the
    value is a mean, not a raw sum, colors remain comparable across
    strategies however often each of the 169 cells happened to be dealt.

    Layout follows the poker-chart convention: diagonal = pocket pairs, upper
    triangle = suited, lower triangle = offsuit. `hands_light` must already
    hold ONLY the strategy's pure (six_<strategy>) sessions — the pure-only
    source is the approved attribution rule.
    """
    gridsum = np.zeros((13, 13), dtype=float)
    counts = np.zeros((13, 13), dtype=float)
    bb = float(hands_light["bb"].iloc[0])
    for _, row in hands_light.iterrows():
        hole = np.asarray(row["hole"], dtype=np.int64)
        net = np.asarray(row["net"], dtype=float)
        # card id -> (rank, suit) per card.py's canonical encoding
        # (id = (rank-2)*4 + suit, so rank = id//4 + 2, suit = id % 4).
        # Inline instead of card.rank_of/card.suit_of to stay vectorized.
        ranks = hole // 4 + 2
        suits = hole % 4
        lo = np.minimum(ranks[:, 0], ranks[:, 1])
        hi = np.maximum(ranks[:, 0], ranks[:, 1])
        same = ranks[:, 0] == ranks[:, 1]
        suited = suits[:, 0] == suits[:, 1]
        rows = np.where(same, lo - 2, np.where(suited, lo - 2, hi - 2))
        cols = np.where(same, lo - 2, np.where(suited, hi - 2, lo - 2))
        # np.add.at handles duplicate (row, col) targets, so a full table of
        # identical holes still tallies every deal exactly once per seat.
        np.add.at(gridsum, (rows, cols), net)
        np.add.at(counts, (rows, cols), np.ones(len(net)))
    dealt = counts > 0
    grid = np.full((13, 13), np.nan)
    grid[dealt] = gridsum[dealt] / counts[dealt] / bb

    fig, ax = plt.subplots(figsize=(9, 8))
    vmax = max(float(np.nanmax(np.abs(grid))), 1e-9) if np.isfinite(grid).any() else 1.0
    # Symmetric diverging scale so winning (positive) and losing (negative)
    # cells share a palette anchored at zero — directly readable across
    # strategies because the value is already per-dealt-hand BB.
    im = ax.imshow(grid, cmap="RdYlGn", origin="lower", aspect="equal",
                   vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label="mean BB per dealt hand")
    ax.set_xticks(range(13))
    ax.set_yticks(range(13))
    ax.set_xticklabels(RANK_LABELS)
    ax.set_yticklabels(RANK_LABELS)
    ax.set_title(f"{strategy} start-hand profitability (mean BB / dealt hand)")
    return fig, ax


def stack_boxplots(manifest, combo):
    """Per-seat final-stack boxplot across the combo's sessions.

    Final stacks come from the persisted session_stats.json (the last stack a
    seat recorded, or the starting buy-in if it never played) rather than a
    second pass over the raw hand logs — compute_session_stats already pinned
    these values, so recomputing them here would add work, not truth.
    """
    finals = [[] for _ in range(6)]
    for s in manifest["sessions"]:
        if s["combo"] != combo:
            continue
        st = json.load(open(s["stats_log"]))["final_stacks"]
        for rank in range(6):
            finals[rank].append(float(st[str(rank)]))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(finals, tick_labels=[f"{i} - {SEATS[i]}" for i in range(6)])
    ax.set_title(f"{combo} - final stacks by seat")
    ax.set_ylabel("final stack (chips)")
    ax.set_xlabel("seat")
    return fig, ax


def stack_evolution(manifest, combo):
    """Per-seat mean-stack curves over hands, averaged across the combo's
    sessions and truncated to the common hand count when sessions are ragged
    (an early-terminated session contributes only the hands it actually ran).
    """
    sessions = [s for s in manifest["sessions"] if s["combo"] == combo]
    timelines = []
    for s in sessions:
        t = []
        for line in open(s["raw_log"]):
            rec = json.loads(line)
            hand = np.full(6, np.nan)
            # Records carry `seats` so post-elimination hands (fewer entries)
            # still map each stack_after entry back to its original rank.
            for pos, rank in enumerate(rec["seats"]):
                hand[rank] = rec["stacks_after"][pos]
            t.append(hand)
        timelines.append(t)
    horizon = min(len(t) for t in timelines) if timelines else 0
    mat = np.stack([np.asarray(t, dtype=float)[:horizon] for t in timelines])
    with warnings.catch_warnings():
        # All-NaN slices can occur once every session has busted a seat; that
        # is a real gap, not an error, so silence the RuntimeWarning.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(mat, axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    for rank in range(6):
        ax.plot(range(horizon), mean[:, rank], label=f"{rank} - {SEATS[rank]}")
    ax.set_xlabel("hand")
    ax.set_ylabel("mean stack (chips)")
    ax.set_title(f"{combo} - mean stack per seat over hands")
    ax.legend(loc="best")
    return fig, ax


def ruin_curves(manifest, combo):
    """Per-seat cumulative first-ruin probability over the planned session
    length: at hand h, the fraction of the combo's sessions in which that
    seat was already ruined (first-passage, stack < BB). Flattens at 1 once
    every session has busted the seat; seats that never bust stay at 0.
    """
    sessions = [s for s in manifest["sessions"] if s["combo"] == combo]
    horizon = manifest["num_hands"]
    ruin_hands = []
    for s in sessions:
        ru = json.load(open(s["stats_log"]))["ruined"]
        ruin_hands.append({int(k): ru[str(k)] for k in ru})   # None = never
    x = range(horizon)
    fig, ax = plt.subplots(figsize=(9, 5))
    n_sessions = max(len(sessions), 1)
    for rank in range(6):
        frac = [sum(1 for rh in ruin_hands
                    if rh.get(rank) is not None and rh[rank] <= h) / n_sessions
                for h in x]
        ax.plot(x, frac, label=f"{rank} - {SEATS[rank]}")
    ax.set_xlabel("hand")
    ax.set_ylabel("cumulative ruin fraction")
    ax.set_title(f"{combo} - first-ruin curves by seat")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    return fig, ax


def bb100_compare(combo_df):
    """Headline bar chart: per combo, the mean BB/100 flattened across the six
    seats and the seeds behind them, with 95% CI (2 sigma / sqrt(seeds)) error
    bars, sorted losing-combo-first. The mean and CI are both simply flattened
    over seats from build_summary_combo_df's per-seat columns: the bar value is
    the per-seat means averaged and the error bar is the per-seat CIs averaged.
    (A full covariance-aware combination would need the per-seat session rows;
    per-seat CI flattening is the documented approximation.)
    """
    names, means, cis = [], [], []
    for _, row in combo_df.iterrows():
        m = [row[f"bb100_{i}_mean"] for i in range(6)]
        c = [row[f"bb100_{i}_ci"] for i in range(6)]
        names.append(row["combo"])
        means.append(float(np.mean([x for x in m if pd.notna(x)] or [0.0])))
        cis.append(float(np.mean([x for x in c if pd.notna(x)] or [0.0])))
    order = np.argsort(means)                       # ascending: losers first
    xs = np.arange(len(names))
    ys = np.asarray(means)[order]
    errs = np.asarray(cis)[order]
    # Taller than the other figures: the ten combo labels sit on a 45-degree
    # baseline, so the figure needs height budget for that rotated row or the
    # labels clip at the canvas edge (reported in the README review).
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.bar(xs, ys, yerr=errs, capsize=4, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8)     # zero line: above = profit
    ax.set_xticks(xs)
    ax.set_xticklabels([names[i] for i in order], rotation=45, ha="right")
    ax.set_ylabel("mean BB/100 (seats x seeds, flattened)")
    ax.set_title("Headline BB/100 by combo with 95% CI")
    fig.tight_layout()
    return fig, ax


def render_all_plots(manifest, out_dir="output/reports/plots"):
    """Build every §7.2 artifact for the manifest and persist the PNGs.

    File naming:
      heatmap_<strategy>.png       per pure combo (six_<strategy>) only
      stack_boxplot_<combo>.png    final-stack boxes,    per combo family
      stack_evolution_<combo>.png  per-hand mean stacks, per combo family
      ruin_curve_<combo>.png       cumulative ruin,      per combo family
      bb100_compare.png            headline BB/100 bar with CIs (global)

    Every figure is closed immediately after saving (no handle leaks) and all
    rendering uses the Agg backend at a fixed 150 dpi, so the output set is
    deterministic for a given manifest.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    combos = sorted({s["combo"] for s in manifest["sessions"]})

    for combo in combos:
        for stem, builder in (("stack_boxplot", stack_boxplots),
                              ("stack_evolution", stack_evolution),
                              ("ruin_curve", ruin_curves)):
            fig, _ = builder(manifest, combo)
            fig.savefig(out / f"{stem}_{combo}.png", dpi=150)
            plt.close(fig)

    records = load_all_records(manifest)
    for strategy in STRATEGIES:
        pure = f"six_{strategy}"
        if pure in combos:
            hands = records[records["combo"] == pure]
            fig, _ = heatmap_of(hands, strategy)
            fig.savefig(out / f"heatmap_{strategy}.png", dpi=150)
            plt.close(fig)

    fig, _ = bb100_compare(build_summary_combo_df(manifest))
    fig.savefig(out / "bb100_compare.png", dpi=150)
    plt.close(fig)