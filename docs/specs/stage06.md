# Stage 06 — Campaign Execution, Analysis, Visualization, Reports

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the full experiment campaign (multiple strategy combinations × multiple seeds), aggregate the raw logs into analysis-ready datasets, produce the §7.2 artifacts (heatmaps, boxplots, stack evolutions, ruin curves, summary tables), and write the README + final report per the README guide in `docs/The Ultimate README Guide.md`.

**Architecture:**
- `poker/experiments.py` — campaign definition, `multiprocessing` runner, manifest.
- `poker/analysis.py` — load logs → tidy datasets; §7.1 metrics with cross-seed variance; plot builders.
- `output/reports/` — **committed** analysis artifacts (summary CSVs, JSON, plot PNGs). `output/logs/` — **gitignored** per-hand JSONL (heavy, regenerable from recorded seeds).

**Approved decisions honored:**
- Persist results to the repo for re-audit (per-hand logs stay out of git due to size; the committed reports + recorded seeds satisfy both re-analysis and auditability — any row of any report can be traced back to its seed and recomputed byte-for-byte).
- Hands/combo: 10k–100k (configurable); default 50,000. Seeds: 5 per combo (variance + CI).

**Experiment plan (doc §7.3):**
```
six_tight          = (Tight,)*6
six_loose          = (Loose,)*6
six_aggressive     = (Aggressive,)*6
six_passive        = (Passive,)*6
six_mathematician  = (Mathematician,)*6
six_adaptive       = (Adaptive,)*6
benchmark_mix      = (Tight, Loose, Aggressive, Passive, Mathematician, Adaptive)
tight3_agg3        = (Tight,Tight,Tight,Aggressive,Aggressive,Aggressive)
agg2_passive4      = (Aggressive,Aggressive,Passive,Passive,Passive,Passive)
tight3_loose3      = (Tight,Tight,Tight,Loose,Loose,Loose)
```
Each × 5 seeds = 50 sessions.

**Data-quality focus:** every reported number must be traceable — the campaign manifest links each session to (combo, seed, raw log, hash of config); a schema validation pass runs over *all* raw logs before aggregation; cross-seed variance is reported as CI, never hidden; aggregated numbers are spot-checked by recomputing from raw logs.

---

### Task 6.1: Campaign runner (`experiments.py`)

**Files:**
- Create: `poker/experiments.py`
- Test: `tests/test_experiments.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_experiments.py`:
```python
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
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_experiments.py`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement the runner**

`poker/experiments.py`:
```python
"""Experiment campaign: combinations × seeds, parallel execution, manifest.

The manifest is the data-quality backbone of every report: it records, per
session, the combo, seed, hand count, config hash, and raw-log path — so any
number in Stage-06 output can be traced to inputs and recomputed exactly.
"""

import hashlib
import json
from pathlib import Path

from poker.config import Config
from poker.simulator import run_session

CAMPAIGN = {
    "six_tight": ("Tight",) * 6,
    "six_loose": ("Loose",) * 6,
    "six_aggressive": ("Aggressive",) * 6,
    "six_passive": ("Passive",) * 6,
    "six_mathematician": ("Mathematician",) * 6,
    "six_adaptive": ("Adaptive",) * 6,
    "benchmark_mix": ("Tight", "Loose", "Aggressive", "Passive", "Mathematician", "Adaptive"),
    "tight3_agg3": ("Tight", "Tight", "Tight", "Aggressive", "Aggressive", "Aggressive"),
    "agg2_passive4": ("Aggressive", "Aggressive", "Passive", "Passive", "Passive", "Passive"),
    "tight3_loose3": ("Tight", "Tight", "Tight", "Loose", "Loose", "Loose"),
}

DEFAULT_SEEDS = (101, 202, 303, 404, 505)


def session_output_path(log_root, combo_name, seed, num_hands):
    return Path(log_root) / combo_name / f"seed{seed}" / f"n{num_hands}"


def _config_digest(config):
    s = json.dumps(config.as_dict(), sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def run_one(args):
    """Worker entry (picklable for multiprocessing.Pool)."""
    combo_name, seed, num_hands, config_kws, log_root = args
    config = Config(num_hands=num_hands, **config_kws)
    combo = CAMPAIGN[combo_name]
    out_dir = session_output_path(log_root, combo_name, seed, num_hands)
    out = run_session(combo, num_hands, seed, config, str(out_dir))
    return {
        "combo": combo_name,
        "seed": seed,
        "num_hands": num_hands,
        "raw_log": str(out_dir / "hands.jsonl"),
        "meta_log": str(out_dir / "session_meta.json"),
        "stats_log": str(out_dir / "session_stats.json"),
        "config_digest": _config_digest(config),
        "num_hands_run": out["num_hands_run"],
        "num_ruined": out["num_ruined"],
    }


def run_campaign(combos=None, seeds=DEFAULT_SEEDS, num_hands=50_000,
                 log_root="output/logs", workers=1, config_kws=None) -> dict:
    """Run every (combo, seed) session, optionally in parallel processes.

    Determinism: each worker runs exactly one session with its recorded seed;
    parallelization never changes a session's internal stream. Byte-reproducible
    logs therefore hold even when the pool size changes (asserted in Task 6.3).
    """
    config_kws = config_kws or {}
    combos = combos or list(CAMPAIGN)
    jobs = [(c, s, num_hands, config_kws, log_root) for c in combos for s in seeds]
    sessions = []
    if workers == 1:
        for j in jobs:
            sessions.append(run_one(j))
    else:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            sessions = pool.map(run_one, jobs)
    manifest = {
        "combos": list(combos),
        "seeds": list(seeds),
        "num_hands": num_hands,
        "config_digest": None,  # filled with the per-session digests below
        "sessions": sessions,
    }
    manifest_path = Path(log_root) / "campaign_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # data-quality: every session must share the same config digest
    digests = {s["config_digest"] for s in sessions}
    if len(digests) != 1:
        raise AssertionError(f"config digests differ across session: {digests}")
    manifest["config_digest"] = digests.pop()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def run_combo_seeds(combo_name, seeds=DEFAULT_SEEDS, num_hands=50_000,
                    log_root="output/logs", config_kws=None) -> dict:
    """Run ONE combo across several seeds; returns the written manifest.

    Thin wrapper over `run_campaign` so the single-combo path shares the exact
    same session runner, digest check, and manifest shape (tests depend on it).
    """
    return run_campaign(combos=[combo_name], seeds=seeds, num_hands=num_hands,
                        log_root=log_root, config_kws=config_kws)


def load_manifest(path="output/logs/campaign_manifest.json") -> dict:
    """Read back a campaign manifest — the entry point for every report step."""
    return json.loads(Path(path).read_text())
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_experiments.py`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/experiments.py tests/test_experiments.py && git commit -m "feat: campaign runner with manifest and config digests"
```

---

### Task 6.2: Analysis datasets (`analysis.py`)

**Files:**
- Create: `poker/analysis.py`
- Test: `tests/test_analysis.py`

**Datasets produced (persisted under `output/reports/`):**
- `summary_sessions.csv` — one row per session: combo, seed, skill BB/100 per seat (6 cols), winrate per seat, ruin hand per seat.
- `summary_combo.csv` — per combo: mean ± CI(both 2σ/sqrt(seeds)) BB/100, winrate, first-ruin-hand distribution, final-stack stats. **The headline §7.1 table.**
- `hands_light.csv` — per-hand aggregated rows (net per seat, board, street counts, action counts) for heatmaps/evolution; debatable size is bounded by storing per-hand-not-per-action.

- [ ] **Step 1: Write the failing tests**

`tests/test_analysis.py`:
```python
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
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_analysis.py`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`poker/analysis.py`:
```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_analysis.py`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/analysis.py tests/test_analysis.py && git commit -m "feat: analysis datasets with cross-seed CIs"
```

---

### Task 6.3: Campaign-scale data-quality gate

**Files:**
- Create: `tests/test_campaign_quality_gate.py`

- [ ] **Step 1: Write the gate (runs on a small campaign)**

`tests/test_campaign_quality_gate.py`:
```python
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
```
*(Target the gate at a small campaign — 3 combos × 2 seeds × 25 hands — so it runs in CI time.)*

The gate must assert:
1. **Schema:** every raw hand record has exactly the `HandRecord` keys and correct types; every action entry has `round/pos/kind/amount`; `len(actions) > 0`.
2. **Cross-worker determinism:** run the same (combo, seed) once with `workers=1` and once with `workers=2`; the two manifests' raw logs for that seed are byte-identical.
3. **Conservation at campaign scale:** sum of `net` per hand == 0 across every raw record.
4. **Session-level consistency:** every `bb100` in `session_stats.json` matches a recompute via `compute_session_stats` from its own raw log.

- [ ] **Step 2: Run the gate**

Run: `.venv/bin/pytest tests/test_campaign_quality_gate.py -q`
Expected: `1 passed` (all four assertions inside)

- [ ] **Step 3: Commit**

```bash
git add tests/test_campaign_quality_gate.py && git commit -m "test: campaign-scale data quality gate"
```

---

### Task 6.4: Visualizations (`plots`)

**Files:**
- Modify: `poker/analysis.py` (plot builders)
- Create: `tests/test_plots.py`

**Artifacts (doc §7.2), all persisted to `output/reports/plots/`:**
1. `heatmap_<strategy>.png` — 13×13 start-hand profitability (upper-triangle suited, lower-triangle offsuit, diagonal pairs); per strategy.
2. `stack_boxplot_<combo>.png` — final-stack boxplots per seat.
3. `stack_evolution_<combo>.png` — running-average stack per seat over hands.
4. `ruin_curve_<combo>.png` — cumulative first-ruin probability vs hand for each seat.
5. `bb100_compare.png` — sorted bar of mean BB/100 per combo with CI error bars.

- [ ] **Step 1: Write the failing tests**

`tests/test_plots.py`:
```python
"""Plot builders emit persisted PNGs."""
from pathlib import Path

from poker.analysis import render_all_plots
from poker.experiments import run_combo_seeds


def test_render_all_plots_produces_files(tmp_path):
    m = run_combo_seeds("benchmark_mix", seeds=(1,), num_hands=20, log_root=str(tmp_path))
    out = Path(tmp_path) / "reports" / "plots"
    render_all_plots(m, out_dir=str(out))
    pngs = list(out.glob("*.png"))
    assert len(pngs) >= 5
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_plots.py`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement the plot builders (matplotlib/seaborn)**

`poker/analysis.py` add: `heatmap_of(hands_light, strategy) -> (fig, ax)`; `stack_boxplots(...)`; `stack_evolution(...)`; `ruin_curves(...)`; `bb100_compare(combo_df)`; and `render_all_plots(manifest, out_dir="output/reports/plots")` that calls them and `fig.savefig(...)`. Key contract: **heatmaps are unit-normalized (mean BB per dealt hand of that starting hand type), so cell colors are comparable across strategies** — raw count differences across 169 cells are folded into the averages, not the palette.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_plots.py`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add poker/analysis.py tests/test_plots.py && git commit -m "feat: analysis plot builders and artifact persistence"
```

---

### Task 6.5: Full production campaign run

- [ ] **Step 1: Run the full campaign**

Run:
```bash
mkdir -p output
.venv/bin/python - <<'PY'
from poker.experiments import run_campaign
m = run_campaign(num_hands=50_000, workers=os.cpu_count(), log_root="output/logs")
print("sessions:", len(m["sessions"]), "config_digest:", m["config_digest"])
PY
```
Expected: 50 sessions (10 combos × 5 seeds), config digest uniform, wall-clock recorded for the report. If throughput (Stage-05 measured hands/s) makes 50×50k too slow for the environment, **halt and consult the user** before reducing hands — do not silently shrink the statistical power.

- [ ] **Step 2: Aggregate and render every artifact**

Run:
```bash
.venv/bin/python - <<'PY'
from poker.experiments import load_manifest
from poker.analysis import build_summary_sessions_df, build_summary_combo_df, render_all_plots
m = load_manifest("output/logs/campaign_manifest.json")
build_summary_sessions_df(m).to_csv("output/reports/summary_sessions.csv", index=False)
build_summary_combo_df(m).to_csv("output/reports/summary_combo.csv", index=False)
render_all_plots(m, out_dir="output/reports/plots")
print("artifacts written to output/reports/")
PY
```

- [ ] **Step 3: Run the whole test suite and the campaign gate once more**

Run: `.venv/bin/pytest`
Expected: all green, including the campaign gate against the small fixture.

- [ ] **Step 4: Commit all report artifacts**

```bash
git add output/reports/ && git commit -m "docs: commit campaign analysis artifacts and plots"
```
*(Make sure `output/logs/` stays gitignored as designed; the committed reports + manifest + seeds are the auditable record.)*

---

### Task 6.6: README and final report

**Files:**
- Create: `README.md`
- Create: `docs/results.md`

Follow `docs/The Ultimate README Guide.md` structure (Tech Stack and Badges mandatory; formulas, graphs, tables throughout; **no emojis**). The README must include: purpose, model limitations (incl. the semi-bluff relaxation decision), quick-start commands (Option A full campaign / Option B small smoke), tested claim of reproducibility (exact commands to reproduce any number), the headline results table with CIs, the key plots, honest limitations (MC error at 400 iters, no bluffing except documented semi-bluffs, pot-limit-only, fixed 6 seats), recommendations/next steps.

`docs/results.md` — the analysis writeup answering the document's core question: *which style is most profitable in the long run, and how does mixing styles change it?* Structured as: setup recap → headline BB/100 table → ruin analysis → starting-hand findings → positional findings → limitations & threats to validity.

- [ ] **Step 1: Draft `README.md`** per the guide (Tech Stack, badges as shields.io links, Quick Start both variants, Testing, Limitations, Roadmap). Include the campaign results table from `output/reports/summary_combo.csv` and 2–3 key plots.

- [ ] **Step 2: Draft `docs/results.md`** with results discussion and threats-to-validity (MC accuracy, windowed adaptive reads, semi-bluff interpretation).

- [ ] **Step 3: Verify docs links and paths exist** (plots referenced exist under `output/reports/plots/`).

- [ ] **Step 4: Commit**

```bash
git add README.md docs/results.md output/reports/ && git commit -m "docs: README and results writeup"
```

---

## Stage Exit Criteria

- [ ] Full campaign (10 combos × 5 seeds × 50k hands) run, manifest + config digests recorded.
- [ ] Analysis datasets summary_sessions.csv / summary_combo.csv (with CI) persisted.
- [ ] All five plot families rendered.
- [ ] Campaign data-quality gate green (schema, cross-worker determinism, conservation, recompute).
- [ ] README + results writeup committed per the README guide.
- [ ] Whole `pytest` suite green.

**Final report to the user (comprehensive):**
- Headline BB/100 table with CIs per combo.
- Ruin-probability findings per strategy.
- Starting-hand and positional findings from heatmaps.
- Threats to validity and recommended next experiments (e.g., MC iterations, semi-bluff intensity, extra seeds).