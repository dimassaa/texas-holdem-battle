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