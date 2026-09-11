"""Raw session persistence.

Single source of truth for all downstream analysis. Files:
  <log_dir>/hands.jsonl        one JSON object per hand (action-by-action audit)
  <log_dir>/session_meta.json  config + seeds + timing (provenance)
  <log_dir>/session_stats.json aggregated per-session numbers

Writing is append-only JSONL: a partially-written session is still parseable
up to the last complete line, so a crashed run loses only the trailing hand.

Append-only JSONL is chosen over a single snapshot because Stage-06 (and any
future re-analysis) must be able to replay the exact sequence of hands without
recomputing anything from memory, a crash must never corrupt an earlier hand,
and the per-line framing keeps the file trivially tail-able while a session is
still running.

One log_dir = one session: hands.jsonl APPENDS across runs on the same path
while the meta/stats files overwrite, so re-running on an existing directory
would pair both runs' hands with only the last run's provenance. Callers must
use a fresh directory per session (the simulator and the reproducibility tests
all do).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def _ser(o):
    # Convert numpy scalars/arrays to plain Python so the whole session tree
    # is json.dumps-safe; anything else passes through untouched.
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, tuple):
        return list(o)
    return o


@dataclass
class HandRecord:
    hand_id: int
    rng_seed: int
    dealer_pos: int
    round_idx_start: int
    board: np.ndarray
    stacks_before: list
    stacks_after: list
    hole: list
    actions: list
    side_pots: list
    pot_total: int
    net: list
    ruined: list
    seats: list        # seat rank per entry position, so post-elimination
                       # records (with fewer entries) still map to seats
    showdown: bool

    def to_dict(self):
        # Explicit array/ndarray conversion (not plain asdict) because asdict
        # leaves numpy types in place and they are not JSON serializable.
        d = asdict(self)
        d["board"] = _ser(self.board)
        d["hole"] = [_ser(h) for h in self.hole]
        return d

    @classmethod
    def from_dict(cls, d):
        # Reconstruct numpy arrays on the way in so analysis code can rely on
        # the in-memory schema (int32 board/hole) regardless of I/O path.
        d = dict(d)
        d["board"] = np.array(d.pop("board"), dtype=np.int32)
        d["hole"] = [np.array(h, dtype=np.int32) for h in d.pop("hole")]
        return cls(**d)


def hand_record_from_json(d):
    """Deserialize a raw dict into a numpy-bearing HandRecord."""
    return HandRecord.from_dict(d)


class JsonlHandLog:
    def __init__(self, path):
        self.path = Path(path)
        # Create the log directory up front so a fresh session can open its
        # log/report files without the caller staging the tree.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Append mode (not write) preserves any earlier hands from a crashed or
        # restarted session; a stale file never silently truncates history.
        self._fh = open(self.path, "a", encoding="utf-8")
        self.count = 0

    def write(self, record: HandRecord) -> None:
        # One JSON object per physical line: a torn write at worst truncates
        # this line, leaving every previously written line parseable.
        # Flush per line so even a hard kill loses at most ONE hand (the
        # buffered-tail docstring guarantee); the logger is not hot enough to
        # make this cost meaningful.
        self._fh.write(json.dumps(record.to_dict()) + "\n")
        self._fh.flush()
        self.count += 1

    def write_meta(self, meta: dict) -> None:
        # Session provenance lives next to the log, named from the log file's
        # stem: hands.jsonl -> session_meta.json (same directory).
        meta_path = self.path.with_name("session_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(_ser(meta), f, indent=2)

    def write_stats(self, stats: dict) -> None:
        stats_path = self.path.with_name("session_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(_ser(stats), f, indent=2)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()