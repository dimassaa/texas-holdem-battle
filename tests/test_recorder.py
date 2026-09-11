"""Raw JSONL hand log: schema, round-trip, and integrity of session files."""
import json

import numpy as np

from poker.recorder import (
    HandRecord, JsonlHandLog, hand_record_from_json,
)
from poker.card import card_id as H


def sample_record():
    return HandRecord(
        hand_id=7,
        rng_seed=1234,
        dealer_pos=2,
        round_idx_start=0,
        board=np.array([H(2, 0), H(9, 1), H(3, 2)], dtype=np.int32),
        stacks_before=[200] * 6,
        stacks_after=[212, 190, 205, 180, 211, 202],
        hole=[np.array([H(14, 0), H(14, 1)])] + [np.array([H(r, 0), H(r, 1)]) for r in (13, 12, 11, 10, 9)],
        actions=[{"round": 0, "pos": 3, "kind": "raise", "amount": 14}],
        side_pots=[{"amount": 42, "eligible": [0, 1, 3]}],
        pot_total=42,
        net=[12.0, -10.0, 5.0, -20.0, 11.0, 2.0],
        ruined=[],
        seats=[0, 1, 2, 3, 4, 5],
        showdown=True,
    )


def test_record_roundtrips_through_json():
    # NOTE: the plan's literal passed `hand_record_from_json(...)` — a HandRecord
    # instance — straight into json.dumps, which raises TypeError (not JSON
    # serializable). Fix: serialize the already-_ser'ed to_dict() output instead,
    # then deserialize through from_json -> from_dict to restore numpy arrays.
    as_json = json.dumps(sample_record().to_dict())
    rec2 = hand_record_from_json(json.loads(as_json))
    assert rec2.hand_id == 7 and rec2.board.tolist() == sample_record().board.tolist()
    assert rec2.net == sample_record().net


def test_jsonl_log_writes_and_reloads(tmp_path):
    p = tmp_path / "hands.jsonl"
    log = JsonlHandLog(p)
    log.write(sample_record())
    log.write(sample_record())
    log.close()
    rows = [json.loads(line) for line in open(p)]
    assert len(rows) == 2 and rows[0]["hand_id"] == 7


def test_session_meta_and_stats_are_persisted(tmp_path):
    p = tmp_path / "session"
    with JsonlHandLog(p / "hands.jsonl") as log:
        log.write(sample_record())
        log.write_meta({"seed": 1, "combo": ["Tight"] * 6})
        log.write_stats({"bb100": {"0": 1.2}})
    assert json.load(open(p / "session_meta.json"))["seed"] == 1
    assert json.load(open(p / "session_stats.json"))["bb100"]["0"] == 1.2