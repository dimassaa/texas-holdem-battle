"""Session manager: seats the declared strategy combo, plays num_hands hands,
freezes first-passage-ruined seats, feeds adaptive brains from PUBLIC hand
summaries, and writes the raw session artifacts (task 5.3 format)."""

import time
from pathlib import Path

from poker.game import run_hand
from poker.player import AdaptiveStrategy, Player, for_name
from poker.recorder import HandRecord, JsonlHandLog
from poker.rng import Rng, subseed


def _hand_summary(player, hand_id, actions, dealer_pos, seats):
    """Build the PUBLIC per-hand summary the adaptive brain observes.

    Only actions a real opponent could have seen: who raised, who faced a
    raise and folded, who c-bet, plus aggregate aggression/calls/folds. Never
    hole cards, never showdown results (doc §1 limitation 2). Keys match
    `_aggregate_window` exactly so real and synthetic windows merge.

    `seats` is the hand's per-position rank list ([p.rank for each played
    seat]). Engine actions log POSITIONS — indices into the hand's live seat
    list — which diverge from original ranks as soon as eliminations shrink
    the table (e.g. active ranks [0, 2, 3, 5] log positions 0-3). Every action
    is therefore attributed by mapping its recorded position back to the
    hero's rank through `seats`.
    """
    seat = player.rank
    rank_of = dict(enumerate(seats))

    def is_mine(a):
        # Showdown entries carry a winner LIST in `pos`; only scalar positions
        # belong to a single seat, so guard the type before the rank lookup.
        return isinstance(a["pos"], int) and rank_of.get(a["pos"]) == seat

    aggro = ("bet", "raise", "all-in")   # all-in wagers are aggression
    preflop_raises = [a for a in actions if a["round"] == 0 and a["kind"] in aggro]
    mine = [a for a in actions if is_mine(a)]
    summary = {"seat": seat, "hand_id": hand_id}
    summary["faces_raise"] = len(preflop_raises)
    summary["folds_to_raise"] = int(any(a["kind"] == "fold" and a["round"] == 0
                                        and is_mine(a) for a in actions))
    summary["cbets"] = int(any(a["kind"] in aggro and a["round"] == 1
                               and is_mine(a) for a in actions))
    # c-bet chance exists only when hero was the preflop aggressor (raised/bet).
    summary["cbet_chances"] = int(any(a["kind"] in aggro and a["round"] == 0
                                      and is_mine(a) for a in actions))
    summary["agg_actions"] = sum(1 for a in mine if a["kind"] in aggro)
    summary["raises"] = summary["agg_actions"]
    summary["calls"] = sum(1 for a in mine if a["kind"] == "call")
    summary["folds"] = int(any(a["kind"] == "fold" and is_mine(a) for a in actions))
    summary["plays"] = int(any(a["kind"] in ("call", "raise", "bet", "check", "all-in")
                               and is_mine(a) for a in actions))
    return summary


def run_session(strategy_combo, num_hands, seed, config, log_dir,
                recorder_factory=None, initial_stacks=None):
    """Run a full session; returns the provenance/stats dict and persists logs.

    `initial_stacks`, when given, is a per-seat stack list matching
    `strategy_combo` (used by the ruin test to seat a guaranteed-bust player);
    defaults to `config.initial_stack` for every seat.

    Reproducibility (approved decision): every random consume in the play of a
    hand flows from `Rng(subseed(seed, hand))` (button draw, shuffle, and all
    per-decision equity Monte Carlo), so two runs with equal seeds produce
    byte-identical hands.jsonl. The only non-seeded values are `meta`'s wall-
    clock floats, which live in provenance, never in the hand log.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    recorder = (recorder_factory or JsonlHandLog)(log_dir / "hands.jsonl")
    started = time.time()

    stacks = initial_stacks or [config.initial_stack] * len(strategy_combo)
    active = [Player(name=f"seat{i}", strategy=name, stack=stacks[i],
                     rank=i, brain=for_name(name))
              for i, name in enumerate(strategy_combo)]
    dealer = int(subseed(seed, 9000) % len(active))
    records = []
    hand = 0
    num_ruined_total = 0
    while hand < num_hands and len(active) >= 2:
        hand_seed = subseed(seed, hand)
        played = list(active)                     # pre-ruin seating for this hand
        result = run_hand(played, dealer, Rng(hand_seed), config)
        hand_dealer = dealer                      # used by this hand's record

        ruined = _mark_ruined(active, config)     # first-passage stack < bb
        if ruined:
            num_ruined_total += len(ruined)
            active = [p for p in active if not p.ruined]   # freeze seats

        record = _to_record(hand, hand_seed, result, played, hand_dealer, ruined)
        recorder.write(record)
        records.append(record)
        _feed_adaptives(played, result, hand, hand_dealer)
        dealer = (dealer + 1) % len(active)   # button advances; modulo tolerates
        hand += 1                              # eliminations below

    finished = time.time()
    meta = {
        "combo": list(strategy_combo),
        "num_hands": num_hands,
        "num_players": len(strategy_combo),   # seats keep their original ranks
        "seed": seed,
        "config": config.as_dict(),
        "started": started,
        "finished": finished,
        "elapsed": finished - started,
    }
    stats = compute_session_stats([r.to_dict() for r in records], meta)
    recorder.write_meta(meta)
    recorder.write_stats(stats)
    recorder.close()
    return {"meta": meta, "stats": stats, "log_path": str(log_dir / "hands.jsonl"),
            "num_hands_run": hand, "num_ruined": num_ruined_total}


def _mark_ruined(players, config):
    """First-passage ruin: flag every live seat whose stack fell below BB.

    Returns the ruined ranks (already marked `p.ruined = True` so the caller
    can freeze them) or an empty list when nobody busted this hand.
    """
    ruined = [p for p in players if not p.ruined and p.stack < config.bb]
    for p in ruined:
        p.ruined = True
    return [p.rank for p in ruined]


def _feed_adaptives(players, result, hand_id, dealer_pos):
    """Feeds each Adaptive brain a PUBLIC summary of this hand (doc §1.2).

    The hand's `seats` list accompanies the summary so actions recorded by
    engine position are attributed to the brain's own rank even after the
    table has shrunk (positions no longer equal ranks post-elimination).
    """
    seats = [p.rank for p in players]
    for p in players:
        if isinstance(p.brain, AdaptiveStrategy):
            p.brain.observe(_hand_summary(p, hand_id, result.actions, dealer_pos, seats))


def _to_record(hand, hand_seed, result, played, dealer_pos, ruined):
    """Serialize one hand into the pinned HandRecord schema.

    `played` is the pre-ruin seating so the record covers exactly the seats
    that took part; `ruined` marks the seats that busted on THIS hand.
    """
    return HandRecord(
        hand_id=hand,
        rng_seed=hand_seed,
        dealer_pos=dealer_pos,
        round_idx_start=0,
        board=result.board,
        stacks_before=result.stacks_before,
        stacks_after=result.stacks_after,
        hole=[p.hole for p in played],
        actions=result.actions,
        side_pots=result.side_pots,
        pot_total=result.pot_total,
        net=[float(x) for x in result.net],
        ruined=list(ruined),
        seats=[p.rank for p in played],
        showdown=any(a["kind"] == "showdown" for a in result.actions),
    )


def _pos_of(dealer_pos, seat_pos, n):
    """Same zone mapping as `relative_position`: button=late, +1/+2 blinds,
    +3/+4 early, else middle. Computed from a raw record (no live state).

    `seat_pos` is the seat's POSITION in the hand's live seating (per the
    raw record's `seats` field), matching the engine's action/position indices
    and `relative_position`, which is defined over exactly those indices;
    `n` is the number of seats that played that hand.
    """
    dist = (seat_pos - dealer_pos) % n
    if dist == 0:
        return "late"
    if dist in (1, 2):
        return "blinds"
    if dist in (3, 4):
        return "early"
    return "middle"


def compute_session_stats(records, meta) -> dict:
    """§7.1 metrics from the raw JSONL records + meta.

    This is the SINGLE stats implementation: `run_session` returns exactly this,
    so recomputing the same metrics from the persisted log is trivially equal.
    BB/100 denominator = hands the seat was live for (before first-passage
    ruin); seats with no live hands report None, never 0.0.
    """
    n = meta["num_players"]
    bb = meta["config"]["bb"]
    seats = range(n)
    net = {s: 0.0 for s in seats}
    live_hands = {s: 0 for s in seats}
    won_hands = {s: 0 for s in seats}
    ruin_hand = {s: None for s in seats}
    last_stack = {s: meta["config"]["initial_stack"] for s in seats}
    aggression = {s: {"bet": 0, "raise": 0, "call": 0, "fold": 0, "check": 0} for s in seats}
    pos_net = {s: {p: 0.0 for p in ("early", "middle", "late", "blinds")} for s in seats}
    pos_hands = {s: {p: 0 for p in ("early", "middle", "late", "blinds")} for s in seats}

    for hand_no, rec in enumerate(records):
        # map position-in-record -> seat rank; post-elimination records are
        # shorter, so zip against the record's own seats field.
        k = len(rec["seats"])                     # live seats for THIS hand
        ruined_in_hand = set(rec.get("ruined", []))
        for pos, rank in enumerate(rec["seats"]):
            last_stack[rank] = rec["stacks_after"][pos]
            # A seat's bust hand DOES count as a live hand — it played and lost
            # it; first-passage ruin freezes the seat only from the NEXT hand.
            live_hands[rank] += 1
            net[rank] += rec["net"][pos]
            if rec["net"][pos] > 0:
                won_hands[rank] += 1
            # Zones are computed in ENGINE POSITION space (as relative_position
            # does), because `dealer_pos` in the record is a position too.
            zone = _pos_of(rec["dealer_pos"], pos, k)
            pos_net[rank][zone] += rec["net"][pos] / bb
            pos_hands[rank][zone] += 1
            for a in rec["actions"]:
                # Actions log the seat's POSITION in the hand's live list, so
                # attribute by `pos` — comparing against `rank` silently skips
                # every action once eliminations diverge them (the reason
                # records carry `seats`).
                if a.get("pos") != pos:
                    continue
                # an all-in wager is aggression consuming a stack, so fold
                # it into the bet counter rather than dropping it silent.
                key = "bet" if a["kind"] == "all-in" else a["kind"]
                if key in aggression[rank]:
                    aggression[rank][key] += 1
            if rank in ruined_in_hand:
                ruin_hand[rank] = hand_no     # first hand the seat busted on

    bb100 = {}
    winrate = {}
    for s in seats:
        bb100[str(s)] = (net[s] / bb / live_hands[s] * 100) if live_hands[s] else None
        winrate[str(s)] = (won_hands[s] / live_hands[s]) if live_hands[s] else None
    positional_ev = {
        str(s): {z: (pos_net[s][z] / pos_hands[s][z]) if pos_hands[s][z] else None
                 for z in ("early", "middle", "late", "blinds")} for s in seats}
    return {
        "bb100": bb100,
        "winrate": winrate,
        "ruined": {str(s): ruin_hand[s] for s in seats},
        "final_stacks": {str(s): last_stack[s] for s in seats},
        "positional_ev": positional_ev,
        "aggression": {str(s): aggression[s] for s in seats},
    }