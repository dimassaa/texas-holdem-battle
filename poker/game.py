"""The hand & betting engine.

Owns GameState, the betting-round loop, side-pot construction, showdown, and
the run_hand driver. All pot math lives here; strategies only emit Actions.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from poker.actions import ALLIN, BET, CALL, CHECK, FOLD, RAISE
from poker.card import new_deck, shuffle_deck
from poker.config import Config
from poker.hand_evaluator import hand_score
from poker.player import Player


@dataclass
class GameState:
    players: List[Player]
    dealer_pos: int
    config: Config
    board: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    round_idx: int = 0          # 0=preflop, 1=flop, 2=turn, 3=river
    pot: int = 0                # chips already committed before this round
    round_bets: Dict[int, int] = field(default_factory=dict)
    aggro_count: int = 0        # count of bet/raise events this round
    history: List[dict] = field(default_factory=list)
    preflop_raise_by: int = -1  # seat that last raised on round 0 (Stage-04 c-bet signal)

    def to_call(self, player_idx: int) -> int:
        """Chips the player must wager to match the current round's highest bet."""
        max_round = max(self.round_bets.values(), default=0)
        return max(0, max_round - self.round_bets.get(player_idx, 0))

    def num_opponents(self, player_idx: int) -> int:
        """Number of non-folded players excluding the given player."""
        active = [i for i, p in enumerate(self.players) if not p.folded]
        return len(active) - 1

    def can_raise(self, player_idx: int) -> bool:
        """Raises are bounded by the 4-aggression cap and the stack."""
        return self.aggro_count < self.config.max_aggressions


def starting_bets(state: GameState) -> None:
    """Post the small and big blinds into the preflop round.

    The button posts SB, the next player posts BB (blinds are on real seats;
    eliminated seats are skipped at the session layer, so `players` here are
    exactly the live ones). Bets are always capped by stack.

    The blinds live in `round_bets`, and `pot` stays at 0 until the street
    closes: `run_betting_round` moves the whole street into `pot` exactly once
    at finalization, which prevents the blinds from being counted twice.
    """
    sb_idx = state.dealer_pos
    bb_idx = (state.dealer_pos + 1) % len(state.players)
    state.round_bets = {}
    for idx, amt in ((sb_idx, state.config.sb), (bb_idx, state.config.bb)):
        paid = min(state.players[idx].stack, amt)
        state.players[idx].stack -= paid
        state.players[idx].contributed += paid
        state.round_bets[idx] = paid
    state.pot = 0


def max_raise_amount(state: GameState, player_idx: int) -> int:
    """Pot-limit maximum total wager this action (doc rule 2.2).

    MaxRaise = (prior streets' pot + all current street bets) + 2 * to_call.
    `state.pot` holds only previous streets; the current street lives in
    `round_bets`. The formula equals: call, then re-raise by the new pot.
    """
    call = state.to_call(player_idx)
    return state.pot + sum(state.round_bets.values()) + 2 * call


def _setting_first_actor(state: "GameState") -> int:
    """Index of the first actor for this street.

    Preflop: UTG (left of the BB). Postflop: the SB seat (left of the button).
    Eliminated seats are removed upstream at the session layer, so live indices
    are contiguous; blinds still sit on real seats.
    """
    if state.round_idx == 0:
        return (state.dealer_pos + 2) % len(state.players)
    return (state.dealer_pos + 1) % len(state.players)


def _round_finished(state: "GameState") -> bool:
    """Every non-folded, non-all-in player has matched the highest open wager."""
    open_bet = max(state.round_bets.values(), default=0)
    for idx, p in enumerate(state.players):
        if p.folded or p.all_in:
            continue
        if state.round_bets.get(idx, 0) < open_bet:
            return False
    return True


def run_betting_round(state: "GameState", actions_override=None) -> None:
    """Execute one betting street with the full rule set.

    Reopen rule (doc requirement): a short all-in raise does not reopen players
    who already matched the larger bet. The loop walks seats forever forward;
    a seat gets a new turn only when the open wager increased *after* its last
    turn (i.e. a full raise happened behind it) — that is exactly when the round
    is legally still open for it. When the walk returns to a seat whose
    last-acting open bet equals the current one AND everyone has matched, the
    street closes. This avoids lap counters entirely and matches real poker:
    a matched player with nothing to do is never asked to "act" again.

    `actions_override` is the Stage-02 test harness: {player_idx: (kind, amount)}
    consumed when set. Production passes None and strategies act via
    `Player.brain.act(...)` (wired in Stage 04).
    """
    n = len(state.players)
    idx = _setting_first_actor(state)
    last_full_raise = state.config.bb   # next bet/raise increment must reach this
    acted_at = {}                       # seat -> open_bet when it last acted
    safety = 0
    while True:
        safety += 1
        if safety > n * 8:
            raise RuntimeError("betting round did not converge")
        open_bet = max(state.round_bets.values(), default=0)
        # Street closed when either:
        #  (a) one player is left to collect an uncontested pot, or
        #  (b) nothing is left that could act (every seat folded or all-in), or
        #  (c) action is back at a seat that already acted under this exact
        #      open wager and everyone has matched it.
        live = [p for p in state.players if not p.folded]
        if len(live) <= 1:
            break
        if _round_finished(state) and (open_bet == acted_at.get(idx)
                                       or all(p.folded or p.all_in
                                              for p in state.players)):
            break
        p = state.players[idx]
        if p.folded or p.all_in:
            idx = (idx + 1) % n
            continue

        if actions_override is not None:
            kind, amount = actions_override[idx]
        else:
            kind, amount = p.act(state, idx)   # resolved in Stage 04

        owed = max(0, open_bet - state.round_bets.get(idx, 0))
        if kind == FOLD:
            p.folded = True
        elif kind == CHECK:
            # legal only when the player owes nothing
            assert owed == 0, "CHECK with a bet outstanding"
        elif kind in (CALL, ALLIN):
            # CALL is capped by the stack; the cap turns it into an all-in call
            paid = min(p.stack, owed if kind == CALL else p.stack)
            bet_kind = kind
        elif kind == BET:
            assert owed == 0, "BET into a live bet"
            paid = min(amount, p.stack, max_raise_amount(state, idx))
            assert paid >= state.config.min_bet, "bet below minimum"
            state.aggro_count += 1
            bet_kind = "BET"
        elif kind == RAISE:
            assert owed > 0, "RAISE with no bet to raise"
            gross = min(amount, max_raise_amount(state, idx)) - state.round_bets.get(idx, 0)
            assert gross <= p.stack, "raise above stack"
            assert gross >= last_full_raise or gross == p.stack, "raise below minimum increment"
            paid = gross
            state.aggro_count += 1
            bet_kind = "RAISE"
        else:
            raise ValueError(f"unknown action kind: {kind!r}")

        if kind in (CALL, ALLIN, BET, RAISE):
            paid = min(paid, p.stack)
            p.stack -= paid
            p.contributed += paid
            state.round_bets[idx] = state.round_bets.get(idx, 0) + paid
            if p.stack == 0:
                p.all_in = True
            if bet_kind in ("BET", "RAISE"):
                # A full raise (or an all-in that reaches it) resets the minimum
                # increment; a short all-in raise keeps it and does not reopen.
                if paid >= last_full_raise:
                    last_full_raise = paid
            state.history.append({"round": state.round_idx, "pos": idx,
                                  "kind": kind, "amount": paid})
        acted_at[idx] = open_bet if kind in (FOLD, CHECK) else max(state.round_bets.values())
        idx = (idx + 1) % n
    # Finalize: move the whole street into the pot exactly once.
    state.pot += sum(state.round_bets.values())
    state.round_bets = {}


@dataclass
class SidePot:
    """A layer of the pot plus the player indices eligible to win it."""
    amount: int
    eligible: tuple[int, ...]


def build_side_pots(players) -> list[SidePot]:
    """Layered side-pot construction from per-player total contributions.

    Level widths follow ALL contributors (folded money included); eligibility
    is restricted to players who have NOT folded — a folded hand forfeits every
    layer it stands in. When no live player covers a layer (everyone there
    folded), that width is dead money: it rolls into the live pot beneath it,
    so the amounts always sum to total contributions (conservation invariant).
    """
    contrib = [p.contributed for p in players]
    levels = sorted(set(c for c in contrib if c > 0))
    pots = []
    prev = 0
    dead = 0   # chips forfeited by folded players above every live contribution
    for level in levels:
        width = level - prev
        layer = width * len([c for c in contrib if c >= level])
        eligible = tuple(i for i, c in enumerate(contrib)
                         if c >= level and not players[i].folded)
        if eligible:
            pots.append(SidePot(amount=layer + dead, eligible=eligible))
            dead = 0
        else:
            dead += layer
        prev = level
    if dead and pots:
        pots[-1] = SidePot(amount=pots[-1].amount + dead, eligible=pots[-1].eligible)
    elif dead:
        # Everyone folded: there is no live pot, but the dead chips must still
        # be preserved so amounts sum to total contributions (conservation).
        pots = [SidePot(amount=dead, eligible=tuple())]
    return pots or [SidePot(0, tuple())]


def _winner_indices(scores, eligible) -> tuple[int, ...]:
    """Indices among `eligible` holding the highest score (ties included)."""
    best = max((scores[i] for i in eligible))
    return tuple(i for i in eligible if scores[i] == best)


def pay_out(state: GameState, scores) -> None:
    """Distribute every side pot to the best hand among its eligible players.

    `scores` maps player index -> encoded hand score (higher better). Ties
    split equally. Mutates players' stacks by the net win. Conservation is
    enforced by construction (amounts sum to total contributions).
    """
    pots = build_side_pots(state.players)
    for pot in pots:
        winners = _winner_indices(scores, pot.eligible)
        share = pot.amount / len(winners)
        for w in winners:
            state.players[w].stack += share


@dataclass
class HandResult:
    """Immutable record of one complete hand, consumed by the recorder."""
    dealer_pos: int
    rng_seed: int
    board: np.ndarray
    stacks_before: list
    stacks_after: list
    hole_cards: list
    actions: list
    side_pots: list
    pot_total: int
    net: list
    ruined: list = field(default_factory=list)   # filled in by the Stage-05 session layer


def deal_hole(state: GameState, deck, cursor=0) -> int:
    """Deal two unique hole cards to each live player from `deck` (top-down)."""
    for p in state.players:
        p.hole = deck[cursor:cursor + 2]
        cursor += 2
    return cursor


def run_hand(players, dealer_pos, rng, config, actions_override=None):
    """Play one complete hand (doc §6.2) and return its HandResult.

    Street sequence: preflop -> flop -> turn -> river, each street betting
    after its board cards are dealt. The run-out shortcut skips betting on any
    street where at most one player can act (all others folded or all-in) and
    deals the remaining board through the river. `pay_out` distributes layered
    side pots; a lone survivor collects the uncontested pot. Chips are
    conserved by construction (stacks only move via blinds/bets/payouts).

    `actions_override` maps a street index to a `run_betting_round`-style
    {player_idx: (kind, amount)} dict (Stage-02/03 test harness). A street that
    still requires betting but is absent from the map raises a KeyError — the
    harness must script every betting street explicitly.
    """
    deck = shuffle_deck(rng, new_deck())
    before = [p.stack for p in players]
    for p in players:
        p.folded = False
        p.all_in = False
        p.contributed = 0

    state = GameState(players=players, dealer_pos=dealer_pos, config=config)
    board_index = deal_hole(state, deck)
    actions_log = []

    for street in range(4):
        state.round_idx = street
        # Preflop opens with the blinds; later streets open with an empty tray.
        if street == 0:
            starting_bets(state)
        else:
            state.round_bets = {}

        non_folded = [i for i, p in enumerate(players) if not p.folded]
        if len(non_folded) <= 1:
            break                       # everyone else folded: uncontested pot
        actable = [i for i, p in enumerate(players) if not p.folded and not p.all_in]

        # Deal the street's board cards (none on the preflop).
        if street == 1:
            state.board = np.concatenate([state.board, deck[board_index:board_index + 3]])
            board_index += 3
        elif street in (2, 3):
            state.board = np.concatenate([state.board, deck[board_index:board_index + 1]])
            board_index += 1

        if len(actable) <= 1:
            continue                    # run-out shortcut: no betting possible

        street_script = actions_override.get(street) if actions_override else None
        run_betting_round(state, street_script)
        actions_log.extend(state.history)
        state.history = []

    survivors = [i for i, p in enumerate(players) if not p.folded]
    if len(survivors) == 1:
        # Uncontested pot: the lone survivor takes everything already in.
        winner = survivors[0]
        players[winner].stack += state.pot + sum(state.round_bets.values())
        actions_log.append({"round": state.round_idx, "pos": [winner],
                            "kind": "showdown", "amount": state.pot,
                            "winners": (winner,), "board": state.board.tolist()})
    elif len(survivors) == 0:
        # Degenerate scripted hand: every player folded, so nobody remains to
        # beat. Award the blinds to the big blind (last unraised seat) — the
        # same ruling the lone-survivor branch produces when the BB is the last
        # one standing (test_fold_to_big_blind pins net [-1, +1, ...]).
        winner = (state.dealer_pos + 1) % len(players)
        players[winner].stack += state.pot + sum(state.round_bets.values())
        actions_log.append({"round": state.round_idx, "pos": [winner],
                            "kind": "showdown", "amount": state.pot,
                            "winners": (winner,), "board": state.board.tolist()})
    else:
        # Showdown: top up an all-in board through the river, then score hands.
        while len(state.board) < 5:
            state.board = np.concatenate([state.board, deck[board_index:board_index + 1]])
            board_index += 1
        scores = {i: hand_score(np.concatenate([p.hole, state.board]))
                  for i, p in enumerate(players) if not p.folded}
        pay_out(state, scores)
        best = max(scores.values())
        winners = tuple(i for i, s in scores.items() if s == best)
        actions_log.append({"round": state.round_idx, "pos": list(winners),
                            "kind": "showdown", "amount": state.pot,
                            "winners": winners, "board": state.board.tolist(),
                            "scores": {i: int(s) for i, s in scores.items()}})

    return HandResult(
        dealer_pos=dealer_pos,
        rng_seed=getattr(rng, "seed", None),
        board=state.board,
        stacks_before=before,
        stacks_after=[p.stack for p in players],
        hole_cards=[p.hole.tolist() for p in players],
        actions=actions_log,
        side_pots=[{"amount": p.amount, "eligible": list(p.eligible)}
                   for p in build_side_pots(players)],
        pot_total=sum(p.contributed for p in players),
        net=[p.stack - b for p, b in zip(players, before)],
    )
