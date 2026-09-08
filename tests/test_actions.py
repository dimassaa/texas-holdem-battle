"""Action vocabulary is a stable constant set."""
from poker.actions import (
    ALLIN, BET, CALL, CHECK, FOLD, RAISE, ALL_ACTION_KINDS, Action,
)


def test_action_is_namedtuple_with_kind_and_amount():
    a = Action(BET, 10)
    assert a.kind == BET and a.amount == 10


def test_all_kinds_are_distinct():
    assert len(set(ALL_ACTION_KINDS)) == 6
    assert FOLD not in (CHECK, CALL, BET, RAISE, ALLIN)
