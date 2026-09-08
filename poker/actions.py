"""Action vocabulary.

A stable, documented set of string constants so strategy code and logging agree
on spellings and no typo can corrupt a hand's recorded history.
"""

from collections import namedtuple

FOLD = "fold"
CHECK = "check"
CALL = "call"
BET = "bet"
RAISE = "raise"
ALLIN = "all-in"

ALL_ACTION_KINDS = (FOLD, CHECK, CALL, BET, RAISE, ALLIN)

# amount: size of the chip wager for BET/RAISE/CALL/ALLIN, else 0.
Action = namedtuple("Action", "kind amount")
