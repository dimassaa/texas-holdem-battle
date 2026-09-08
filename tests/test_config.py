"""Config is immutable and serializable for experiment metadata."""
import pytest

from poker.config import Config


def test_defaults_match_rules_document():
    c = Config()
    assert c.num_players == 6
    assert c.sb == 1 and c.bb == 2
    assert c.initial_stack == 200
    assert c.min_bet == c.bb
    assert c.max_aggressions == 4  # bet + 3 raises per round
    assert len(c.strategy_combo) == 6


def test_config_is_frozen():
    c = Config()
    with pytest.raises(Exception):
        c.num_players = 5


def test_config_serializes_to_plain_dict():
    d = Config().as_dict()
    assert d["bb"] == 2 and d["mc_iterations"] == 400
