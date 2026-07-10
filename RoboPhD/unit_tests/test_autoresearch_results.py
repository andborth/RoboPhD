"""Tests for autoresearch result extraction from the agent-authored
_experiment_log.jsonl — untrusted free-form JSON."""

from RoboPhD.engines.autoresearch import _best_kept_val_score


def test_sentinel_string_among_floats_is_skipped():
    """Regression: sudoku_20260709_215531 crashed at the finish line
    with TypeError because one kept entry carried the agent-invented
    sentinel val_score='not_captured_tail_error' and max() compared
    str with float — discarding a completed 2-hour run's results."""
    log = [
        {"kept": True, "val_score": 0.026436},
        {"kept": True, "val_score": 0.758705},
        {"kept": False, "val_score": 0.754686},
        {"kept": True, "val_score": "not_captured_tail_error"},
        {"kept": True, "val_score": 0.848928},
    ]
    assert _best_kept_val_score(log) == 0.848928


def test_numeric_string_is_accepted():
    log = [{"kept": True, "val_score": "0.9"}, {"kept": True, "val_score": 0.5}]
    assert _best_kept_val_score(log) == 0.9


def test_bool_is_not_a_score():
    """float(True) == 1.0 would register a perfect score from a
    truthy flag; booleans are skipped despite subclassing int."""
    log = [{"kept": True, "val_score": True}, {"kept": True, "val_score": 0.5}]
    assert _best_kept_val_score(log) == 0.5


def test_null_and_missing_are_skipped_quietly():
    log = [
        {"kept": True, "val_score": None},
        {"kept": True},
        {"kept": True, "val_score": 0.3},
    ]
    assert _best_kept_val_score(log) == 0.3


def test_unkept_entries_are_ignored():
    log = [{"kept": False, "val_score": 0.99}, {"kept": True, "val_score": 0.4}]
    assert _best_kept_val_score(log) == 0.4


def test_empty_or_all_invalid_yields_zero():
    assert _best_kept_val_score([]) == 0.0
    assert _best_kept_val_score([{"kept": True, "val_score": "n/a"}]) == 0.0
