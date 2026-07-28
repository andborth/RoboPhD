"""Unit tests for runner_utils.parse_dollars_or_percent.

The function is a CLI-boundary converter for money knobs that may be
written relative to another knob (asta_paper_finder's --cost-per-error as
a percentage of --cost-threshold). Two properties matter beyond plain
parsing:

  - it returns dollars, always — the percentage form must not leak past
    the front end into scoring, checkpoints, or interpolated docs;
  - its output survives an equality comparison against a stored value,
    because resolve_run_immutable rejects a *disagreeing* value on
    --resume and binary float noise would otherwise manufacture a
    disagreement between two spellings of the same number.
"""
import pytest

from RoboPhD.runner_utils import parse_dollars_or_percent, resolve_run_immutable


# --- absolute form ------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("0.006", 0.006),
    ("0.02", 0.02),
    ("1", 1.0),
    ("1e-3", 0.001),
    ("  0.006  ", 0.006),
])
def test_plain_amounts_pass_through(text, expected):
    assert parse_dollars_or_percent(text, of=0.06, flag="cost-per-error") == expected


def test_absolute_form_ignores_the_reference_knob():
    """A dollar amount means the same thing regardless of what it sits
    beside — only the percent form consults `of`."""
    a = parse_dollars_or_percent("0.006", of=0.06, flag="x")
    b = parse_dollars_or_percent("0.006", of=99.0, flag="x")
    assert a == b == 0.006


# --- percent form -------------------------------------------------------------


@pytest.mark.parametrize("text,of,expected", [
    ("10%", 0.06, 0.006),
    ("10%", 0.033, 0.0033),
    ("100%", 0.06, 0.06),
    ("7.5%", 0.08, 0.006),
    ("150%", 0.02, 0.03),      # over 100% is legal: a very soft penalty
    ("  10 %  ", 0.06, 0.006),
])
def test_percentages_resolve_against_the_reference(text, of, expected):
    got = parse_dollars_or_percent(text, of=of, flag="cost-per-error")
    assert got == pytest.approx(expected)


def test_percentage_returns_dollars_not_a_fraction():
    """Guards the leak this function exists to prevent: 10% of $0.06 is
    six-tenths of a cent, not 0.10."""
    got = parse_dollars_or_percent("10%", of=0.06, flag="cost-per-error")
    assert got == 0.006
    assert got != 0.10


def test_rounding_defeats_float_noise():
    """0.10 * 0.033 is 0.0033000000000000004 in binary floating point.
    Unrounded, that would not compare equal to a stored 0.0033."""
    got = parse_dollars_or_percent("10%", of=0.033, flag="cost-per-error")
    assert got == 0.0033
    assert repr(got) == "0.0033"


def test_percent_form_agrees_with_the_equivalent_absolute_form_on_resume():
    """End-to-end on the property the rounding protects: a run started
    with the dollar spelling and resumed with the equivalent percentage
    must not trip the immutability check."""
    stored = parse_dollars_or_percent("0.0033", of=0.033, flag="cost-per-error")
    resumed = parse_dollars_or_percent("10%", of=0.033, flag="cost-per-error")
    # Would raise SystemExit if the two spellings disagreed.
    assert resolve_run_immutable(
        cli_value=resumed, stored_value=stored, default_value=None,
        flag="cost-per-error", on_resume=True,
    ) == stored


def test_small_values_are_not_flattened_by_the_rounding():
    """12 decimals is far below any plausible knob, so the noise fix must
    not clip legitimately tiny slopes."""
    got = parse_dollars_or_percent("10%", of=0.003, flag="cost-per-error")
    assert got == 0.0003


# --- rejections ---------------------------------------------------------------


@pytest.mark.parametrize("text", ["abc", "", "$0.006", "10 percent", "--"])
def test_malformed_absolute_values_exit_cleanly(text):
    with pytest.raises(SystemExit) as exc:
        parse_dollars_or_percent(text, of=0.06, flag="cost-per-error")
    assert "--cost-per-error" in str(exc.value)


@pytest.mark.parametrize("text", ["%", "ten%", "1.2.3%", " %"])
def test_malformed_percentages_exit_cleanly(text):
    with pytest.raises(SystemExit) as exc:
        parse_dollars_or_percent(text, of=0.06, flag="cost-per-error")
    assert "percentage" in str(exc.value)


def test_error_names_the_flag_it_was_called_for():
    """The message is the user's only clue about which knob it came
    from — the parser is generic and reusable across flags."""
    with pytest.raises(SystemExit) as exc:
        parse_dollars_or_percent("bogus", of=1.0, flag="some-other-knob")
    assert "--some-other-knob" in str(exc.value)
