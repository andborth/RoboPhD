"""Unit tests for runner_utils' cost-knob validators.

These live in the framework suite because the validators do, and they do
because the blame logic below is subtle enough that two copies would drift.
Both AstaBench examples route through them; a third adopting the knob gets
the same messages for free rather than copying fifty lines a third time.

What makes the logic worth centralising is that the RIGHT flag to name
depends on how the slope reached zero:

  - a dollar amount is the user's own number, so say so plainly;
  - a percentage — including the DEFAULT one, which nobody typed — is a
    consequence of the threshold it multiplied, so naming
    --cost-per-error there sends the reader to a flag they never passed.

Per-example wiring (that each main.py actually calls these, with its own
constants) is tested in that example's own suite.
"""
import pytest

from RoboPhD.runner_utils import validate_cost_slope, validate_cost_threshold


def _fmt(x: float) -> str:
    """Stand-in for the examples' _fmt_cost, which they pass in."""
    return f"${x:.2f}" if x == round(x, 2) else f"${x:.4f}".rstrip("0")


# --- threshold ----------------------------------------------------------------


def test_negative_threshold_is_rejected_by_name():
    """Otherwise it reaches the slope derivation and draws a message about a
    free zone having no width, which is not that input's problem."""
    with pytest.raises(SystemExit, match="--cost-threshold"):
        validate_cost_threshold(-1.0, _fmt)


@pytest.mark.parametrize("threshold", [0.0, 0.003, 0.06, 1.0])
def test_non_negative_thresholds_pass(threshold):
    """Zero is legal — it means 'no free zone, penalize from the first cent'."""
    validate_cost_threshold(threshold, _fmt)


def test_threshold_error_states_that_zero_is_allowed():
    """A user who wanted a zero free zone should not read this and conclude
    the whole idea is rejected."""
    with pytest.raises(SystemExit) as exc:
        validate_cost_threshold(-0.01, _fmt)
    assert "0 means no free zone at all" in str(exc.value)


# --- slope: which flag gets blamed --------------------------------------------


_TASK = dict(fraction=0.10, suggested_slope=0.006, fmt=_fmt)


def test_zero_threshold_with_the_default_slope_blames_the_threshold():
    """The case the validators exist for. Once the default slope is a
    fraction of the threshold, a zero threshold produces a zero slope the
    user never typed — so the message must name the threshold."""
    with pytest.raises(SystemExit) as exc:
        validate_cost_slope(0.0, 0.0, None, **_TASK)
    msg = str(exc.value)
    assert "--cost-threshold" in msg, (
        "the zero threshold is the cause; naming only --cost-per-error sends "
        "the reader to a flag they did not pass"
    )
    assert "the default (10%)" in msg, "say where the untyped slope came from"
    assert "--cost-per-error 0.006" in msg, "no actionable way out"


def test_zero_threshold_with_an_explicit_percentage_blames_the_threshold():
    with pytest.raises(SystemExit) as exc:
        validate_cost_slope(0.0, 0.0, "10%", **_TASK)
    msg = str(exc.value)
    assert "--cost-threshold" in msg
    assert "'10%'" in msg, "quote what the user actually passed"
    assert "the default" not in msg, "they did type it; do not call it a default"


@pytest.mark.parametrize("resolved,spec", [
    (0.0, "0"),        # dollars, typed as zero
    (0.0, "0%"),       # a percentage the user chose to be zero
    (-0.5, "-0.5"),    # negative dollars
])
def test_a_typed_non_positive_slope_is_not_blamed_on_the_threshold(resolved, spec):
    """At a healthy threshold the user asked for this; the threshold is
    innocent and the message must not implicate it."""
    with pytest.raises(SystemExit) as exc:
        validate_cost_slope(resolved, 0.06, spec, **_TASK)
    msg = str(exc.value)
    assert "--cost-per-error must be > 0" in msg
    assert "no free zone" not in msg, (
        f"{spec!r} at a healthy threshold wrongly blamed the threshold"
    )


def test_a_positive_slope_passes():
    validate_cost_slope(0.006, 0.06, None, **_TASK)
    validate_cost_slope(0.02, 0.06, "0.02", **_TASK)


def test_a_zero_threshold_stays_reachable_with_an_explicit_dollar_slope():
    """The escape hatch the rejection message points at.

    A zero free zone is a legitimate configuration ("penalize from the
    first cent"), so the validator must only reject the case where the
    slope was *derived* from that zero — not the case where the user
    supplied one. Otherwise the advice it prints would be a dead end.
    """
    validate_cost_slope(0.006, 0.0, "0.006", **_TASK)


# --- the per-task parameters actually reach the message -----------------------


def test_the_fraction_shown_is_the_calling_task_s():
    """A hardcoded 10% would be wrong for any task that tunes it."""
    with pytest.raises(SystemExit) as exc:
        validate_cost_slope(0.0, 0.0, None, fraction=0.25,
                            suggested_slope=0.01, fmt=_fmt)
    assert "the default (25%)" in str(exc.value)


def test_the_suggested_slope_shown_is_the_calling_task_s():
    """Each example derives this from its own default threshold, so the
    advice cannot drift away from that example's constants."""
    for slope, expected in ((0.006, "--cost-per-error 0.006"),
                            (0.005, "--cost-per-error 0.005"),
                            (0.0003, "--cost-per-error 0.0003")):
        with pytest.raises(SystemExit) as exc:
            validate_cost_slope(0.0, 0.0, None, fraction=0.10,
                                suggested_slope=slope, fmt=_fmt)
        assert expected in str(exc.value)


def test_fmt_defaults_to_str_so_the_helpers_work_without_one():
    """Callers that have no money formatter should still get a message."""
    with pytest.raises(SystemExit, match="--cost-threshold"):
        validate_cost_threshold(-1.0)
