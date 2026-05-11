"""Pin `_head_tail_truncate` behavior on the asta_ds1000 evaluator.

The truncation helper feeds every per-problem `error` diagnostic file
when an Inspect sample fails (evaluator.py call site). Easy-to-regress
properties: slice arithmetic, marker formatting, the +50 short-string
pass-through threshold. A flat `s[:N]` regression silently hides
upstream provider errors behind the request JSON — exactly the failure
mode that led to commit e7e2348 (`asta_ds1000: keep error head+tail
instead of flat 1000-char prefix`).

Test fixture imports evaluator once per module via sys.path
manipulation (same pattern as the other asta_ds1000 test files).
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ASTA_DS1000_DIR = REPO_ROOT / "examples" / "asta_ds1000"


@pytest.fixture(scope="module")
def truncate():
    """Import `_head_tail_truncate` from evaluator.py once per module.

    Evaluator imports astabench / inspect_evals which are heavy; the
    module-scope fixture amortizes that across all tests here. Same
    sys.path dance as test_asta_ds1000_main.py.
    """
    sys.path.insert(0, str(ASTA_DS1000_DIR))
    try:
        from evaluator import _head_tail_truncate  # noqa: E402
        return _head_tail_truncate
    finally:
        sys.path.remove(str(ASTA_DS1000_DIR))


# --- Pass-through (short strings) ------------------------------------------


def test_empty_string_passes_through(truncate):
    assert truncate("") == ""


def test_short_string_passes_through(truncate):
    s = "x" * 100
    assert truncate(s) == s


def test_exactly_head_chars_passes_through(truncate):
    """Length exactly HEAD: no truncation."""
    s = "x" * 200
    assert truncate(s) == s


def test_exactly_head_plus_tail_chars_passes_through(truncate):
    """Length exactly HEAD+TAIL: no truncation."""
    s = "x" * (200 + 1500)
    assert truncate(s) == s


def test_at_pass_through_threshold(truncate):
    """Length exactly HEAD+TAIL+50: the threshold is `<=` so this still
    passes through unchanged. Catches an off-by-one regression to `<`."""
    s = "x" * (200 + 1500 + 50)
    assert truncate(s) == s


# --- Truncation triggered (long strings) -----------------------------------


def test_just_above_threshold_triggers_truncation(truncate):
    """Length HEAD+TAIL+51 is the first length above the pass-through
    threshold. Output should contain the truncation marker."""
    s = "H" * 200 + "M" * 51 + "T" * 1500
    out = truncate(s)
    assert out != s
    assert out.startswith("H" * 200), "head preserved"
    assert out.endswith("T" * 1500), "tail preserved"
    assert "chars truncated" in out, "marker present"


def test_long_string_keeps_head_and_tail_intact(truncate):
    """For a clearly-long string, head and tail must be exact slices.
    Marker contains the count of middle chars dropped."""
    head_marker = "BEGINMARK" + "h" * 191    # 200 chars total
    tail_marker = "t" * 1491 + "ENDMARK01"   # 1500 chars total
    middle = "M" * 10000                      # gets truncated
    s = head_marker + middle + tail_marker
    out = truncate(s)
    assert out.startswith(head_marker), f"head not preserved: {out[:50]!r}"
    assert out.endswith(tail_marker), f"tail not preserved: {out[-50:]!r}"
    assert "10000 chars truncated" in out


def test_marker_format_is_stable(truncate):
    """The marker is `\\n... (N chars truncated) ...\\n`. Format change
    would break grep patterns / log analyzers that key on it. Pin the
    exact string for a known truncation count."""
    s = "x" * (200 + 1500 + 1000)  # 1000 middle chars get dropped
    out = truncate(s)
    assert "\n... (1000 chars truncated) ...\n" in out


def test_output_length_smaller_than_input_when_truncated(truncate):
    """The whole point of truncation: output must be strictly smaller
    than input. If the marker ever ends up larger than the savings,
    this catches it."""
    s = "x" * 10_000
    out = truncate(s)
    assert len(out) < len(s)
    # Sanity: output should be roughly head + tail + small-marker
    assert len(out) < 200 + 1500 + 50  # generous marker headroom


# --- Custom head/tail parameters --------------------------------------------


def test_custom_head_tail_parameters(truncate):
    """Helper accepts head/tail kwargs. Caller may want a different
    split for a different context."""
    s = "H" * 10 + "M" * 1000 + "T" * 20
    out = truncate(s, head=10, tail=20)
    assert out.startswith("H" * 10)
    assert out.endswith("T" * 20)
    assert "1000 chars truncated" in out


def test_custom_parameters_respect_pass_through_threshold(truncate):
    """The +50 pass-through threshold scales with the head+tail args.

    With head=5, tail=10, the threshold is 5+10+50 = 65 chars. The
    helper uses `<=` so:
      - len(s) == 65 → pass-through (exactly at threshold)
      - len(s) == 66 → truncation triggered (first length above)
    """
    # At-threshold: pass-through.
    at_threshold = "x" * 65
    assert truncate(at_threshold, head=5, tail=10) == at_threshold

    # One char above threshold: truncation triggered.
    above_threshold = "x" * 66
    out = truncate(above_threshold, head=5, tail=10)
    assert out != above_threshold
    assert "chars truncated" in out
