"""Pin `_head_tail_truncate` behavior on the asta_ds1000 evaluator.

The truncation helper feeds every per-problem `error` diagnostic file
when an Inspect sample fails (evaluator.py call site). Easy-to-regress
properties: slice arithmetic, marker formatting, the +50 short-string
pass-through threshold. A flat `s[:N]` regression silently hides
upstream provider errors behind the request JSON — exactly the failure
mode that led to commit e7e2348 (`asta_ds1000: keep error head+tail
instead of flat 1000-char prefix`).

Also pins the subprocess-isolation timeout machinery in
`_evaluate_via_subprocess`: a future refactor that drops
`start_new_session=True`, the `os.killpg` path, or the bounded
post-kill drain would silently re-introduce the orphan-grandchild
hang that wedged the parent for 60+ minutes on
`asta_ds1000_20260511_162205`. AST-level checks catch the structural
regression without needing a multi-minute integration repro.

Test fixture imports evaluator once per module via sys.path
manipulation (same pattern as the other asta_ds1000 test files).
"""
import ast
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


# --- _evaluate_via_subprocess timeout-machinery regression -----------------


@pytest.fixture(scope="module")
def evaluate_via_subprocess_node():
    """Parse evaluator.py and return the FunctionDef AST for
    `_evaluate_via_subprocess`. Pure AST parse — no import — so the
    fixture is cheap and doesn't trigger evaluator.py's env-var checks.
    """
    src = (ASTA_DS1000_DIR / "evaluator.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_evaluate_via_subprocess":
            return node
    raise AssertionError("_evaluate_via_subprocess not found in evaluator.py")


def test_popen_uses_start_new_session(evaluate_via_subprocess_node):
    """The worker must be spawned with start_new_session=True so the
    timeout-path killpg signals the whole group. A refactor back to
    `subprocess.run(...)` (no session control) silently re-introduces
    the orphan-grandchild hang."""
    popen_calls = [
        c for c in ast.walk(evaluate_via_subprocess_node)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute)
        and c.func.attr == "Popen"
        and isinstance(c.func.value, ast.Name)
        and c.func.value.id == "subprocess"
    ]
    assert len(popen_calls) == 1, (
        f"Expected exactly one subprocess.Popen call in "
        f"_evaluate_via_subprocess; found {len(popen_calls)}"
    )
    kwargs = {kw.arg: kw.value for kw in popen_calls[0].keywords}
    assert "start_new_session" in kwargs, (
        "subprocess.Popen call must pass start_new_session=True so we "
        "can killpg the worker's process group on timeout"
    )
    val = kwargs["start_new_session"]
    assert isinstance(val, ast.Constant) and val.value is True, (
        f"start_new_session must be literal True; got {ast.dump(val)}"
    )


def test_killpg_used_on_timeout(evaluate_via_subprocess_node):
    """Timeout handling must call os.killpg — plain proc.kill() only
    signals the immediate child, leaving grandchildren that can keep
    stdout/stderr pipes open and wedge the post-kill drain."""
    killpg_calls = [
        c for c in ast.walk(evaluate_via_subprocess_node)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute)
        and c.func.attr == "killpg"
        and isinstance(c.func.value, ast.Name)
        and c.func.value.id == "os"
    ]
    assert killpg_calls, (
        "os.killpg call required in _evaluate_via_subprocess: timeout "
        "handling must kill the worker's process group, not just the "
        "immediate child"
    )


def test_all_communicate_calls_have_timeout(evaluate_via_subprocess_node):
    """Every proc.communicate() call must pass an explicit timeout.
    The first communicate() (subprocess_timeout) bounds the main run;
    the second (the post-kill drain) bounds how long we're willing to
    wait for pipes to EOF after killpg. An unbounded communicate()
    would re-introduce the hang we're guarding against."""
    communicate_calls = [
        c for c in ast.walk(evaluate_via_subprocess_node)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute)
        and c.func.attr == "communicate"
    ]
    assert len(communicate_calls) >= 2, (
        f"Expected at least two communicate() calls (main run + "
        f"post-kill drain); found {len(communicate_calls)}"
    )
    for call in communicate_calls:
        kw_names = {kw.arg for kw in call.keywords}
        assert "timeout" in kw_names, (
            f"communicate() at line {call.lineno} must pass "
            f"timeout=N — every drain needs a hard cap"
        )
