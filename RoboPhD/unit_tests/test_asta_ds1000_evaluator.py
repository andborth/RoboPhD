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


# --- fallback_candidate machinery regression -------------------------------


@pytest.fixture(scope="module")
def evaluator_tree():
    """Parse evaluator.py once per module for AST tests that walk the
    whole file (not just one function). Pure parse — no import."""
    return ast.parse((ASTA_DS1000_DIR / "evaluator.py").read_text())


def _find_method(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"{class_name}.{method_name} not found")


def test_constructor_accepts_fallback_candidate(evaluator_tree):
    """`Ds1000Evaluator.__init__` must accept `fallback_candidate` with
    a default of `None`. Default-None matters: training instantiates
    without specifying it and must retain today's behavior (error signals
    visible to the evolution loop)."""
    init = _find_method(evaluator_tree, "Ds1000Evaluator", "__init__")
    arg_names = {a.arg for a in init.args.args} | {a.arg for a in init.args.kwonlyargs}
    assert "fallback_candidate" in arg_names, (
        "Ds1000Evaluator.__init__ must accept a `fallback_candidate` kwarg"
    )
    # Default must be None — find the index of fallback_candidate in args
    # and check the corresponding default. Python aligns defaults to the
    # tail of args.
    pos_args = init.args.args
    defaults = init.args.defaults
    pos_arg_names = [a.arg for a in pos_args]
    if "fallback_candidate" in pos_arg_names:
        idx = pos_arg_names.index("fallback_candidate")
        default_idx = idx - (len(pos_args) - len(defaults))
        assert default_idx >= 0, "fallback_candidate must have a default value"
        default_node = defaults[default_idx]
        assert isinstance(default_node, ast.Constant) and default_node.value is None, (
            f"fallback_candidate default must be literal None; "
            f"got {ast.dump(default_node)}"
        )


def test_with_overrides_propagates_fallback_candidate(evaluator_tree):
    """`with_overrides` builds a `base` dict that's spread into a fresh
    Ds1000Evaluator. Every new constructor field must be added to this
    dict or it silently resets to default on every test-evaluator
    derivation. Catches the same trap that already exists for every
    other field."""
    method = _find_method(evaluator_tree, "Ds1000Evaluator", "with_overrides")
    # Find the `base = {...}` dict literal in the method body.
    base_dict_keys = []
    for node in ast.walk(method):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant):
                    base_dict_keys.append(k.value)
    assert "fallback_candidate" in base_dict_keys, (
        f"with_overrides()'s base dict must include 'fallback_candidate' "
        f"to propagate it across .with_overrides() calls; "
        f"found keys: {sorted(set(base_dict_keys))}"
    )


def test_call_checks_fallback_candidate_on_error(evaluator_tree):
    """`__call__` must re-invoke the dispatch path when a primary error
    is observed AND fallback_candidate is set. Looks for an `if`
    statement whose test references both `self.fallback_candidate` and
    the string literal `"error"` (the diagnostics dict key). Regression
    catch: a refactor that drops the conditional re-invocation."""
    call_method = _find_method(evaluator_tree, "Ds1000Evaluator", "__call__")
    found_conditional = False
    for node in ast.walk(call_method):
        if not isinstance(node, ast.If):
            continue
        # Walk the condition's AST directly — robust against quote-style
        # differences from ast.unparse (single vs double quotes).
        names_fallback = False
        names_error = False
        for child in ast.walk(node.test):
            if (isinstance(child, ast.Attribute)
                    and child.attr == "fallback_candidate"
                    and isinstance(child.value, ast.Name)
                    and child.value.id == "self"):
                names_fallback = True
            if isinstance(child, ast.Constant) and child.value == "error":
                names_error = True
        if names_fallback and names_error:
            found_conditional = True
            break
    assert found_conditional, (
        "__call__ must contain `if self.fallback_candidate is not None "
        "and \"error\" in diagnostics:` (or equivalent) to gate the "
        "fallback re-invocation"
    )


# --- fallback runtime behavior ----------------------------------------------


@pytest.fixture
def evaluator_with_fallback(monkeypatch):
    """Construct a Ds1000Evaluator with fallback_candidate set, mocking
    the env-var pre-flight and skipping the Docker check. Stubs dummy
    provider keys; no real API calls happen because the test
    monkey-patches `_evaluate_via_subprocess` directly."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "sk-test")
    sys.path.insert(0, str(ASTA_DS1000_DIR))
    try:
        from evaluator import Ds1000Evaluator
        ev = Ds1000Evaluator(
            skip_docker_check=True,
            apply_cost_penalty=False,
            fallback_candidate={"agent.py": "# pretend seed"},
        )
        yield ev
    finally:
        sys.path.remove(str(ASTA_DS1000_DIR))


def test_fallback_fires_on_primary_error(evaluator_with_fallback, monkeypatch):
    """End-to-end behavioral test: when the primary returns an error
    diagnostic, __call__ must re-invoke the dispatch with the fallback
    candidate, return the fallback's score, and mark fallback_used.
    AST tests pin the structural shape; this catches a refactor that
    makes the gate more clever but silently skips the fallback in real
    conditions."""
    from evaluator import Ds1000Evaluator
    calls = []

    def fake_eval(self, candidate, example):
        calls.append(candidate)
        if len(calls) == 1:
            return 0.0, {"error": "synthetic primary failure"}
        return 1.0, {"score": 1.0, "library": "numpy"}

    monkeypatch.setattr(Ds1000Evaluator, "_evaluate_via_subprocess", fake_eval)

    primary = {"agent.py": "# pretend primary"}
    example = {"id": "test_0", "input": "x"}
    score, diag = evaluator_with_fallback(primary, example)

    assert score == 1.0
    assert diag["fallback_used"] is True
    assert diag["primary_error"] == "synthetic primary failure"
    assert len(calls) == 2
    assert calls[0] is primary
    assert calls[1] is evaluator_with_fallback.fallback_candidate


def test_fallback_skipped_when_primary_succeeds(evaluator_with_fallback, monkeypatch):
    """When the primary succeeds (no `"error"` key), the fallback must
    NOT fire. Single call to _evaluate_via_subprocess; no fallback_used
    marker; primary diagnostics pass through unchanged."""
    from evaluator import Ds1000Evaluator
    calls = []

    def fake_eval(self, candidate, example):
        calls.append(candidate)
        return 1.0, {"score": 1.0, "library": "numpy", "agent_cost_usd": 0.05}

    monkeypatch.setattr(Ds1000Evaluator, "_evaluate_via_subprocess", fake_eval)

    score, diag = evaluator_with_fallback({"agent.py": "primary"}, {"id": "test_0"})

    assert score == 1.0
    assert "fallback_used" not in diag
    assert "primary_error" not in diag
    assert diag["agent_cost_usd"] == 0.05
    assert len(calls) == 1


def test_fallback_sums_primary_cost(evaluator_with_fallback, monkeypatch):
    """When the primary somehow reports cost before erroring (future-
    proof case — today's error paths return no cost field, but if a
    later commit adds one, the fallback layer should sum primary +
    fallback so the aggregate isn't under-reported)."""
    from evaluator import Ds1000Evaluator
    calls = []

    def fake_eval(self, candidate, example):
        calls.append(candidate)
        if len(calls) == 1:
            return 0.0, {
                "error": "synthetic, but cost was tracked",
                "agent_cost_usd": 0.03,
                "cost_by_model_usd": {"primary_model": 0.03},
            }
        return 1.0, {
            "score": 1.0,
            "agent_cost_usd": 0.02,
            "cost_by_model_usd": {"fallback_model": 0.02},
        }

    monkeypatch.setattr(Ds1000Evaluator, "_evaluate_via_subprocess", fake_eval)

    score, diag = evaluator_with_fallback({"agent.py": "primary"}, {"id": "test_0"})

    assert score == 1.0
    assert diag["fallback_used"] is True
    # Cost is summed across primary + fallback so the aggregate isn't lost.
    assert diag["agent_cost_usd"] == pytest.approx(0.05)
    assert diag["primary_agent_cost_usd"] == pytest.approx(0.03)
    assert diag["cost_by_model_usd"] == {
        "primary_model": pytest.approx(0.03),
        "fallback_model": pytest.approx(0.02),
    }


def test_fallback_also_errors_passes_through(evaluator_with_fallback, monkeypatch):
    """If the fallback also errors, its result is returned as-is. No
    third retry, no recursion. Primary error is still preserved under
    primary_error for the audit trail."""
    from evaluator import Ds1000Evaluator
    calls = []

    def fake_eval(self, candidate, example):
        calls.append(candidate)
        if len(calls) == 1:
            return 0.0, {"error": "primary boom"}
        return 0.0, {"error": "fallback boom too"}

    monkeypatch.setattr(Ds1000Evaluator, "_evaluate_via_subprocess", fake_eval)

    score, diag = evaluator_with_fallback({"agent.py": "primary"}, {"id": "test_0"})

    assert score == 0.0
    assert diag["error"] == "fallback boom too"
    assert diag["primary_error"] == "primary boom"
    assert diag["fallback_used"] is True
    assert len(calls) == 2  # exactly two, no third retry


# --- eval_wall_clock_seconds: every post-timing return must carry it -------
# Timeouts otherwise surface as score 0 / $0 / no stdout — an invisible
# cliff the evolution loop has historically misdiagnosed as a reasoning
# regression. Pin that EVERY return after the timing starts records the
# wall-clock, especially the timeout branch (the highest-value case).


def test_evaluate_via_subprocess_records_wall_clock_on_every_path(
    evaluate_via_subprocess_node,
):
    """After `_t0 = time.monotonic()` (timing start), every `return`
    in `_evaluate_via_subprocess` must surface `eval_wall_clock_seconds`
    — either as a key in a returned dict literal, or, for the
    `return score, diagnostics` tuple, via a
    `diagnostics["eval_wall_clock_seconds"] = ...` assignment in the
    function body. Returns *before* timing starts (the malformed-input
    guard) are exempt — there is no eval to time there."""
    fn = evaluate_via_subprocess_node

    # Locate the timing-start line: `_t0 = time.monotonic()`.
    t0_line = None
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_t0" for t in node.targets)
        ):
            t0_line = node.lineno
            break
    assert t0_line is not None, (
        "No `_t0 = time.monotonic()` timing anchor in "
        "_evaluate_via_subprocess — wall-clock recording was removed."
    )

    # Does the function assign diagnostics["eval_wall_clock_seconds"]?
    has_subscript_assign = any(
        isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Subscript)
            and isinstance(t.value, ast.Name)
            and isinstance(t.slice, ast.Constant)
            and t.slice.value == "eval_wall_clock_seconds"
            for t in n.targets
        )
        for n in ast.walk(fn)
    )

    def _dict_has_key(d: ast.Dict) -> bool:
        return any(
            isinstance(k, ast.Constant) and k.value == "eval_wall_clock_seconds"
            for k in d.keys
        )

    # Defensive: returns inside any nested helper def are not
    # _evaluate_via_subprocess return paths, so exclude them. (The
    # wall-clock computation is currently the module-level
    # `_elapsed_seconds` helper, so there's no in-function closure
    # today — but keep this so a future closure refactor can't produce
    # a false positive.)
    nested_returns = {
        id(r)
        for inner in ast.walk(fn)
        if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef))
        and inner is not fn
        for r in ast.walk(inner)
        if isinstance(r, ast.Return)
    }

    offenders = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or node.lineno <= t0_line:
            continue
        if id(node) in nested_returns:
            continue
        v = node.value
        # Returns here are all `return <score>, <payload>` tuples.
        payload = v.elts[1] if isinstance(v, ast.Tuple) and len(v.elts) == 2 else v
        if isinstance(payload, ast.Dict):
            if not _dict_has_key(payload):
                offenders.append((node.lineno, ast.unparse(node)))
        elif isinstance(payload, ast.Name):
            # e.g. `return score, diagnostics` — covered iff the function
            # sets diagnostics["eval_wall_clock_seconds"] somewhere.
            if not has_subscript_assign:
                offenders.append((node.lineno, ast.unparse(node)))
        else:
            offenders.append((node.lineno, ast.unparse(node)))

    assert not offenders, (
        "These post-timing returns in _evaluate_via_subprocess do NOT "
        "record `eval_wall_clock_seconds` — a latency failure on these "
        "paths would be an invisible score-0 cliff:\n  "
        + "\n  ".join(f"line {ln}: {src}" for ln, src in offenders)
    )


def test_evaluate_non_subprocess_path_records_wall_clock():
    """Parity pin for the non-isolated `evaluate()` path (used when
    subprocess_isolation=False, e.g. some eval-only flows). It's a
    lower-traffic path than _evaluate_via_subprocess, but the
    test-rigor asymmetry is itself a regression risk: a future
    refactor of evaluate() could silently drop the field with no
    fence. Pin both the crash-path dict and the success-path
    `diagnostics[...] = ...` assignment, anchored to evaluate()'s own
    `_t0 = time.monotonic()`."""
    src = (ASTA_DS1000_DIR / "evaluator.py").read_text()
    tree = ast.parse(src)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
            fn = node
            break
    assert fn is not None, "evaluate() not found in evaluator.py"

    has_timing_anchor = any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_t0" for t in n.targets)
        for n in ast.walk(fn)
    )
    assert has_timing_anchor, (
        "evaluate() lost its `_t0 = time.monotonic()` timing anchor — "
        "wall-clock recording on the non-subprocess path is gone."
    )

    # Crash-path return dict must carry the key.
    crash_dict_has_key = any(
        isinstance(n, ast.Return)
        and isinstance(n.value, ast.Tuple)
        and len(n.value.elts) == 2
        and isinstance(n.value.elts[1], ast.Dict)
        and any(
            isinstance(k, ast.Constant) and k.value == "eval_wall_clock_seconds"
            for k in n.value.elts[1].keys
        )
        for n in ast.walk(fn)
    )
    # Success path sets diagnostics["eval_wall_clock_seconds"] = ...
    success_assign_has_key = any(
        isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Subscript)
            and isinstance(t.value, ast.Name)
            and isinstance(t.slice, ast.Constant)
            and t.slice.value == "eval_wall_clock_seconds"
            for t in n.targets
        )
        for n in ast.walk(fn)
    )
    assert crash_dict_has_key and success_assign_has_key, (
        "evaluate() must record `eval_wall_clock_seconds` on BOTH its "
        f"crash return (have={crash_dict_has_key}) and its success path "
        f"(have={success_assign_has_key}). Asymmetric coverage vs "
        "_evaluate_via_subprocess is a silent-drop risk."
    )
