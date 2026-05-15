"""Pin asta_ds1000 main.py invariants that the type system can't enforce.

Currently covers three bug classes:

  - Test-pipeline eval calls (`eval_candidate` / `eval_run`) must pass
    `config=` explicitly. Without it, they silently inherit
    `RoboPhDEvalConfig()` defaults — which include a 300 s eval_timeout
    that historically didn't match training's 600 s and let test
    problems silently fail under a stricter limit. Fixed in 37a7a7b
    by consolidating a single `test_eval_config = RoboPhDEvalConfig(
    eval_timeout=EVAL_TIMEOUT)` near `test_evaluator` construction; this
    test guards against re-introduction.

  - On `--resume`, main.py must auto-detect imports of gated stronger-
    tier handles in the resumed run's agent pool and set
    ASTA_DS1000_ALLOW_STRONGER_MODELS=1 even when
    --allow-stronger-models was omitted. Without it, eval workers
    ImportError on every sample, yielding a uniform 0.000 test score
    (silent — no exception bubbles up). The gated names live in
    model_registry as `GATED_HANDLE_NAMES` (the single source of
    truth); main.py's `_resume_needs_stronger_flag` consumes that
    constant via AST `ImportFrom` matching on `agents/*/agent.py`.

  - --new-agent-test-rounds must reach engine_overrides AND drive the
    framing in objective.md. The wording in objective.md paragraph 3
    swaps to a Round-2-aware variant when test rounds >= 1, giving the
    agent extra incentive to avoid overfitting to the iteration's
    visible batch. If main.py reverts to a hardcoded
    `"new_agent_test_rounds": 0` or objective.md drops the
    `${TEST_ROUNDS_FRAMING}` placeholder, the framing silently drifts
    from the runtime — agent gets a weaker prompt than the runtime
    actually warrants.
"""
import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = REPO_ROOT / "examples" / "asta_ds1000" / "main.py"
ASTA_DS1000_DIR = REPO_ROOT / "examples" / "asta_ds1000"

EVAL_FUNCS = {"eval_candidate", "eval_run"}


@pytest.fixture(scope="module")
def resume_helper():
    """Import main.py's `_resume_needs_stronger_flag` once per module.

    sys.path manipulation rather than a package install because
    examples/asta_ds1000/ isn't an installable package. Importing
    main.py runs argparse setup but not main() — safe at import time.
    """
    sys.path.insert(0, str(ASTA_DS1000_DIR))
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import main as asta_main  # noqa: E402
        return asta_main._resume_needs_stronger_flag
    finally:
        sys.path.remove(str(ASTA_DS1000_DIR))
        sys.path.remove(str(REPO_ROOT))


@pytest.fixture(scope="module")
def framing_helper():
    """Import main.py's `_test_rounds_framing` once per module.

    Same sys.path dance as `resume_helper`. Module-cached, so the actual
    `import main` only re-runs once per test session even with multiple
    fixtures.
    """
    sys.path.insert(0, str(ASTA_DS1000_DIR))
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import main as asta_main  # noqa: E402
        return asta_main._test_rounds_framing
    finally:
        sys.path.remove(str(ASTA_DS1000_DIR))
        sys.path.remove(str(REPO_ROOT))


@pytest.fixture(scope="module")
def stronger_rows_helper():
    """Import main.py's `_build_stronger_rows` once per module."""
    sys.path.insert(0, str(ASTA_DS1000_DIR))
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import main as asta_main  # noqa: E402
        return asta_main._build_stronger_rows
    finally:
        sys.path.remove(str(ASTA_DS1000_DIR))
        sys.path.remove(str(REPO_ROOT))


def _make_fake_resume(tmp_path: Path, agent_src: str) -> Path:
    """Build a minimal resume_dir with one agent.py at the conventional
    path. Returns the resume_dir."""
    fake_run = tmp_path / "fake_run"
    agent_dir = fake_run / "agents" / "fake_agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "agent.py").write_text(agent_src)
    return fake_run


def _calls_to(funcs: set[str], src: str) -> list[ast.Call]:
    """All ast.Call nodes calling a name in `funcs` (top-level Name only;
    we don't follow Attribute access since main.py imports the eval
    helpers as bare names)."""
    tree = ast.parse(src)
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in funcs:
                found.append(node)
    return found


def test_test_eval_calls_pass_config_explicitly():
    """Every eval_candidate / eval_run call in main.py must pass `config=`.

    Bug class: silent inheritance of RoboPhDEvalConfig() defaults causes
    the test pipeline's eval_timeout to drift from training's
    EVAL_TIMEOUT. Catching the missing kwarg at the call site is enough
    — *what* is passed is a separate concern handled by the source-of-
    truth `test_eval_config` variable in main.py.
    """
    src = MAIN_PY.read_text()
    calls = _calls_to(EVAL_FUNCS, src)
    assert calls, (
        "expected eval_candidate / eval_run call sites in main.py, found "
        "none — has main.py been refactored?"
    )

    missing: list[str] = []
    for call in calls:
        kwarg_names = {kw.arg for kw in call.keywords}
        if "config" not in kwarg_names:
            missing.append(f"line {call.lineno}: {call.func.id}(...)")

    assert not missing, (
        "These eval_candidate / eval_run call sites in main.py don't "
        "pass `config=` explicitly, so they silently inherit "
        "RoboPhDEvalConfig() defaults (300 s eval_timeout — different "
        "from training's 600 s). Pass `config=test_eval_config` (the "
        "consolidated test-side config defined near test_evaluator) at "
        "each site:\n  " + "\n  ".join(missing)
    )


def test_test_eval_config_consolidated_to_single_construction():
    """The test-side RoboPhDEvalConfig should be constructed exactly once.

    Three eval call sites currently reuse one shared `test_eval_config`.
    If a future edit re-inlines `RoboPhDEvalConfig(...)` at a call site,
    the count goes to 2+ and the consolidation has been violated —
    future test-side knobs (max_workers, test_repeats, ...) would need
    to be added in multiple places.
    """
    src = MAIN_PY.read_text()
    # AST-based: count direct constructor calls. More robust than a
    # text-search since string occurrences in comments/docstrings don't
    # match `ast.Call`.
    tree = ast.parse(src)
    constructions = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "RoboPhDEvalConfig"
    ]
    assert len(constructions) == 1, (
        f"Expected exactly 1 RoboPhDEvalConfig(...) construction in "
        f"main.py (the consolidated `test_eval_config`), found "
        f"{len(constructions)} on lines "
        f"{[c.lineno for c in constructions]}. Did someone re-add an "
        f"inline construction at a test eval call site?"
    )


# --- Resume-time gated-handle auto-detection --------------------------------
# Direct unit tests of `_resume_needs_stronger_flag`. The helper is a
# pure function (Path + Iterable[str] → bool) with no side effects, so
# we can exercise it in-process without env-var pollution and without
# coupling to main.py's import order.


def test_resume_helper_true_when_agent_imports_gated_handle(resume_helper, tmp_path):
    """Agent imports a gated handle → True."""
    fake = _make_fake_resume(
        tmp_path,
        "from model_registry import CLAUDE_OPUS_4_7, CLAUDE_SONNET_4_6\n",
    )
    assert resume_helper(fake, ("CLAUDE_OPUS_4_7", "GPT_5_5")) is True


def test_resume_helper_false_when_only_cheap_tier_imported(resume_helper, tmp_path):
    """No gated import → False."""
    fake = _make_fake_resume(
        tmp_path,
        "from model_registry import GPT_5_4_MINI, CLAUDE_SONNET_4_6\n",
    )
    assert resume_helper(fake, ("CLAUDE_OPUS_4_7", "GPT_5_5")) is False


def test_resume_helper_ignores_comment_mentions(resume_helper, tmp_path):
    """Comment-only mention must not trigger detection — the AST scan is
    the substring-scan upgrade. This is the test that fails under the
    pre-refactor implementation, where `# unlike CLAUDE_OPUS_4_7, ...`
    in a docstring would have set the env var unnecessarily."""
    fake = _make_fake_resume(
        tmp_path,
        "# unlike CLAUDE_OPUS_4_7, this seed only uses the cheap tier\n"
        '"""docstring mentions GPT_5_5 in passing."""\n'
        "from model_registry import GPT_5_4_MINI\n",
    )
    assert resume_helper(fake, ("CLAUDE_OPUS_4_7", "GPT_5_5")) is False


def test_resume_helper_handles_missing_agents_dir(resume_helper, tmp_path):
    """Resume dir without agents/ subdir → False, no exception. Defensive
    for partial / malformed resume dirs."""
    fake_run = tmp_path / "no_agents_dir"
    fake_run.mkdir()
    assert resume_helper(fake_run, ("CLAUDE_OPUS_4_7",)) is False


def test_resume_helper_skips_unparseable_agent(resume_helper, tmp_path):
    """A syntactically broken agent.py shouldn't blow up the helper —
    skip it and continue. The next agent in the pool may still trigger."""
    fake_run = tmp_path / "mixed_pool"
    broken_dir = fake_run / "agents" / "broken_agent"
    broken_dir.mkdir(parents=True)
    (broken_dir / "agent.py").write_text("def : invalid syntax\n")
    good_dir = fake_run / "agents" / "good_agent"
    good_dir.mkdir(parents=True)
    (good_dir / "agent.py").write_text(
        "from model_registry import CLAUDE_OPUS_4_7\n"
    )
    assert resume_helper(fake_run, ("CLAUDE_OPUS_4_7",)) is True


def test_main_imports_gated_names_from_model_registry():
    """main.py must consume `GATED_HANDLE_NAMES` from `model_registry`
    rather than hardcoding its own list. This is the wiring check that
    keeps the gated set as a single source of truth: if a fourth gated
    handle is added to model_registry, main.py picks it up
    automatically; if a future commit re-inlines a hardcoded tuple in
    main.py instead of importing the constant, this test fails.
    """
    tree = ast.parse(MAIN_PY.read_text())
    found = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "model_registry"
            and any(a.name == "GATED_HANDLE_NAMES" for a in node.names)
        ):
            found = True
            break
    assert found, (
        "main.py doesn't import `GATED_HANDLE_NAMES` from model_registry. "
        "The resume-time auto-detect must consume the registry's "
        "single-source-of-truth constant — re-inlining a hardcoded "
        "list creates drift between main.py and model_registry that "
        "silently re-introduces the uniform-0.000 bug if the gated set "
        "ever changes."
    )


# --- --new-agent-test-rounds plumbing ---------------------------------------
# Two checks: (1) objective.md keeps the `${TEST_ROUNDS_FRAMING}`
# placeholder so main.py's interpolation can swap the wording, and
# (2) main.py plumbs `args.new_agent_test_rounds` into engine_overrides
# rather than reverting to a hardcoded literal. Both regressions would
# silently weaken the anti-overfit prompt the agent sees in Round-2
# mode while leaving the runtime semantically correct, so neither shows
# up in functional tests.


def test_objective_md_uses_test_rounds_framing_placeholder():
    """objective.md must contain `${TEST_ROUNDS_FRAMING}` so main.py's
    interpolation can swap the wording based on
    --new-agent-test-rounds. Without the placeholder, the framing in
    the rendered prompt drifts silently from the runtime test-rounds
    value — the agent always sees the rounds==0 wording even when the
    engine actually runs Round 2."""
    obj = (REPO_ROOT / "examples" / "asta_ds1000" / "objective.md").read_text()
    assert "${TEST_ROUNDS_FRAMING}" in obj, (
        "objective.md no longer contains `${TEST_ROUNDS_FRAMING}`. "
        "Re-add the placeholder in paragraph 3 (replacing the "
        "hardcoded sentence about future iterations) so main.py can "
        "swap framings based on --new-agent-test-rounds."
    )


def test_main_plumbs_test_rounds_arg_into_engine_overrides():
    """When the user provides `--new-agent-test-rounds N`, the value
    must reach engine_overrides as `args.new_agent_test_rounds`.

    Post-fix pattern: the CLI value is packed in an `if`-guarded
    assignment so the (now-None) default doesn't silently clobber on
    resume. A separate elif/else branch may pack DS-1000's task-
    specific default (literal `0`) for the initial-run case; that's
    a different invariant tested by
    test_main_does_not_clobber_test_rounds_on_resume.

    This test verifies the user-explicit branch: at least one
    assignment to engine_overrides["new_agent_test_rounds"] must
    have RHS `args.new_agent_test_rounds`. Catches regressions
    where the CLI flag becomes inert (RHS is only a hardcoded
    literal)."""
    tree = ast.parse(MAIN_PY.read_text())
    found: list[tuple[int, ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "engine_overrides"
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "new_agent_test_rounds"
                ):
                    found.append((node.lineno, node.value))
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if (
                    isinstance(k, ast.Constant)
                    and k.value == "new_agent_test_rounds"
                ):
                    found.append((node.lineno, v))

    assert found, (
        "Couldn't find any assignment in main.py setting "
        "engine_overrides[\"new_agent_test_rounds\"] — has the "
        "construction been refactored away?"
    )
    # At least one assignment must source from args.new_agent_test_rounds.
    # (Others may exist for the task-default branch; that's fine.)
    has_args_source = any(
        isinstance(v, ast.Attribute)
        and isinstance(v.value, ast.Name)
        and v.value.id == "args"
        and v.attr == "new_agent_test_rounds"
        for _, v in found
    )
    assert has_args_source, (
        "engine_overrides[\"new_agent_test_rounds\"] must include at "
        "least one assignment sourcing from `args.new_agent_test_rounds` "
        "so the --new-agent-test-rounds CLI flag actually changes "
        "runtime behavior. Found assignments:\n  "
        + "\n  ".join(f"line {lineno}: `{ast.unparse(v)}`" for lineno, v in found)
    )


def test_main_does_not_clobber_test_rounds_on_resume():
    """Regression for the resume-clobber bug + DS-1000 task-default
    preservation.

    Two invariants:
      1. The argparse flag has default=None so an unset CLI value
         arrives as None (distinguishable from "user explicitly
         passed 0").
      2. The engine_overrides assignment is guarded by an `if args.X
         is not None` block so unset values don't propagate.
      3. The else/elif fallback that packs DS-1000's task default
         (0, which differs from RoboPhD's framework default of 1)
         must check `is_resume` / `args.resume` — otherwise the
         initial-run default would clobber the original run's
         setting on resume.

    Without all three: either (a) resume silently overrides the
    user's prior --new-agent-test-rounds choice with the CLI
    default, or (b) initial runs silently use RoboPhD's default 1
    instead of DS-1000's intended 0."""
    tree = ast.parse(MAIN_PY.read_text())

    # (1) argparse default must be None.
    argparse_default = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--new-agent-test-rounds"):
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                argparse_default = kw.value
                break
    assert argparse_default is not None, (
        "Couldn't find p.add_argument('--new-agent-test-rounds', ...) — "
        "has the flag been removed?"
    )
    assert (
        isinstance(argparse_default, ast.Constant)
        and argparse_default.value is None
    ), (
        f"--new-agent-test-rounds must default to None (not {ast.unparse(argparse_default)!r}) "
        "so the engine_overrides packing can distinguish 'user set this' "
        "from 'framework default'. A non-None default re-introduces the "
        "resume-clobber bug."
    )

    # (2) Find the if/elif/else chain that assigns
    # engine_overrides["new_agent_test_rounds"]. The if-test must
    # reference args.new_agent_test_rounds, and any elif branch must
    # reference args.resume / is_resume (so it doesn't fire on resume).
    guarded_if_test = None  # the If node that has the assignment in its body
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not _if_test_references(node.test, "args", "new_agent_test_rounds"):
            continue
        # And the body must assign engine_overrides["new_agent_test_rounds"]
        if any(_is_engine_overrides_test_rounds_assign(stmt) for stmt in node.body):
            guarded_if_test = node
            break
    assert guarded_if_test is not None, (
        "engine_overrides['new_agent_test_rounds'] must be set inside an "
        "`if args.new_agent_test_rounds is not None` block. Without the "
        "guard, the CLI default propagates into engine_overrides and "
        "silently overrides the original run's value on --resume "
        "(api.py:327 reapplies engine_overrides as a delta on resume)."
    )

    # (3) If the if-block has an else/elif that ALSO assigns
    # engine_overrides["new_agent_test_rounds"] (DS-1000's task
    # default path), that branch must be guarded against the resume
    # case. Otherwise the task default fires on resume too and
    # clobbers the original run.
    elif_assigns_default = False
    elif_guards_resume = False
    current = guarded_if_test
    while current.orelse:
        # An `elif X:` in source parses as orelse=[If(test=X, ...)]
        if (len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If)):
            elif_node = current.orelse[0]
            if any(_is_engine_overrides_test_rounds_assign(s) for s in elif_node.body):
                elif_assigns_default = True
                # Check the elif test references resume in some form
                src = ast.unparse(elif_node.test)
                if "is_resume" in src or "args.resume" in src or "resume" in src:
                    elif_guards_resume = True
            current = elif_node
        else:
            # `else:` block (orelse is a list of statements, not a single If)
            for stmt in current.orelse:
                if _is_engine_overrides_test_rounds_assign(stmt):
                    elif_assigns_default = True
                    # A bare else with no guard means it always fires when
                    # the if-test is false — including on resume. That's
                    # the bug.
                    elif_guards_resume = False
            break

    if elif_assigns_default:
        assert elif_guards_resume, (
            "engine_overrides['new_agent_test_rounds'] is assigned in an "
            "elif/else branch (presumably packing DS-1000's task default), "
            "but that branch isn't guarded against the --resume case. On "
            "resume the task default will clobber the original run's setting. "
            "Guard the elif on `is_resume` / `args.resume`."
        )


def _if_test_references(node: ast.expr, root_name: str, attr_name: str) -> bool:
    """Walk an AST expression looking for an attribute access like
    `args.new_agent_test_rounds`. Helper for the test above."""
    for n in ast.walk(node):
        if (
            isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == root_name
            and n.attr == attr_name
        ):
            return True
    return False


def _is_engine_overrides_test_rounds_assign(stmt: ast.stmt) -> bool:
    """True if `stmt` is `engine_overrides["new_agent_test_rounds"] = ...`."""
    if not isinstance(stmt, ast.Assign):
        return False
    for target in stmt.targets:
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "engine_overrides"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "new_agent_test_rounds"
        ):
            return True
    return False


# --- _test_rounds_framing branch coverage -----------------------------------
# Pure-function tests on the framing helper. We assert by SEMANTIC
# CONTENT (which concepts each branch must / must not mention) rather
# than exact strings, so the wording can be tuned without breaking
# tests — what's load-bearing is the agent-prompt-level meaning, not
# the surface phrasing.


def test_framing_rounds_zero_no_in_iteration_revision_promise(framing_helper):
    """rounds=0: the agent gets ONE pass per iteration. The framing
    must not promise an in-iteration revision opportunity — that would
    be a false promise that distorts behavior (the agent might leave
    obvious bugs unfixed expecting a "Round 2" that won't happen). The
    long-arc 'future iterations' framing is shared with rounds>=1 and
    should still appear."""
    text = framing_helper(0).lower()
    assert "future iterations" in text, (
        f"rounds=0 framing must keep the long-arc framing about future "
        f"iterations; got: {text!r}"
    )
    forbidden = (
        "refine", "revise", "different batch", "another batch",
        "second pass", "second round", "round 2",
    )
    leaked = [w for w in forbidden if w in text]
    assert not leaked, (
        f"rounds=0 framing must not promise an in-iteration revision "
        f"loop, but mentions {leaked!r} in: {text!r}"
    )


def test_framing_rounds_one_promises_in_iteration_revision(framing_helper):
    """rounds>=1: the anti-overfit incentive depends on telling the
    agent BOTH that it'll see results on a different batch within the
    iteration AND that it'll get to refine before the agent is tested
    in future iterations. Drop either half and the prompt loses bite —
    just the future-iterations framing without the in-iteration
    revision promise reduces to the rounds=0 wording."""
    text = framing_helper(1).lower()
    assert "different batch" in text or "another batch" in text, (
        f"rounds>=1 framing must promise a view on a different batch "
        f"of examples within the iteration; got: {text!r}"
    )
    assert any(w in text for w in ("refine", "revise", "improve")), (
        f"rounds>=1 framing must promise a refinement opportunity; "
        f"got: {text!r}"
    )
    assert "future iterations" in text, (
        f"rounds>=1 framing must keep the long-arc framing about "
        f"future iterations; got: {text!r}"
    )


def test_framing_higher_values_use_same_branch(framing_helper):
    """rounds=2, 3, ... share the rounds>=1 wording — the runtime
    semantics differ (more refinement passes) but the prompt-level
    framing is binary. A regression that only branches on rounds==1
    would break this."""
    one = framing_helper(1)
    assert framing_helper(2) == one
    assert framing_helper(5) == one


# --- 5-column table shape + temperature-free seed ---------------------------
# Two invariants from the "reasoning_effort + max_tokens table" change:
# (1) the stronger-tier rows in main.py must keep the 5-column shape that
#     matches the cheap-tier table in background.md (regression to the old
#     3-column shape would render a broken markdown table); and
# (2) the seed agent's only .generate() call must not pass `temperature`
#     (the seed is the anchoring pattern evolution copies; passing
#     temperature there leaks the footgun into Opus/GPT-5.5 agents).

SEED_AGENT_PY = REPO_ROOT / "examples" / "asta_ds1000" / "seeds" / "baseline" / "agent.py"


def test_stronger_rows_off_mode_is_empty(stronger_rows_helper):
    """With --allow-stronger-models off, the helper returns empty string
    so the ${STRONGER_MODELS_TABLE_ROWS}\\n placeholder collapses
    cleanly (see the trailing-newline absorption trick in
    _interpolate()). A regression that returned anything non-empty
    here would re-introduce stronger-tier handle names into off-mode
    background.md."""
    assert stronger_rows_helper(False) == ""


def test_stronger_rows_on_mode_uses_five_column_shape(stronger_rows_helper):
    """With --allow-stronger-models on, every row must match the
    cheap-tier 5-column shape (Handle | Input | Output | Default
    reasoning_effort | Available overrides). Tests the helper output
    directly — survives any string-construction refactor (f-strings,
    .join(), list-of-rows, ...) since we're parsing rendered markdown
    rather than scanning source. A regression to the pre-5-column
    shape would break the rendered table when interpolated into
    background.md."""
    rendered = stronger_rows_helper(True)
    row_lines = [
        line for line in rendered.split("\n")
        if line.startswith("| `") and line.endswith(" |")
    ]
    assert row_lines, (
        f"Helper returned non-empty output but no parseable rows:\n"
        f"{rendered!r}"
    )

    # 5 columns ⇒ 6 pipe separators per row
    bad: list[str] = []
    for line in row_lines:
        if line.count("|") != 6:
            bad.append(f"{line.count('|')} pipes (expected 6): {line!r}")
    assert not bad, (
        "stronger_rows helper returned rows with the wrong number of "
        "columns:\n  " + "\n  ".join(bad)
    )


def test_stronger_rows_includes_all_three_handle_names(stronger_rows_helper):
    """All three gated handles must appear in the on-mode output. Pins
    against accidental row removal."""
    rendered = stronger_rows_helper(True)
    for name in ("GPT_5_5", "CLAUDE_OPUS_4_7", "GEMINI_3_1_PRO_PREVIEW"):
        assert name in rendered, (
            f"Handle {name!r} missing from stronger_rows output:\n"
            f"{rendered!r}"
        )


def test_seed_generate_call_omits_temperature():
    """The seed's `.generate(...)` call must not pass `temperature`.

    Evolution copies the seed's call shape into derived agents. Passing
    temperature in the seed leaks the param into Opus 4.7 / GPT-5.5
    calls in derived agents — those handles 400 on temperature
    (`asta_ds1000_20260510_170547` is the captured incident). Scan for
    every Call where `.generate` is called and assert no keyword arg
    named `temperature`."""
    tree = ast.parse(SEED_AGENT_PY.read_text())
    generate_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate"
        ):
            generate_calls.append(node)
    assert generate_calls, (
        "Couldn't find any `.generate(...)` call in the seed — has the "
        "seed been refactored away from the model_registry handle pattern?"
    )

    offenders: list[str] = []
    for call in generate_calls:
        # Check direct keyword args
        for kw in call.keywords:
            if kw.arg == "temperature":
                offenders.append(f"line {call.lineno}: direct temperature kwarg")
        # Check config=GenerateConfig(...) for temperature kwarg
        for kw in call.keywords:
            if kw.arg == "config" and isinstance(kw.value, ast.Call):
                for inner_kw in kw.value.keywords:
                    if inner_kw.arg == "temperature":
                        offenders.append(
                            f"line {call.lineno}: temperature inside "
                            f"config=GenerateConfig(...)"
                        )
    assert not offenders, (
        "The seed's .generate() call passes `temperature`, which "
        "anchors evolution toward a parameter that's rejected on "
        "Opus 4.7 / GPT-5.5 and silently stripped when combined with "
        "reasoning_effort elsewhere. Drop the temperature kwarg:\n  "
        + "\n  ".join(offenders)
    )


def test_write_test_results_reads_fallback_keys_from_diagnostics():
    """`_write_test_results` constructs the per-problem JSON from a
    hardcoded key list — the same anti-pattern that originally dropped
    `fallback_used`/`primary_error` on the way out. This test asserts
    both keys are read from the diagnostics dict (via `diag.get(...)`)
    so a future contributor who adds new diagnostic keys can't silently
    re-introduce the drop without also failing this test.

    Mirrors `test_with_overrides_propagates_fallback_candidate` in the
    evaluator test file — same trap, opposite side of the boundary.
    """
    src = MAIN_PY.read_text()
    tree = ast.parse(src)
    write_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_write_test_results":
            write_fn = node
            break
    assert write_fn is not None, "_write_test_results function not found"

    # Collect string-constant args passed to any `<name>.get("...")` call
    # inside the function. The diagnostics dict is named `diag` in this
    # function, but be tolerant of refactors that rename it — assert on
    # any `.get(<str>)` call.
    read_keys = set()
    for node in ast.walk(write_fn):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "get"):
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            read_keys.add(first_arg.value)

    required = {"fallback_used", "primary_error"}
    missing = required - read_keys
    assert not missing, (
        f"_write_test_results does not read these diagnostic keys: "
        f"{sorted(missing)}. The hardcoded per-problem-key list "
        f"silently drops them from test_results_<phase>.per_problem.json. "
        f"Add `<key>: diag.get(\"<key>\")` entries (and the matching "
        f"aggregate fields in the summary block) before the JSON dump."
    )
