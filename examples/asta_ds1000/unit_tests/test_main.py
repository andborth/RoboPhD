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

REPO_ROOT = Path(__file__).resolve().parents[3]
MAIN_PY = REPO_ROOT / "examples" / "asta_ds1000" / "main.py"
ASTA_DS1000_DIR = REPO_ROOT / "examples" / "asta_ds1000"

EVAL_FUNCS = {"eval_candidate", "eval_run"}


@pytest.fixture(scope="module")
def framing_helper():
    """Import main.py's `_test_rounds_framing` once per module.

    sys.path manipulation rather than a package install because
    examples/asta_ds1000/ isn't an installable package. Importing
    main.py runs argparse setup but not main() — safe at import time.
    Module-cached so the actual `import main` only re-runs once per
    test session even with multiple fixtures.
    """
    sys.path.insert(0, str(ASTA_DS1000_DIR))
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import main as asta_main  # noqa: E402
        return asta_main._test_rounds_framing
    finally:
        sys.path.remove(str(ASTA_DS1000_DIR))
        sys.path.remove(str(REPO_ROOT))


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
                # AST-walk the elif test for the resume guard. Substring
                # matching on ast.unparse would false-positive on any
                # unrelated identifier containing "resume" (e.g.
                # `resume_kwargs`, `resumed_run`, or a docstring).
                if _expression_references_resume(elif_node.test):
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


def _expression_references_resume(node: ast.expr) -> bool:
    """True if `node` is an expression that references the resume
    state in one of the canonical forms used in main.py:
      - bare Name `is_resume` (the local boolean main.py computes)
      - attribute access `args.resume` (the raw argparse value)
    Anything else returns False — including unrelated identifiers
    that happen to contain the substring "resume"."""
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id == "is_resume":
            return True
        if (
            isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "args"
            and n.attr == "resume"
        ):
            return True
    return False


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


# --- background.md model-handle table + temperature-free seed --------------
# Two invariants:
# (1) The model-handle table in background.md must list all 10 handles in
#     5-column shape and the 3 Gemini handles must share an identical
#     Available-overrides cell. Pins the table the agent reads.
# (2) The seed agent's only .generate() call must not pass `temperature`
#     (the seed is the anchoring pattern evolution copies; passing
#     temperature there leaks the footgun into Opus/GPT-5.5 agents).

SEED_AGENT_PY = REPO_ROOT / "examples" / "asta_ds1000" / "seeds" / "baseline" / "agent.py"

EXPECTED_HANDLES = (
    "GPT_5_4_MINI", "GPT_5_4", "GPT_5_5",
    "CLAUDE_HAIKU_4_5", "CLAUDE_SONNET_4_6", "CLAUDE_OPUS_4_7",
    "CLAUDE_FABLE_5",
    "GEMINI_3_1_FLASH_LITE", "GEMINI_3_5_FLASH",
    "GEMINI_3_1_PRO_PREVIEW",
)


def test_background_md_lists_all_ten_handles_in_five_column_shape():
    """background.md's model-handle table must include all 10 handles in
    the 5-column shape (Handle | Input | Output | Default reasoning_effort
    | Available overrides). Regression to fewer columns would break the
    rendered markdown table; missing handles would hide them from the
    evolution agent's menu."""
    background_md = (ASTA_DS1000_DIR / "background.md").read_text()
    # Match the model-handle rows specifically — other tables (e.g.
    # `state.input` row in the inspect-ai input doc) also start with
    # "| `" but their cells don't begin with an UPPERCASE family prefix.
    handle_rows = [
        line for line in background_md.split("\n")
        if (
            line.endswith(" |")
            and any(
                line.startswith(f"| `{prefix}")
                for prefix in ("GPT_", "CLAUDE_", "GEMINI_")
            )
        )
    ]
    # 5 columns ⇒ 6 pipe separators per row
    bad = [
        f"{line.count('|')} pipes (expected 6): {line!r}"
        for line in handle_rows if line.count("|") != 6
    ]
    assert not bad, "wrong column count:\n  " + "\n  ".join(bad)
    # Each handle must appear in some row
    missing = [
        h for h in EXPECTED_HANDLES
        if not any(f"`{h}`" in row for row in handle_rows)
    ]
    assert not missing, f"handles missing from background.md table: {missing}"


def test_all_gemini_rows_share_available_overrides_cell():
    """All three Gemini handles share the same default `"low"` / opt-up
    `"high"` reasoning_effort shape, so their Available-overrides cells
    in the rendered model-handle table must be identical.

    Pins the symmetry that the "describe Pro Preview like the other
    Geminis" change (commit fe36199) established.
    """
    background_md = (ASTA_DS1000_DIR / "background.md").read_text()
    gemini_rows = [
        line for line in background_md.split("\n")
        if line.startswith("| `GEMINI_") and line.endswith(" |")
    ]
    assert len(gemini_rows) == 3, (
        f"Expected 3 GEMINI_* rows in background.md, found "
        f"{len(gemini_rows)}:\n  " + "\n  ".join(gemini_rows)
    )

    # Row shape: | handle | input | output | default | overrides |
    # split("|") -> ['', ' handle ', ' input ', ' output ', ' default ',
    #                 ' overrides ', '']
    # so [-2] is the Available-overrides cell.
    overrides_by_handle = {
        row.split("|")[1].strip(): row.split("|")[-2].strip()
        for row in gemini_rows
    }
    unique_cells = set(overrides_by_handle.values())
    assert len(unique_cells) == 1, (
        "GEMINI_* rows must share an identical Available-overrides "
        "cell — Pro Preview should be described in the same boat as "
        "the cheap-tier Geminis (all three default to `\"low\"` with "
        "`\"high\"` as the only opt-up). Cells found:\n  "
        + "\n  ".join(f"{h}: {c!r}" for h, c in overrides_by_handle.items())
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


def test_write_test_results_summary_includes_required_aggregate_keys():
    """The summary JSON dict literal in `_write_test_results` must
    carry the aggregate stats that downstream tooling reads (cleanup
    scripts, dev-eval reports, leaderboard exports). The list of
    required keys is a structural contract — a refactor that silently
    drops one (e.g. `mean_test_agent_cost_usd` going missing in a
    rename) breaks consumers without failing the smoke run that
    produced the file.

    Same AST-walk shape as the per-problem fallback-keys test above —
    this is the matching pin on the summary side of the boundary.
    """
    src = MAIN_PY.read_text()
    tree = ast.parse(src)
    write_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_write_test_results":
            write_fn = node
            break
    assert write_fn is not None, "_write_test_results function not found"

    # Find the summary dict literal. Multiple dict literals can appear
    # in this function (e.g. per-problem rows built inside the loop);
    # anchor on "mean_test_score" + "total_test_problems" — the unique
    # pair that identifies the top-level summary dict passed to
    # json.dump for the *.json (not *.per_problem.json) file.
    summary_keys = None
    for node in ast.walk(write_fn):
        if not isinstance(node, ast.Dict):
            continue
        keys = {
            k.value for k in node.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }
        if "mean_test_score" in keys and "total_test_problems" in keys:
            summary_keys = keys
            break
    assert summary_keys is not None, (
        "summary dict literal (anchored by mean_test_score + "
        "total_test_problems) not found in _write_test_results"
    )

    required = {
        "mean_test_score",
        "total_test_score",
        "total_test_problems",
        "test_eval_cost_usd",         # total infra/eval spend
        "test_eval_agent_cost_usd",   # total agent spend
        "mean_test_agent_cost_usd",   # per-problem mean agent spend
    }
    missing = required - summary_keys
    assert not missing, (
        f"_write_test_results summary JSON is missing required keys: "
        f"{sorted(missing)}. Add the field to the dict literal "
        f"passed to json.dump(..., summary_path, ...)."
    )


# --- per-problem wall-clock budget --------------------------------------------
# The per-problem timeout was raised 20→30 min and surfaced agent-side
# via background.md. These pins keep the constant, the placeholder, and
# the interpolation in sync so the agent-facing doc can never silently
# drift from the real budget (the exact stale-constraint failure mode
# this work addressed).


def test_eval_timeout_is_30_minutes():
    """EVAL_TIMEOUT must be 1800s (30 min). Pinned because the
    agent-facing background.md derives its stated budget from this
    constant; a silent change here would desync the doc from reality."""
    tree = ast.parse(MAIN_PY.read_text())
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "EVAL_TIMEOUT":
                    found = node.value
    assert found is not None, "EVAL_TIMEOUT assignment not found in main.py"
    assert isinstance(found, ast.Constant) and found.value == 1800, (
        f"EVAL_TIMEOUT must be the literal 1800 (30 min); got "
        f"{ast.unparse(found)!r}. The agent's true budget is "
        f"EVAL_TIMEOUT-30s = 1770s = 29 min (reaper buffer)."
    )


def test_background_md_eval_timeout_placeholder_and_interpolation():
    """Three coupled invariants:
      1. background.md contains the literal `${EVAL_TIMEOUT_MIN}`.
      2. main.py's `_interpolate` substitutes it with an expression
         derived from `EVAL_TIMEOUT` minus the 30s reaper buffer,
         floored to whole minutes.
      3. The arithmetic resolves to 29 at EVAL_TIMEOUT=1800.
    Together these guarantee the rendered prompt states the *true*
    floored kill budget and can never go stale relative to the
    constant."""
    bg = (REPO_ROOT / "examples" / "asta_ds1000" / "background.md").read_text()
    assert "${EVAL_TIMEOUT_MIN}" in bg, (
        "background.md no longer contains `${EVAL_TIMEOUT_MIN}`; the "
        "Time budget section must keep the placeholder so main.py "
        "interpolates the real (constant-derived) value."
    )

    tree = ast.parse(MAIN_PY.read_text())
    repl = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "replace"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "${EVAL_TIMEOUT_MIN}"
        ):
            repl = node
    assert repl is not None, (
        "main.py `_interpolate` has no .replace(\"${EVAL_TIMEOUT_MIN}\", ...) "
        "call — the placeholder would survive into the rendered prompt."
    )
    val_src = ast.unparse(repl.args[1])
    assert "EVAL_TIMEOUT" in val_src and "- 30" in val_src and "// 60" in val_src, (
        f"${{EVAL_TIMEOUT_MIN}} must be derived from "
        f"`(EVAL_TIMEOUT - 30) // 60` (true kill budget, floored). "
        f"Got: {val_src}"
    )
    # Arithmetic intent, documented and pinned.
    assert (1800 - 30) // 60 == 29


# ----------------------------------------------------------------------
# DS-1000 runtime sidecar (cost_threshold / cost_per_error). The
# framework's ConfigManager doesn't persist these task-specific knobs,
# so without the sidecar they'd silently revert to evaluator defaults
# on --resume, which for cost_threshold *changes the scoring function*
# mid-run. The cost knobs are immutable across a run: passing the CLI
# flag on --resume is a hard error, and a missing sidecar (pre-fix
# historical run) is also an error rather than silent fallback.
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def sidecar_helpers():
    """Import main.py's sidecar helpers once per module."""
    sys.path.insert(0, str(ASTA_DS1000_DIR))
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import main as asta_main  # noqa: E402
        return {
            "read": asta_main._read_ds1000_runtime_config,
            "write": asta_main._write_ds1000_runtime_config,
            "enforce_immutable": asta_main._enforce_immutable_on_resume,
            "filename": asta_main.DS1000_RUNTIME_CONFIG_FILENAME,
        }
    finally:
        sys.path.remove(str(ASTA_DS1000_DIR))
        sys.path.remove(str(REPO_ROOT))


def test_sidecar_roundtrip(sidecar_helpers, tmp_path):
    """write → read returns exactly what was written."""
    payload = {"cost_threshold": 0.06, "cost_per_error": 0.01}
    sidecar_helpers["write"](tmp_path, payload)
    assert sidecar_helpers["read"](tmp_path) == payload


def test_sidecar_missing_file_returns_empty_dict(sidecar_helpers, tmp_path):
    """An absent sidecar must NOT raise from the reader — the immutability
    enforcement at the call site is where the historical-run error fires."""
    assert sidecar_helpers["read"](tmp_path) == {}


def test_sidecar_malformed_file_returns_empty_dict(sidecar_helpers, tmp_path):
    """Best-effort read: malformed JSON falls back to {} rather than crashing
    a resume. The error then surfaces from _enforce_immutable_on_resume
    with a clear "no stored value" message."""
    (tmp_path / sidecar_helpers["filename"]).write_text("{not: valid json")
    assert sidecar_helpers["read"](tmp_path) == {}


def test_enforce_fresh_run_uses_cli_when_set(sidecar_helpers):
    """Fresh run + CLI flag set → returns CLI value."""
    result = sidecar_helpers["enforce_immutable"](
        cli_value=0.08, stored_value=None, default_value=0.04,
        name="cost-threshold", on_resume=False,
    )
    assert result == 0.08


def test_enforce_fresh_run_falls_back_to_default(sidecar_helpers):
    """Fresh run + no CLI flag → returns evaluator default."""
    result = sidecar_helpers["enforce_immutable"](
        cli_value=None, stored_value=None, default_value=0.04,
        name="cost-threshold", on_resume=False,
    )
    assert result == 0.04


def test_enforce_resume_with_matching_cli_silently_uses_stored(sidecar_helpers):
    """Resume + CLI flag matching stored value → silent no-op.

    A user who passes the same flag they used on the original run
    shouldn't get an error — the immutability rule is about preventing
    *changes*, not about banning the flag entirely.
    """
    result = sidecar_helpers["enforce_immutable"](
        cli_value=0.06, stored_value=0.06, default_value=0.04,
        name="cost-threshold", on_resume=True,
    )
    assert result == 0.06


def test_enforce_resume_with_conflicting_cli_is_hard_error(sidecar_helpers):
    """Resume + CLI flag differing from stored → SystemExit.

    This is the load-bearing immutability rule: changing cost-threshold
    mid-run silently invalidates Elo continuity with prior iterations.
    """
    with pytest.raises(SystemExit) as exc_info:
        sidecar_helpers["enforce_immutable"](
            cli_value=0.10, stored_value=0.06, default_value=0.04,
            name="cost-threshold", on_resume=True,
        )
    msg = str(exc_info.value)
    assert "cost-threshold" in msg
    assert "cannot be changed on --resume" in msg
    assert "stored value:" in msg


def test_enforce_resume_with_no_cli_uses_stored(sidecar_helpers):
    """Resume + no CLI flag + sidecar value → returns stored.

    Without this, the user's original --cost-threshold setting
    silently reverts to evaluator defaults on resume.
    """
    result = sidecar_helpers["enforce_immutable"](
        cli_value=None, stored_value=0.06, default_value=0.04,
        name="cost-threshold", on_resume=True,
    )
    assert result == 0.06


def test_enforce_resume_bootstraps_when_sidecar_missing(sidecar_helpers):
    """Resume + CLI flag + missing sidecar → returns CLI value (bootstrap).

    The sidecar is only written after iteration 1 completes, so a run
    interrupted before then has no sidecar. Allowing the CLI flag to
    bootstrap a fresh sidecar is the recovery path; the value is then
    locked for all subsequent resumes.
    """
    result = sidecar_helpers["enforce_immutable"](
        cli_value=0.10, stored_value=None, default_value=0.04,
        name="cost-threshold", on_resume=True,
    )
    assert result == 0.10


def test_enforce_resume_missing_stored_and_no_cli_is_hard_error(sidecar_helpers):
    """Resume + no CLI flag + missing sidecar → SystemExit.

    Silent default fallback here is the bug class that produced the
    $0.10 → $0.04 drift in asta_ds1000_20260526_120002. Error and
    tell the user how to recover (pass --cost-threshold to bootstrap,
    or restart). The error must also flag that both knobs are checked
    independently — bootstrapping only one when both are missing
    triggers a second error.
    """
    with pytest.raises(SystemExit) as exc_info:
        sidecar_helpers["enforce_immutable"](
            cli_value=None, stored_value=None, default_value=0.04,
            name="cost-threshold", on_resume=True,
        )
    msg = str(exc_info.value)
    assert "no stored cost-threshold" in msg
    assert "bootstrap" in msg
    assert "--cost-per-error" in msg, (
        "missing-sidecar error must mention that both cost knobs need "
        "to be supplied together — without this, users who pass only "
        "--cost-threshold hit a second error and don't know why"
    )


def test_background_md_prices_match_litellm_registry():
    """The model-handle table's advertised $/MTok rates must match
    litellm's price registry for the underlying model IDs.

    The table is hand-maintained, and providers reprice or re-alias
    models upstream: the asta_ds1000_20260609_144857 run shopped from a
    table advertising Gemini Flash at $0.50/$3.00 while Google had
    already routed the preview ID to gemini-3.5-flash billed at
    $1.50/$9.00. The cost *penalty* was unaffected (it uses measured
    rates keyed by the response model name), but evolution made routing
    decisions against a 3x-wrong menu price. This test fails the day
    the advertised menu and the billed reality diverge.

    Handle -> model ID comes from AST-parsing model_registry.py's
    `_<HANDLE>_ID` constants rather than importing the module (import
    requires an Anthropic key at construction time). Models absent from
    litellm's registry (e.g. claude-fable-5) are covered by RoboPhD's
    own SUPPORTED_MODELS registration, which is itself the pricing
    source the rest of the framework trusts.

    Failure-class split: a price MISMATCH always hard-fails — that's
    the drift signal. A model MISSING from litellm's registry skips
    instead: offline runs (and stale bundled maps on the litellm <=1.83
    that agent-eval pins) lack recent models like gemini-3.5-flash
    while containing older ones, which is registry staleness, not table
    drift — hard-failing there would cry wolf in any freshly-installed
    env. The cost: an upstream model *removal* surfaces as a visible
    skip rather than a failure; repricing (the actual incident class)
    is always caught when the registry has the model.
    """
    import litellm

    from RoboPhD.runner_utils import register_supported_model_pricing

    # Cover models newer than litellm's registry (e.g. claude-fable-5).
    register_supported_model_pricing()

    # Handle name -> "provider/model" from _<HANDLE>_ID assignments.
    registry_src = (ASTA_DS1000_DIR / "model_registry.py").read_text()
    handle_to_id = {}
    for node in ast.walk(ast.parse(registry_src)):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("_")
            and node.targets[0].id.endswith("_ID")
            and isinstance(node.value, ast.Constant)
        ):
            handle_to_id[node.targets[0].id[1:-3]] = node.value.value

    background_md = (ASTA_DS1000_DIR / "background.md").read_text()
    handle_rows = [
        line for line in background_md.split("\n")
        if (
            line.endswith(" |")
            and any(
                line.startswith(f"| `{prefix}")
                for prefix in ("GPT_", "CLAUDE_", "GEMINI_")
            )
        )
    ]
    assert handle_rows, "no model-handle rows found in background.md"

    mismatches = []     # genuine drift — always a hard failure
    unverifiable = []   # registry staleness / offline — skip, don't cry wolf
    broken_rows = []    # repo-internal inconsistency — hard failure
    for row in handle_rows:
        cells = [c.strip() for c in row.split("|")]
        # ['', '`HANDLE`', input, output, default, overrides, '']
        handle = cells[1].strip("`")
        advertised_in, advertised_out = float(cells[2]), float(cells[3])

        model_id = handle_to_id.get(handle)
        if model_id is None:
            broken_rows.append(f"{handle}: no _<HANDLE>_ID in model_registry.py")
            continue

        bare = model_id.split("/", 1)[1]
        key = next(
            (k for k in (model_id, bare, f"gemini/{bare}") if k in litellm.model_cost),
            None,
        )
        if key is None:
            unverifiable.append(f"{handle} ({model_id})")
            continue

        mc = litellm.model_cost[key]
        actual_in = mc["input_cost_per_token"] * 1e6
        actual_out = mc["output_cost_per_token"] * 1e6
        if advertised_in != pytest.approx(actual_in, abs=1e-6) or (
            advertised_out != pytest.approx(actual_out, abs=1e-6)
        ):
            mismatches.append(
                f"{handle}: background.md advertises "
                f"${advertised_in}/{advertised_out} per MTok but litellm "
                f"({key}) says ${actual_in:.2f}/{actual_out:.2f} — the "
                f"agent-facing menu has drifted from billed reality"
            )

    assert not broken_rows, (
        "table rows without a model_registry constant:\n  "
        + "\n  ".join(broken_rows)
    )
    assert not mismatches, "price-table drift:\n  " + "\n  ".join(mismatches)
    if unverifiable:
        pytest.skip(
            "rows unverifiable — models absent from litellm's price "
            "registry (stale bundled map or offline; not table drift): "
            + ", ".join(unverifiable)
        )
