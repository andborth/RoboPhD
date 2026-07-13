"""Unit tests for asta_paper_finder's main.py wiring.

Source/AST-level guards for the modernization invariants:

  - every eval_candidate/eval_run call passes config= (finding E: the
    archived example ran test evals at the framework's default worker
    count, ignoring --max-workers);
  - the doc files carry every ${...} placeholder main.py interpolates,
    and main.py interpolates every placeholder the docs carry;
  - task_config_extras persistence is wired (finding B);
  - --model is gone (the nine-handle registry replaced it) and the
    cost/tool knobs exist;
  - the background.md price table matches litellm's registry for the
    underlying model IDs (menu-vs-billed-reality drift).
"""
import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PFB_DIR = REPO_ROOT / "examples" / "asta_paper_finder"
MAIN_SRC = (PFB_DIR / "main.py").read_text()
MAIN_TREE = ast.parse(MAIN_SRC)


# --- eval calls pass config= --------------------------------------------------


def test_every_eval_call_passes_config():
    offenders = []
    for node in ast.walk(MAIN_TREE):
        if isinstance(node, ast.Call):
            func_name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if func_name in ("eval_candidate", "eval_run"):
                kwargs = {kw.arg for kw in node.keywords}
                if "config" not in kwargs:
                    offenders.append(f"line {node.lineno}: {func_name}")
    assert not offenders, (
        "eval calls without config= (test eval ignores --max-workers/"
        "eval_timeout): " + "; ".join(offenders)
    )


# --- interpolation coverage -----------------------------------------------------


EXPECTED_PLACEHOLDERS = {
    "${COST_PENALTY_TABLE}",
    "${COST_THRESHOLD}",
    "${COST_PER_ERROR}",
    "${EVAL_TIMEOUT_MIN}",
}


def _doc_placeholders() -> set[str]:
    found = set()
    for doc in ("background.md", "objective.md"):
        text = (PFB_DIR / doc).read_text()
        found |= set(re.findall(r"\$\{[A-Z_]+\}", text))
    return found


def test_docs_use_only_interpolated_placeholders():
    """Any ${...} in the docs that main.py doesn't substitute would ship
    to the evolution AI as literal template text."""
    unknown = _doc_placeholders() - EXPECTED_PLACEHOLDERS
    assert not unknown, f"placeholders with no _interpolate substitution: {unknown}"


def test_main_interpolates_every_expected_placeholder():
    for ph in EXPECTED_PLACEHOLDERS:
        assert f'.replace("{ph}"' in MAIN_SRC, (
            f"main.py's _interpolate no longer substitutes {ph}"
        )


def test_docs_carry_the_cost_placeholders():
    """The cost framing must actually reach the evolution AI: the table
    in background.md and the threshold/slope in objective.md."""
    background = (PFB_DIR / "background.md").read_text()
    objective = (PFB_DIR / "objective.md").read_text()
    assert "${COST_PENALTY_TABLE}" in background
    assert "${COST_THRESHOLD}" in background
    assert "${EVAL_TIMEOUT_MIN}" in background
    assert "${COST_THRESHOLD}" in objective
    assert "${COST_PER_ERROR}" in objective


# --- task_config_extras / CLI surface -------------------------------------------


def test_task_config_extras_wired():
    assert "task_config_extras={PFB_TASK_CONFIG_KEY: resolved_runtime}" in MAIN_SRC


def _argparse_flags() -> set[str]:
    flags = set()
    for node in ast.walk(MAIN_TREE):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            flags.add(node.args[0].value)
    return flags


def test_model_flag_removed():
    """--model was replaced by the model_registry menu; its presence
    would mean a solver-model bypass around the registry."""
    assert "--model" not in _argparse_flags()


def test_cost_and_tool_knobs_present():
    flags = _argparse_flags()
    for required in ("--cost-threshold", "--cost-per-error",
                     "--max-workers", "--eval-only", "--eval-agent"):
        assert required in flags, f"missing CLI flag: {required}"


def test_max_workers_defaults_none():
    """argparse default must be None so the resolution ladder can
    distinguish user-explicit from default (finding E: a concrete
    default silently clobbers the checkpoint's value on resume)."""
    for node in ast.walk(MAIN_TREE):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--max-workers"
        ):
            defaults = [kw.value for kw in node.keywords if kw.arg == "default"]
            assert defaults and isinstance(defaults[0], ast.Constant) \
                and defaults[0].value is None
            return
    pytest.fail("--max-workers flag not found")


def _fresh_run_override_assigns() -> dict[str, ast.expr]:
    """Value expressions assigned to engine_overrides["<key>"] inside the
    `if not is_resume:` branch, keyed by subscript key.

    AST-based (not a source-text match) so behavior-preserving rewrites
    of the value expressions don't break the guard, while a move of an
    assignment OUT of the fresh-run-only branch — the regression that
    matters, because it would clobber the checkpoint's value on resume —
    does break it."""
    for node in ast.walk(MAIN_TREE):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Name)
            and node.test.operand.id == "is_resume"
        ):
            assigns = {}
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Subscript)
                    and isinstance(stmt.targets[0].value, ast.Name)
                    and stmt.targets[0].value.id == "engine_overrides"
                    and isinstance(stmt.targets[0].slice, ast.Constant)
                ):
                    assigns[stmt.targets[0].slice.value] = stmt.value
            return assigns
    pytest.fail("no `if not is_resume:` branch found in main.py")


def _eval_expr(expr: ast.expr, train_size: int):
    """Evaluate an assignment's value expression against a fake train
    pool, so the test pins the RESULT (behavior) rather than the
    spelling of the arithmetic."""
    import math
    code = compile(ast.fix_missing_locations(ast.Expression(body=expr)), "<ast>", "eval")
    return eval(  # noqa: S307 — evaluating our own parsed source
        code,
        {"__builtins__": {"len": len}, "math": math, "train": [None] * train_size},
    )


def test_fresh_run_packs_task_defaults():
    """PFB's task defaults that differ from the framework's must be
    stamped into engine_overrides on fresh runs ONLY (so --resume
    inherits the checkpoint): new_agent_test_rounds=0 (framework
    default 1) and examples_per_iteration=ceil(train/5) (framework
    default 20 — too much per-example reuse for a 66-sample pool)."""
    assigns = _fresh_run_override_assigns()
    assert set(assigns) >= {"new_agent_test_rounds", "examples_per_iteration"}, (
        f"fresh-run branch packs {sorted(assigns)}; a missing key means the "
        f"framework default silently applies"
    )
    assert _eval_expr(assigns["new_agent_test_rounds"], 66) == 0
    # ceil(66/5) = 14 for the current pool; ceil(50/5) = 10 pins that the
    # value tracks the pool size (e.g. a future thermometer holdout)
    # rather than being a hardcoded constant.
    assert _eval_expr(assigns["examples_per_iteration"], 66) == 14
    assert _eval_expr(assigns["examples_per_iteration"], 50) == 10


def test_main_requires_asta_tool_key_up_front():
    """main.py preflights ASTA_TOOL_KEY with a SystemExit before dataset
    loading (the evaluator's constructor backstops it with the same
    requirement). The search fallback and --tool-source knob are gone;
    neither may reappear."""
    assert 'os.environ.get("ASTA_TOOL_KEY")' in MAIN_SRC
    assert "raise SystemExit" in MAIN_SRC
    assert "--tool-source" not in MAIN_SRC
    assert "tool_source" not in MAIN_SRC


def test_gepa_and_autoresearch_apply_engine_config():
    """Finding D: --engine-config must not be a silent no-op on the
    non-default engines."""
    assert MAIN_SRC.count("apply_engine_config(cfg, parsed_engine_config)") >= 2


# --- doc <-> code name agreement --------------------------------------------------
#
# background.md tells the evolution AI about concrete field/file names it
# will encounter in diagnostics and must produce in output. Each name is
# owned by a different piece of code, and nothing else ties the doc to the
# emitters — a rename would leave the doc pointing at a nonexistent name
# with every other test green (the silent-staleness failure mode).


def _background() -> str:
    return (PFB_DIR / "background.md").read_text()


def _evaluator_emitted_keys() -> set[str]:
    """String keys the evaluator writes into diagnostics: subscript
    assignments on `diagnostics` plus dict-literal keys anywhere in the
    module (the error-path returns and the initial diagnostics literal)."""
    src = (PFB_DIR / "evaluator.py").read_text()
    keys: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "diagnostics"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.add(node.slice.value)
        elif isinstance(node, ast.Dict):
            keys |= {
                k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
    return keys


def test_doc_diagnostic_names_match_evaluator():
    """Names background.md attributes to per-problem diagnostics must be
    keys the evaluator actually emits (and must still appear in the doc,
    so this list can't silently outlive the prose it guards)."""
    background = _background()
    emitted = _evaluator_emitted_keys()
    for name in ("gold_criteria.md", "agent_stdout", "eval_wall_clock_seconds",
                  "judge_verdicts.md"):
        assert name in background, f"{name} no longer mentioned in background.md — prune it here"
        assert name in emitted, (
            f"background.md documents diagnostic {name!r} but evaluator.py "
            f"never emits that key — the doc points at a nonexistent field"
        )


def test_doc_other_cost_name_matches_domain_mapping():
    """background.md tells the agent the judge spend appears in
    result.json as `other_cost`. That name is produced by the FRAMEWORK:
    domain.py maps the evaluator's other_cost_usd diagnostic into a
    result field literally named other_cost. Pin both halves of the
    contract so a rename on either side fails here instead of silently
    stranding the doc."""
    assert "`other_cost`" in _background()
    domain_src = (REPO_ROOT / "RoboPhD" / "domains" / "external" / "domain.py").read_text()
    assert '"other_cost_usd"' in domain_src, (
        "domain.py no longer reads the other_cost_usd diagnostic bucket"
    )
    assert '"other_cost"' in domain_src, (
        "domain.py no longer writes an other_cost result field — update "
        "background.md's judge section to the new name"
    )


def test_doc_output_schema_names_match_astabench():
    """`paper_id` and `markdown_evidence` are astabench's output-schema
    contract (paper_finder/datamodel.py); the doc and the seed both spell
    them. Introspect the installed datamodel so an upstream rename fails
    loudly on the version bump that introduces it."""
    import astabench.evals.paper_finder.datamodel as dm

    fields: set[str] = set()
    for obj in vars(dm).values():
        if isinstance(obj, type) and hasattr(obj, "model_fields"):
            fields |= set(obj.model_fields)
    background = _background()
    for name in ("paper_id", "markdown_evidence"):
        assert name in background
        assert name in fields, (
            f"astabench's paper_finder datamodel no longer has a {name!r} "
            f"field — background.md's output schema and the seed are stale"
        )

def test_doc_examples_dont_model_the_openai_token_trap():
    """background.md's code examples are what evolution copies — the run
    record shows iter2-iter5 shipped the reasoning_effort + tight
    max_tokens combination on OpenAI handles straight from the doc's
    example, silently emptying their most important LLM call for four
    generations. The doc's rule is now blanket: max_tokens is not
    recommended on OpenAI handles at all (GPT_5_5's model-managed
    reasoning can trip the shared cap with no opt-in). Pin: the rule
    sentence survives, and no python code block may attach a
    GenerateConfig with max_tokens to a GPT_* handle call."""
    import re
    background = (PFB_DIR / "background.md").read_text()
    assert "not recommended on OpenAI handles" in background, (
        "the OpenAI max_tokens rule was removed from background.md"
    )
    for block in re.findall(r"```python\n(.*?)```", background, re.DOTALL):
        # Associate each GenerateConfig(...) with the nearest preceding
        # handle call. GenerateConfig args never nest parens, so a
        # non-greedy match to the first ')' is exact.
        calls = [(m.start(), m.group(1)) for m in re.finditer(r"await ([A-Z0-9_]+)\.generate\(", block)]
        for cfg in re.finditer(r"GenerateConfig\(([^)]*)\)", block):
            handle = next((h for pos, h in reversed(calls) if pos < cfg.start()), None)
            args = cfg.group(1)
            if handle and handle.startswith("GPT_") and "max_tokens" in args:
                pytest.fail(
                    f"doc example sets max_tokens on OpenAI handle {handle} — "
                    f"contradicts the doc's own rule (shared reasoning cap; "
                    f"the iter2-iter5 silent-empty-completion trap)"
                )


def test_no_macro_mean_claim_in_docs():
    """The scoring objective is a plain mean over queries — the headline
    metric is adjusted_f1_micro_avg, built with grouped(mean(),
    all="samples"). background.md claimed a macro-by-group mean for a
    month and the iteration-6 evolution session built strategy on it
    ("each query type is worth ~1/3"). Pin the word out of the
    agent-facing docs; if per-group weighting ever genuinely returns,
    this test is the reminder to re-verify against the scorer first."""
    for doc in ("background.md", "objective.md"):
        text = (PFB_DIR / doc).read_text().lower()
        assert "macro" not in text, (
            f"{doc} mentions 'macro' — the scorer's headline is a plain "
            f"per-query mean (adjusted_f1_micro_avg); verify against "
            f"astabench paper_finder/task.py before reintroducing this"
        )


# --- price-table consistency -----------------------------------------------------


def test_background_md_prices_match_litellm_registry():
    """The model-handle table's advertised $/MTok rates must match
    litellm's price registry for the underlying model IDs.

    The table is hand-maintained, and providers reprice or re-alias
    models upstream. The cost *penalty* is unaffected by table drift
    (it uses measured rates keyed by the response model name), but
    evolution makes routing decisions against the menu — an incident
    class asta_ds1000 hit when Flash repriced 3x.

    Handle -> model ID comes from AST-parsing model_registry.py's
    `_<HANDLE>_ID` constants rather than importing the module (import
    requires an Anthropic key at construction time).

    Failure-class split: a price MISMATCH always hard-fails — that's
    the drift signal. A model MISSING from litellm's registry skips
    instead: offline runs and stale bundled maps lack recent models,
    which is registry staleness, not table drift.

    "Billed reality" is defined by evaluator._estimate_cost: litellm's
    BUNDLED price snapshot first (the leaderboard's billing basis),
    falling back to the live map only for models the snapshot lacks.
    The lookup order here mirrors the evaluator exactly.
    """
    import json

    import litellm

    from RoboPhD.runner_utils import register_supported_model_pricing

    register_supported_model_pricing()

    try:
        bundled_map = json.loads(
            (Path(litellm.__file__).parent
             / "model_prices_and_context_window_backup.json").read_text()
        )
    except Exception:
        bundled_map = {}

    registry_src = (PFB_DIR / "model_registry.py").read_text()
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

    background_md = (PFB_DIR / "background.md").read_text()
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
    assert len(handle_rows) == 9, (
        f"expected 9 model-handle rows in background.md, found {len(handle_rows)}"
    )

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
        candidates = (model_id, bare, f"gemini/{bare}")
        key = next(
            (k for k in candidates
             if bundled_map.get(k, {}).get("input_cost_per_token") is not None),
            None,
        )
        source = bundled_map
        if key is None:
            key = next(
                (k for k in candidates if k in litellm.model_cost), None
            )
            source = litellm.model_cost
        if key is None:
            unverifiable.append(f"{handle} ({model_id})")
            continue

        mc = source[key]
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
            "registry (staleness, not drift): " + ", ".join(unverifiable)
        )
