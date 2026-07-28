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
import os
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PFB_DIR = REPO_ROOT / "examples" / "asta_paper_finder"
MAIN_SRC = (PFB_DIR / "main.py").read_text()
MAIN_TREE = ast.parse(MAIN_SRC)


@pytest.fixture(scope="module")
def main_mod():
    """Import main.py once per module (pulls in the RoboPhD framework).

    Needed by the _set_test_cache_env behavior tests; the rest of this
    file stays source/AST-level.
    """
    if str(PFB_DIR) not in sys.path:
        sys.path.insert(0, str(PFB_DIR))
    import main  # noqa: E402
    return main


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
    "${EVIDENCE_CAP_NOTE}",
    "${TOOL_LAUNCH_NOTE}",
    "${SESSION_ACCESS_NOTE}",
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
                     "--cap-judge-to-estimate", "--tool-launch-rate",
                     "--max-workers", "--eval-only", "--eval-agent"):
        assert required in flags, f"missing CLI flag: {required}"


def test_cap_judge_is_run_immutable():
    """The top-estimate cap changes the training scoring basis, so it must be
    persisted (in resolved_runtime) and resolved through the same immutable
    machinery as the cost knobs — otherwise a resume could flip it mid-run."""
    assert '"cap_judge_to_estimate": cap_judge_to_estimate' in MAIN_SRC
    assert 'name="cap-judge-to-estimate"' in MAIN_SRC


# --- test-eval judge env (_set_test_cache_env) -----------------------------------
#
# Internal test evals default to a persistent shared verdict cache and the
# top-estimate judging cap; --no-shared-judge-cache / --no-cap-judge-to-estimate
# restore submission-exact semantics. These are behavior tests against the
# module-level helper: the env vars it sets are the ONLY channel to the
# grounded judge, so a wiring slip silently changes the recorded metric.


@pytest.fixture
def _judge_env(monkeypatch, main_mod):
    """Pre-register the three judge env vars with monkeypatch so whatever
    _set_test_cache_env does to them is rolled back at teardown."""
    from evaluator import CACHE_PATH_ENV, CAP_JUDGE_ENV, TRAINING_GRADER_ENV
    for var in (CACHE_PATH_ENV, CAP_JUDGE_ENV, TRAINING_GRADER_ENV):
        monkeypatch.setenv(var, "sentinel")
    return CACHE_PATH_ENV, CAP_JUDGE_ENV, TRAINING_GRADER_ENV


def test_test_cache_env_shared_capped_default(main_mod, _judge_env, tmp_path):
    cache_env, cap_env, grader_env = _judge_env
    from astabench.evals.paper_finder.relevance import GRADER_MODEL_NAME

    mode = main_mod._set_test_cache_env(
        "test_set", runs_dir=str(tmp_path), shared=True, cap=True,
        judge=GRADER_MODEL_NAME, stock=GRADER_MODEL_NAME,
    )
    path = Path(os.environ[cache_env])
    assert path.parent == tmp_path / ".judge_cache"
    assert path.name == f"shared_test_{main_mod._judge_slug(GRADER_MODEL_NAME)}.json"
    # Dedicated test file: never training's shared_<slug>.json, so test
    # verdict provenance stays pure even if the split query-id namespaces
    # ever stop being disjoint.
    assert path.name != f"shared_{main_mod._judge_slug(GRADER_MODEL_NAME)}.json"
    assert os.environ.get(cap_env) == "1"
    assert grader_env not in os.environ, "test evals must use the stock grader"
    assert mode == {
        "judge_model": GRADER_MODEL_NAME,
        "judge_prompt": "stock",
        "judge_cache": "shared",
        "judge_cache_path": str(path),
        "cap_judge_to_estimate": True,
    }


def test_test_cache_env_pristine_uncapped(main_mod, _judge_env, tmp_path):
    """--no-shared-judge-cache + --no-cap-judge-to-estimate is the
    submission-exact combination: fresh empty cache, judge everything."""
    cache_env, cap_env, grader_env = _judge_env
    from astabench.evals.paper_finder.relevance import GRADER_MODEL_NAME

    mode = main_mod._set_test_cache_env(
        "eval_only", runs_dir=str(tmp_path), shared=False, cap=False,
        judge=GRADER_MODEL_NAME, stock=GRADER_MODEL_NAME,
    )
    path = Path(os.environ[cache_env])
    assert "pf_pristine_eval_only_" in path.name
    assert not path.exists(), "pristine cache must start absent (init treats missing as {})"
    assert tmp_path not in path.parents, "pristine cache must not be a persistent runs-dir file"
    assert cap_env not in os.environ
    assert grader_env not in os.environ
    assert mode["judge_cache"] == "pristine"
    assert mode["cap_judge_to_estimate"] is False


def test_test_cache_env_nonstock_judge(main_mod, _judge_env, tmp_path):
    """A non-stock test judge (luna) must: set the override env (the
    evaluator then installs the lenient normalizer), scope the cache by
    the alternate judge's slug (verdicts from different judges never
    mix), and mark the record as official-incomparable."""
    cache_env, cap_env, grader_env = _judge_env
    from astabench.evals.paper_finder.relevance import GRADER_MODEL_NAME

    luna = "openai/gpt-5.6-luna"
    mode = main_mod._set_test_cache_env(
        "eval_only", runs_dir=str(tmp_path), shared=True, cap=True,
        judge=luna, stock=GRADER_MODEL_NAME,
    )
    path = Path(os.environ[cache_env])
    assert path.name == f"shared_test_{main_mod._judge_slug(luna)}.json"
    assert os.environ.get(grader_env) == luna
    assert mode["judge_model"] == luna
    assert "NOT comparable" in mode["judge_note"]


def test_test_results_filename_judge_suffix(main_mod):
    """Stock evals keep the historical filenames; a non-stock judge gets a
    .judge_<model> suffix so its (official-incomparable) scores can never
    collide with a stock GPT-4o eval of the same agent."""
    stock = "openai/gpt-4o-2024-11-20"
    luna = "openai/gpt-5.6-luna"
    f = main_mod._test_results_filename
    assert f(None, stock, stock) == "test_results.json"
    assert f("iter5_x", stock, stock) == "test_results_iter5_x.json"
    assert f(None, luna, stock) == "test_results.judge_gpt-5.6-luna.json"
    assert f("iter5_x", luna, stock) == "test_results_iter5_x.judge_gpt-5.6-luna.json"
    # The per-problem sibling derives via with_suffix(".per_problem.json")
    # in _write_test_results — the judge suffix must survive that.
    p = Path(f(None, luna, stock)).with_suffix(".per_problem.json")
    assert p.name == "test_results.judge_gpt-5.6-luna.per_problem.json"


def test_write_test_results_persists_problem_diagnostics(main_mod, tmp_path):
    """Test evals must persist per-problem diagnostics (submission.json,
    judge_verdicts.md, ...) like training iterations do — discarding them
    forced judge A/B studies to re-run the agent instead of re-judging
    stored submissions. Judge-suffixed evals get their own problems tree."""
    from types import SimpleNamespace
    eval_result = SimpleNamespace(
        per_example_diagnostics=[{
            "sample_id": "semantic_9",
            "score_type": "semantic_f1",
            "submission.json": '{"output": {"results": []}}',
            "judge_verdicts.md": "1. 123 — Perfectly Relevant",
            "agent_cost_usd": 0.05,
            "other_cost_usd": 0.10,
            "usage": {"m": {"input_tokens": 1}},  # non-str: must be skipped
        }],
        per_example_scores=[0.5],
        num_examples=1,
        mean_score=0.5,
        total_score=0.5,
        aggregate_explanation="",
    )
    fake_evaluator = SimpleNamespace(total_eval_cost=0.05)
    main_mod._write_test_results(
        eval_result, fake_evaluator, tmp_path, "best",
        "test_results.judge_gpt-5.6-luna.json", scoring_mode={},
    )
    pdir = tmp_path / "test_problems.judge_gpt-5.6-luna" / "semantic_9"
    assert (pdir / "submission.json").read_text() == '{"output": {"results": []}}'
    assert "Perfectly Relevant" in (pdir / "judge_verdicts.md").read_text()
    assert not (pdir / "usage").exists()
    row = __import__("json").loads((pdir / "result.json").read_text())
    assert row["score"] == 0.5 and row["sample_id"] == "semantic_9"
    # Stock evals land in the unsuffixed tree.
    main_mod._write_test_results(
        eval_result, fake_evaluator, tmp_path, "best",
        "test_results.json", scoring_mode={},
    )
    assert (tmp_path / "test_problems" / "semantic_9" / "submission.json").exists()


def test_evidence_cap_is_run_immutable_and_training_only():
    """--evidence-char-cap shapes what the judge sees during training →
    run-immutable like the other basis knobs; test evals must clear the
    env (official runs have no cap — the evolved behavior transfers)."""
    resolver = [
        node for node in ast.walk(MAIN_TREE)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_enforce_immutable_on_resume"
        and any(kw.arg == "name" and getattr(kw.value, "value", None) == "evidence-char-cap"
                for kw in node.keywords)
    ]
    assert resolver, "evidence-char-cap must go through _enforce_immutable_on_resume"
    packs = [
        node for node in ast.walk(MAIN_TREE)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "resolved_runtime" for t in node.targets)
    ]
    assert any(getattr(k, "value", None) == "evidence_char_cap" for k in packs[0].value.keys)
    assert "DEFAULT_EVIDENCE_CHAR_CAP = 2500" in MAIN_SRC


def test_test_cache_env_clears_evidence_cap(main_mod, _judge_env, tmp_path, monkeypatch):
    from astabench.evals.paper_finder.relevance import GRADER_MODEL_NAME
    from evaluator import EVIDENCE_CAP_ENV
    monkeypatch.setenv(EVIDENCE_CAP_ENV, "2500")
    main_mod._set_test_cache_env(
        "test_set", runs_dir=str(tmp_path), shared=True, cap=True,
        judge=GRADER_MODEL_NAME, stock=GRADER_MODEL_NAME,
    )
    import os as _os
    assert EVIDENCE_CAP_ENV not in _os.environ, (
        "test evals must be uncapped — they measure the agent as-is"
    )


def test_training_cache_env_names_resolve():
    """_set_training_cache_env is a closure inside main() executed only at
    real launch time — a name it references but main() never imports is a
    NameError that unit tests of the module-level helpers can't catch
    (this exact bug shipped: TRAINING_GRADER_PROMPT_ENV was used but not
    imported, crashing the first full-stack launch). Statically verify
    every evaluator env-constant the closure loads is imported in main()."""
    main_fn = next(n for n in ast.walk(MAIN_TREE)
                   if isinstance(n, ast.FunctionDef) and n.name == "main")
    imported = {
        alias.asname or alias.name
        for node in ast.walk(main_fn)
        if isinstance(node, ast.ImportFrom) and node.module == "evaluator"
        for alias in node.names
    }
    cache_fn = next(n for n in ast.walk(main_fn)
                    if isinstance(n, ast.FunctionDef) and n.name == "_set_training_cache_env")
    used = {
        node.id for node in ast.walk(cache_fn)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        and (node.id.endswith("_ENV"))
    }
    missing = used - imported
    assert not missing, (
        f"_set_training_cache_env references env constants main() never "
        f"imports (NameError at launch): {sorted(missing)}"
    )


def test_judge_prompt_basis_slug_and_filenames(main_mod):
    stock = "openai/gpt-4o-2024-11-20"
    luna = "openai/gpt-5.6-luna"
    assert main_mod._judge_basis_slug(luna, "stock") == main_mod._judge_slug(luna)
    assert main_mod._judge_basis_slug(luna, "no-prose").endswith("_noprose")
    f = main_mod._test_results_filename
    assert f(None, luna, stock, judge_prompt="no-prose") == \
        "test_results.judge_gpt-5.6-luna-noprose.json"
    assert f(None, luna, stock, judge_prompt="stock") == \
        "test_results.judge_gpt-5.6-luna.json"
    assert f(None, stock, stock) == "test_results.json"


def test_test_cache_env_no_prose_basis(main_mod, _judge_env, tmp_path):
    """luna + no-prose is a distinct verdict basis: its own cache file,
    both env vars set, and the profile recorded in scoring_mode."""
    cache_env, cap_env, grader_env = _judge_env
    import os as _os
    from astabench.evals.paper_finder.relevance import GRADER_MODEL_NAME
    from evaluator import TRAINING_GRADER_PROMPT_ENV
    luna = "openai/gpt-5.6-luna"
    mode = main_mod._set_test_cache_env(
        "eval_only", runs_dir=str(tmp_path), shared=True, cap=True,
        judge=luna, stock=GRADER_MODEL_NAME, judge_prompt="no-prose",
    )
    path = Path(_os.environ[cache_env])
    assert path.name == f"shared_test_{main_mod._judge_slug(luna)}_noprose.json"
    assert _os.environ.get(grader_env) == luna
    assert _os.environ.get(TRAINING_GRADER_PROMPT_ENV) == "no-prose"
    assert mode["judge_prompt"] == "no-prose"
    assert "NOT comparable" in mode["judge_note"]


def test_judge_prompt_is_derived_not_settable_and_still_persisted():
    """The judge-prompt profile is a property of the judge, not a knob.

    It could only ever take one correct value per judge (no-prose is
    validated for luna and rejected for gpt-4o), so exposing it added a
    way to mispair the two and nothing else. It still rides
    paper_finder_runtime, because the resolved value scopes the verdict
    cache and the test-result filename, and resume must restore it.
    """
    import ast as _ast
    packs = [
        node for node in _ast.walk(MAIN_TREE)
        if isinstance(node, _ast.Assign)
        and any(getattr(t, "id", None) == "resolved_runtime" for t in node.targets)
    ]
    keys = {getattr(k, "value", None) for k in packs[0].value.keys}
    assert "judge_prompt" in keys, "profile must stay persisted for resume"

    flags = _argparse_flags()
    assert "--judge-prompt" not in flags, "profile is derived, not settable"
    assert "--training-judge" in flags, "the judge itself stays selectable"


def test_prompt_derivation_pairs_each_judge_with_its_validated_profile(main_mod):
    """gpt-4o -> stock, luna -> no-prose, and nothing else is reachable."""
    from astabench.evals.paper_finder.relevance import GRADER_MODEL_NAME

    assert main_mod._prompt_for_judge(GRADER_MODEL_NAME) == "stock"
    assert main_mod._prompt_for_judge("openai/gpt-5.6-luna") == "no-prose"
    # Every selectable judge derives a profile in JUDGE_PROMPT_CHOICES.
    for judge in main_mod.JUDGE_CHOICES:
        assert main_mod._prompt_for_judge(judge) in main_mod.JUDGE_PROMPT_CHOICES


def test_stock_judge_never_derives_no_prose(main_mod):
    """The rejected pairing is unreachable by construction.

    gpt-4o under no-prose inflates its Perfect rate ~18.5%, which is a
    silent scoring error rather than a loud failure -- so the guard that
    used to catch a mispaired flag is now an assertion against a
    hand-edited checkpoint, and the derivation itself must never produce
    that pairing.
    """
    from astabench.evals.paper_finder.relevance import GRADER_MODEL_NAME

    assert main_mod._prompt_for_judge(GRADER_MODEL_NAME) != "no-prose"
    assert "is not a supported basis" in MAIN_SRC, "keep the checkpoint guard"


def test_training_and_test_judges_are_independent_flags():
    """Separate knobs, separate defaults — training cheap, test official.

    They answer different questions: training wants the cheapest basis
    that ranks agents faithfully, a test eval wants the basis the
    leaderboard uses. One flag serving both created an
    explicit-default-is-not-default trap, where `--training-judge <luna>`
    moved the test eval but relying on the same value as a default did
    not.
    """
    flags = _argparse_flags()
    assert "--training-judge" in flags
    assert "--test-judge" in flags
    assert "_DEFAULT_TRAINING_JUDGE = " in MAIN_SRC
    assert "_DEFAULT_TEST_JUDGE = " in MAIN_SRC


def test_test_eval_sites_never_read_the_training_judge():
    """The trap regression test.

    Every test-eval call site must resolve from `test_judge`. If one
    reads `args.training_judge`, stating the training default explicitly
    silently changes what a test score means -- and the score still looks
    like a normal number.
    """
    import ast as _ast

    offenders = []
    for node in _ast.walk(MAIN_TREE):
        if not (isinstance(node, _ast.Call) and getattr(node.func, "id", None)
                in ("_set_test_cache_env", "_test_results_filename")):
            continue
        src = _ast.unparse(node)
        if "training_judge" in src:
            offenders.append(f"line {node.lineno}: {getattr(node.func, 'id')}")
    assert not offenders, (
        "test-eval sites reading the training judge: " + "; ".join(offenders)
    )


def _rendered_help(main_mod) -> str:
    import contextlib, io, sys

    argv = sys.argv
    sys.argv = ["main.py", "--help"]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            main_mod.parse_args()
    except SystemExit:
        pass
    finally:
        sys.argv = argv
    return buf.getvalue()


def test_help_never_prints_default_none(main_mod):
    """"(default: None)" is a contradiction, not information.

    Fourteen flags default to None so the resume path can tell "user
    passed nothing" from "user passed X"; the real default is resolved
    later and stated in prose. Letting ArgumentDefaultsHelpFormatter
    append "(default: None)" prints it directly beneath sentences like
    "Default: openai/gpt-5.6-luna".
    """
    help_text = _rendered_help(main_mod)
    rendered = re.findall(r"\(default: ([^)]*)\)", help_text)
    # Both halves stated over the rendered set rather than any one flag's
    # value: the property is "no default renders as None, and the
    # formatter still renders the real ones", which no particular flag
    # should be able to break by changing its own default.
    assert "None" not in rendered, (
        "a flag's default rendered as None; the real default is resolved "
        "after argparse and stated in prose, so printing None contradicts it"
    )
    assert rendered, (
        "no defaults rendered at all — the formatter suppressed every "
        "default, not just the None ones"
    )


def test_training_judge_help_does_not_claim_test_evals(main_mod):
    """The help is where a user decides which flag to pass, so a stale
    sentence there re-teaches the exact coupling the split removed."""
    help_text = _rendered_help(main_mod)
    # Slice the options block, not the usage banner at the top (which
    # lists every flag name and would otherwise be what we matched).
    options = help_text[help_text.index("\noptions:"):]
    start = options.index("--training-judge {")
    end = options.index("--test-judge {")
    training_help = options[start:end]
    # Redirect rather than exact phrasing: the negatives below would also
    # pass on an EMPTY help string, so something positive is needed — but
    # pinning a sentence would fail on a harmless reword. Pointing at the
    # flag that does govern test evals is the substantive content.
    assert "--test-judge" in training_help, (
        "--training-judge help should point at the flag that governs test evals"
    )
    for stale in ("--eval-test-set", "--eval-only", "internal test evals"):
        assert stale not in training_help, (
            f"--training-judge help still references {stale!r}; it governs "
            "training only since the --test-judge split"
        )


# --- --cost-per-error: dollars or percent-of-threshold ------------------------
#
# The percentage form is a front-end convenience only. These guard the two
# ways it could stop being one: argparse coercing the string away before the
# parser sees it, and the conversion binding to the wrong threshold.


def test_cost_per_error_is_not_coerced_to_float_by_argparse():
    """type=float would reject "10%" at the argparse layer, before
    parse_dollars_or_percent ever runs — the percentage form would be
    dead on arrival with an argparse-generated error."""
    for node in ast.walk(MAIN_TREE):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "add_argument"
            and node.args
            and getattr(node.args[0], "value", None) == "--cost-per-error"
        ):
            types = [kw.value for kw in node.keywords if kw.arg == "type"]
            assert types, "--cost-per-error declares no type"
            assert getattr(types[0], "id", None) == "str", (
                "--cost-per-error must take a str so a percentage survives "
                "argparse; got type=" + ast.unparse(types[0])
            )
            return
    raise AssertionError("--cost-per-error not found in main.py")


def test_percentage_resolves_against_the_resolved_threshold():
    """`of=` must be the resolved cost_threshold, not MIN_COST_THRESHOLD.

    On --resume the threshold in force is the stored one, so binding the
    percentage to the module constant would silently mis-scale the
    penalty for every run that moved --cost-threshold.
    """
    assert "of=cost_threshold" in MAIN_SRC, (
        "--cost-per-error's percentage must resolve against the run's "
        "resolved threshold"
    )
    assert "of=MIN_COST_THRESHOLD" not in MAIN_SRC


def test_default_slope_is_derived_from_the_resolved_threshold():
    """Same property for the default: `default_cost_per_error(cost_threshold)`,
    never a frozen dollar constant."""
    assert "default_value=default_cost_per_error(cost_threshold)" in MAIN_SRC


def test_only_dollars_are_persisted_and_interpolated():
    """The percentage never reaches a checkpoint or an agent-facing doc.

    Both sites must read the resolved float `cost_per_error`, not the raw
    `args.cost_per_error` string.
    """
    assert '"cost_per_error": cost_per_error' in MAIN_SRC
    assert '.replace("${COST_PER_ERROR}", _fmt_cost(cost_per_error))' in MAIN_SRC
    assert "args.cost_per_error" not in MAIN_SRC.split("resolved_runtime = {")[1], (
        "the raw CLI string leaks past the resolution point"
    )


def test_cli_default_matches_the_evaluator_default(main_mod):
    """Guards the explicit-default-is-not-default trap: passing the
    default value by hand must be indistinguishable from omitting it."""
    if str(PFB_DIR) not in sys.path:
        sys.path.insert(0, str(PFB_DIR))
    import evaluator

    from RoboPhD.runner_utils import parse_dollars_or_percent

    for threshold in (0.06, 0.033):
        omitted = evaluator.default_cost_per_error(threshold)
        spelled_out = parse_dollars_or_percent(
            f"{evaluator.COST_PER_ERROR_FRACTION:.0%}",
            of=threshold, flag="cost-per-error",
        )
        assert omitted == spelled_out, (
            f"at threshold {threshold}, omitting --cost-per-error gives "
            f"{omitted} but passing its documented default gives {spelled_out}"
        )


def test_cost_per_error_help_documents_both_forms(main_mod):
    """A user cannot discover the percentage form from anywhere else."""
    help_text = _rendered_help(main_mod)
    options = help_text[help_text.index("\noptions:"):]
    start = options.index("--cost-per-error AMOUNT")
    end = options.index("--max-workers")
    flag_help = options[start:end]
    assert "%" in flag_help, "help never mentions the percentage form"
    assert "--cost-threshold" in flag_help, (
        "help must say what the percentage is a percentage OF"
    )


# --- README vs the constants it documents -------------------------------------
#
# The judge section went stale twice without anything failing: it claimed
# GPT-4o was still the training default months after the flip, and kept
# documenting --judge-prompt as a selectable flag after its removal. Both
# were readable contradictions with the paragraphs directly above them. The
# --help guards below caught neither, because none of it is in --help.


def _readme_row(flag: str) -> str:
    """The judge table's row for `flag`, as written in the README."""
    readme = (PFB_DIR / "README.md").read_text()
    for line in readme.splitlines():
        if line.startswith(f"| `{flag}` |"):
            return line
    raise AssertionError(f"no README judge-table row for {flag}")


@pytest.mark.parametrize("flag,attr", [
    ("--training-judge", "_DEFAULT_TRAINING_JUDGE"),
    ("--test-judge", "_DEFAULT_TEST_JUDGE"),
])
def test_readme_states_the_defaults_the_code_actually_uses(main_mod, flag, attr):
    """Pins the model id to its own row, not the surrounding prose — the
    row is free to be reworded, but it cannot name the wrong judge."""
    expected = getattr(main_mod, attr)
    row = _readme_row(flag)
    assert expected in row, (
        f"README's {flag} row does not name its actual default {expected!r}: {row}"
    )
    stale = [j for j in main_mod.JUDGE_CHOICES if j != expected]
    for other in stale:
        assert other not in row, (
            f"README's {flag} row names {other!r}, which is not its default"
        )


def test_readme_does_not_present_judge_prompt_as_selectable():
    """--judge-prompt was removed in favor of deriving the profile from the
    judge. The README may still *mention* the flag to say it is gone; what
    it must not do is show a value being passed to it."""
    readme = (PFB_DIR / "README.md").read_text()
    usages = re.findall(r"--judge-prompt[ =]+(\S+)", readme)
    assert not usages, (
        "README shows --judge-prompt being passed a value "
        f"({usages}), but main.py derives the profile from the judge"
    )
    assert "--judge-prompt" not in _argparse_flags(), (
        "the flag came back; this guard and the README both need revisiting"
    )


def test_default_test_judge_is_the_official_one(main_mod):
    """An unflagged test eval must land on the leaderboard's basis."""
    from astabench.evals.paper_finder.relevance import GRADER_MODEL_NAME

    assert main_mod._DEFAULT_TEST_JUDGE == GRADER_MODEL_NAME
    assert main_mod._DEFAULT_TRAINING_JUDGE != GRADER_MODEL_NAME, \
        "training default is the cheap basis; if these ever match, the " \
        "cost saving is gone and this test should be the thing that says so"


def test_resume_restores_stored_profile_rather_than_deriving_it():
    """A resume must never re-base an existing campaign.

    Two legacy shapes both have to keep resolving to "stock", and the
    derivation would send both to no-prose if it were allowed to run:

      * checkpoints predating the knob (the feature did not exist, so
        every one of them ran the stock prompt);
      * the one prose-luna campaign (20260721_215631), which ran luna
        under the stock prompt while that pairing was still selectable.

    Both are covered by reading the stored value with a "stock" fallback,
    so this pins that the resume branch does exactly that and does not
    consult _prompt_for_judge.
    """
    import ast as _ast

    fn = next(
        n for n in _ast.walk(MAIN_TREE)
        if isinstance(n, _ast.FunctionDef) and n.name == "main"
    )
    assign = next(
        n for n in _ast.walk(fn)
        if isinstance(n, _ast.If)
        and _ast.unparse(n.test) == "checkpoint_pfb is not None"
        and "judge_prompt" in _ast.unparse(n)
    )
    resume_branch, fresh_branch = _ast.unparse(assign.body), _ast.unparse(assign.orelse)

    assert "checkpoint_pfb.get('judge_prompt', 'stock')" in resume_branch
    assert "_prompt_for_judge" not in resume_branch, \
        "resume must restore the stored basis, never re-derive it"
    assert "_prompt_for_judge" in fresh_branch, \
        "a fresh run derives the profile from its judge"


def test_training_judge_is_run_immutable_and_persisted():
    """--training-judge must ride paper_finder_runtime like the cost knobs:
    resolved via _enforce_immutable_on_resume (so a flagless --resume of a
    luna-trained run hard-errors instead of silently flipping the judge —
    and the cache namespace — back to stock mid-campaign), packed into
    resolved_runtime for checkpoint persistence, and consumed by
    _set_training_cache_env as the RESOLVED value, not the raw CLI flag."""
    # 1. Resolved through the immutability resolver with the stored key.
    resolver_calls = [
        node for node in ast.walk(MAIN_TREE)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_enforce_immutable_on_resume"
        and any(
            kw.arg == "name" and getattr(kw.value, "value", None) == "training-judge"
            for kw in node.keywords
        )
    ]
    assert resolver_calls, "training-judge must go through _enforce_immutable_on_resume"
    stored = next(
        ast.unparse(kw.value) for kw in resolver_calls[0].keywords
        if kw.arg == "stored_value"
    )
    assert "training_judge" in stored and "checkpoint_pfb" in stored

    # 2. Packed into resolved_runtime (→ task_config_extras → checkpoint).
    packs = [
        node for node in ast.walk(MAIN_TREE)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "resolved_runtime" for t in node.targets)
    ]
    assert packs and any(
        getattr(k, "value", None) == "training_judge" for k in packs[0].value.keys
    )

    # 3. The training cache env fn reads the resolved variable, not args.
    cache_fn = next(
        n for n in ast.walk(MAIN_TREE)
        if isinstance(n, ast.FunctionDef) and n.name == "_set_training_cache_env"
    )
    src = ast.unparse(cache_fn)
    assert "args.training_judge" not in src, (
        "_set_training_cache_env must consume the resolved run-immutable "
        "value so resumes restore the run's original judge"
    )


def test_training_judge_choices_pinned(main_mod):
    """The CLI's two allowed judges: element 0 must be astabench's stock
    grader (the default basis), and every choice must be in the
    evaluator's JUDGE_MODEL_IDS (else its spend would misbill to the
    agent and _apply_training_grader would reject it at run time)."""
    from astabench.evals.paper_finder.relevance import GRADER_MODEL_NAME
    from evaluator import JUDGE_MODEL_IDS

    choices = main_mod.JUDGE_CHOICES
    assert choices[0] == GRADER_MODEL_NAME
    assert set(choices) <= JUDGE_MODEL_IDS
    assert "openai/gpt-5.6-luna" in choices


def _calls_to(name: str) -> list[ast.Call]:
    return [
        node for node in ast.walk(MAIN_TREE)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == name
    ]


def test_eval_paths_thread_scoring_mode():
    """Every test call site must pass the resolved cap and the shared-cache
    choice into _set_test_cache_env and thread a scoring-mode record into
    _write_test_results — otherwise recorded scores are ambiguous about the
    basis (capped/shared vs official) that produced them. Per-call-site AST
    checks, so a future third eval path can't slip in unwired."""
    setter_calls = _calls_to("_set_test_cache_env")
    assert len(setter_calls) >= 2, "expected the --eval-only and --eval-test-set call sites"
    for call in setter_calls:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        cap = kwargs.get("cap")
        assert isinstance(cap, ast.Name) and cap.id == "cap_judge_to_estimate", (
            f"line {call.lineno}: _set_test_cache_env must take cap= from the "
            f"resolved cap_judge_to_estimate knob"
        )
        shared = kwargs.get("shared")
        assert (
            isinstance(shared, ast.UnaryOp)
            and isinstance(shared.op, ast.Not)
            and isinstance(shared.operand, ast.Attribute)
            and shared.operand.attr == "no_shared_judge_cache"
        ), (
            f"line {call.lineno}: _set_test_cache_env must take shared= from "
            f"`not args.no_shared_judge_cache`"
        )

    writer_calls = _calls_to("_write_test_results")
    assert len(writer_calls) >= 2, "expected the --eval-only and --eval-test-set call sites"
    for call in writer_calls:
        assert "scoring_mode" in {kw.arg for kw in call.keywords}, (
            f"line {call.lineno}: _write_test_results call misses scoring_mode= — "
            f"this test summary would not record its scoring basis"
        )


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
