"""Pin asta_ds1000 main.py invariants that the type system can't enforce.

Currently covers two bug classes:

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
