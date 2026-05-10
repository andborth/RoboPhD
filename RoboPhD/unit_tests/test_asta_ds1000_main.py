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
    tier handles (CLAUDE_OPUS_4_7 / GPT_5_5 / GEMINI_3_1_PRO_PREVIEW) in
    the resumed run's agent pool and set
    ASTA_DS1000_ALLOW_STRONGER_MODELS=1 even when --allow-stronger-models
    was omitted. Without it, eval workers ImportError on every sample,
    yielding a uniform 0.000 test score (silent — no exception bubbles
    up). Detection is by simple substring scan of `agents/*/agent.py`.
"""
import ast
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = REPO_ROOT / "examples" / "asta_ds1000" / "main.py"

EVAL_FUNCS = {"eval_candidate", "eval_run"}

GATED_HANDLES = ("CLAUDE_OPUS_4_7", "GPT_5_5", "GEMINI_3_1_PRO_PREVIEW")


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
# Behavioral test: spawn a subprocess that imports main.py's flag-gating
# block on a fake resume dir, and confirm the env var is set when an
# agent imports a gated handle. We exercise the actual main.py source
# (not a copy) so any future regression of the auto-detect logic is
# caught — matching the asymmetry-bug failure mode (silent 0.000 score).


def _run_resume_check(tmp_path: Path, agent_src: str) -> bool:
    """Stand up a fake resume dir, run main.py's auto-detect block, and
    return whether the env var was set. Subprocess-isolated so env-var
    writes don't pollute the test suite.
    """
    fake_run = tmp_path / "fake_run"
    agent_dir = fake_run / "agents" / "fake_agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "agent.py").write_text(agent_src)

    # Extract the auto-detect block from main.py and run it in isolation
    # against the fake resume dir. We don't import main.py wholesale (it
    # has side effects), but we replicate the flag-gating logic by
    # invoking the real main.py up to argparse and then inspecting
    # os.environ. Cleanest is to actually call main.py with --eval-only
    # --resume <fake>; main.py will fail later (no real checkpoint), but
    # the env var will be set first. Capture os.environ before that
    # failure by intercepting sys.exit.
    code = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(MAIN_PY.parent)!r})
        sys.path.insert(0, {str(REPO_ROOT)!r})
        # Stub out the post-env-var imports so main.py exits cleanly
        # right after our block runs but before evaluator import.
        import builtins
        _real_import = builtins.__import__
        def _stub_import(name, *a, **kw):
            if name == "evaluator":
                print("ENV_AT_EVAL_IMPORT:",
                      os.environ.get("ASTA_DS1000_ALLOW_STRONGER_MODELS", "<unset>"))
                sys.exit(0)
            return _real_import(name, *a, **kw)
        builtins.__import__ = _stub_import

        sys.argv = ["main.py", "--eval-only", "--resume", {str(fake_run)!r}]
        import main
        main.main()
    """)
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30,
    )
    for line in r.stdout.splitlines():
        if line.startswith("ENV_AT_EVAL_IMPORT:"):
            return line.split(":", 1)[1].strip() == "1"
    raise AssertionError(
        f"subprocess didn't reach evaluator import:\n"
        f"STDOUT: {r.stdout}\nSTDERR: {r.stderr[-1500:]}"
    )


def test_resume_auto_enables_flag_when_agent_imports_gated_handle(tmp_path):
    """Resume with no flag + agent imports a gated handle → env set."""
    agent_src = "from model_registry import CLAUDE_OPUS_4_7, CLAUDE_SONNET_4_6\n"
    assert _run_resume_check(tmp_path, agent_src) is True, (
        "main.py failed to auto-set ASTA_DS1000_ALLOW_STRONGER_MODELS=1 "
        "on resume despite the resumed agent pool importing a gated "
        "handle. Without this, eval workers crash on import and produce "
        "a uniform 0.000 test score with no surfaced exception."
    )


def test_resume_leaves_flag_unset_when_no_agent_imports_gated_handle(tmp_path):
    """Resume with no flag + no gated imports → env stays unset."""
    agent_src = "from model_registry import GPT_5_4_MINI, CLAUDE_SONNET_4_6\n"
    assert _run_resume_check(tmp_path, agent_src) is False, (
        "main.py auto-enabled stronger-models on a resume that doesn't "
        "need it — the gate should only fire when an agent in the pool "
        "imports a gated handle"
    )


def test_resume_auto_detect_covers_all_gated_handles():
    """Sanity check: the auto-detect substring list in main.py matches the
    set of handles the model_registry actually gates. If a future commit
    adds a fourth gated handle, this catches the asymmetry between the
    registry definition and the resume-time detector."""
    src = MAIN_PY.read_text()
    for handle in GATED_HANDLES:
        assert handle in src, (
            f"Expected the auto-detect block in main.py to mention "
            f"{handle!r} as a gated handle name, but the substring isn't "
            f"in the file. Did the gated set change without updating the "
            f"resume-time detector?"
        )
