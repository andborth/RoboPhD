"""Static invariant: the Autoresearch engine must never launch a Claude CLI
session without the workspace sandbox wired in.

The sandbox *hook* is exhaustively covered by ``test_sandbox_hook.py`` and
``test_sandbox_integration.py``. What those cannot catch is the engine
silently ceasing to *activate* it — a cleanup pass that drops
``extra_env=sandbox_env`` from a ``call_claude_cli`` call, reverts
``--settings`` to a bare ``{"autoCompact": True}`` dict, or removes the
``install_evolution_sandbox`` call reopens the exact unrestricted-repo-read
hole (agent reads the simulator source, reconstructs the evaluator, and
bypasses the evaluation budget — see the paper's Ethics Statement) with
every hook test still green.

This test parses ``engines/autoresearch.py`` and asserts, over EVERY Claude
CLI launch site in ``run_autoresearch`` (both ``call_claude_cli`` and the
``subprocess.run`` fallbacks, for both the main and reflection sessions):

  * ``install_evolution_sandbox(...)`` is called (the PreToolUse hook).
  * every ``call_claude_cli(...)`` passes ``extra_env=sandbox_env``.
  * every ``subprocess.run(...)`` passes ``env=`` that references ``sandbox_env``.
  * every ``--settings`` value in a launch command is ``session_settings``
    (i.e. ``claude_cli_settings()``), never an inline dict.

Framed as "no launch site is unsandboxed", so a NEW launch site added later
fails this test by default rather than shipping unguarded.
"""

import ast
from pathlib import Path

import pytest

AUTORESEARCH_SRC = (
    Path(__file__).resolve().parents[1] / "engines" / "autoresearch.py"
)
SANDBOX_ENV_NAME = "sandbox_env"
SETTINGS_NAME = "session_settings"


def _run_autoresearch_node() -> ast.FunctionDef:
    tree = ast.parse(AUTORESEARCH_SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_autoresearch":
            return node
    raise AssertionError("run_autoresearch not found in engines/autoresearch.py")


def _calls_named(node: ast.AST, name: str) -> list:
    """All ast.Call nodes under `node` whose callee is the bare name `name`."""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name:
            out.append(n)
    return out


def _subtree_references_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(n, ast.Name) and n.id == name for n in ast.walk(node)
    )


def _keyword(call: ast.Call, arg: str):
    for kw in call.keywords:
        if kw.arg == arg:
            return kw
    return None


@pytest.fixture(scope="module")
def run_fn():
    return _run_autoresearch_node()


def test_sandbox_hook_is_installed(run_fn):
    assert _calls_named(run_fn, "install_evolution_sandbox"), (
        "run_autoresearch no longer calls install_evolution_sandbox — the "
        "workspace read-scope hook is not installed and the session runs with "
        "unrestricted filesystem access."
    )


def test_every_call_claude_cli_passes_sandbox_env(run_fn):
    calls = _calls_named(run_fn, "call_claude_cli")
    assert calls, "expected at least one call_claude_cli launch site"
    for call in calls:
        kw = _keyword(call, "extra_env")
        assert kw is not None, (
            f"a call_claude_cli launch (line {call.lineno}) is missing "
            f"extra_env — the sandbox env vars won't reach the CLI subprocess."
        )
        assert isinstance(kw.value, ast.Name) and kw.value.id == SANDBOX_ENV_NAME, (
            f"call_claude_cli (line {call.lineno}) passes extra_env but not the "
            f"{SANDBOX_ENV_NAME} dict — a wrong/empty env silently reopens the hole."
        )


def test_every_subprocess_run_passes_sandbox_env(run_fn):
    # subprocess.run inside run_autoresearch is only ever the Claude CLI
    # fallback path; the git subprocess.run calls live in _setup_workspace,
    # a different function, so scoping to run_fn excludes them.
    runs = [
        n for n in ast.walk(run_fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "run"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "subprocess"
    ]
    assert runs, "expected at least one subprocess.run Claude CLI fallback"
    for call in runs:
        kw = _keyword(call, "env")
        assert kw is not None, (
            f"a subprocess.run Claude launch (line {call.lineno}) is missing "
            f"env — the sandbox env vars won't reach the CLI subprocess."
        )
        assert _subtree_references_name(kw.value, SANDBOX_ENV_NAME), (
            f"subprocess.run (line {call.lineno}) passes env but it does not "
            f"reference {SANDBOX_ENV_NAME}."
        )


def test_settings_arg_is_the_hardened_settings(run_fn):
    """Every '--settings' in a launch command list is followed by
    session_settings (claude_cli_settings()), never an inline dict — a revert
    to {"autoCompact": True} would drop the Read(/<repo>/**) deny."""
    checked = 0
    for lst in ast.walk(run_fn):
        if not isinstance(lst, ast.List):
            continue
        elts = lst.elts
        for i, el in enumerate(elts[:-1]):
            if isinstance(el, ast.Constant) and el.value == "--settings":
                nxt = elts[i + 1]
                assert isinstance(nxt, ast.Name) and nxt.id == SETTINGS_NAME, (
                    f"a --settings arg (line {getattr(nxt, 'lineno', lst.lineno)}) "
                    f"is not {SETTINGS_NAME}; the repo-read deny may be gone."
                )
                checked += 1
    assert checked >= 1, "no '--settings' arg found in any launch command list"


def test_settings_source_is_claude_cli_settings(run_fn):
    assert _calls_named(run_fn, "claude_cli_settings"), (
        "run_autoresearch no longer calls claude_cli_settings() — the "
        "autoCompact + Read(/<repo>/**) deny settings are gone."
    )
