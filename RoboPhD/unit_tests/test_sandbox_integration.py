"""Integration tests for the evolution sandbox wiring.

The unit suite in ``test_sandbox_hook.py`` invokes the hook script
directly with a hand-crafted env. These tests instead exercise the full
chain that's bitten us in production:

  build_evolution_env  →  ROBOPHD_EXPERIMENT_DIR
  _install_evolution_sandbox  →  <exp>/.claude/settings.local.json
  install_iteration_sandbox  →  <exp>/<iter>/.claude/settings.local.json
  shell-spawn the command from settings.local.json  →  hook process

Past bugs that would have surfaced here, and didn't because the unit
tests stop short of the real wiring:

  * ``f75228e`` — ``extra_env`` was ``None`` for Anthropic models, so
    ``ROBOPHD_EXPERIMENT_DIR`` never reached the hook subprocess.
    Caught by parametrizing over an Anthropic model name.
  * ``aa548a4`` — ``self.experiment_dir`` was a relative path, the env
    var inherited that relativity, and the hook resolved it against
    its own cwd (= iteration dir) into a bogus location. Caught by
    constructing experiment_dir as a relative Path under ``tmp_path``.
  * ``1b5f9da`` — denials silently routed to ``/tmp`` instead of the
    experiment-dir log because the env-pointed dir didn't exist.
    Caught by asserting the denial record lands in
    ``<experiment_dir>/sandbox_denials.jsonl``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from RoboPhD.config import build_evolution_env
from RoboPhD.researcher import _install_evolution_sandbox, install_iteration_sandbox


@pytest.fixture
def wired_experiment(tmp_path, monkeypatch):
    """Build a realistic experiment + iteration layout and wire the sandbox.

    ``experiment_dir`` is intentionally constructed as a *relative* path
    to mirror ``researcher.py``'s ``runs_dir = Path('../robophd_runs')``.
    ``build_evolution_env`` is responsible for resolving it to absolute;
    if it doesn't, the hook breaks (= regression of ``aa548a4``).
    """
    monkeypatch.chdir(tmp_path)
    rel_exp = Path("runs") / "robophd" / "task_20260506_000000"
    rel_exp.mkdir(parents=True)
    iteration = rel_exp / "evolution_output" / "iteration_001"
    iteration.mkdir(parents=True)

    _install_evolution_sandbox(rel_exp, extra_read_paths=None)
    install_iteration_sandbox(iteration, rel_exp)

    return {
        "rel_exp": rel_exp,
        "abs_exp": rel_exp.resolve(),
        "iteration": iteration.resolve(),
    }


def _hook_command_from_settings(settings_dir: Path) -> str:
    """Read the hook command string Claude CLI would actually run."""
    payload = json.loads((settings_dir / "settings.local.json").read_text())
    return payload["hooks"]["PreToolUse"][0]["hooks"][0]["command"]


def _run_hook_via_settings(setup: dict, model: str, envelope: dict):
    """Run the hook the way Claude CLI does: shell-spawn the settings
    command, env from build_evolution_env, cwd = iteration dir."""
    env = dict(os.environ)
    env.update(build_evolution_env(model, setup["rel_exp"]))
    cmd = _hook_command_from_settings(setup["iteration"] / ".claude")
    proc = subprocess.run(
        cmd, shell=True,
        input=json.dumps(envelope),
        capture_output=True, text=True, env=env, timeout=10,
        cwd=str(setup["iteration"]),
    )
    decision = None
    if proc.stdout.strip():
        try:
            decision = (
                json.loads(proc.stdout)
                .get("hookSpecificOutput", {})
                .get("permissionDecision")
            )
        except json.JSONDecodeError:
            pass
    return proc, decision


@pytest.mark.parametrize("model", ["opus-4.7", "qwen/qwen3-coder-30b"])
def test_in_scope_read_allowed_end_to_end(wired_experiment, model):
    """Full chain: env-var built absolute, settings command parses, hook
    fires, in-scope read is allowed.

    Parametrized so an Anthropic model (which makes
    ``get_lmstudio_env`` return ``None``) and a non-Anthropic model
    both flow through the env-build helper correctly.
    """
    seed = wired_experiment["abs_exp"] / "agents" / "seed" / "agent.py"
    seed.parent.mkdir(parents=True)
    seed.write_text("# seed\n")
    envelope = {
        "tool_name": "Read",
        "tool_input": {"file_path": str(seed)},
        "cwd": str(wired_experiment["iteration"]),
    }
    proc, decision = _run_hook_via_settings(wired_experiment, model, envelope)
    assert proc.returncode == 0, (
        f"hook errored. stderr={proc.stderr!r}"
    )
    assert decision != "deny", (
        f"in-scope read was denied. stderr={proc.stderr!r}"
    )


def test_env_var_resolves_to_absolute(wired_experiment):
    """``build_evolution_env`` must canonicalize relative experiment dirs.

    The hook resolves ``ROBOPHD_EXPERIMENT_DIR`` via realpath against
    its own cwd; a relative env value points at a wrong location.
    """
    extra = build_evolution_env("opus-4.7", wired_experiment["rel_exp"])
    assert os.path.isabs(extra["ROBOPHD_EXPERIMENT_DIR"])
    assert (
        Path(extra["ROBOPHD_EXPERIMENT_DIR"])
        == wired_experiment["abs_exp"]
    )


def test_iteration_dir_env_set_when_passed(wired_experiment):
    """When the harness passes iteration_dir, build_evolution_env
    sets ROBOPHD_ITERATION_DIR (absolute). The sandbox hook reads
    this to anchor write scope at the iteration root instead of the
    runtime cwd."""
    # Pass a relative iteration_dir to mirror the experiment_dir
    # relative-path pattern; the helper must resolve to absolute.
    rel_iter = wired_experiment["rel_exp"] / "evolution_output" / "iteration_001"
    extra = build_evolution_env(
        "opus-4.7", wired_experiment["rel_exp"], iteration_dir=rel_iter,
    )
    assert "ROBOPHD_ITERATION_DIR" in extra
    assert os.path.isabs(extra["ROBOPHD_ITERATION_DIR"])
    assert Path(extra["ROBOPHD_ITERATION_DIR"]) == wired_experiment["iteration"]


def test_iteration_dir_env_unset_by_default(wired_experiment):
    """When iteration_dir is omitted, ROBOPHD_ITERATION_DIR is NOT
    set — preserves the legacy fallback (write scope = cwd) for
    callers that haven't migrated."""
    extra = build_evolution_env("opus-4.7", wired_experiment["rel_exp"])
    assert "ROBOPHD_ITERATION_DIR" not in extra


def test_denial_lands_in_experiment_dir_not_tmp_fallback(wired_experiment):
    """A real denial routes to ``<experiment_dir>/sandbox_denials.jsonl``.

    The hook's ``append_denial_record`` falls back to
    ``/tmp/robophd_sandbox_denials.jsonl`` when the env-pointed dir
    isn't found. If the wiring is broken (env var missing or pointing
    nowhere) every denial silently goes to /tmp, where nothing tails
    it. Asserting the experiment-dir log gets the record proves the
    happy path.
    """
    sibling = (
        wired_experiment["abs_exp"].parent / "other_run" / "agent.py"
    )
    sibling.parent.mkdir(parents=True)
    sibling.write_text("# out of read scope\n")
    envelope = {
        "tool_name": "Read",
        "tool_input": {"file_path": str(sibling)},
        "cwd": str(wired_experiment["iteration"]),
    }
    proc, decision = _run_hook_via_settings(
        wired_experiment, "opus-4.7", envelope
    )
    assert decision == "deny", f"expected deny. stderr={proc.stderr!r}"
    primary_log = wired_experiment["abs_exp"] / "sandbox_denials.jsonl"
    assert primary_log.exists(), (
        "denial did not reach the experiment-dir log; likely "
        "fell back to /tmp (regression of 1b5f9da)"
    )
    records = [
        json.loads(line)
        for line in primary_log.read_text().splitlines()
        if line.strip()
    ]
    assert any(r.get("scope") == "read" for r in records)
