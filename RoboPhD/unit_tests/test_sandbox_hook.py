"""Unit tests for utilities.sandbox_hook.

Exercises the hook as a subprocess (which is how Claude CLI actually
calls it) so the stdin/stdout JSON protocol is part of the contract
under test, not just the internal classifier.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HOOK = REPO_ROOT / "utilities" / "sandbox_hook.py"


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def experiment_layout(tmp_path):
    """Build a realistic experiment-dir layout under tmp_path.

    Returns (experiment_dir, cwd, sibling_run, repo_root_stub).
    """
    runs_root = tmp_path / "runs"
    engine = runs_root / "robophd"
    engine.mkdir(parents=True)

    experiment_dir = engine / "task_20260101_000000"
    cwd = experiment_dir / "evolution_output" / "iteration_002"
    cwd.mkdir(parents=True)

    # Existing pieces evolution legitimately reads.
    (experiment_dir / "evolution_output" / "CLAUDE.md").write_text("# CLAUDE.md\n")
    agents_dir = experiment_dir / "agents" / "iter009_some_agent"
    agents_dir.mkdir(parents=True)
    (agents_dir / "agent.py").write_text("# prior agent\n")
    seed_dir = experiment_dir / "agents" / "seed_07w6nmhv"
    seed_dir.mkdir(parents=True)
    (seed_dir / "agent.py").write_text("# seed\n")

    # Sibling run we want to block.
    sibling = engine / "task_20251231_120000"
    sibling_agent = sibling / "agents" / "iter015_x" / "agent.py"
    sibling_agent.parent.mkdir(parents=True)
    sibling_agent.write_text("# do not read\n")

    # Source-repo stub somewhere outside both runs and tmp's experiment.
    repo_stub = tmp_path / "RoboPhD" / "examples" / "asta_ds1000"
    repo_stub.mkdir(parents=True)
    (repo_stub / "model_registry.py").write_text("# private\n")

    return {
        "experiment_dir": experiment_dir,
        "cwd": cwd,
        "sibling_agent": sibling_agent,
        "repo_stub": repo_stub,
        "agents_dir": agents_dir,
        "seed_dir": seed_dir,
    }


def run_hook(envelope: dict, experiment_dir: Path) -> dict:
    """Invoke the hook as a subprocess and return parsed result.

    Returns a dict with keys: rc, stdout, stderr, decision (or None),
    reason (or None).
    """
    env = dict(os.environ)
    env["ROBOPHD_EXPERIMENT_DIR"] = str(experiment_dir)
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(envelope),
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    decision = None
    reason = None
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
            inner = payload.get("hookSpecificOutput", {})
            decision = inner.get("permissionDecision")
            reason = inner.get("permissionDecisionReason")
        except json.JSONDecodeError:
            pass
    return {
        "rc": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "decision": decision,
        "reason": reason,
    }


def make_envelope(tool_name: str, tool_input: dict, cwd: Path) -> dict:
    return {
        "tool_name": tool_name,
        "tool_input": tool_input,
        "cwd": str(cwd),
        "session_id": "test",
        "hook_event_name": "PreToolUse",
    }


# ---------------------------------------------------------------------
# Read tool
# ---------------------------------------------------------------------


def test_read_inside_experiment_dir_allows(experiment_layout):
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"
    res = run_hook(
        make_envelope("Read", {"file_path": str(target)}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["rc"] == 0
    assert res["decision"] is None  # silent allow


def test_read_seed_inside_experiment_dir_allows(experiment_layout):
    layout = experiment_layout
    target = layout["seed_dir"] / "agent.py"
    res = run_hook(
        make_envelope("Read", {"file_path": str(target)}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["rc"] == 0
    assert res["decision"] is None


def test_read_sibling_run_denies(experiment_layout):
    layout = experiment_layout
    res = run_hook(
        make_envelope("Read", {"file_path": str(layout["sibling_agent"])},
                      layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["rc"] == 0
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]
    assert str(layout["sibling_agent"].resolve()) in res["reason"]


def test_read_source_repo_denies(experiment_layout):
    layout = experiment_layout
    target = layout["repo_stub"] / "model_registry.py"
    res = run_hook(
        make_envelope("Read", {"file_path": str(target)}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


# ---------------------------------------------------------------------
# Write tool
# ---------------------------------------------------------------------


def test_write_into_cwd_allows(experiment_layout):
    layout = experiment_layout
    target = layout["cwd"] / "agent.py"
    res = run_hook(
        make_envelope("Write", {"file_path": str(target), "content": "x"},
                      layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["rc"] == 0
    assert res["decision"] is None


def test_edit_into_cwd_allows(experiment_layout):
    layout = experiment_layout
    target = layout["cwd"] / "notes.md"
    target.write_text("a\n")
    res = run_hook(
        make_envelope("Edit",
                      {"file_path": str(target), "old_string": "a", "new_string": "b"},
                      layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] is None


def test_write_into_experiment_dir_outside_cwd_denies(experiment_layout):
    """Writing to a prior agent's dir is inside read scope but outside write scope."""
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"  # write to a prior iteration
    res = run_hook(
        make_envelope("Write", {"file_path": str(target), "content": "x"},
                      layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside write scope" in res["reason"]


# ---------------------------------------------------------------------
# Bash: read/write classification
# ---------------------------------------------------------------------


def test_bash_cat_inside_experiment_allows(experiment_layout):
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"
    res = run_hook(
        make_envelope("Bash", {"command": f"cat {target}"}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] is None


def test_bash_cat_sibling_denies(experiment_layout):
    layout = experiment_layout
    res = run_hook(
        make_envelope("Bash", {"command": f"cat {layout['sibling_agent']}"},
                      layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_bash_find_escape_denies(experiment_layout):
    """`find ../../../` resolves outside the experiment dir."""
    layout = experiment_layout
    res = run_hook(
        make_envelope("Bash", {"command": "find ../../../ -name agent.py"},
                      layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"


def test_bash_find_inside_iteration_allows(experiment_layout):
    layout = experiment_layout
    # Searching ./ from cwd is fine.
    res = run_hook(
        make_envelope("Bash", {"command": "find . -name agent.py"},
                      layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] is None


def test_bash_cp_copyin_pattern_allows(experiment_layout):
    """Routine: cp prior-agent into cwd. Source under read scope, dst under write scope."""
    layout = experiment_layout
    src = layout["agents_dir"] / "agent.py"
    res = run_hook(
        make_envelope("Bash", {"command": f"cp {src} ./agent.py"}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] is None


def test_bash_cp_target_outside_cwd_denies(experiment_layout):
    """cp into a different iteration dir → write target outside cwd."""
    layout = experiment_layout
    src = layout["seed_dir"] / "agent.py"
    dst = layout["agents_dir"] / "agent.py"
    res = run_hook(
        make_envelope("Bash", {"command": f"cp {src} {dst}"}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside write scope" in res["reason"]


def test_bash_rm_prior_iteration_denies(experiment_layout):
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"
    res = run_hook(
        make_envelope("Bash", {"command": f"rm {target}"}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside write scope" in res["reason"]


def test_bash_rm_inside_cwd_allows(experiment_layout):
    layout = experiment_layout
    target = layout["cwd"] / "scratch.txt"
    target.write_text("x")
    res = run_hook(
        make_envelope("Bash", {"command": f"rm {target}"}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] is None


def test_bash_redirect_outside_cwd_denies(experiment_layout):
    layout = experiment_layout
    target = layout["agents_dir"] / "stolen.txt"
    res = run_hook(
        make_envelope("Bash", {"command": f"echo hello > {target}"}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside write scope" in res["reason"]


def test_bash_redirect_into_cwd_allows(experiment_layout):
    layout = experiment_layout
    target = layout["cwd"] / "out.txt"
    res = run_hook(
        make_envelope("Bash", {"command": f"echo hello > {target}"}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] is None


def test_bash_sed_inplace_outside_cwd_denies(experiment_layout):
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"
    res = run_hook(
        make_envelope("Bash", {"command": f"sed -i 's/a/b/' {target}"},
                      layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside write scope" in res["reason"]


def test_bash_sed_inplace_inside_cwd_allows(experiment_layout):
    layout = experiment_layout
    target = layout["cwd"] / "agent.py"
    target.write_text("a\n")
    res = run_hook(
        make_envelope("Bash", {"command": f"sed -i 's/a/b/' {target}"},
                      layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] is None


# ---------------------------------------------------------------------
# Bash: passthrough commands
# ---------------------------------------------------------------------


@pytest.mark.parametrize("cmd", [
    "pwd",
    "whoami",
    "python --version",
    "python3 -V",
    "which python",
    "git status",
    "echo hello world",
    "date",
])
def test_bash_passthrough_no_path_tokens(cmd, experiment_layout):
    layout = experiment_layout
    res = run_hook(
        make_envelope("Bash", {"command": cmd}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] is None, f"{cmd}: {res}"


# ---------------------------------------------------------------------
# Bash: unknown commands
# ---------------------------------------------------------------------


def test_bash_unknown_cmd_with_path_denies(experiment_layout):
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"
    res = run_hook(
        make_envelope("Bash", {"command": f"frobnicate {target}"}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "not in classifier" in res["reason"]


def test_bash_unknown_cmd_no_path_allows(experiment_layout):
    layout = experiment_layout
    res = run_hook(
        make_envelope("Bash", {"command": "frobnicate --help"}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] is None


# ---------------------------------------------------------------------
# Symlink bypass
# ---------------------------------------------------------------------


def test_symlink_bypass_blocked_via_realpath(experiment_layout):
    """An agent that creates a symlink inside cwd to a sibling-run file
    must still be blocked when reading through the alias."""
    layout = experiment_layout
    alias = layout["cwd"] / "alias_agent.py"
    alias.symlink_to(layout["sibling_agent"])
    res = run_hook(
        make_envelope("Read", {"file_path": str(alias)}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


# ---------------------------------------------------------------------
# Denial log
# ---------------------------------------------------------------------


def test_denial_appends_to_log(experiment_layout):
    layout = experiment_layout
    log_path = layout["experiment_dir"] / "sandbox_denials.jsonl"
    res = run_hook(
        make_envelope("Read", {"file_path": str(layout["sibling_agent"])},
                      layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["tool"] == "Read"
    assert rec["scope"] == "read"
    assert rec["blocked_path"] == str(layout["sibling_agent"].resolve())
    assert "ts" in rec


def test_allow_does_not_append_to_log(experiment_layout):
    layout = experiment_layout
    log_path = layout["experiment_dir"] / "sandbox_denials.jsonl"
    target = layout["agents_dir"] / "agent.py"
    res = run_hook(
        make_envelope("Read", {"file_path": str(target)}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] is None
    assert not log_path.exists()


# ---------------------------------------------------------------------
# Hook errors
# ---------------------------------------------------------------------


def test_missing_env_var_fails_closed(experiment_layout):
    """Hook without ROBOPHD_EXPERIMENT_DIR must exit 2."""
    layout = experiment_layout
    env = dict(os.environ)
    env.pop("ROBOPHD_EXPERIMENT_DIR", None)
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(make_envelope("Read", {"file_path": "/etc/passwd"},
                                       layout["cwd"])),
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert proc.returncode == 2


def test_bad_stdin_fails_closed(experiment_layout):
    """Garbage on stdin must exit 2, not silently allow."""
    layout = experiment_layout
    env = dict(os.environ)
    env["ROBOPHD_EXPERIMENT_DIR"] = str(layout["experiment_dir"])
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input="not json{",
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert proc.returncode == 2
