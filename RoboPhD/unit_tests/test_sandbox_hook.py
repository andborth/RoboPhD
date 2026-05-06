"""Unit tests for utilities.sandbox_hook.

Exercises the hook as a subprocess (which is how Claude CLI actually
calls it) so the stdin/stdout JSON protocol is part of the contract
under test, not just the internal classifier.

**Integration gap, by design:** these tests invoke the hook script
directly. They do NOT verify that Claude CLI actually loads
.claude/settings.local.json from a given location — that's a
property of Claude CLI's settings-resolution behavior, which is
external. Past wiring bugs (e.g., assuming Claude CLI walks up from
cwd to find settings) passed all unit tests while the sandbox was
silently no-op in production. After any change to
_install_evolution_sandbox / install_iteration_sandbox / the
hook-command construction in researcher.py, run an end-to-end smoke
against the real CLI as documented in the sandbox memory entry.
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


def run_hook(envelope: dict, experiment_dir: Path,
             extra_reads: list = None) -> dict:
    """Invoke the hook as a subprocess and return parsed result.

    Returns a dict with keys: rc, stdout, stderr, decision (or None),
    reason (or None).
    """
    env = dict(os.environ)
    env["ROBOPHD_EXPERIMENT_DIR"] = str(experiment_dir)
    cmd = [sys.executable, str(HOOK)]
    for p in extra_reads or []:
        cmd.append(f"--extra-read={p}")
    proc = subprocess.run(
        cmd,
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
# dd: if=src is read, of=dst is write
# ---------------------------------------------------------------------


def test_bash_dd_legitimate_copy_inside_scope_allows(experiment_layout):
    """dd if=<read-scope-path> of=<cwd-path> copies legitimately."""
    layout = experiment_layout
    src = layout["agents_dir"] / "agent.py"
    dst = layout["cwd"] / "copy.py"
    res = run_hook(
        make_envelope("Bash", {"command": f"dd if={src} of={dst} bs=4096"},
                      layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] is None, res


def test_bash_dd_read_source_outside_read_scope_denies(experiment_layout):
    """dd if=<sibling-run> of=<cwd> — source is a read-scope violation."""
    layout = experiment_layout
    dst = layout["cwd"] / "copy.py"
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": f"dd if={layout['sibling_agent']} of={dst}"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_bash_dd_write_target_outside_cwd_denies(experiment_layout):
    """dd if=<read-scope> of=<outside-cwd> — target is a write-scope violation."""
    layout = experiment_layout
    src = layout["agents_dir"] / "agent.py"
    dst = layout["agents_dir"] / "stolen.py"  # in read scope, NOT in write scope
    res = run_hook(
        make_envelope("Bash", {"command": f"dd if={src} of={dst}"},
                      layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside write scope" in res["reason"]


# ---------------------------------------------------------------------
# find -exec / xargs: bypass closure
# ---------------------------------------------------------------------


def test_bash_find_exec_fails_closed(experiment_layout):
    """find -exec invokes commands per-match — those subprocesses
    bypass per-command path classification. Treat as unknown-with-paths."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": "find . -name agent.py -exec cat {} ;"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "not in classifier" in res["reason"]


@pytest.mark.parametrize("variant", ["-exec", "-execdir", "-ok", "-okdir"])
def test_bash_find_exec_variants_all_fail_closed(variant, experiment_layout):
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": f"find . -name '*.py' {variant} cat {{}} ;"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny", f"{variant}: {res}"


def test_bash_find_without_exec_still_allowed(experiment_layout):
    """Plain find without -exec is fine and stays in BASH_READ_COMMANDS."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash", {"command": "find . -name agent.py"}, layout["cwd"]
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] is None


def test_bash_xargs_fails_closed(experiment_layout):
    """xargs reads paths from stdin — bypasses per-command classification."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": "find . -name '*.sqlite' | xargs cat"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "not in classifier" in res["reason"]


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


# ---------------------------------------------------------------------
# Per-task read carve-outs (--extra-read=PATH)
# ---------------------------------------------------------------------


@pytest.fixture
def carveout_layout(tmp_path):
    """Layout with an external resource dir + a symlink from inside the
    experiment dir to it. Models text2sql's BIRD database setup.
    """
    runs_root = tmp_path / "runs"
    engine = runs_root / "robophd"
    engine.mkdir(parents=True)
    experiment_dir = engine / "task_20260101_000000"
    cwd = experiment_dir / "evolution_output" / "iteration_002"
    cwd.mkdir(parents=True)

    # External resource (mirrors RoboPhD/benchmark_resources/...)
    resource_root = tmp_path / "resources"
    resource_root.mkdir()
    real_db = resource_root / "talkingdata.sqlite"
    real_db.write_text("BIRD database stub\n")

    # Per-problem dir with symlink mirroring the text2sql layout.
    problem_dir = experiment_dir / "iteration_002" / "agent_x" / "problems" / "452"
    problem_dir.mkdir(parents=True)
    db_link = problem_dir / "database.sqlite"
    db_link.symlink_to(real_db)

    # A sibling run that should still be blocked even with the carve-out.
    sibling = engine / "task_20251231_120000"
    sibling_agent = sibling / "agents" / "iter1" / "agent.py"
    sibling_agent.parent.mkdir(parents=True)
    sibling_agent.write_text("# sibling\n")

    return {
        "experiment_dir": experiment_dir,
        "cwd": cwd,
        "resource_root": resource_root,
        "real_db": real_db,
        "db_link": db_link,
        "problem_dir": problem_dir,
        "sibling_agent": sibling_agent,
    }


def test_carveout_allows_read_via_symlink(carveout_layout):
    """Read of a symlink whose target lives under a carved-out root."""
    L = carveout_layout
    res = run_hook(
        make_envelope("Read", {"file_path": str(L["db_link"])}, L["cwd"]),
        L["experiment_dir"],
        extra_reads=[str(L["resource_root"])],
    )
    assert res["decision"] is None, res


def test_carveout_allows_bash_through_symlink(carveout_layout):
    """Bash cat through a symlinked-in resource — the text2sql case."""
    L = carveout_layout
    res = run_hook(
        make_envelope("Bash", {"command": f"cat {L['db_link']}"}, L["cwd"]),
        L["experiment_dir"],
        extra_reads=[str(L["resource_root"])],
    )
    assert res["decision"] is None, res


def test_carveout_allows_direct_read_of_resource(carveout_layout):
    """Reading the resource directly by absolute path is also OK."""
    L = carveout_layout
    res = run_hook(
        make_envelope("Read", {"file_path": str(L["real_db"])}, L["cwd"]),
        L["experiment_dir"],
        extra_reads=[str(L["resource_root"])],
    )
    assert res["decision"] is None, res


def test_carveout_does_not_grant_writes(carveout_layout):
    """A read carve-out must NOT grant writes into the same path."""
    L = carveout_layout
    target = L["resource_root"] / "stolen.txt"
    res = run_hook(
        make_envelope("Write", {"file_path": str(target), "content": "x"},
                      L["cwd"]),
        L["experiment_dir"],
        extra_reads=[str(L["resource_root"])],
    )
    assert res["decision"] == "deny"
    assert "outside write scope" in res["reason"]


def test_carveout_does_not_unblock_sibling_runs(carveout_layout):
    """Carve-outs are additive — sibling runs stay denied."""
    L = carveout_layout
    res = run_hook(
        make_envelope("Read", {"file_path": str(L["sibling_agent"])}, L["cwd"]),
        L["experiment_dir"],
        extra_reads=[str(L["resource_root"])],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_no_carveout_blocks_resource(carveout_layout):
    """Without the carve-out, the symlinked-in DB is denied."""
    L = carveout_layout
    res = run_hook(
        make_envelope("Bash", {"command": f"cat {L['db_link']}"}, L["cwd"]),
        L["experiment_dir"],
        # extra_reads omitted intentionally
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_multiple_carveouts(carveout_layout, tmp_path):
    """Hook accepts repeated --extra-read args."""
    L = carveout_layout
    extra2 = tmp_path / "extra"
    extra2.mkdir()
    extra_db = extra2 / "another.sqlite"
    extra_db.write_text("x")
    res = run_hook(
        make_envelope("Read", {"file_path": str(extra_db)}, L["cwd"]),
        L["experiment_dir"],
        extra_reads=[str(L["resource_root"]), str(extra2)],
    )
    assert res["decision"] is None, res


def test_deny_message_lists_extra_roots(carveout_layout):
    """When extra-reads exist, the denial reason should hint at them."""
    L = carveout_layout
    res = run_hook(
        make_envelope("Read", {"file_path": str(L["sibling_agent"])}, L["cwd"]),
        L["experiment_dir"],
        extra_reads=[str(L["resource_root"])],
    )
    assert res["decision"] == "deny"
    assert "task-specific resource roots" in res["reason"]
    assert str(L["resource_root"].resolve()) in res["reason"]
