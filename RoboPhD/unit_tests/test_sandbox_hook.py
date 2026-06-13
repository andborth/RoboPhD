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

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from utilities.sandbox_hook import (  # noqa: E402
    project_slug,
    auto_memory_dir,
    auto_session_dirs,
    split_statement_newlines,
    evaluate_bash,
    replay_denial_record,
)


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
             extra_reads: list = None,
             iteration_dir: Path = None) -> dict:
    """Invoke the hook as a subprocess and return parsed result.

    Returns a dict with keys: rc, stdout, stderr, decision (or None),
    reason (or None).

    When ``iteration_dir`` is None, ROBOPHD_EVOLUTION_ITERATION_DIR is NOT set
    in the env and the hook falls back to cwd-rooted write scope
    (the legacy behavior). Existing tests use this default to pin
    backward compat. Pass ``iteration_dir`` to exercise the new
    iteration-rooted write-scope behavior.
    """
    env = dict(os.environ)
    env["ROBOPHD_EXPERIMENT_DIR"] = str(experiment_dir)
    # Default: do NOT inherit a stray ROBOPHD_EVOLUTION_ITERATION_DIR from the
    # outer process env — tests must opt in by passing iteration_dir.
    env.pop("ROBOPHD_EVOLUTION_ITERATION_DIR", None)
    if iteration_dir is not None:
        env["ROBOPHD_EVOLUTION_ITERATION_DIR"] = str(iteration_dir)
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
# Run-scoped Claude auto-memory carve-out
#
# Claude Code's built-in memory feature writes/recalls under
# ~/.claude/projects/<project-slug>/memory/. The hook grants exactly
# this one run's <slug>/memory/ (read + Write-tool write), for both the
# raw-env and realpath slug forms (symlinked-root robustness).
#
# Containment is NOT structural (project_slug is lossy); it holds
# because experiment-dir names are timestamp-unique and RoboPhD owns
# EXPERIMENT_DIR. These tests lock down BOTH sides: a genuinely
# different slug is denied (sibling/source-repo), AND the lossy-slug
# collision boundary is pinned explicitly so the known limitation is
# fenced rather than merely asserted. See sandbox_hook.auto_memory_dir
# for why this is fragile-but-fail-safe and only an interim measure.
# ---------------------------------------------------------------------


def _memory_path(experiment_dir: Path, name: str = "insight.md") -> str:
    """The carve-out path the hook computes for this experiment dir."""
    slug = project_slug(os.path.realpath(str(experiment_dir)))
    return os.path.join(
        os.path.expanduser("~"), ".claude", "projects", slug, "memory", name
    )


def test_project_slug_ground_truth():
    """Pin the path->slug transform against OBSERVED Claude Code output.

    If Claude Code changes its slug transform this fails loudly HERE
    (our-bug surface). That is deliberately distinct from an anchor
    drift, which fails safe as an ordinary runtime deny recorded in
    sandbox_denials.jsonl.
    """
    assert project_slug(
        "/Users/a/Desktop/cc/robophd_runs/robophd/asta_ds1000_20260514_231614"
    ) == "-Users-a-Desktop-cc-robophd-runs-robophd-asta-ds1000-20260514-231614"
    # Case preserved; '_' folds to '-'.
    assert project_slug("/Users/a/Desktop/cc/RoboPhD") == "-Users-a-Desktop-cc-RoboPhD"


def test_auto_memory_write_into_run_slug_allows(experiment_layout):
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Write",
            {"file_path": _memory_path(layout["experiment_dir"]), "content": "x"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["rc"] == 0
    assert res["decision"] is None  # silent allow, like a cwd write


def test_auto_memory_read_into_run_slug_allows(experiment_layout):
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Read",
            {"file_path": _memory_path(layout["experiment_dir"])},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["rc"] == 0
    assert res["decision"] is None


def test_auto_memory_write_sibling_run_slug_denies(experiment_layout):
    """A memory write whose slug is a DIFFERENT run must NOT escape."""
    layout = experiment_layout
    sibling_dir = layout["experiment_dir"].parent / "task_20251231_120000"
    slug = project_slug(os.path.realpath(str(sibling_dir)))
    target = os.path.join(
        os.path.expanduser("~"), ".claude", "projects", slug, "memory", "x.md"
    )
    res = run_hook(
        make_envelope("Write", {"file_path": target, "content": "x"},
                      layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside write scope" in res["reason"]


def test_auto_memory_write_source_repo_slug_denies(experiment_layout):
    """A memory write slugged to the source repo must NOT escape."""
    layout = experiment_layout
    slug = project_slug("/Users/a/Desktop/cc/RoboPhD")
    target = os.path.join(
        os.path.expanduser("~"), ".claude", "projects", slug, "memory", "evil.md"
    )
    res = run_hook(
        make_envelope("Write", {"file_path": target, "content": "x"},
                      layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"


def test_auto_memory_bash_write_still_denied(experiment_layout):
    """Bash write path deliberately does NOT honor the carve-out.

    Only Claude's legitimate Write-tool auto-memory is allowed; an
    injection using `bash -c '... > ~/.claude/.../memory/x'` stays
    denied so the documented Bash hole isn't widened.
    """
    layout = experiment_layout
    mem = _memory_path(layout["experiment_dir"], "viaBash.md")
    res = run_hook(
        make_envelope("Bash", {"command": f"echo pwned > {mem}"},
                      layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"


def test_project_slug_collision_is_known_inherited_limitation():
    """Pin the lossy-slug collision boundary explicitly.

    project_slug folds '/', '_', '.' to '-', so it is non-injective:
    distinct paths CAN map to one slug, hence to one carve-out dir.
    This is inherited from Claude Code's own transform (Claude would
    likewise merge their memory), not introduced by the hook. We assert
    the limitation rather than a false structural guarantee, so the
    docstring's claim is fenced by a test instead of merely stated.
    """
    a = "/x/run_1/exp"
    b = "/x/run/1/exp"
    assert a != b
    assert project_slug(a) == project_slug(b) == "-x-run-1-exp"
    # ...and the collision propagates to the carve-out dir: two
    # different experiment dirs that slug-collide share ONE memory dir.
    # Containment does NOT rest on injectivity; it rests on EXP names
    # being timestamp-unique and RoboPhD-owned (covered by the
    # sibling/source-repo deny tests above).
    assert auto_memory_dir(a) == auto_memory_dir(b)


# ---------------------------------------------------------------------
# Session-project READ carve-out (Claude CLI tool-results / transcript)
#
# Claude Code spills large tool outputs to
# ~/.claude/projects/<slug(cwd)>/<session>/tool-results/<id>.txt, and
# the model later Reads it back. The slug is cwd-keyed (different from
# auto-memory's experiment-root key). These tests pin the carve-out
# AND the containment: a sibling iteration's slug stays denied, and
# the carve-out is READ-only (Write/Bash-write to the same path stay
# denied so the documented ~/.claude write boundary is unchanged).
# ---------------------------------------------------------------------


def _tool_results_path(cwd: Path, name: str = "spill.txt",
                       config_dir_name: str = ".claude") -> str:
    slug = project_slug(os.path.realpath(str(cwd)))
    return os.path.join(
        os.path.expanduser("~"), config_dir_name, "projects",
        slug, "session-abc", "tool-results", name,
    )


def test_auto_session_dirs_uses_both_homes_when_env_unset(tmp_path, monkeypatch):
    """Without $CLAUDE_CONFIG_DIR, both conventional homes are
    whitelisted (~/.claude AND ~/.claude-secondary) because we don't
    know which Claude will use."""
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    d = tmp_path / "iter"
    d.mkdir()
    out = auto_session_dirs(str(d))
    slug = project_slug(os.path.realpath(str(d)))
    home = os.path.expanduser("~")
    expected_default = os.path.realpath(os.path.join(home, ".claude", "projects", slug))
    expected_secondary = os.path.realpath(os.path.join(home, ".claude-secondary", "projects", slug))
    assert expected_default in out
    assert expected_secondary in out


def test_auto_session_dirs_prefers_env_when_set(tmp_path, monkeypatch):
    """With $CLAUDE_CONFIG_DIR set, only that dir is whitelisted —
    the fallbacks aren't needed and shouldn't widen the surface."""
    custom = tmp_path / "custom_claude"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
    d = tmp_path / "iter"
    d.mkdir()
    out = auto_session_dirs(str(d))
    slug = project_slug(os.path.realpath(str(d)))
    expected = os.path.realpath(os.path.join(str(custom), "projects", slug))
    home = os.path.expanduser("~")
    fallback = os.path.realpath(os.path.join(home, ".claude", "projects", slug))
    assert expected in out
    assert fallback not in out  # fallbacks NOT added when env is set


def test_tool_results_same_slug_different_session_id_allows(experiment_layout):
    """Carve-out is intentionally the WHOLE <slug>/ dir, not just one
    session-id subdir — Claude may legitimately spawn multiple sessions
    in the same cwd (resume/extend), and all of them sit under this
    one timestamp-unique slug. Pins the intentional breadth so a
    future narrowing has to deliberate."""
    layout = experiment_layout
    slug = project_slug(os.path.realpath(str(layout["cwd"])))
    home = os.path.expanduser("~")
    other_session = os.path.join(
        home, ".claude", "projects", slug,
        "another-session-uuid", "tool-results", "x.txt",
    )
    res = run_hook(
        make_envelope("Read", {"file_path": other_session}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["rc"] == 0
    assert res["decision"] is None


def test_tool_results_read_under_cwd_slug_allows(experiment_layout):
    """The CLI's own spill readback (this iteration's slug) must succeed."""
    layout = experiment_layout
    target = _tool_results_path(layout["cwd"])
    res = run_hook(
        make_envelope("Read", {"file_path": target}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["rc"] == 0
    assert res["decision"] is None


def test_tool_results_read_under_secondary_home_allows(experiment_layout):
    """Same carve-out for the .claude-secondary home variant
    (observed in alt installations)."""
    layout = experiment_layout
    target = _tool_results_path(layout["cwd"], config_dir_name=".claude-secondary")
    res = run_hook(
        make_envelope("Read", {"file_path": target}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["rc"] == 0
    assert res["decision"] is None


def test_tool_results_sibling_iteration_slug_denies(experiment_layout):
    """A *different* iteration's slug (here: a sibling-run iter dir) is
    not on the carve-out — read denied. Pins containment: the carve-out
    is exactly this cwd's slug, not all of ~/.claude/projects/."""
    layout = experiment_layout
    other_cwd = (layout["experiment_dir"].parent / "task_20251231_120000"
                 / "evolution_output" / "iteration_002")
    other_slug = project_slug(os.path.realpath(str(other_cwd)))
    target = os.path.join(
        os.path.expanduser("~"), ".claude", "projects",
        other_slug, "session-xyz", "tool-results", "leak.txt",
    )
    res = run_hook(
        make_envelope("Read", {"file_path": target}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_tool_results_write_tool_still_denied(experiment_layout):
    """Carve-out is READ-only. Writing to the same path must still
    deny — keeps the documented ~/.claude write boundary intact."""
    layout = experiment_layout
    target = _tool_results_path(layout["cwd"])
    res = run_hook(
        make_envelope("Write", {"file_path": target, "content": "x"},
                      layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside write scope" in res["reason"]


def test_tool_results_bash_write_still_denied(experiment_layout):
    """Bash write redirect into the carve-out dir must still deny —
    the Bash write path deliberately doesn't honor session carve-outs,
    same asymmetry as the memory carve-out."""
    layout = experiment_layout
    target = _tool_results_path(layout["cwd"], name="evil.txt")
    res = run_hook(
        make_envelope("Bash", {"command": f"echo pwned > {target}"},
                      layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"


def test_auto_memory_symlinked_root_literal_slug_allows(tmp_path):
    """Fix-2 regression: carve-out must hold when literal != realpath.

    Builds an experiment root reached via a symlink so the raw-env path
    and its realpath produce DIFFERENT slugs. The expected memory path
    is constructed from the *literal* (symlinked) slug — NOT realpath —
    so this test fails under a single-realpath-slug implementation and
    is therefore not green-by-construction.
    """
    real_root = tmp_path / "real_runs" / "robophd" / "task_20260101_000000"
    cwd = real_root / "evolution_output" / "iteration_002"
    cwd.mkdir(parents=True)
    link_root = tmp_path / "link_runs"
    link_root.symlink_to(tmp_path / "real_runs")

    # Env value reaches the experiment dir THROUGH the symlink.
    exp_literal = link_root / "robophd" / "task_20260101_000000"
    cwd_literal = exp_literal / "evolution_output" / "iteration_002"

    literal_slug = project_slug(str(exp_literal))
    realpath_slug = project_slug(os.path.realpath(str(exp_literal)))
    assert literal_slug != realpath_slug  # the symlink really diverges

    literal_mem = os.path.join(
        os.path.expanduser("~"), ".claude", "projects",
        literal_slug, "memory", "insight.md",
    )
    res = run_hook(
        make_envelope("Write", {"file_path": literal_mem, "content": "x"},
                      cwd_literal),
        exp_literal,  # ROBOPHD_EXPERIMENT_DIR = the literal symlinked path
    )
    assert res["rc"] == 0
    assert res["decision"] is None
    # The realpath-form slug must ALSO still be honored.
    realpath_mem = os.path.join(
        os.path.expanduser("~"), ".claude", "projects",
        realpath_slug, "memory", "insight.md",
    )
    res2 = run_hook(
        make_envelope("Write", {"file_path": realpath_mem, "content": "x"},
                      cwd_literal),
        exp_literal,
    )
    assert res2["rc"] == 0
    assert res2["decision"] is None


# ---------------------------------------------------------------------
# Iteration-rooted write scope ($ROBOPHD_EVOLUTION_ITERATION_DIR)
#
# When the harness declares an iteration root, write scope anchors on
# that root rather than on the runtime cwd — so an agent can edit
# <iter>/agent.py from any subdir it has `cd`'d into for testing.
# Security boundaries (sibling iterations / sibling runs / source repo
# / ~/.claude write) are unchanged. Without the env var, the hook
# falls back to the legacy cwd-rooted policy (existing tests above
# fence that backward-compat path).
# ---------------------------------------------------------------------


def test_iteration_root_write_from_nested_cwd_allows(experiment_layout):
    """The motivating case: cwd has moved into a nested test subdir,
    but the agent wants to refine its own iteration's agent.py.
    Under the iteration-rooted policy this MUST be allowed; under the
    legacy cwd-rooted policy (same call without iteration_dir set) it
    was denied (see test_legacy_cwd_rooted_write_outside_cwd_denies
    below for the contrast)."""
    layout = experiment_layout
    nested = layout["cwd"] / "iteration_013_test" / "agent_iter15_x"
    nested.mkdir(parents=True)
    target = layout["cwd"] / "agent.py"
    target.write_text("# pre-existing\n")
    res = run_hook(
        make_envelope("Write", {"file_path": str(target), "content": "new"},
                      nested),
        layout["experiment_dir"],
        iteration_dir=layout["cwd"],  # declare iteration root
    )
    assert res["rc"] == 0
    assert res["decision"] is None


def test_legacy_cwd_rooted_write_outside_cwd_denies(experiment_layout):
    """Contrast / regression guard: same envelope but WITHOUT
    iteration_dir set falls back to cwd-rooted write scope and must
    deny — pins the backward-compat path the existing test suite has
    been exercising all along."""
    layout = experiment_layout
    nested = layout["cwd"] / "iteration_013_test" / "agent_iter15_x"
    nested.mkdir(parents=True)
    target = layout["cwd"] / "agent.py"
    target.write_text("# pre-existing\n")
    res = run_hook(
        make_envelope("Write", {"file_path": str(target), "content": "new"},
                      nested),
        layout["experiment_dir"],
        # iteration_dir NOT set -> falls back to cwd
    )
    assert res["decision"] == "deny"
    assert "outside write scope" in res["reason"]


def test_iteration_root_deny_message_names_iteration_directory(experiment_layout):
    """When iteration_dir is declared, the deny message must
    (a) show the iteration root as the write-scope path, AND
    (b) explicitly call it the "evolution iteration directory" so
    the agent doesn't think it needs to `cd` there. Without this
    label, agents reading "Write only under <path>" plausibly
    interpret it as 'I must be in that dir'."""
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"  # outside iteration root
    res = run_hook(
        make_envelope("Write", {"file_path": str(target), "content": "x"},
                      layout["cwd"]),
        layout["experiment_dir"],
        iteration_dir=layout["cwd"],
    )
    assert res["decision"] == "deny"
    assert str(layout["cwd"].resolve()) in res["reason"]
    # The label ("evolution iteration directory") plus the word
    # "anywhere" already carries the cwd-independence — no further
    # explanatory parenthetical needed (and any redundant prose costs
    # tokens on every denial).
    assert "anywhere under your evolution iteration directory" in res["reason"]


def test_legacy_deny_message_omits_iteration_label(experiment_layout):
    """In legacy mode (no iteration_dir env var), the deny message
    should NOT claim "evolution iteration directory" — that label
    would be wrong (the displayed path is the cwd, not necessarily
    the iteration root)."""
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"
    res = run_hook(
        make_envelope("Write", {"file_path": str(target), "content": "x"},
                      layout["cwd"]),
        layout["experiment_dir"],
        # iteration_dir NOT set -> legacy fallback
    )
    assert res["decision"] == "deny"
    assert "evolution iteration directory" not in res["reason"]


def test_iteration_root_does_not_allow_sibling_iteration_write(experiment_layout):
    """Security boundary unchanged: even with iteration_dir declared,
    writing into a sibling iteration's (or prior agent's) dir must
    still deny — those dirs are NOT under the declared iteration
    root, just under the broader experiment dir."""
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"  # prior agent dir
    res = run_hook(
        make_envelope("Write", {"file_path": str(target), "content": "x"},
                      layout["cwd"]),
        layout["experiment_dir"],
        iteration_dir=layout["cwd"],
    )
    assert res["decision"] == "deny"
    assert "outside write scope" in res["reason"]


def test_iteration_root_does_not_allow_sibling_run_write(experiment_layout):
    """Security boundary unchanged: a sibling RUN's dir is outside
    even with iteration_dir declared."""
    layout = experiment_layout
    target = layout["sibling_agent"]  # different run
    res = run_hook(
        make_envelope("Write", {"file_path": str(target), "content": "x"},
                      layout["cwd"]),
        layout["experiment_dir"],
        iteration_dir=layout["cwd"],
    )
    assert res["decision"] == "deny"


def test_iteration_root_bash_redirect_to_iter_root_from_nested_cwd_allows(experiment_layout):
    """Bash write path inherits the iteration-rooted policy too —
    `echo … > <iter>/agent.py` from a nested cwd is allowed when
    iteration_dir is declared."""
    layout = experiment_layout
    nested = layout["cwd"] / "scratch"
    nested.mkdir()
    target = layout["cwd"] / "agent.py"
    cmd = f"echo content > {target}"
    res = run_hook(
        make_envelope("Bash", {"command": cmd}, nested),
        layout["experiment_dir"],
        iteration_dir=layout["cwd"],
    )
    assert res["rc"] == 0
    assert res["decision"] is None


def test_iteration_root_bash_write_to_sibling_iter_still_denied(experiment_layout):
    """Bash write to a sibling iteration's dir stays denied even
    under the iteration-rooted policy."""
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"
    cmd = f"echo x > {target}"
    res = run_hook(
        make_envelope("Bash", {"command": cmd}, layout["cwd"]),
        layout["experiment_dir"],
        iteration_dir=layout["cwd"],
    )
    assert res["decision"] == "deny"


def test_iteration_root_memory_carveout_still_works(experiment_layout):
    """Auto-memory write under <experiment-slug>/memory/ stays
    allowed regardless of which write-scope mode is in effect."""
    layout = experiment_layout
    slug = project_slug(os.path.realpath(str(layout["experiment_dir"])))
    target = os.path.join(
        os.path.expanduser("~"), ".claude", "projects", slug,
        "memory", "insight.md",
    )
    res = run_hook(
        make_envelope("Write", {"file_path": target, "content": "x"},
                      layout["cwd"]),
        layout["experiment_dir"],
        iteration_dir=layout["cwd"],
    )
    assert res["rc"] == 0
    assert res["decision"] is None


def test_iteration_root_bash_write_to_claude_dir_still_denied(experiment_layout):
    """The documented Bash hole closure stays — even with
    iteration_dir set, a Bash redirect into ~/.claude/.../memory/
    stays denied (Bash write path doesn't honor any of the carve-outs
    or the iteration-root expansion for ~/.claude)."""
    layout = experiment_layout
    slug = project_slug(os.path.realpath(str(layout["experiment_dir"])))
    target = os.path.join(
        os.path.expanduser("~"), ".claude", "projects", slug,
        "memory", "viaBash.md",
    )
    res = run_hook(
        make_envelope("Bash", {"command": f"echo x > {target}"},
                      layout["cwd"]),
        layout["experiment_dir"],
        iteration_dir=layout["cwd"],
    )
    assert res["decision"] == "deny"


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


def test_bash_unknown_cmd_with_path_in_read_scope_allows(experiment_layout):
    """Unknown commands default to treating path tokens as reads.

    Policy decision (2026-05-06): rather than fail-close on every
    novel command shape (`frobnicate`, `mlr`, `dasel`, etc.) and
    play whack-a-mole with the agent's creativity, treat path tokens
    as reads and let the read-scope check do the gating. If the path
    is in scope, allow; if outside, deny via the standard read-scope
    deny path.
    """
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"
    res = run_hook(
        make_envelope("Bash", {"command": f"frobnicate {target}"}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] is None, res


def test_bash_unknown_cmd_with_path_outside_read_scope_denies(experiment_layout):
    """Unknown command + path outside read scope → standard read-scope deny."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": f"frobnicate {layout['sibling_agent']}"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


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
# find -exec / xargs: subprocess_bypass retired — visible paths are still
# checked, but the structural deny on these constructs is gone.
# ---------------------------------------------------------------------


def test_bash_find_exec_in_scope_allows(experiment_layout):
    """find -exec on the iteration's own files is now allowed. The
    structural subprocess_bypass deny was retired; the inner cat
    operates on files under the in-scope find root."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": "find . -name agent.py -exec cat {} ;"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] is None, res


def test_bash_find_exec_with_out_of_scope_root_still_denies(experiment_layout):
    """find's search root tokenizes as a path argument; if it's
    outside read scope, the path-token check still denies regardless
    of -exec."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": f"find {layout['sibling_agent'].parent} -exec cat {{}} ;"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_bash_find_exec_with_out_of_scope_hardcoded_arg_still_denies(experiment_layout):
    """Hardcoded path inside the -exec command (not the {} substitution)
    still tokenizes as a path argument and gets read-scope-checked. We
    lose recall on what the inner command does *with* {}, not on
    explicit hardcoded paths."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": f"find . -exec cp {{}} {layout['sibling_agent'].parent}/ ;"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_bash_find_without_exec_still_allowed(experiment_layout):
    """Plain find without -exec stays allowed, same as before."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash", {"command": "find . -name agent.py"}, layout["cwd"]
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] is None


def test_bash_xargs_in_scope_allows(experiment_layout):
    """xargs in a pipeline whose visible paths are all in scope is
    now allowed. The subprocess_bypass deny was retired."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": "find . -name '*.py' | xargs cat"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] is None, res


def test_bash_xargs_with_out_of_scope_predecessor_still_denies(experiment_layout):
    """If the pipeline predecessor's visible path is out of scope,
    the path-token check on that segment still denies, independent of
    xargs in the next segment."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": f"find {layout['sibling_agent'].parent} | xargs cat"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_bash_unknown_cmd_routes_through_read_scope(experiment_layout):
    """Unknown command + out-of-scope path tokenizes as a normal read-
    scope violation. Same as before — the subprocess_bypass deny path
    was retired but this one stays."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": f"frobnicate {layout['sibling_agent']}"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


# ---------------------------------------------------------------------
# Shell control flow (for / while / if / case): strip leading keywords
# ---------------------------------------------------------------------


def test_bash_for_loop_multiline_body_in_scope_allows(experiment_layout):
    """Multi-line for-loop reading files in the iteration's own dir.

    Reproduces the 2026-05-06 production denial: shlex collapses the
    newlines, so the body lands in a `;`-segment whose leading token
    is `do`. Stripping `do` lets the classifier reach `cat <path>`.
    """
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"
    cmd = (
        "for p in 1 2 3; do\n"
        "  echo === $p ===\n"
        f"  cat {target}\n"
        "  echo\n"
        "done"
    )
    res = run_hook(
        make_envelope("Bash", {"command": cmd}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] is None, res


def test_bash_for_loop_singleline_body_in_scope_allows(experiment_layout):
    """Same shape with `;`-delimited body — should classify each
    body command independently and allow."""
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": f"for p in 1 2 3; do cat {target}; done"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] is None, res


def test_bash_for_loop_body_path_outside_scope_denies(experiment_layout):
    """Loop body cat'ing a sibling-run file → must still deny."""
    layout = experiment_layout
    cmd = f"for p in 1; do cat {layout['sibling_agent']}; done"
    res = run_hook(
        make_envelope("Bash", {"command": cmd}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_bash_if_then_fi_in_scope_allows(experiment_layout):
    """`if [ ... ]; then cat <in-scope>; fi` — strip `then`, classify cat."""
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": f"if true; then cat {target}; fi"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] is None, res


def test_bash_while_loop_with_body_write_outside_cwd_denies(experiment_layout):
    """Strip `do`, classify the body's `rm` against write scope."""
    layout = experiment_layout
    # rm target is in read scope but outside write scope (= cwd) — should deny.
    target = layout["agents_dir"] / "agent.py"
    cmd = f"while false; do rm {target}; done"
    res = run_hook(
        make_envelope("Bash", {"command": cmd}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside write scope" in res["reason"]


# ---------------------------------------------------------------------
# Variable assignments: `x=/path/a; cat $x` and `x=/path/a cat $x`
# ---------------------------------------------------------------------


def test_bash_assignment_prefix_in_scope_path_allows(experiment_layout):
    """`x=<in-scope> cat $x` — the assignment value gets read-checked
    even though it's not in `args`. In-scope path → allow."""
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": f"x={target} cat $x"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] is None, res


def test_bash_assignment_prefix_out_of_scope_path_denies(experiment_layout):
    """`x=<sibling-run> cat $x` — assignment-stripper captures the
    sibling path as a read target; read-scope check denies."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": f"x={layout['sibling_agent']} cat $x"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_bash_assignment_then_use_out_of_scope_denies(experiment_layout):
    """`x=<sibling>; cat $x` — assignment is its own segment; same
    classification logic applies."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": f"x={layout['sibling_agent']}; cat $x"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_bash_multiple_assignments_each_checked(experiment_layout):
    """`A=<in-scope> B=<sibling> cmd` — every assignment value is
    extracted; B's sibling path triggers the deny."""
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": f"A={target} B={layout['sibling_agent']} cmd"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_bash_assignment_with_non_path_value_allows(experiment_layout):
    """`x=hello cmd` — value isn't path-shaped; nothing to check."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash", {"command": "FOO=bar BAZ=42 echo hi"}, layout["cwd"]
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] is None, res


# ---------------------------------------------------------------------
# Command substitution: $(...), `...`, subshells (...)
# ---------------------------------------------------------------------


def test_bash_dollar_paren_with_out_of_scope_path_denies(experiment_layout):
    """`cat $(echo /etc/passwd)` — the structural subprocess_bypass
    deny was retired; the visible path token /etc/passwd still gets
    read-scope-checked and denied. The path is what the inner command
    would emit anyway, so catching the visible token is what matters."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": "cat $(echo /etc/passwd)"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_bash_dollar_paren_in_scope_allows(experiment_layout):
    """`cat $(ls *.py | head -1)` with all visible paths in scope is
    now allowed."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": "cat $(ls *.py | head -1)"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] is None, res


def test_bash_backtick_substitution_with_path_outside_scope_denies(experiment_layout):
    """`cat \`echo /sibling\`` — the path token tokenizes standalone
    (shlex doesn't preserve the backticks as a single quoted unit), so
    the standard read-scope check still catches it. This is the only
    backtick-substitution shape we still deny; the corresponding
    inline-in-double-quotes shape (echo "x: \`cat /sibling\`") is now
    allowed, symmetric with our existing $()-inside-double-quotes hole."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": f"cat `echo {layout['sibling_agent']}`"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_bash_legitimate_backtick_substitution_allows(experiment_layout):
    """`echo \"today: \`date\`\"` — legitimate backtick substitution that
    doesn't reference any out-of-scope path. Used to be denied by the
    has_unquoted_backtick rule; now allowed since we accept the residual
    risk in exchange for not blocking legit substitution patterns."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": 'echo "today: `date`"'},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] is None, res


def test_bash_backtick_inside_single_quotes_allows(experiment_layout):
    """Single-quoted backticks are literal in bash; this used to require
    the raw-string quote tracker to allow. After dropping that check,
    single-quoted backticks pass through trivially."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": "echo 'literal backtick: ` symbol'"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] is None, res


# ---------------------------------------------------------------------
# Heredoc body stripping: opaque content, only redirect target is checked
# ---------------------------------------------------------------------


def test_bash_heredoc_with_python_parens_in_body_allows(experiment_layout):
    """`cat <<EOF > script.py\\nprint(1)\\nEOF` — without body-stripping the
    `(` and `)` from `print(1)` would trip the subshell check. Body is
    opaque, redirect target is in cwd, should allow."""
    layout = experiment_layout
    cmd = (
        f"cat <<EOF > {layout['cwd']}/script.py\n"
        "print(1)\n"
        "EOF"
    )
    res = run_hook(
        make_envelope("Bash", {"command": cmd}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] is None, res


def test_bash_heredoc_with_backticks_in_body_allows(experiment_layout):
    """Heredoc body with markdown-style backticks in a comment. Used to
    be denied by has_unquoted_backtick; now opaque thanks to body stripping."""
    layout = experiment_layout
    cmd = (
        f"cat <<EOF > {layout['cwd']}/script.py\n"
        "# Run with `python script.py`\n"
        "print('hello')\n"
        "EOF"
    )
    res = run_hook(
        make_envelope("Bash", {"command": cmd}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] is None, res


def test_bash_heredoc_quoted_delimiter_allows(experiment_layout):
    """`<<'EOF'` — single-quoted delimiter form. Body is treated as
    literal by bash and as opaque by us. Should allow."""
    layout = experiment_layout
    cmd = (
        f"cat <<'EOF' > {layout['cwd']}/script.py\n"
        "# Anything goes here: `\\$ ()\n"
        "EOF"
    )
    res = run_hook(
        make_envelope("Bash", {"command": cmd}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] is None, res


def test_bash_heredoc_redirect_target_outside_cwd_still_denies(experiment_layout):
    """Body is opaque but the `> file` redirect target is still
    write-scope-checked. Writing the heredoc output outside cwd → deny."""
    layout = experiment_layout
    cmd = (
        f"cat <<EOF > {layout['agents_dir']}/stolen.py\n"
        "print('overwrite')\n"
        "EOF"
    )
    res = run_hook(
        make_envelope("Bash", {"command": cmd}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside write scope" in res["reason"]


def test_bash_heredoc_dash_form_strips_body(experiment_layout):
    """`<<-EOF` strips leading tabs from body lines. We don't care about
    the body content but should still parse the heredoc start so the
    body doesn't trip downstream classifiers."""
    layout = experiment_layout
    cmd = (
        f"cat <<-EOF > {layout['cwd']}/script.py\n"
        "\tline with leading tab `print(x)`\n"
        "\tEOF"
    )
    res = run_hook(
        make_envelope("Bash", {"command": cmd}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["decision"] is None, res


def test_bash_process_substitution_with_out_of_scope_path_denies(experiment_layout):
    """`<(cat /etc/passwd)` — the visible /etc/passwd token still gets
    read-scope-checked and denied via the standard path-classifier
    path; the structural deny on `(`/`)` is gone."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": "diff <(cat /etc/passwd) <(echo)"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_bash_subshell_with_out_of_scope_path_denies(experiment_layout):
    """`(cat /etc/passwd)` — subshell wrapping. Visible path token
    still gets read-scope-checked and denied."""
    layout = experiment_layout
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": "(cat /etc/passwd)"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_bash_subshell_in_scope_allows(experiment_layout):
    """In-scope subshell now allowed; structural deny was retired."""
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"
    res = run_hook(
        make_envelope(
            "Bash",
            {"command": f"(cat {target})"},
            layout["cwd"],
        ),
        layout["experiment_dir"],
    )
    assert res["decision"] is None, res


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


# ---------------------------------------------------------------------
# Tokenizer-flattening false positives (observed in asta_ds1000 runs)
#
# shlex collapses newlines and substitution boundaries, so the leading
# command's classifier used to consume *other* commands' tokens. These
# pin the fixes AND guard that the added precision never HIDES a path:
# out-of-scope reads on later statements / sed file operands must still
# be denied, and `find /` must stay blocked (it is a correct catch, not
# a false positive).
# ---------------------------------------------------------------------


def test_split_statement_newlines_unit():
    assert split_statement_newlines("a\nb\nc") == "a;b;c"
    # Newlines inside quotes are data, not separators.
    assert split_statement_newlines("echo '1\n2'") == "echo '1\n2'"
    assert split_statement_newlines('echo "1\n2"') == 'echo "1\n2"'
    # bash line-continuation JOINS with no separator (a\<nl>b -> ab),
    # NOT a space. A space would mis-split a path token and is
    # exploitable (see test_backslash_newline_join_keeps_path_denied).
    assert split_statement_newlines("a\\\nb") == "ab"
    assert split_statement_newlines("a \\\nb") == "a b"  # real space kept


def test_backslash_newline_join_keeps_path_denied(experiment_layout):
    """Escape regression — must be NOT green-by-construction.

    The split lands exactly at the experiment-dir prefix boundary:
      token-1 = <experiment_dir>          -> == root, IN scope
      token-2 = _escape/secret            -> relative, resolves under
                                             cwd, IN scope
      joined  = <experiment_dir>_escape/secret -> sibling-of-root,
                                             OUT of scope (bash reads
                                             this).
    Under the OLD space-emitting bug both tokens pass -> ALLOW (escape).
    Under the JOIN fix it is one out-of-scope token -> DENY. So this
    test FAILS on the buggy impl and PASSES on the fixed one, unlike a
    mid-`sibling_agent` split whose every prefix is already out of
    scope (green under both).
    """
    layout = experiment_layout
    exp = str(layout["experiment_dir"])
    # `<exp>_escape/secret` shares the string prefix but is NOT under
    # <exp> (next char is '_' , not os.sep), so it is out of scope.
    cmd = f"cat {exp}\\\n_escape/secret"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_backslash_newline_test_is_not_vacuous(experiment_layout):
    """Control for the test above: the same `cat` of a genuinely
    in-scope file (no continuation) is allowed — proving the deny
    above is the join behavior, not the harness denying everything."""
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"  # in read scope
    res = run_hook(
        make_envelope("Bash", {"command": f"cat {target}"}, layout["cwd"]),
        layout["experiment_dir"],
    )
    assert res["rc"] == 0
    assert res["decision"] is None


def test_multiline_interpreter_after_cd_allows(experiment_layout):
    """rec2: `cd <in-scope>` then a newline then an absolute-path
    interpreter must NOT make the interpreter an argument of `cd`."""
    layout = experiment_layout
    cmd = (
        f"cd {layout['cwd']}\n"
        "/opt/anaconda3/envs/robophd_demo/bin/python -c "
        "\"import ast; ast.parse(open('agent.py').read()); print('ok')\""
    )
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["rc"] == 0
    assert res["decision"] is None


def test_multiline_second_statement_still_scoped(experiment_layout):
    """Guard: newline-splitting must not let line 2 escape scope.

    If it merely *ignored* later statements this would wrongly allow;
    it must classify line 2 on its own and deny the sibling read."""
    layout = experiment_layout
    cmd = f"cd {layout['cwd']}\ncat {layout['sibling_agent']}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_sed_script_in_process_substitution_allows(experiment_layout):
    """rec3: `diff <(sed -n '/re/,$p' a) <(sed -n '/re/,$p' b)` with
    both files in scope — the sed script must not read as a path."""
    layout = experiment_layout
    a = layout["agents_dir"] / "agent.py"   # in read scope
    b = layout["cwd"] / "agent.py"
    b.write_text("x\n")
    cmd = (f"diff <(sed -n '/^FORMAT_VAR = /,$p' {a}) "
           f"<(sed -n '/^FORMAT_VAR = /,$p' {b}) && echo IDENT")
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["rc"] == 0
    assert res["decision"] is None


def test_sed_file_operand_still_scoped(experiment_layout):
    """Guard: only the sed SCRIPT is exempted — a real out-of-scope
    file operand must still be denied."""
    layout = experiment_layout
    cmd = f"sed -n '/re/,$p' {layout['sibling_agent']}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_sed_inplace_outside_cwd_still_write_denied(experiment_layout):
    """Guard: sed -i FILE is still a write; a prior-agent dir is in
    read scope but out of write scope."""
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"
    cmd = f"sed -i 's/a/b/' {target}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] == "deny"
    assert "outside write scope" in res["reason"]


def test_find_root_still_denied(experiment_layout):
    """Security regression guard: `find /` is a full-filesystem scan
    and MUST stay denied — it is a correct catch, not a false
    positive, and the precision changes must not relax it."""
    layout = experiment_layout
    cmd = 'find / -name "test_760.py" 2>/dev/null | head'
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_tr_slash_operand_allows(experiment_layout):
    """`tr`'s operand is a character SET, not a path, so the bare '/' in
    `tr -d /` must NOT read-deny (was a pinned FP; now fixed via
    BASH_NO_PATH_OPERANDS). `tr` only reads stdin / writes stdout, so an
    operand is never opened as a file. `find /` (full-FS scan) is
    unaffected — that path flows under the `find` branch, not tr's."""
    layout = experiment_layout
    for cmd in ("echo abc | tr -d /",
                "ls -d */ | tr -d '/'",
                "for p in $(ls -d */ | tr -d /); do echo $p; done"):
        res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                       layout["experiment_dir"])
        assert res["decision"] is None, (cmd, res)


def test_tr_does_not_open_redirect_or_substitution_paths(experiment_layout):
    """Guard: the tr exemption covers only its SET operands. A redirect
    target and a $()-substituted out-of-scope read alongside tr must
    still be scope-checked (redirect extracted before dispatch;
    substitution split into its own segment)."""
    layout = experiment_layout
    # Redirect target outside cwd is still a write-deny.
    sib = layout["sibling_agent"]
    res = run_hook(
        make_envelope("Bash", {"command": f"echo abc | tr a b > {sib}/x.txt"},
                      layout["cwd"]),
        layout["experiment_dir"])
    assert res["decision"] == "deny", res
    # Out-of-scope read inside a substitution alongside tr still denies.
    res = run_hook(
        make_envelope("Bash", {"command": "tr -d $(cat /etc/passwd)"},
                      layout["cwd"]),
        layout["experiment_dir"])
    assert res["decision"] == "deny", res
    assert "outside read scope" in res["reason"]


def test_sed_bundled_inplace_flag_is_write(experiment_layout):
    """`-ni` is a bundled `-n -i` on GNU sed — it MUST be detected as
    in-place, else the operand is read-classified and an in-place
    write outside write scope is wrongly ALLOWED (the one
    unsafe-direction gap; now closed)."""
    layout = experiment_layout
    target = layout["agents_dir"] / "agent.py"  # read scope, NOT write
    cmd = f"sed -ni 's/a/b/' {target}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] == "deny"
    assert "outside write scope" in res["reason"]


def test_sed_inplace_target_in_cwd_allows(experiment_layout):
    """Counterpart: bundled in-place on a cwd target is a legit write."""
    layout = experiment_layout
    target = layout["cwd"] / "agent.py"
    target.write_text("a\n")
    cmd = f"sed -i.bak 's/a/b/' {target}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["rc"] == 0
    assert res["decision"] is None


def test_sed_f_scriptfile_is_read_not_write_under_inplace(experiment_layout):
    """`sed -i -f SCRIPT.sed TARGET`: SCRIPT.sed is a READ even under
    -i (only TARGET is the write). An out-of-scope script file must be
    denied as a *read*, proving it isn't mis-sunk into write_paths."""
    layout = experiment_layout
    sib_script = layout["sibling_agent"]  # out of scope
    target = layout["cwd"] / "agent.py"
    target.write_text("x\n")
    cmd = f"sed -i -f {sib_script} {target}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] == "deny"
    assert "outside read scope" in res["reason"]


def test_sed_f_scriptfile_in_scope_allows(experiment_layout):
    """`sed -i -f SCRIPT.sed TARGET` with the script file in read scope
    and the target in write scope — both classified correctly -> ok."""
    layout = experiment_layout
    script = layout["agents_dir"] / "prog.sed"  # in read scope
    script.write_text("s/a/b/\n")
    target = layout["cwd"] / "agent.py"
    target.write_text("a\n")
    cmd = f"sed -i -f {script} {target}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["rc"] == 0
    assert res["decision"] is None


# ---- awk: inline program is not a path (mirrors sed) ----

def test_awk_range_program_with_in_scope_file_allows(experiment_layout):
    """`awk '/^## Step 2/,/^## Step 3/' FILE` — the `/regex/,/regex/`
    range is the awk PROGRAM, not a path. With FILE in read scope the
    whole command must be allowed (was a recurring FP on in-scope
    agent_trace.md reads)."""
    layout = experiment_layout
    target = layout["agents_dir"] / "trace.md"  # in read scope
    target.write_text("## Step 1\n## Step 2\nx\n## Step 3\n")
    cmd = f"awk '/^## Step 2/,/^## Step 3/' {target}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] is None, res


def test_awk_action_program_no_file_allows(experiment_layout):
    """`... | awk '/[multiblock]/{c++} END{print c}'` — program-only awk
    on piped stdin (no file operand). The `/.../{...}` action guards
    start with `/` but are program text, not paths."""
    layout = experiment_layout
    cmd = "cat in.txt | awk '/foo/{c++} END{print c}'"
    (layout["cwd"] / "in.txt").write_text("foo\n")
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] is None, res


def test_awk_file_operand_out_of_scope_still_denies(experiment_layout):
    """Guard: only the awk PROGRAM is exempt — a real out-of-scope FILE
    operand after the program is still read-scope-checked."""
    layout = experiment_layout
    cmd = f"awk '/re/' {layout['sibling_agent']}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] == "deny", res
    assert "outside read scope" in res["reason"]


def test_awk_progfile_is_read_with_in_scope_file(experiment_layout):
    """`awk -f PROG.awk FILE` — PROG.awk is a READ (the program comes
    from a file), and the trailing FILE is a read operand. Both in scope
    -> allowed."""
    layout = experiment_layout
    prog = layout["agents_dir"] / "p.awk"
    prog.write_text("{print}\n")
    data = layout["agents_dir"] / "d.txt"
    data.write_text("x\n")
    cmd = f"awk -f {prog} {data}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] is None, res


def test_awk_progfile_out_of_scope_denies(experiment_layout):
    """Guard: `awk -f PROG.awk` with an out-of-scope program file is a
    read-deny (the -f operand is a real file read)."""
    layout = experiment_layout
    cmd = f"awk -f {layout['sibling_agent']}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] == "deny", res


def test_awk_v_assign_not_treated_as_path(experiment_layout):
    """`awk -v p=/x/y '{...}' FILE` — the `-v VAR=val` assignment value
    is not a path operand even when it looks path-shaped."""
    layout = experiment_layout
    data = layout["agents_dir"] / "d.txt"
    data.write_text("x\n")
    cmd = f"awk -v p=/sibling/secret '{{print p}}' {data}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] is None, res


# ---- $((arithmetic)) must not split a surrounding path token ----

def test_arithmetic_expansion_in_path_allows(experiment_layout):
    """`f=.../iter${v}_v$((v-2))/agent.py; head $f` — the `$((v-2))`
    expansion must not split the path token so the trailing `/agent.py`
    read-denies as an absolute path. The expansion collapses to a digit;
    the (in-scope) path stays intact."""
    layout = experiment_layout
    # Build an in-scope path with an arithmetic expansion glued into a
    # path component: agents_dir is `.../agents/iter009_some_agent`, so
    # `.../agents/iter00$((8+1))_some_agent/agent.py` collapses to the
    # real in-scope file. Without the collapse, split_compound breaks on
    # `((`/`))` and the trailing `/agent.py` read-denies as an absolute
    # path.
    agents_parent = str(layout["agents_dir"].parent)
    cmd = f'head {agents_parent}/iter00$((8+1))_some_agent/agent.py'
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] is None, res


def test_arithmetic_expansion_does_not_hide_out_of_scope_path(experiment_layout):
    """Guard: collapsing `$((...))` must not let an out-of-scope path
    slip through — a sibling path with an embedded expansion still
    denies."""
    layout = experiment_layout
    sib_dir = layout["sibling_agent"].parent
    cmd = f"head {sib_dir}/a$((0))gent.py"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] == "deny", res


# ---------------------------------------------------------------------
# evaluate_bash / replay_denial_record — the pure decision core shared
# with the catalog's live-replay classifier. These must agree with the
# subprocess hook (same pipeline) and reconstruct a logged denial's
# verdict faithfully.
# ---------------------------------------------------------------------


def test_evaluate_bash_allows_in_scope_read(experiment_layout):
    layout = experiment_layout
    exp = str(layout["experiment_dir"].resolve())
    cwd = str(layout["cwd"].resolve())
    agent = str((layout["agents_dir"] / "agent.py").resolve())
    decision, scope, blocked = evaluate_bash(
        f"cat {agent}", cwd, exp, cwd, [])
    assert decision == "allow"


def test_evaluate_bash_denies_out_of_scope_read(experiment_layout):
    layout = experiment_layout
    exp = str(layout["experiment_dir"].resolve())
    cwd = str(layout["cwd"].resolve())
    sib = str(layout["sibling_agent"].resolve())
    decision, scope, blocked = evaluate_bash(
        f"cat {sib}", cwd, exp, cwd, [])
    assert decision == "deny"
    assert scope == "read"
    assert blocked == sib


def test_evaluate_bash_parse_fail_is_deny(experiment_layout):
    layout = experiment_layout
    exp = str(layout["experiment_dir"].resolve())
    cwd = str(layout["cwd"].resolve())
    # Unbalanced quote -> shlex fails -> fail closed.
    decision, scope, _ = evaluate_bash('cat "unterminated', cwd, exp, cwd, [])
    assert decision == "deny"
    assert scope == "parse"


def test_replay_bash_record_reconstructs_deny(experiment_layout):
    """A logged out-of-scope Bash read replays as still-denying."""
    layout = experiment_layout
    sib = str(layout["sibling_agent"].resolve())
    rec = {
        "tool": "Bash",
        "scope": "read",
        "blocked_path": sib,
        "command": f"cat {sib}",
        "cwd": str(layout["cwd"].resolve()),
    }
    still, blocked = replay_denial_record(rec, str(layout["experiment_dir"]))
    assert still is True
    assert blocked == sib


def test_replay_bash_record_now_allows_when_in_scope(experiment_layout):
    """A logged denial whose path is actually in scope replays as fixed
    (the verdict the catalog reads to mark FP-FIXED)."""
    layout = experiment_layout
    agent = str((layout["agents_dir"] / "agent.py").resolve())
    rec = {
        "tool": "Bash",
        "scope": "read",
        "blocked_path": agent,
        "command": f"cat {agent}",
        "cwd": str(layout["cwd"].resolve()),
    }
    still, _ = replay_denial_record(rec, str(layout["experiment_dir"]))
    assert still is False


def test_replay_read_tool_record(experiment_layout):
    """Read-tool records (empty command, blocked_path set) replay via the
    read-scope check."""
    layout = experiment_layout
    sib = str(layout["sibling_agent"].resolve())
    rec = {"tool": "Read", "scope": "read", "blocked_path": sib,
           "command": "", "cwd": str(layout["cwd"].resolve())}
    still, blocked = replay_denial_record(rec, str(layout["experiment_dir"]))
    assert still is True
    assert blocked == sib


def test_replay_unreplayable_record_raises(experiment_layout):
    """A parse record (no blocked_path) can't be replayed; the function
    raises so the catalog falls back to static classification."""
    layout = experiment_layout
    rec = {"tool": "Bash", "scope": "parse", "blocked_path": "",
           "command": "", "cwd": str(layout["cwd"].resolve())}
    # Empty command -> evaluate_bash returns allow (no deny to confirm),
    # which is a faithful "no longer denies" rather than a raise. A truly
    # unreplayable shape is a non-Bash tool with no blocked_path:
    rec2 = {"tool": "Read", "scope": "read", "blocked_path": "",
            "command": "", "cwd": str(layout["cwd"].resolve())}
    import pytest as _pytest
    with _pytest.raises(ValueError):
        replay_denial_record(rec2, str(layout["experiment_dir"]))


# ---------------------------------------------------------------------
# Command-wrapper prefix stripping: timeout/time/nice/... relocate the
# real command into argument position. An interpreter so relocated must
# be treated like the bare leading invocation it stands in for (command
# token not scope-checked), WITHOUT dropping any path-shaped scaffolding.
# ---------------------------------------------------------------------


def test_timeout_interpreter_allows(experiment_layout):
    """`timeout 600 /opt/.../python in_scope.py` — DURATION consumed,
    interpreter becomes the (unchecked) command token; the in-scope arg
    is fine. Was the wrapper-prefix FP-OPEN."""
    layout = experiment_layout
    target = layout["cwd"] / "run.py"
    target.write_text("print(1)\n")
    cmd = f"timeout 600 /opt/anaconda3/envs/x/bin/python {target}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] is None, res


def test_time_interpreter_allows(experiment_layout):
    """`time /opt/.../python in_scope.py` — no leading operand, just the
    wrapped command."""
    layout = experiment_layout
    target = layout["cwd"] / "run.py"
    target.write_text("print(1)\n")
    cmd = f"time /opt/anaconda3/envs/x/bin/python {target}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] is None, res


def test_nice_bare_interpreter_allows(experiment_layout):
    """`nice /opt/.../python in_scope.py` — bare wrapper (no options) is
    stripped to the wrapped command."""
    layout = experiment_layout
    target = layout["cwd"] / "run.py"
    target.write_text("print(1)\n")
    cmd = f"nice /opt/anaconda3/envs/x/bin/python {target}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] is None, res


def test_wrapper_with_options_bails_does_not_escape(experiment_layout):
    """A wrapper invoked WITH options (`nice -n 10 CMD`) is NOT stripped —
    we don't guess per-flag arity. The strip bails, leaving CMD in
    argument position. The safety property: an OUT-OF-SCOPE path among the
    bailed tokens is still read-checked and denied (no escape). Use a real
    out-of-scope sibling path (NOT an interpreter, which is now read-scope
    exempt by design)."""
    layout = experiment_layout
    sib = layout["sibling_agent"]
    cmd = f"nice -n 10 cat {sib}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] == "deny", res


def test_wrapper_with_flag_bails_permissive_bounded_to_experiment_dir(experiment_layout):
    """A flagged wrapper bails to the normal unknown-command branch, which
    is deliberately PERMISSIVE (path args default to read scope) — matching
    that branch's own policy and the owner's preference for false negatives
    over false positives.

    Consequence (documented, bounded, 0 real occurrences): a flagged-
    wrapper command that is actually a WRITE into experiment_dir is allowed
    — e.g. `timeout -k 5 rm <sibling-iter file>` is read-classified, the
    path is under experiment_dir, so it's allowed. This is a run-to-run
    hygiene leak, NOT an escape: the write stays confined to experiment_dir.
    The two assertions below pin BOTH halves of that boundary."""
    layout = experiment_layout
    # (a) Inside experiment_dir but outside the iteration write root:
    #     permissively ALLOWED (the accepted bounded leak).
    in_exp = layout["agents_dir"] / "agent.py"
    res = run_hook(make_envelope("Bash", {"command": f"timeout -k 5 rm {in_exp}"},
                                 layout["cwd"]),
                   layout["experiment_dir"], iteration_dir=layout["cwd"])
    assert res["decision"] is None, res
    # (b) OUTSIDE experiment_dir (sibling RUN): still DENIED — the leak is
    #     bounded to experiment_dir; nothing escapes the read scope.
    out_exp = layout["sibling_agent"]
    res2 = run_hook(make_envelope("Bash", {"command": f"timeout -k 5 rm {out_exp}"},
                                  layout["cwd"]),
                    layout["experiment_dir"], iteration_dir=layout["cwd"])
    assert res2["decision"] == "deny", res2


def test_wrapper_does_not_drop_out_of_scope_command_arg(experiment_layout):
    """Safety: stripping the wrapper must NOT exempt the wrapped command's
    own arguments — `timeout 600 cat /sibling/secret` still read-denies
    the out-of-scope arg."""
    layout = experiment_layout
    cmd = f"timeout 600 cat {layout['sibling_agent']}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] == "deny", res
    assert "outside read scope" in res["reason"]


def test_wrapper_separated_path_option_still_denies(experiment_layout):
    """Safety: `time -o /sibling/out CMD` — `-o` is a flag, so the strip
    bails; `/sibling/out` then flows through normal scope-checking as a
    separate token and is denied."""
    layout = experiment_layout
    sib_out = layout["sibling_agent"].parent / "out.txt"
    cmd = f"time -o {sib_out} /opt/anaconda3/envs/x/bin/python run.py"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] == "deny", res


def test_timeout_path_where_duration_expected_does_not_strip_blindly(experiment_layout):
    """`timeout` only consumes a numeric DURATION. A path in the duration
    slot is not a duration, so it's NOT consumed — `timeout /sibling/x
    600` leaves `/sibling/x` as the (unchecked) command token, identical
    to a bare leading `/sibling/x` (command tokens are never scope-checked
    by policy). No path is dropped or newly exposed."""
    layout = experiment_layout
    # The point is robustness of the DURATION gate, not a deny assertion:
    # a leading command path is allowed with or without the wrapper.
    cmd = f"timeout {layout['sibling_agent']} 600"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] is None, res


def test_known_limitation_bundled_path_option_not_caught(experiment_layout):
    """KNOWN LIMITATION (pre-existing, NOT wrapper-specific): a path fused
    into an option token — `-o/sib/x`, `--output=/sib/x` — is treated as
    a flag by looks_like_path (which rejects any '-'-prefixed token) and
    so is NOT scope-checked, anywhere in the classifier. Demonstrated
    here with a bare command (no wrapper) so the locus is unambiguous:
    the out-of-scope path fused into `-o...` is the ONLY out-of-scope
    token, and it leaks. The identical leak exists with or without a
    wrapper; the wrapper strip itself is not the cause (it correctly
    bails on the flag). 0 occurrences across all real logged commands.
    A real fix belongs in looks_like_path. If one lands, flip to deny.
    """
    layout = experiment_layout
    sib = layout["sibling_agent"].parent / "secret"
    in_scope = layout["cwd"] / "in.txt"
    in_scope.write_text("x\n")
    # Only the bundled -o target is out of scope; the command and its
    # other operand are in scope. Current behavior: allowed (leak).
    cmd = f"grep -f{sib} foo {in_scope}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] is None, res
    # Guard the framing: the SEPARATED form of the same option IS caught,
    # confirming the gap is specifically the fused spelling.
    cmd2 = f"grep -f {sib} foo {in_scope}"
    res2 = run_hook(make_envelope("Bash", {"command": cmd2}, layout["cwd"]),
                    layout["experiment_dir"])
    assert res2["decision"] == "deny", res2


def test_wrapped_interpreter_with_out_of_scope_script_still_denies(experiment_layout):
    """The interpreter (command token) is exempt, but a SCRIPT arg that's
    out of scope is still denied: `timeout 60 python /sibling/x.py`."""
    layout = experiment_layout
    sib = layout["sibling_agent"].parent / "x.py"
    cmd = f"timeout 60 /opt/anaconda3/envs/x/bin/python {sib}"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] == "deny", res


# ---------------------------------------------------------------------
# Interpreter-binary read-scope exemption: running a system interpreter
# is benign, and a bare leading `/opt/.../python foo.py` is already
# allowed (command token unchecked). Denying it only when the path lands
# in a non-command position (assignment, find -exec, loop list) is a
# pure false positive. INTERPRETER_BIN_RE exempts it — matched on the
# REALPATH so a symlink to a secret can't bypass.
# ---------------------------------------------------------------------


def test_interpreter_assignment_then_use_allows(experiment_layout):
    """`PY=/opt/.../python; $PY ...` — the dominant real shape. The
    assignment value is an interpreter path, read-scope exempt, allowed."""
    layout = experiment_layout
    cmd = ("PY=/opt/anaconda3/envs/robophd_demo/bin/python\n"
           "$PY -c 'import ast; ast.parse(open(\"agent.py\").read())'")
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] is None, res


def test_interpreter_in_find_exec_allows(experiment_layout):
    """`find ... -exec /opt/.../python -c ... {} \\;` — interpreter in arg
    position is exempt."""
    layout = experiment_layout
    cmd = ("find . -name '*.py' -exec "
           "/opt/anaconda3/envs/x/bin/python -c 'pass' {} \\;")
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] is None, res


def test_interpreter_in_for_loop_list_allows(experiment_layout):
    """`for p in /opt/.../python /usr/bin/python3; do ...` — interpreter
    paths as loop-list values are exempt."""
    layout = experiment_layout
    cmd = ("for p in /opt/anaconda3/envs/x/bin/python /usr/bin/python3; "
           "do $p -c 1 && break; done")
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] is None, res


def test_non_interpreter_assignment_still_denies(experiment_layout):
    """Safety: the exemption is interpreter-specific. Assigning an
    out-of-scope NON-interpreter path and reading it still denies —
    `X=/sibling/secret; cat $X` (the exfil shape the assignment-value
    guard exists for) is unaffected."""
    layout = experiment_layout
    sib = layout["sibling_agent"]
    cmd = f"X={sib}; cat $X"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] == "deny", res


def test_interpreter_symlink_to_secret_still_denies(experiment_layout):
    """Safety: the exemption matches the REALPATH. A symlink named like an
    interpreter but pointing at an out-of-scope secret resolves to the
    secret and is NOT exempt — no exfil bypass."""
    layout = experiment_layout
    secret = layout["sibling_agent"]  # out of scope
    link = layout["cwd"] / "python3.11"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        import pytest as _pytest
        _pytest.skip("symlinks unavailable")
    # The link sits in cwd, but its realpath is the out-of-scope secret.
    cmd = f"PY={link}; cat $PY"
    res = run_hook(make_envelope("Bash", {"command": cmd}, layout["cwd"]),
                   layout["experiment_dir"])
    assert res["decision"] == "deny", res
