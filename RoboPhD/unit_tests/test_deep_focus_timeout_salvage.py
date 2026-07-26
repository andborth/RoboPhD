"""Timeout handling in DeepFocusEvolutionManager: budget prompts and salvage.

Motivated by asta_paper_finder run 20260724_193339, which died at iteration 11.
That session was not hung — it wrote a valid 3613-line agent.py and a
reasoning.md eight minutes before the 3600s ceiling fired, then ran its own
validation suite (which passed, four minutes after the kill). The framework
raised before it ever looked at the disk, discarding both artifacts and ending
a 22-iteration campaign at 10.

These tests pin the two halves of the fix: sessions are told their budget, and
a timeout falls through to the artifact verification a clean exit would have
gone through. They are the first tests to touch this module, so they also
establish that a timeout is distinguishable from every other failure — the
distinction the salvage rests on.
"""

import subprocess
from pathlib import Path

import pytest

from RoboPhD.deep_focus_evolution_manager import (
    _RECOVERY_TIMEOUT_S,
    DeepFocusEvolutionManager,
)

VALID_AGENT = "def make_solver():\n    return None\n"
# Truncated mid-function, exactly as a kill during a write leaves it.
TRUNCATED_AGENT = "def make_solver():\n    x = {'a': 1,\n"


def _manager(tmp_path, timeout=3600):
    """A manager wired far enough to run one round, with no CLI behind it."""
    mgr = DeepFocusEvolutionManager(timeout=timeout, domain=None)
    mgr._file_mapping = {"agent": "agent.py"}
    mgr.working_dir = tmp_path
    mgr.experiment_dir = tmp_path.parent
    mgr.current_iteration = 11
    mgr.session_id = "test-session"
    return mgr


def _write(tmp_path, *, agent=VALID_AGENT, reasoning="# plan\nName: test-agent\n"):
    if agent is not None:
        (tmp_path / "agent.py").write_text(agent)
    if reasoning is not None:
        (tmp_path / "reasoning.md").write_text(reasoning)


def _round1(mgr):
    return mgr._round1_analysis_and_implementation(
        "strategy guidance", continue_from_analyzer=False
    )


@pytest.fixture
def timed_out(monkeypatch):
    """Make the CLI raise TimeoutExpired with no completion marker anywhere."""
    def _raise(*a, **kw):
        raise subprocess.TimeoutExpired(cmd=["claude"], timeout=kw.get("timeout", 3600))

    monkeypatch.setattr(
        "RoboPhD.deep_focus_evolution_manager.call_claude_cli", _raise
    )
    monkeypatch.setattr(
        DeepFocusEvolutionManager, "_get_claude_cli_path", lambda self: Path("/bin/true")
    )
    monkeypatch.setattr(
        DeepFocusEvolutionManager, "_marker_in_transcript", lambda self, marker: False
    )


# --- salvage: the iteration-11 case ----------------------------------------


def test_timeout_with_intact_artifacts_survives(tmp_path, timed_out):
    """The regression this whole change exists for: complete work on disk at
    the ceiling must not end the run."""
    _write(tmp_path)
    _round1(_manager(tmp_path))  # must not raise
    assert (tmp_path / "agent.py").read_text() == VALID_AGENT


def test_timeout_with_truncated_artifact_raises(tmp_path, timed_out):
    """A half-written agent.py imports cleanly up to the truncation point, so
    it must be rejected rather than shipped into the pool."""
    _write(tmp_path, agent=TRUNCATED_AGENT)
    with pytest.raises(RuntimeError, match="does not parse"):
        _round1(_manager(tmp_path))


def test_timeout_with_empty_artifact_raises(tmp_path, timed_out):
    _write(tmp_path, agent="")
    with pytest.raises(RuntimeError, match="is empty"):
        _round1(_manager(tmp_path))


def test_non_timeout_failure_still_raises(tmp_path, monkeypatch):
    """Salvage is timeout-only. A CLI that never launched has no claim on
    whatever happens to be on disk."""
    _write(tmp_path)
    monkeypatch.setattr(
        DeepFocusEvolutionManager, "_get_claude_cli_path", lambda self: Path("/bin/true")
    )
    monkeypatch.setattr(
        "RoboPhD.deep_focus_evolution_manager.call_claude_cli",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("cli missing")),
    )
    with pytest.raises(RuntimeError, match="Round 1 .* failed"):
        _round1(_manager(tmp_path))


def test_timeout_with_missing_artifact_reaches_recovery(tmp_path, timed_out):
    """The tie-in: a missing file on the timeout path routes into the existing
    recovery, bounded rather than given another full evolution timeout."""
    _write(tmp_path, agent=None)
    mgr = _manager(tmp_path)
    calls = []

    def _fake_recover(expected_path, filename, description):
        calls.append(filename)
        Path(expected_path).write_text(VALID_AGENT)
        return True

    mgr._recover_missing_file = _fake_recover
    _round1(mgr)
    assert calls == ["agent.py"]


def test_recovery_call_is_bounded(tmp_path, monkeypatch):
    """_recover_missing_file must not inherit the full evolution timeout."""
    monkeypatch.setattr(
        DeepFocusEvolutionManager, "_get_claude_cli_path", lambda self: Path("/bin/true")
    )
    seen = {}

    def _capture(*a, **kw):
        seen["timeout"] = kw["timeout"]
        raise subprocess.TimeoutExpired(cmd=["claude"], timeout=kw["timeout"])

    monkeypatch.setattr(
        "RoboPhD.deep_focus_evolution_manager.call_claude_cli", _capture
    )
    monkeypatch.setattr(
        DeepFocusEvolutionManager, "_marker_in_transcript", lambda self, marker: False
    )
    mgr = _manager(tmp_path, timeout=3600)
    mgr._recover_missing_file(tmp_path / "agent.py", "agent.py", "the agent")
    assert seen["timeout"] == _RECOVERY_TIMEOUT_S
    assert seen["timeout"] < mgr.timeout


# --- the budget line -------------------------------------------------------


@pytest.mark.parametrize("timeout,minutes", [(3600, "60"), (300, "5")])
def test_round1_prompt_states_the_budget(tmp_path, timed_out, timeout, minutes):
    """Parameterized so a hardcoded number cannot pass: the stated minutes
    have to track the configured timeout."""
    _write(tmp_path)
    _round1(_manager(tmp_path, timeout=timeout))
    prompt = (tmp_path / "evolution_prompt.md").read_text()
    assert f"capped at {minutes} minutes" in prompt


def test_recovery_prompt_states_no_budget(tmp_path, monkeypatch):
    """Enforce-don't-announce: the recovery call is bounded in code, but the
    prompt stays silent — a stated allowance invites a one-file task to fill
    it. Asserted on both halves so a later edit can't reintroduce the number.
    """
    monkeypatch.setattr(
        DeepFocusEvolutionManager, "_get_claude_cli_path", lambda self: Path("/bin/true")
    )
    seen = {}

    def _capture(cmd, **kw):
        seen["prompt"] = cmd[cmd.index("--print") + 1]
        seen["timeout"] = kw["timeout"]
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kw["timeout"])

    monkeypatch.setattr(
        "RoboPhD.deep_focus_evolution_manager.call_claude_cli", _capture
    )
    monkeypatch.setattr(
        DeepFocusEvolutionManager, "_marker_in_transcript", lambda self, marker: False
    )
    mgr = _manager(tmp_path)
    mgr._recover_missing_file(tmp_path / "agent.py", "agent.py", "the agent")
    assert "minutes of wall clock" not in seen["prompt"]
    assert str(_RECOVERY_TIMEOUT_S) not in seen["prompt"]
    assert seen["timeout"] == _RECOVERY_TIMEOUT_S


# --- the flag the salvage rests on -----------------------------------------


def test_timeout_flag_resets_between_calls(tmp_path, monkeypatch):
    """A stale True would let a later non-timeout failure salvage artifacts it
    has no claim on."""
    monkeypatch.setattr(
        DeepFocusEvolutionManager, "_get_claude_cli_path", lambda self: Path("/bin/true")
    )
    monkeypatch.setattr(
        DeepFocusEvolutionManager, "_marker_in_transcript", lambda self, marker: False
    )
    mgr = _manager(tmp_path)

    monkeypatch.setattr(
        "RoboPhD.deep_focus_evolution_manager.call_claude_cli",
        lambda *a, **kw: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(cmd=["claude"], timeout=1)
        ),
    )
    mgr._call_claude_code("p", continue_session=False, expected_completion="X")
    assert mgr._last_call_timed_out is True

    monkeypatch.setattr(
        "RoboPhD.deep_focus_evolution_manager.call_claude_cli",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("boom")),
    )
    mgr._call_claude_code("p", continue_session=False, expected_completion="X")
    assert mgr._last_call_timed_out is False
