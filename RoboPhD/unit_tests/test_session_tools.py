"""Tests for the session_tools materialization mechanism.

Session tools are task-shipped helper scripts copied into
<experiment>/session_tools/ at researcher startup: readable and runnable
by evolution sessions, outside their write root, refreshed on resume so
repo-side fixes propagate.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from RoboPhD.researcher import ParallelAgentEvolver, ParallelAgentResearcher


def _stub(experiment_dir: Path):
    return SimpleNamespace(experiment_dir=experiment_dir)


def _materialize(stub, paths):
    return ParallelAgentResearcher._materialize_session_tools(stub, paths)


def test_copies_files_into_session_tools(tmp_path):
    src = tmp_path / "probe.py"
    src.write_text("print('hi')\n")
    exp = tmp_path / "exp"
    exp.mkdir()

    _materialize(_stub(exp), [str(src)])

    assert (exp / "session_tools" / "probe.py").read_text() == "print('hi')\n"


def test_overwrites_on_restart(tmp_path):
    """A resumed run must pick up repo-side fixes to the shipped scripts."""
    src = tmp_path / "probe.py"
    src.write_text("v1\n")
    exp = tmp_path / "exp"
    exp.mkdir()
    _materialize(_stub(exp), [str(src)])

    src.write_text("v2\n")
    _materialize(_stub(exp), [str(src)])

    assert (exp / "session_tools" / "probe.py").read_text() == "v2\n"


def test_missing_source_fails_loudly(tmp_path):
    exp = tmp_path / "exp"
    exp.mkdir()
    with pytest.raises(FileNotFoundError, match="session_tools"):
        _materialize(_stub(exp), [str(tmp_path / "nope.py")])


def test_duplicate_basenames_rejected_before_any_copy(tmp_path):
    """Flat-by-filename materialization must refuse colliding basenames
    loudly instead of letting the last entry silently clobber."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "probe.py").write_text("A")
    (b / "probe.py").write_text("B")
    exp = tmp_path / "exp"
    exp.mkdir()

    with pytest.raises(ValueError, match="basename"):
        _materialize(_stub(exp), [str(a / "probe.py"), str(b / "probe.py")])

    # Validation precedes copying: nothing was materialized.
    assert not (exp / "session_tools").exists()


def test_no_paths_creates_nothing(tmp_path):
    exp = tmp_path / "exp"
    exp.mkdir()
    _materialize(_stub(exp), None)
    _materialize(_stub(exp), [])
    assert not (exp / "session_tools").exists()


def test_experiment_structure_lists_session_tools(tmp_path):
    """The evolution prompt's structure block advertises shipped scripts,
    generated from what is actually on disk (no hardcoded filenames)."""
    exp = tmp_path / "exp"
    (exp / "iteration_001" / "agent_seed").mkdir(parents=True)
    (exp / "session_tools").mkdir()
    (exp / "session_tools" / "tool_probe.py").write_text("x")

    stub = SimpleNamespace(
        experiment_dir=exp,
        domain=SimpleNamespace(experiment_structure_docs="DOCS"),
    )
    text = ParallelAgentEvolver._get_experiment_structure(stub, 1)

    assert "session_tools" in text
    assert "tool_probe.py" in text


def test_experiment_structure_silent_without_session_tools(tmp_path):
    exp = tmp_path / "exp"
    (exp / "iteration_001" / "agent_seed").mkdir(parents=True)

    stub = SimpleNamespace(
        experiment_dir=exp,
        domain=SimpleNamespace(experiment_structure_docs="DOCS"),
    )
    text = ParallelAgentEvolver._get_experiment_structure(stub, 1)

    assert "session_tools" not in text
