"""Tests for researcher.build_prior_evolution_section — the
include_prior_evolution prompt section pointing evolution sessions at
prior sessions' reasoning.md / evolution_reflection.md."""

from RoboPhD.researcher import build_prior_evolution_section

OPINIONS = (
    "These prior session analyses are available to you. "
    "Feel free to form your own opinions."
)


def _mk_iter(base, num, files=("reasoning.md", "evolution_reflection.md")):
    d = base / "evolution_output" / f"iteration_{num:03d}"
    d.mkdir(parents=True, exist_ok=True)
    for name in files:
        (d / name).write_text(f"# {name} for iteration {num}\n")
    return d


def test_no_evolution_output_dir_yields_empty(tmp_path):
    assert build_prior_evolution_section(tmp_path, 2) == ""


def test_first_evolution_round_yields_empty(tmp_path):
    """Iteration 2 is the first evolution round — its own dir may already
    exist (created for the write target), but nothing PRIOR does."""
    _mk_iter(tmp_path, 2)
    assert build_prior_evolution_section(tmp_path, 2) == ""


def test_points_at_latest_prior_with_wording(tmp_path):
    for n in (2, 3, 4):
        _mk_iter(tmp_path, n)
    section = build_prior_evolution_section(tmp_path, 5)
    assert section.startswith("## Prior Evolution Sessions")
    assert "evolution_output/iteration_004/reasoning.md" in section
    assert "evolution_output/iteration_004/evolution_reflection.md" in section
    assert OPINIONS in section
    # Earlier iterations noted collectively, not enumerated as paths
    assert "iterations 2–3" in section
    assert "iteration_002/reasoning.md" not in section


def test_current_and_future_iterations_excluded(tmp_path):
    _mk_iter(tmp_path, 2)
    _mk_iter(tmp_path, 5)  # current
    _mk_iter(tmp_path, 6)  # stale leftover beyond current
    section = build_prior_evolution_section(tmp_path, 5)
    assert "iteration_002" in section
    assert "iteration_005" not in section
    assert "iteration_006" not in section


def test_latest_prior_is_existence_checked_not_arithmetic(tmp_path):
    """If iteration 4's session died before writing artifacts, the section
    points at 3 — not at a nonexistent N-1."""
    _mk_iter(tmp_path, 2)
    _mk_iter(tmp_path, 3)
    (tmp_path / "evolution_output" / "iteration_004").mkdir()  # empty dir
    section = build_prior_evolution_section(tmp_path, 5)
    assert "iteration_003/reasoning.md" in section
    assert "iteration_004" not in section


def test_only_allowlisted_artifacts_are_listed(tmp_path):
    _mk_iter(
        tmp_path, 2,
        files=("reasoning.md", "evolution_reflection.md",
               "session_summary.md", "evolution_prompt.md"),
    )
    section = build_prior_evolution_section(tmp_path, 3)
    assert "session_summary.md" not in section
    assert "evolution_prompt.md" not in section
    assert "reasoning.md" in section


def test_dir_with_only_nonallowlisted_files_treated_as_empty(tmp_path):
    _mk_iter(tmp_path, 2, files=("session_summary.md", "evolution_prompt.md"))
    assert build_prior_evolution_section(tmp_path, 3) == ""


def test_partial_artifacts_list_only_what_exists(tmp_path):
    _mk_iter(tmp_path, 2, files=("reasoning.md",))
    section = build_prior_evolution_section(tmp_path, 3)
    assert "iteration_002/reasoning.md" in section
    assert "evolution_reflection.md" not in section


def test_non_iteration_entries_ignored(tmp_path):
    base = tmp_path / "evolution_output"
    base.mkdir()
    (base / "CLAUDE.md").write_text("# not a dir\n")
    (base / "scratch").mkdir()
    _mk_iter(tmp_path, 2)
    section = build_prior_evolution_section(tmp_path, 3)
    assert "iteration_002" in section
    assert "scratch" not in section


def test_single_earlier_iteration_uses_singular_wording(tmp_path):
    _mk_iter(tmp_path, 2)
    _mk_iter(tmp_path, 3)
    section = build_prior_evolution_section(tmp_path, 4)
    assert "iteration 2)" in section  # singular, no dash range
    assert "–" not in section
