"""Tests for optimize_anything's seed_agents parameter and the agent-directory
resolution behind it.

A run's seed pool is `{agent name: directory-or-artifacts}`. The engine always
supported several seeds (ParallelAgentResearcher.load_initial_agents takes a
list); only the API layer was wired to one, and it had to invent that agent's
name because a bare artifacts dict carries no identity.

The tests stop each run at researcher.run() rather than letting evolution
spawn, so they assert on what the API hands the engine.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RoboPhD.api import (  # noqa: E402
    AutoresearchConfig,
    GEPAConfig,
    RoboPhDConfig,
    _resolve_seed_agents,
    optimize_anything,
)
from RoboPhD.candidate_utils import read_agent_dir  # noqa: E402
from RoboPhD.runner_utils import _resolve_agent_dir  # noqa: E402


class _Stop(Exception):
    """Raised from the stubbed run() to end the call after pool setup."""


@pytest.fixture
def launch(tmp_path):
    """Run optimize_anything up to researcher.run() and capture the wiring.

    Returns a callable taking optimize_anything kwargs and returning
    {"initial_agents", "config", "agents_directory"}.
    """

    def _launch(**kwargs):
        captured = {}

        class FakeResearcher:
            def __init__(self, **kw):
                captured["config"] = kw["config_manager"].get_config(1)
                self.experiment_dir = tmp_path / "experiment"

            def run(self, initial_agents=None):
                captured["initial_agents"] = initial_agents
                raise _Stop()

        with patch("RoboPhD.researcher.ParallelAgentResearcher", FakeResearcher):
            with pytest.raises(_Stop):
                optimize_anything(
                    evaluator=lambda candidate, example: (0.0, {}),
                    dataset=[{"x": 1}],
                    config=RoboPhDConfig(parent_experiments_dir=str(tmp_path)),
                    **kwargs,
                )
        captured["agents_directory"] = Path(captured["config"]["agents_directory"])
        return captured

    return _launch


@pytest.fixture
def agent_dir(tmp_path):
    """A realistic seed directory: two artifacts plus build residue."""
    source = tmp_path / "source" / "baseline"
    (source / "__pycache__").mkdir(parents=True)
    (source / "agent.py").write_text("AGENT")
    (source / "analyze_db.py").write_text("ANALYZE")
    (source / "__pycache__" / "agent.cpython-311.pyc").write_bytes(b"\x00\x01binary")
    (source / ".DS_Store").write_bytes(b"\x00")
    return source


# -- materialization ----------------------------------------------------


def test_every_seed_is_materialized_under_one_container(launch):
    seeds = {
        "seed_a": {"agent.py": "AAA"},
        "seed_b": {"agent.py": "BBB"},
        "seed_c": {"agent.py": "CCC"},
    }
    captured = launch(seed_agents=seeds)

    root = captured["agents_directory"]
    assert root.name.startswith("seeds_"), (
        "the container is per-run so the seed dirs inside it can carry the "
        "caller's names verbatim"
    )
    assert sorted(d.name for d in root.iterdir()) == ["seed_a", "seed_b", "seed_c"]
    for name, candidate in seeds.items():
        assert (root / name / "agent.py").read_text() == candidate["agent.py"]


def test_all_seeds_reach_the_engine_in_order(launch):
    """Both the config and the run() argument must carry every seed: the
    config drives agents_directory resolution, the argument is what
    load_initial_agents actually iterates."""
    seeds = {f"seed_{i}": {"agent.py": str(i)} for i in range(4)}
    captured = launch(seed_agents=seeds)

    expected = ["seed_0", "seed_1", "seed_2", "seed_3"]
    assert captured["initial_agents"] == expected
    assert captured["config"]["initial_agents"] == expected


def test_one_seed_takes_the_same_path_and_keeps_its_name(launch):
    """The single-seed case is not special-cased. It also must not acquire a
    random suffix — the whole point of the per-run container is that
    `baseline` stays `baseline` in Elo tables, reports and --eval-agent."""
    captured = launch(seed_agents={"baseline": {"agent.py": "SEED"}})

    assert captured["initial_agents"] == ["baseline"]
    root = captured["agents_directory"]
    assert root.name.startswith("seeds_")
    assert (root / "baseline" / "agent.py").read_text() == "SEED"


def test_directory_values_are_read(launch, agent_dir):
    captured = launch(seed_agents={"baseline": agent_dir})

    materialized = captured["agents_directory"] / "baseline"
    assert (materialized / "agent.py").read_text() == "AGENT"
    assert (materialized / "analyze_db.py").read_text() == "ANALYZE"


def test_directory_and_dict_seeds_can_be_mixed(launch, agent_dir):
    captured = launch(
        seed_agents={
            "from_disk": agent_dir,
            "in_memory": {"agent.py": "X", "analyze_db.py": "Y"},
        }
    )

    assert captured["initial_agents"] == ["from_disk", "in_memory"]
    root = captured["agents_directory"]
    assert (root / "from_disk" / "agent.py").read_text() == "AGENT"
    assert (root / "in_memory" / "analyze_db.py").read_text() == "Y"


def test_nested_artifact_paths_survive(launch):
    """Keys are paths within the agent dir, so subdirectories round-trip."""
    captured = launch(seed_agents={"baseline": {"agent.py": "A", "lib/util.py": "U"}})

    root = captured["agents_directory"] / "baseline"
    assert (root / "lib" / "util.py").read_text() == "U"


def test_the_engine_loads_every_seed_from_the_container(launch, tmp_path):
    """The API's wiring is only half the contract — load_initial_agents has
    to resolve all of it. Exercised against the real method (bound to a stub
    holding just the attributes it touches) so a container layout the
    researcher can't read fails here rather than at run time."""
    from types import SimpleNamespace

    from RoboPhD.researcher import ParallelAgentResearcher

    seeds = {f"seed_{label}": {"agent.py": label} for label in "abcd"}
    captured = launch(seed_agents=seeds)

    stub = SimpleNamespace(
        agents_directory=str(captured["agents_directory"]),
        domain=SimpleNamespace(file_mapping={"agent.py": "agent.py"}),
        experiment_dir=tmp_path / "loaded",
        agent_pool={},
        performance_records={},
        _is_valid_agent_dir=lambda d: ParallelAgentResearcher._is_valid_agent_dir(
            stub, d
        ),
    )
    stub.experiment_dir.mkdir()

    ParallelAgentResearcher.load_initial_agents(stub, captured["initial_agents"])

    assert sorted(stub.agent_pool) == sorted(seeds)
    assert all(record["elo"] == 1500 for record in stub.performance_records.values())
    for name, candidate in seeds.items():
        copied = stub.experiment_dir / "agents" / name / "agent.py"
        assert copied.read_text() == candidate["agent.py"]


# -- selection ----------------------------------------------------------


def _selection_stub(agent_ids, agents_per_iteration=3, **overrides):
    """Minimal stand-in carrying only what select_agents_for_iteration reads."""
    from types import SimpleNamespace

    from RoboPhD.researcher import ParallelAgentResearcher

    records = {
        agent_id: {
            "test_count": 0, "elo": 1500,
            "last_win_iteration": None, "last_test_iteration": None,
        }
        for agent_id in agent_ids
    }
    for agent_id, fields in overrides.items():
        records[agent_id].update(fields)
    stub = SimpleNamespace(
        agent_pool={agent_id: {} for agent_id in agent_ids},
        performance_records=records,
        agents_per_iteration=agents_per_iteration,
        evolver=SimpleNamespace(
            use_greedy_selection=False, use_challenger_selection=False
        ),
    )
    # Real pending-winner logic, not a fake: it decides how many slots are
    # left for the untested seed.
    stub._get_pending_winners = lambda: ParallelAgentResearcher._get_pending_winners(
        stub
    )
    return stub


def test_a_fourth_seed_is_tested_at_iteration_2():
    """Seeding more agents than agents_per_iteration must not strand one.
    Iteration 1 fills its 3 slots from 4 untested seeds; iteration 2 spends
    two slots on the pending winner and the evolved agent, and Priority 3
    claims the third with the seed that sat out. This is why seeding 4
    agents does NOT call for raising agents_per_iteration to 4."""
    from RoboPhD.researcher import ParallelAgentResearcher

    seeds = ["seed_a", "seed_b", "seed_c", "seed_d"]
    stub = _selection_stub(seeds)

    first = ParallelAgentResearcher.select_agents_for_iteration(stub, 1)
    assert len(first) == 3
    assert set(first) < set(seeds), "iteration 1 can only test 3 of the 4"

    # Iteration 1's outcome: one seed won, all three tested are now tested.
    winner, *rest = first
    (held_back,) = set(seeds) - set(first)
    for agent_id in first:
        stub.performance_records[agent_id].update(
            test_count=1, last_test_iteration=1
        )
    stub.performance_records[winner]["last_win_iteration"] = 1
    stub.agent_pool["iter2_evolved"] = {}
    stub.performance_records["iter2_evolved"] = {
        "test_count": 0, "elo": 1500,
        "last_win_iteration": None, "last_test_iteration": None,
    }

    second = ParallelAgentResearcher.select_agents_for_iteration(
        stub, 2, evolved_agent_id="iter2_evolved"
    )

    assert set(second) == {winner, "iter2_evolved", held_back}


# -- engine scope -------------------------------------------------------


@pytest.mark.parametrize(
    "retired, expected",
    [
        ("seed_candidate", "replaced by seed_agents"),
        ("seed_candidates", "renamed to seed_agents"),
    ],
)
def test_retired_seed_parameters_explain_the_migration(retired, expected):
    """Dropping the old parameters outright leaves callers with a bare
    'unexpected keyword argument' — loud but silent about the fix. This is
    the project's primary public API, so retirement states the replacement,
    the way a retired model handle does."""
    with pytest.raises(TypeError) as exc_info:
        optimize_anything(
            evaluator=lambda candidate, example: (0.0, {}),
            dataset=[{"x": 1}],
            **{retired: {"agent.py": "A"}},
        )

    message = str(exc_info.value)
    assert expected in message
    assert "seed_agents" in message


def test_an_ordinary_typo_keeps_pythons_own_wording():
    """**kwargs must not degrade unrelated mistakes into the migration
    error, or a plain misspelling gets a confusing answer."""
    with pytest.raises(TypeError) as exc_info:
        optimize_anything(
            evaluator=lambda candidate, example: (0.0, {}),
            dataset=[{"x": 1}],
            seed_agentz={"baseline": {"agent.py": "A"}},
        )

    assert (
        str(exc_info.value)
        == "optimize_anything() got an unexpected keyword argument 'seed_agentz'"
    )


def test_no_seed_argument_raises_on_fresh_run(tmp_path):
    with pytest.raises(ValueError, match="required for fresh runs"):
        optimize_anything(
            evaluator=lambda candidate, example: (0.0, {}),
            dataset=[{"x": 1}],
            config=RoboPhDConfig(parent_experiments_dir=str(tmp_path)),
        )


@pytest.mark.parametrize("config_cls", [GEPAConfig, AutoresearchConfig])
def test_multi_seed_pool_rejected_by_single_agent_engines(config_cls):
    """GEPA and Autoresearch evolve one agent. Silently taking the first
    entry would discard the rest without saying so."""
    with pytest.raises(ValueError, match="optimizes a single agent"):
        optimize_anything(
            evaluator=lambda candidate, example: (0.0, {}),
            dataset=[{"x": 1}],
            seed_agents={"a": {"agent.py": "A"}, "b": {"agent.py": "B"}},
            config=config_cls(),
        )


@pytest.mark.parametrize("config_cls", [GEPAConfig, AutoresearchConfig])
def test_single_seed_pool_reaches_single_agent_engines(config_cls, agent_dir):
    """A one-entry pool is unwrapped to the bare candidate those engines take,
    directory form included."""
    engine = "gepa" if config_cls is GEPAConfig else "autoresearch"
    seen = {}

    def fake_engine(evaluator, dataset, seed_candidate, *args):
        seen["candidate"] = seed_candidate
        raise _Stop()

    with patch(f"RoboPhD.engines.{engine}.run_{engine}", fake_engine):
        with pytest.raises(_Stop):
            optimize_anything(
                evaluator=lambda candidate, example: (0.0, {}),
                dataset=[{"x": 1}],
                seed_agents={"baseline": agent_dir},
                config=config_cls(),
            )

    assert seen["candidate"] == {"agent.py": "AGENT", "analyze_db.py": "ANALYZE"}


# -- name and shape validation ------------------------------------------


def test_iter_prefixed_name_raises_and_explains_the_collision():
    """researcher's --from-iteration archival parses `iter<N>_` off any dir
    in agents/ to decide what to move out, so a seed carrying its source
    run's name would be read as an iteration of THIS run and archived away.
    The message has to say so, or the fix looks arbitrary."""
    with pytest.raises(ValueError) as exc_info:
        _resolve_seed_agents({"iter15_verdict_repair": {"agent.py": "A"}})

    message = str(exc_info.value)
    assert "from-iteration" in message
    assert "seed_iter15_verdict_repair" in message, "should suggest the fix"


def test_iter_without_a_digit_is_allowed():
    """The archival parser only claims `iter<N>`; `iterative_foo` is a
    legitimate name and must not be collateral damage."""
    _resolve_seed_agents({"iterative_refiner": {"agent.py": "A"}})


@pytest.mark.parametrize(
    "name", ["a/b", "../escape", "sub/dir", "", "has space", "sneaky\\win"]
)
def test_unsafe_names_raise(name):
    with pytest.raises(ValueError, match="Invalid seed name"):
        _resolve_seed_agents({name: {"agent.py": "A"}})


@pytest.mark.parametrize("name", [".", ".."])
def test_relative_path_names_raise(name):
    """'.' and '..' are made entirely of otherwise-allowed characters, so a
    character-class check alone lets them through — and then
    `container / ".."` materializes the agent OUTSIDE its own container, into
    the seeds directory shared by every run."""
    with pytest.raises(ValueError, match="Invalid seed name"):
        _resolve_seed_agents({name: {"agent.py": "A"}})


@pytest.mark.parametrize("name", [".hidden", "-x"])
def test_names_cannot_start_with_dot_or_dash(name):
    """A leading dot contradicts read_agent_dir, which skips dot-prefixed
    entries; a leading dash is ambiguous as a value to --eval-agent."""
    with pytest.raises(ValueError, match="Invalid seed name"):
        _resolve_seed_agents({name: {"agent.py": "A"}})


def test_leading_underscore_is_allowed():
    """Only the characters with a concrete failure mode are excluded."""
    _resolve_seed_agents({"_private": {"agent.py": "A"}})


def test_artifacts_dict_passed_as_the_pool_raises_pointing_at_the_shape():
    """The likely mistake: passing one agent's artifacts where a pool of
    agents belongs. Before the type check this survived validation (the key
    matched the name regex, and frozenset over a str yields characters) and
    died inside materialize_candidate with 'string indices must be
    integers'."""
    with pytest.raises(ValueError) as exc_info:
        _resolve_seed_agents({"agent.py": "def solve(): pass"})

    message = str(exc_info.value)
    assert "keyed by AGENT name" in message
    assert "str" in message


def test_non_text_artifact_value_raises():
    with pytest.raises(ValueError, match="not text"):
        _resolve_seed_agents({"baseline": {"agent.py": b"bytes"}})


def test_mismatched_artifacts_raise():
    """One file_mapping is derived from one seed and applied to all; a seed
    missing a mapped file is dropped by the researcher's agent-dir check
    with only a printed warning."""
    with pytest.raises(ValueError, match="same artifacts"):
        _resolve_seed_agents(
            {
                "seed_a": {"agent.py": "A", "notes.md": "n"},
                "seed_b": {"agent.py": "B"},
            }
        )


def test_empty_artifacts_raise():
    with pytest.raises(ValueError, match="no artifacts"):
        _resolve_seed_agents({"seed_a": {}})


def test_empty_pool_raises():
    with pytest.raises(ValueError, match="non-empty"):
        _resolve_seed_agents({})


# -- read_agent_dir -----------------------------------------------------


def test_read_agent_dir_skips_build_residue(agent_dir):
    """asta_ds1000's seed dir carries a stray __pycache__/*.pyc on disk.
    Without the filter it would reach evolution as an editable artifact —
    and the binary read would raise before that."""
    artifacts = read_agent_dir(agent_dir)

    assert sorted(artifacts) == ["agent.py", "analyze_db.py"]


def test_read_agent_dir_keeps_nested_paths(tmp_path):
    source = tmp_path / "agent"
    (source / "lib").mkdir(parents=True)
    (source / "agent.py").write_text("A")
    (source / "lib" / "util.py").write_text("U")

    assert read_agent_dir(source) == {"agent.py": "A", "lib/util.py": "U"}


def test_read_agent_dir_rejects_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Seed agent directory not found"):
        read_agent_dir(tmp_path / "nope")


def test_read_agent_dir_rejects_an_empty_directory(tmp_path):
    empty = tmp_path / "empty"
    (empty / "__pycache__").mkdir(parents=True)
    (empty / "__pycache__" / "x.pyc").write_bytes(b"\x00")

    with pytest.raises(ValueError, match="no artifacts"):
        read_agent_dir(empty)


def test_read_agent_dir_reports_binary_artifacts_by_path(tmp_path):
    source = tmp_path / "agent"
    source.mkdir()
    (source / "blob.bin").write_bytes(b"\xff\xfe\x00")

    with pytest.raises(ValueError, match="not UTF-8 text"):
        read_agent_dir(source)


# -- agent-directory resolution -----------------------------------------


def _pool(package_dir):
    return {"winner": {"package_dir": str(package_dir)}}


def test_absolute_package_dir_does_not_escape_a_moved_run(tmp_path):
    """Some checkpoints store package_dir absolute. Since Path(x) / "/abs"
    yields the absolute path, an archived copy of such a run would resolve
    to wherever it originally lived — a directory that can be deleted or
    edited independently of the snapshot being asked about."""
    original = tmp_path / "original_run" / "agents" / "winner"
    original.mkdir(parents=True)
    (original / "agent.py").write_text("STALE")

    archived = tmp_path / "archived_run"
    (archived / "agents" / "winner").mkdir(parents=True)
    (archived / "agents" / "winner" / "agent.py").write_text("SNAPSHOT")

    resolved = _resolve_agent_dir(archived, "winner", _pool(original), {}, None)

    assert resolved == archived / "agents" / "winner"
    assert (resolved / "agent.py").read_text() == "SNAPSHOT"


def test_relative_package_dir_resolves_as_before(tmp_path):
    run = tmp_path / "run"
    (run / "agents" / "winner").mkdir(parents=True)

    resolved = _resolve_agent_dir(run, "winner", _pool("agents/winner"), {}, None)

    assert resolved == run / "agents" / "winner"


def test_non_canonical_layout_still_uses_the_stored_value(tmp_path):
    """The stored package_dir stays the fallback for anything the canonical
    <run>/agents/<id> path doesn't cover."""
    run = tmp_path / "run"
    (run / "elsewhere" / "winner").mkdir(parents=True)

    resolved = _resolve_agent_dir(
        run, "winner", _pool("elsewhere/winner"), {}, None
    )

    assert resolved == run / "elsewhere" / "winner"


def test_missing_directory_still_raises(tmp_path):
    run = tmp_path / "run"
    run.mkdir()

    with pytest.raises(FileNotFoundError, match="Agent directory not found"):
        _resolve_agent_dir(run, "winner", _pool("agents/winner"), {}, None)
