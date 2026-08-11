"""Tests for runner_utils.resolve_seed_runs, the resolver behind the
examples' --seed-runs flag.

Shared by asta_ds1000 and asta_paper_finder, which bind it with their own
example run dir for the malformed-spec message and otherwise use it
unchanged. It lives here rather than in either example for the reason the
cost validators do: two copies means a message or validation fix in one
silently misses the other.

The examples' own suites cover that main.py calls this at all and hands the
result to optimize_anything; the resolution semantics are covered here and
not duplicated there.
"""

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RoboPhD.runner_utils import resolve_seed_runs  # noqa: E402

EXAMPLE = "029=example_runs/robophd/asta_ds1000/v0_0_5_soft_cap_0_05"


def _robophd_run(root: Path, name: str, agents: dict) -> Path:
    """A minimal RoboPhD run dir: checkpoint.json + agents/<name>/agent.py.

    `agents` maps agent name -> Elo, so a test can state which one should win
    without hand-building performance_records.
    """
    run_dir = root / name
    (run_dir / "agents").mkdir(parents=True)
    for agent_name in agents:
        agent_dir = run_dir / "agents" / agent_name
        agent_dir.mkdir()
        (agent_dir / "agent.py").write_text(f"# {agent_name}\n")
    (run_dir / "checkpoint.json").write_text(json.dumps({
        "agent_pool": {
            a: {"package_dir": f"agents/{a}"} for a in agents
        },
        "performance_records": {
            a: {"elo": elo, "mean_score": 0.5, "test_count": 3}
            for a, elo in agents.items()
        },
    }))
    return run_dir


def _single_candidate_run(root: Path, name: str) -> Path:
    """A GEPA / Autoresearch run dir: best_agent/, no checkpoint."""
    run_dir = root / name
    (run_dir / "best_agent").mkdir(parents=True)
    (run_dir / "best_agent" / "agent.py").write_text("# winner\n")
    return run_dir


def test_names_are_prefixed_and_ordered(tmp_path):
    """The seed_ prefix is formed by the resolver, not taken from the
    operator: it keeps seeds visibly distinct from a run's own evolved
    agents, and makes the iter<N>_ name the API rejects unreachable from
    this flag."""
    a = _robophd_run(tmp_path, "run_a", {"iter3_a": 1500})
    b = _robophd_run(tmp_path, "run_b", {"iter7_b": 1500})

    seeds = resolve_seed_runs([f"first={a}", f"second={b}"], example=EXAMPLE)

    assert list(seeds) == ["seed_first", "seed_second"]


def test_a_robophd_run_contributes_its_elo_winner(tmp_path):
    """Resolved from the run rather than from a caller-supplied name, so a
    seed cannot disagree with what that run actually produced."""
    run_dir = _robophd_run(tmp_path, "run", {
        "iter4_early": 1500, "iter9_champion": 1620, "iter11_late": 1580,
    })

    seeds = resolve_seed_runs([f"x={run_dir}"], example=EXAMPLE)

    assert seeds["seed_x"] == run_dir / "agents" / "iter9_champion"


def test_a_single_candidate_run_contributes_best_agent(tmp_path):
    """GEPA and Autoresearch optimize one candidate and write no
    checkpoint.json. Resolving only through find_best_agent would silently
    exclude every run those two engines have ever produced."""
    run_dir = _single_candidate_run(tmp_path, "autoresearch_run")

    seeds = resolve_seed_runs([f"auto={run_dir}"], example=EXAMPLE)

    assert seeds == {"seed_auto": run_dir / "best_agent"}


def test_checkpoint_wins_when_a_run_has_both_shapes(tmp_path):
    """A RoboPhD run that also happens to carry a best_agent/ dir must
    still resolve by Elo — the checkpoint is the authoritative record of
    what that run selected."""
    run_dir = _robophd_run(tmp_path, "run", {"iter5_winner": 1600})
    (run_dir / "best_agent").mkdir()
    (run_dir / "best_agent" / "agent.py").write_text("# stale\n")

    seeds = resolve_seed_runs([f"x={run_dir}"], example=EXAMPLE)

    assert seeds["seed_x"] == run_dir / "agents" / "iter5_winner"


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("nolabel", "not LABEL=RUN_DIR"),
        ("=/some/dir", "not LABEL=RUN_DIR"),
        ("label=", "not LABEL=RUN_DIR"),
        ("label=/definitely/not/a/run", "neither a checkpoint.json"),
    ],
)
def test_rejects_bad_specs(spec, expected):
    with pytest.raises(SystemExit, match=re.escape(expected)):
        resolve_seed_runs([spec], example=EXAMPLE)


def test_the_caller_supplied_example_reaches_the_error():
    """Each example binds its own run dir here; a generic one would send the
    operator looking in the wrong archive."""
    with pytest.raises(SystemExit, match=re.escape(EXAMPLE)):
        resolve_seed_runs(["nolabel"], example=EXAMPLE)


def test_rejects_duplicate_labels(tmp_path):
    """A dict would silently keep only the last one, quietly shrinking the
    pool the operator asked for."""
    a = _robophd_run(tmp_path, "run_a", {"iter3_a": 1500})
    b = _robophd_run(tmp_path, "run_b", {"iter7_b": 1500})

    with pytest.raises(SystemExit, match="given twice"):
        resolve_seed_runs([f"dup={a}", f"dup={b}"], example=EXAMPLE)


def test_a_checkpoint_without_performance_records_exits(tmp_path):
    """find_best_agent raises ValueError there; the resolver must turn it
    into an operator-facing exit rather than a traceback."""
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    (run_dir / "checkpoint.json").write_text(json.dumps({"agent_pool": {}}))

    with pytest.raises(SystemExit, match="No performance records"):
        resolve_seed_runs([f"x={run_dir}"], example=EXAMPLE)
