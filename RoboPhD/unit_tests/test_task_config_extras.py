"""Tests for RoboPhDConfig.task_config_extras and the extra_read_paths
fold-in.

task_config_extras lets callers ride task-specific values on the
checkpoint's task_config dict, which the researcher rewrites every
iteration — the persistence mechanism behind asta_ds1000's cost knobs
surviving mid-run interruption. extra_read_paths moved from a top-level
optimize_anything kwarg to a RoboPhDConfig field at the same time (both
are RoboPhD-engine-only knobs).
"""

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RoboPhD.api import (  # noqa: E402
    RoboPhDConfig,
    _build_resume_kwargs,
    _merge_task_config_extras,
    optimize_anything,
)
from RoboPhD.runner_utils import (  # noqa: E402
    read_task_config_extras,
    resolve_run_immutable,
)


# -- _merge_task_config_extras ------------------------------------------


def test_merge_none_extras_returns_input_unchanged():
    base = {"file_mapping": {"agent.py": "agent.py"}}
    assert _merge_task_config_extras(base, None) is base


def test_merge_adds_caller_keys():
    base = {"file_mapping": {"agent.py": "agent.py"}, "objective": "o"}
    merged = _merge_task_config_extras(
        base, {"ds1000_runtime": {"cost_threshold": 0.044}}
    )
    assert merged["ds1000_runtime"] == {"cost_threshold": 0.044}
    assert merged["file_mapping"] == {"agent.py": "agent.py"}
    assert "ds1000_runtime" not in base, "merge must not mutate the input"


@pytest.mark.parametrize("reserved", ["file_mapping", "objective", "background"])
def test_merge_reserved_key_collision_raises(reserved):
    """file_mapping drives candidate materialization and objective/
    background are recovered on resume — extras shadowing them would
    corrupt resume, so the collision must be loud."""
    with pytest.raises(ValueError) as exc_info:
        _merge_task_config_extras({"file_mapping": {}}, {reserved: "x"})
    assert reserved in str(exc_info.value)


# -- config surface -----------------------------------------------------


def test_robophd_config_carries_engine_specific_fields():
    cfg = RoboPhDConfig()
    assert cfg.extra_read_paths is None
    assert cfg.task_config_extras is None


def test_optimize_anything_no_longer_accepts_extra_read_paths_kwarg():
    """extra_read_paths moved onto RoboPhDConfig; the old top-level kwarg
    widened the public signature with an engine-specific flag (it was
    silently ignored by GEPA/Autoresearch). Pin the removal so it doesn't
    creep back."""
    assert "extra_read_paths" not in inspect.signature(optimize_anything).parameters


# -- resume-path merge --------------------------------------------------


def _write_minimal_checkpoint(experiment_dir: Path, task_config: dict) -> None:
    (experiment_dir / "checkpoint.json").write_text(json.dumps({
        "config_manager": {
            "iteration_configs": {"0": {"domain": "external"}},
            "resolved_configs": {},
            "config_change_history": [],
            "current_iteration": 1,
        },
        "last_completed_iteration": 1,
        "num_iterations": 2,
        "task_config": task_config,
    }))


def test_resume_kwargs_merge_extras_over_checkpoint_task_config(tmp_path):
    """On resume the merged task_config must be passed explicitly (the
    explicit param wins over resume_checkpoint in the researcher), so a
    bootstrap resume of a pre-extras run persists the values at the next
    _save_checkpoint."""
    _write_minimal_checkpoint(tmp_path, {
        "file_mapping": {"agent.py": "agent.py"},
        "objective": "o",
        "background": "b",
    })
    cfg = RoboPhDConfig(
        experiment_dir=tmp_path,
        task_config_extras={"ds1000_runtime": {"cost_threshold": 0.044}},
    )
    _, _, researcher_kwargs, task_config = _build_resume_kwargs(cfg)
    assert researcher_kwargs["task_config"]["ds1000_runtime"] == {
        "cost_threshold": 0.044
    }
    assert researcher_kwargs["task_config"]["file_mapping"] == {
        "agent.py": "agent.py"
    }
    assert task_config == researcher_kwargs["task_config"]


def test_resume_kwargs_without_extras_pass_checkpoint_task_config(tmp_path):
    """No extras → checkpoint's task_config flows through unchanged (and
    is still passed explicitly, which is a no-op vs the resume_checkpoint
    fallback)."""
    stored = {
        "file_mapping": {"agent.py": "agent.py"},
        "ds1000_runtime": {"cost_threshold": 0.08},
    }
    _write_minimal_checkpoint(tmp_path, stored)
    cfg = RoboPhDConfig(experiment_dir=tmp_path)
    _, _, researcher_kwargs, _ = _build_resume_kwargs(cfg)
    assert researcher_kwargs["task_config"] == stored


# -- immutability backstop on resume ------------------------------------


def test_resume_extras_changing_stored_value_raises(tmp_path):
    """The backstop: extras computed from CLI flags by a caller WITHOUT a
    flag-level guard must not silently mutate a persisted knob mid-run.
    Sub-key granularity — the conflict names the changed leaf."""
    _write_minimal_checkpoint(tmp_path, {
        "file_mapping": {"agent.py": "agent.py"},
        "ds1000_runtime": {"cost_threshold": 0.044, "cost_per_error": 0.01},
    })
    cfg = RoboPhDConfig(
        experiment_dir=tmp_path,
        task_config_extras={"ds1000_runtime": {"cost_threshold": 0.08,
                                               "cost_per_error": 0.01}},
    )
    with pytest.raises(ValueError) as exc_info:
        _build_resume_kwargs(cfg)
    msg = str(exc_info.value)
    assert "ds1000_runtime.cost_threshold" in msg
    assert "0.044" in msg and "0.08" in msg
    assert "resolve_run_immutable" in msg


def test_resume_extras_equal_values_pass(tmp_path):
    """Steady-state resume: a guarded caller passes back the stored
    values verbatim — idempotent merge, no error."""
    stored_runtime = {"cost_threshold": 0.044, "cost_per_error": 0.01}
    _write_minimal_checkpoint(tmp_path, {
        "file_mapping": {"agent.py": "agent.py"},
        "ds1000_runtime": dict(stored_runtime),
    })
    cfg = RoboPhDConfig(
        experiment_dir=tmp_path,
        task_config_extras={"ds1000_runtime": dict(stored_runtime)},
    )
    _, _, researcher_kwargs, _ = _build_resume_kwargs(cfg)
    assert researcher_kwargs["task_config"]["ds1000_runtime"] == stored_runtime


def test_resume_extras_new_subkey_merges_and_preserves_stored(tmp_path):
    """One-level-deep semantics: a task that grew a new knob can resume
    its older runs (sub-key added), and extras built without a sub-key
    the checkpoint has don't silently drop it."""
    _write_minimal_checkpoint(tmp_path, {
        "file_mapping": {"agent.py": "agent.py"},
        "ds1000_runtime": {"cost_threshold": 0.044, "old_knob": 7},
    })
    cfg = RoboPhDConfig(
        experiment_dir=tmp_path,
        task_config_extras={"ds1000_runtime": {"cost_threshold": 0.044,
                                               "new_knob": True}},
    )
    _, _, researcher_kwargs, _ = _build_resume_kwargs(cfg)
    merged = researcher_kwargs["task_config"]["ds1000_runtime"]
    assert merged == {"cost_threshold": 0.044, "old_knob": 7, "new_knob": True}


# -- shared example-side helpers (runner_utils) --------------------------


def test_read_task_config_extras_prefers_checkpoint(tmp_path):
    _write_minimal_checkpoint(tmp_path, {"t_runtime": {"k": 1}})
    (tmp_path / "legacy.json").write_text(json.dumps({"k": 99}))
    assert read_task_config_extras(tmp_path, "t_runtime", "legacy.json") == {"k": 1}


def test_read_task_config_extras_falls_back_to_legacy_sidecar(tmp_path):
    _write_minimal_checkpoint(tmp_path, {"file_mapping": {}})
    (tmp_path / "legacy.json").write_text(json.dumps({"k": 99}))
    assert read_task_config_extras(tmp_path, "t_runtime", "legacy.json") == {"k": 99}


def test_read_task_config_extras_missing_everything_returns_empty(tmp_path):
    assert read_task_config_extras(tmp_path, "t_runtime", "legacy.json") == {}


def test_resolve_run_immutable_matrix():
    """The six-cell resolution matrix, shared by all examples."""
    r = resolve_run_immutable
    # fresh: cli wins, else default
    assert r(0.08, None, 0.04, "knob", on_resume=False) == 0.08
    assert r(None, None, 0.04, "knob", on_resume=False) == 0.04
    # resume + stored: match ok, absent-cli ok, conflict exits
    assert r(0.06, 0.06, 0.04, "knob", on_resume=True) == 0.06
    assert r(None, 0.06, 0.04, "knob", on_resume=True) == 0.06
    with pytest.raises(SystemExit) as exc_info:
        r(0.10, 0.06, 0.04, "knob", on_resume=True)
    assert "cannot be changed on --resume" in str(exc_info.value)
    # resume + no stored: cli bootstraps, absent-cli exits with the note
    assert r(0.10, None, 0.04, "knob", on_resume=True) == 0.10
    with pytest.raises(SystemExit) as exc_info:
        r(None, None, 0.04, "knob", on_resume=True, missing_note="EXTRA. ")
    msg = str(exc_info.value)
    assert "no stored knob" in msg and "bootstrap" in msg and "EXTRA." in msg
