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
