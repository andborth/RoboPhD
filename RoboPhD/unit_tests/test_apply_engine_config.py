"""Tests for runner_utils.apply_engine_config — the --engine-config
overlay for GEPA/Autoresearch engine config dataclasses."""

import pytest

from RoboPhD.api import AutoresearchConfig, GEPAConfig
from RoboPhD.runner_utils import apply_engine_config


def test_sets_field_from_json_string():
    cfg = GEPAConfig()
    out = apply_engine_config(cfg, '{"reflection_model": "fable-5"}')
    assert out is cfg
    assert cfg.reflection_model == "fable-5"


def test_sets_field_from_parsed_dict():
    cfg = GEPAConfig()
    apply_engine_config(cfg, {"reflection_model": "fable-5", "val_size": 50})
    assert cfg.reflection_model == "fable-5"
    assert cfg.val_size == 50


def test_none_and_empty_are_noops():
    cfg = GEPAConfig()
    default_model = cfg.reflection_model
    apply_engine_config(cfg, None)
    apply_engine_config(cfg, "")
    apply_engine_config(cfg, {})
    assert cfg.reflection_model == default_model


def test_unknown_key_fails_loudly_and_names_valid_keys():
    """A RoboPhD-engine key like evolution_model must not be silently
    dropped for GEPA — the error names the bad key and the valid ones,
    so the caller learns reflection_model is the knob they wanted."""
    with pytest.raises(ValueError) as excinfo:
        apply_engine_config(GEPAConfig(), '{"evolution_model": "fable-5"}')
    msg = str(excinfo.value)
    assert "evolution_model" in msg
    assert "GEPAConfig" in msg
    assert "reflection_model" in msg


def test_overlay_wins_over_constructor_value():
    cfg = GEPAConfig(max_workers=4)
    apply_engine_config(cfg, {"max_workers": 8})
    assert cfg.max_workers == 8


def test_works_for_autoresearch_config():
    cfg = AutoresearchConfig()
    with pytest.raises(ValueError):
        apply_engine_config(cfg, {"definitely_not_a_field": 1})
