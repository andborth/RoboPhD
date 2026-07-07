"""Tests for runner_utils.apply_engine_config — the --engine-config
overlay for GEPA/Autoresearch engine config dataclasses."""

import ast
from pathlib import Path

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


def test_non_dict_payload_fails_with_clear_message():
    """A JSON array would otherwise pass the unknown-key set check
    (set-iteration over a list works) and die on .items() with a bare
    AttributeError."""
    with pytest.raises(ValueError) as excinfo:
        apply_engine_config(GEPAConfig(), '["reflection_model"]')
    assert "JSON object" in str(excinfo.value)


def test_string_for_int_field_fails_at_the_boundary():
    with pytest.raises(ValueError) as excinfo:
        apply_engine_config(GEPAConfig(), '{"max_workers": "8"}')
    msg = str(excinfo.value)
    assert "max_workers" in msg and "int" in msg and "str" in msg


def test_bool_for_int_field_fails_despite_subclass():
    with pytest.raises(ValueError):
        apply_engine_config(GEPAConfig(), {"val_size": True})


def test_int_accepted_for_float_field():
    cfg = GEPAConfig()
    apply_engine_config(cfg, {"debug_log_probability": 1})
    assert cfg.debug_log_probability == 1


def test_null_accepted_for_optional_field():
    cfg = GEPAConfig(max_workers=4)
    apply_engine_config(cfg, '{"max_workers": null}')
    assert cfg.max_workers is None


def test_structural_fields_pass_through_unchecked():
    cfg = GEPAConfig()
    apply_engine_config(cfg, {"val_dataset": [{"q": 1}]})
    assert cfg.val_dataset == [{"q": 1}]


# --- Cross-example invariant -------------------------------------------------

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
ENGINE_CONFIG_CLASSES = {"GEPAConfig", "AutoresearchConfig"}


def _statement_calls_apply(stmt: ast.stmt) -> bool:
    for node in ast.walk(stmt):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == "apply_engine_config":
                return True
    return False


def _unapplied_constructions(tree: ast.AST) -> list:
    """Line numbers of GEPAConfig/AutoresearchConfig constructions not
    followed by an apply_engine_config call in the same statement block."""
    missing = []
    for node in ast.walk(tree):
        for attr in ("body", "orelse", "finalbody"):
            block = getattr(node, attr, None)
            if not isinstance(block, list):
                continue
            for idx, stmt in enumerate(block):
                if not (
                    isinstance(stmt, ast.Assign)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name)
                    and stmt.value.func.id in ENGINE_CONFIG_CLASSES
                ):
                    continue
                if not any(_statement_calls_apply(s) for s in block[idx + 1:]):
                    missing.append(stmt.lineno)
    return missing


def test_every_example_applies_engine_config_after_construction():
    """Each GEPAConfig/AutoresearchConfig construction in an example's
    main.py must be followed by apply_engine_config in the same branch —
    otherwise that example silently drops --engine-config for that
    engine, the exact bug this helper exists to prevent."""
    mains = sorted(EXAMPLES_DIR.glob("*/main.py"))
    assert mains, f"no example main.py files found under {EXAMPLES_DIR}"
    offenders = {}
    for main_py in mains:
        tree = ast.parse(main_py.read_text())
        missing = _unapplied_constructions(tree)
        if missing:
            offenders[str(main_py.relative_to(EXAMPLES_DIR))] = missing
    assert not offenders, (
        "engine config constructed without a following apply_engine_config "
        f"(file -> line numbers): {offenders}"
    )
