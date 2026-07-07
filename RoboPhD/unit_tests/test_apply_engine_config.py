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

# Compound statements own nested blocks; their inner statements are
# scanned when ast.walk reaches those blocks. Skipped at the outer
# level so a construction inside e.g. an `if` isn't double-reported
# against the outer block, where a same-branch apply wouldn't be seen.
_COMPOUND_STMTS = (
    ast.If, ast.For, ast.While, ast.With, ast.Try,
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
)


def _statement_calls_apply(stmt: ast.stmt) -> bool:
    for node in ast.walk(stmt):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == "apply_engine_config":
                return True
    return False


def _engine_class_names(tree: ast.AST) -> set:
    """Bare names that refer to an engine config class in this module —
    the class names themselves plus any import aliases
    (`from RoboPhD import GEPAConfig as GC`)."""
    names = set(ENGINE_CONFIG_CLASSES)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in ENGINE_CONFIG_CLASSES and alias.asname:
                    names.add(alias.asname)
    return names


def _statement_constructs_engine_config(stmt: ast.stmt, names: set) -> bool:
    """True if any call inside the statement constructs an engine config
    class — bare name (incl. import aliases), attribute access
    (api.GEPAConfig), any statement/assignment shape (plain, annotated,
    tuple-unpacking, expression)."""
    for node in ast.walk(stmt):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in names:
            return True
        if isinstance(func, ast.Attribute) and func.attr in ENGINE_CONFIG_CLASSES:
            return True
    return False


def _scan_constructions(tree: ast.AST) -> tuple:
    """(construction_count, missing_linenos): engine-config constructions
    found, and those not paired with an apply_engine_config call in the
    same statement or a following statement of the same block."""
    names = _engine_class_names(tree)
    found = 0
    missing = []
    for node in ast.walk(tree):
        for attr in ("body", "orelse", "finalbody"):
            block = getattr(node, attr, None)
            if not isinstance(block, list):
                continue
            for idx, stmt in enumerate(block):
                if isinstance(stmt, _COMPOUND_STMTS):
                    continue
                if not _statement_constructs_engine_config(stmt, names):
                    continue
                found += 1
                if _statement_calls_apply(stmt):
                    continue  # constructed and applied in one statement
                if not any(_statement_calls_apply(s) for s in block[idx + 1:]):
                    missing.append(stmt.lineno)
    return found, missing


def test_every_example_applies_engine_config_after_construction():
    """Each GEPAConfig/AutoresearchConfig construction in an example's
    main.py must be followed by apply_engine_config in the same branch —
    otherwise that example silently drops --engine-config for that
    engine, the exact bug this helper exists to prevent."""
    mains = sorted(EXAMPLES_DIR.glob("*/main.py"))
    assert mains, f"no example main.py files found under {EXAMPLES_DIR}"
    total_found = 0
    offenders = {}
    for main_py in mains:
        found, missing = _scan_constructions(ast.parse(main_py.read_text()))
        total_found += found
        if missing:
            offenders[str(main_py.relative_to(EXAMPLES_DIR))] = missing
    # Canary: zero constructions means the detector went stale (class
    # renames, import style the scanner can't see) and the invariant
    # above would pass vacuously.
    assert total_found > 0, (
        "no engine config constructions detected in any example — "
        "detector is stale, not the examples clean"
    )
    assert not offenders, (
        "engine config constructed without a following apply_engine_config "
        f"(file -> line numbers): {offenders}"
    )


@pytest.mark.parametrize(
    "snippet",
    [
        pytest.param("cfg = GEPAConfig(a=1)\n", id="plain-assign"),
        pytest.param("cfg: GEPAConfig = GEPAConfig(a=1)\n", id="annotated-assign"),
        pytest.param("cfg = api.GEPAConfig(a=1)\n", id="attribute-call"),
        pytest.param("cfg, dataset = AutoresearchConfig(a=1), train\n", id="tuple-unpacking"),
        pytest.param(
            "from RoboPhD import GEPAConfig as GC\ncfg = GC(a=1)\n",
            id="aliased-import",
        ),
    ],
)
def test_detector_catches_unapplied_construction_shapes(snippet):
    found, missing = _scan_constructions(ast.parse(snippet))
    assert found == 1 and len(missing) == 1


def test_detector_accepts_applied_constructions():
    compliant = (
        "if engine == 'gepa':\n"
        "    cfg = GEPAConfig(a=1)\n"
        "    cfg = apply_engine_config(cfg, ec)\n"
        "    dataset = train\n"
        "one_liner = apply_engine_config(GEPAConfig(a=1), ec)\n"
    )
    found, missing = _scan_constructions(ast.parse(compliant))
    assert found == 2 and missing == []
