"""Model registry: alias registration, defaults, and the LM Studio guard.

The registry is two tables that must agree (RoboPhD/config.py). A model
present in only one of them still "works" — right up until
``get_lmstudio_env`` sends the evolution CLI to localhost:1234 instead of
Anthropic, which looks like a hung run rather than a config error. These
tests pin both halves plus the guard that turns the misroute into a raise.
"""

import pytest

from RoboPhD.api import AutoresearchConfig, GEPAConfig, RoboPhDConfig
from RoboPhD.config import (
    CLAUDE_CLI_MODEL_MAP,
    SUPPORTED_MODELS,
    build_evolution_env,
    get_lmstudio_env,
    validate_model_alias,
)
from RoboPhD.config_manager import ConfigManager, ConfigSource

DEFAULT_EVOLUTION_MODEL = "opus-5"


def test_default_model_is_registered_in_both_tables():
    assert SUPPORTED_MODELS[DEFAULT_EVOLUTION_MODEL]["name"] == "claude-opus-5"
    assert CLAUDE_CLI_MODEL_MAP[DEFAULT_EVOLUTION_MODEL] == "claude-opus-5[1m]"


def test_every_cli_alias_has_pricing():
    """A CLI-only alias has no pricing, so it silently reports $0 in cost
    rollups and cannot be registered with litellm for GEPA."""
    assert set(CLAUDE_CLI_MODEL_MAP) <= set(SUPPORTED_MODELS)


@pytest.mark.parametrize("model", sorted(CLAUDE_CLI_MODEL_MAP))
def test_registered_models_do_not_route_to_lmstudio(model):
    assert get_lmstudio_env(model) is None


def test_unregistered_non_claude_model_still_routes_to_lmstudio():
    env = get_lmstudio_env("qwen/qwen3-coder-30b")
    assert env["ANTHROPIC_BASE_URL"] == "http://localhost:1234"


@pytest.mark.parametrize(
    "config_default",
    [
        ConfigManager().get_defaults()["evolution_model"],
        ConfigManager().get_defaults()["meta_evolution_model"],
        RoboPhDConfig().evolution_model,
        RoboPhDConfig().meta_evolution_model,
        GEPAConfig().reflection_model,
        AutoresearchConfig().model,
    ],
)
def test_all_engine_defaults_agree(config_default):
    assert config_default == DEFAULT_EVOLUTION_MODEL


@pytest.mark.parametrize(
    "model",
    [
        "opus-4.5",       # retired alias — dropped, never re-pointed
        "opus5",          # typo: missing hyphen
        "claude-opus-5",  # the API id, not the RoboPhD alias
        "sonnet-4.7",     # plausible but nonexistent
    ],
)
def test_claude_family_typos_raise(model):
    with pytest.raises(ValueError, match="Unknown Claude model alias"):
        validate_model_alias(model)


@pytest.mark.parametrize("model", ["opus-5", "haiku-4.5", "qwen/qwen3-coder-30b"])
def test_valid_and_lmstudio_names_pass(model):
    validate_model_alias(model)


def test_error_message_lists_the_valid_aliases():
    """A bare 'what did I mistype' error is useless mid-run — the message
    has to carry the menu."""
    with pytest.raises(ValueError) as excinfo:
        validate_model_alias("opus")
    message = str(excinfo.value)
    assert "opus" in message
    for alias in ("opus-5", "haiku-4.5"):
        assert alias in message


def test_build_evolution_env_rejects_bad_alias(tmp_path):
    """The choke point every CLI engine funnels through."""
    with pytest.raises(ValueError, match="Unknown Claude model alias"):
        build_evolution_env("opus-4.5", tmp_path)


def test_config_manager_rejects_bad_alias_at_config_time():
    manager = ConfigManager()
    with pytest.raises(ValueError, match="evolution_model"):
        manager._validate_parameters(
            {"evolution_model": "opus-4.5"},
            context="Test configuration",
            source=ConfigSource.CLI,
        )
