"""Structural tests for the MetaEvolutionStrategy class hierarchy.

Verifies the contract the manager depends on: registry resolution, metadata
shape, error handling, and the reflection prompt default. These are permanent
contract tests and should pass on any checkout.

The .md → .py encapsulation refactor was additionally validated against a
recorded run's prompt artifacts (`cant_be_late_20260429_211129`) at refactor
time; those snapshot comparisons were scaffolding and have been removed.
"""

import pytest

from RoboPhD.meta_evolution_strategies import (
    MetaEvolutionStrategy,
    available_strategies,
    load_strategy,
)


def test_registry_resolves_three_known_names():
    # Exact-list assertion is intentional: when adding a fourth strategy,
    # update this list explicitly so the addition is visible in review.
    assert available_strategies() == [
        "minimal_guidance",
        "parameter_adjustment",
        "train_a_winner",
    ]


@pytest.mark.parametrize(
    "name",
    ["minimal_guidance", "parameter_adjustment", "train_a_winner"],
)
def test_each_strategy_has_required_metadata(name):
    strategy = load_strategy(name)
    assert isinstance(strategy, MetaEvolutionStrategy)
    assert strategy.name == name
    assert strategy.description, "description must be non-empty"
    body = strategy.instructions_for_llm()
    assert body.startswith("# "), "body should start with a markdown heading"
    assert len(body) > 200, "body should be substantive prose"


def test_unknown_strategy_raises_with_helpful_message():
    with pytest.raises(ValueError, match="minimal_guidance"):
        load_strategy("does_not_exist")


def test_reflection_prompt_default_renders_iter_path():
    s = load_strategy("minimal_guidance")
    rendered = s.reflection_prompt(iteration=42)
    assert "iteration_042/meta_evolution_reflection.md" in rendered
    assert "REFLECTION COMPLETE" in rendered
