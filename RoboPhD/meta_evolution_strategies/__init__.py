"""
Meta-evolution strategy registry.

A meta-evolution strategy is a Python class (subclass of
`MetaEvolutionStrategy`) that bundles every design choice distinguishing one
meta-evolution approach from another — the strategy body, the CLAUDE.md
content, the firing prompts, and the reflection prompt.

Strategies are looked up by their `name` class var (the same string a user
passes via `meta_evolution_strategy: <name>` in config). The registry below
is explicit so a syntax error in any strategy file fails loud at import
time and `grep` can locate every available strategy.
"""

from RoboPhD.meta_evolution_strategies.base import MetaEvolutionStrategy
from RoboPhD.meta_evolution_strategies.minimal_guidance import MinimalGuidance
from RoboPhD.meta_evolution_strategies.parameter_adjustment import ParameterAdjustment
from RoboPhD.meta_evolution_strategies.train_a_winner import TrainAWinner


_STRATEGIES = (MinimalGuidance, ParameterAdjustment, TrainAWinner)
_REGISTRY = {cls.name: cls for cls in _STRATEGIES}


def load_strategy(name: str) -> MetaEvolutionStrategy:
    """Instantiate the meta-evolution strategy with the given config name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown meta-evolution strategy '{name}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]()


def available_strategies() -> list[str]:
    """Sorted list of strategy names registered in this package."""
    return sorted(_REGISTRY)


__all__ = [
    "MetaEvolutionStrategy",
    "load_strategy",
    "available_strategies",
]
