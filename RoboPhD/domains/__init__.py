"""
Domain abstraction for RoboPhD research system.

All domains now use ExternalEvaluatorDomain, which bridges task-registry
evaluator functions into the evolution loop.
"""

from .base import DomainInterface

__all__ = ['DomainInterface', 'get_domain']


def get_domain(domain_name: str, config: dict) -> DomainInterface:
    """
    Factory function to get a domain implementation.

    Args:
        domain_name: Name of the domain (currently only 'external')
        config: Configuration dictionary for the domain

    Returns:
        DomainInterface implementation

    Raises:
        ValueError: If domain_name is not recognized
    """
    if domain_name == 'external':
        from .external.domain import ExternalEvaluatorDomain
        return ExternalEvaluatorDomain(config)
    else:
        raise ValueError(
            f"Unknown domain: {domain_name}. "
            f"Available domains: external"
        )
