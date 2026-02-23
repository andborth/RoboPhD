"""
Domain abstraction for RoboPhD research system.

This module provides a plugin architecture for different evaluation domains
(Text2SQL, Code Generation, etc.) while preserving the core research infrastructure.
"""

from .base import DomainInterface

__all__ = ['DomainInterface', 'get_domain']


def get_domain(domain_name: str, config: dict) -> DomainInterface:
    """
    Factory function to get a domain implementation.

    Args:
        domain_name: Name of the domain ('text2sql', 'codegen')
        config: Configuration dictionary for the domain

    Returns:
        DomainInterface implementation

    Raises:
        ValueError: If domain_name is not recognized
    """
    if domain_name == 'text2sql':
        from .text2sql.domain import Text2SQLDomain
        return Text2SQLDomain(config)
    elif domain_name == 'codegen':
        from .codegen.domain import CodeGenDomain
        return CodeGenDomain(config)
    elif domain_name == 'external':
        from .external.domain import ExternalEvaluatorDomain
        return ExternalEvaluatorDomain(config)
    else:
        raise ValueError(
            f"Unknown domain: {domain_name}. "
            f"Available domains: text2sql, codegen, external"
        )
