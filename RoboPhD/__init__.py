"""
RoboPhD: Autonomous AI Research System for Text-to-SQL

A self-improving research system where AI agents conduct autonomous research
to evolve better database analysis agents through iterative experimentation.
"""

__version__ = "1.0.0"
__author__ = "RoboPhD Authors"

from .core import SQLGenerator, Evaluator, DatabaseManager
from .agent_orchestrator import AgentOrchestrator
from .researcher import ParallelAgentResearcher, ParallelAgentEvolver

__all__ = [
    'SQLGenerator',
    'Evaluator',
    'DatabaseManager',
    'AgentOrchestrator',
    'ParallelAgentResearcher',
    'ParallelAgentEvolver'
]