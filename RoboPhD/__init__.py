"""
RoboPhD: Autonomous AI Research System for Text-to-SQL

A self-improving research system where AI agents conduct autonomous research
to evolve better database analysis agents through iterative experimentation.
"""

# Add project root to sys.path for utilities/evaluation imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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