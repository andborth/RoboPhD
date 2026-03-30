"""
RoboPhD: Multi-Domain Evolution System

A self-improving research system where AI agents conduct autonomous research
to evolve better agents through iterative experimentation.
"""

# Add project root to sys.path for utilities/evaluation imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

__version__ = "2.0.0"
__author__ = "RoboPhD Authors"

from .researcher import ParallelAgentResearcher, ParallelAgentEvolver
from .api import (
    optimize_anything, optimize_task, eval_candidate,
    OptimizeResult, EvalResult,
    RoboPhDConfig, RoboPhDEvalConfig,
)

__all__ = [
    'ParallelAgentResearcher',
    'ParallelAgentEvolver',
    'optimize_anything',
    'optimize_task',
    'eval_candidate',
    'OptimizeResult',
    'EvalResult',
    'RoboPhDConfig',
    'RoboPhDEvalConfig',
]
