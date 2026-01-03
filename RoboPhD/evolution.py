#!/usr/bin/env python3
"""
Evolution strategy selector for RoboPhD research system.
Manages which evolution strategies to use at different iterations.
"""

from pathlib import Path
from typing import Dict, Optional, List
import random


class EvolutionStrategySelector:
    """
    Manages evolution strategy selection for different iterations.
    Dynamically loads strategies from evolution_strategies/ directory.
    """
    
    def __init__(self,
                 default_strategy: str = "use_your_judgment",
                 prompts_dir: Optional[str] = None):
        """
        Initialize the evolution strategy selector.
        
        Args:
            default_strategy: Default evolution strategy to use
            prompts_dir: Optional custom directory for evolution strategies
        """
        self.default_strategy = default_strategy
        
        # Load available evolution strategies
        if prompts_dir:
            self.evolution_prompts_dir = Path(prompts_dir)
        else:
            # Default to evolution_strategies/ in RoboPhD directory
            self.evolution_prompts_dir = Path(__file__).parent / "evolution_strategies"
        
        self.available_strategies = self._load_available_strategies()

        # Validate default strategy exists
        if default_strategy not in self.available_strategies:
            available = sorted(list(self.available_strategies.keys()))
            if available:
                raise ValueError(
                    f"Default evolution strategy '{default_strategy}' not found.\n"
                    f"Available strategies: {available}\n"
                    f"Strategies directory: {self.evolution_prompts_dir}"
                )
            else:
                raise ValueError(
                    f"No evolution strategies found in {self.evolution_prompts_dir}\n"
                    f"Ensure the directory contains subdirectories with strategy.md files"
                )
    
    def _load_available_strategies(self) -> Dict[str, Path]:
        """
        Load all available evolution strategies.

        Strategies are now two-artifact packages in subdirectories:
        - evolution_strategies/strategy_name/strategy.md
        - evolution_strategies/strategy_name/tools/ (optional)

        Returns:
            Dict mapping strategy_name -> Path to strategy.md file
        """
        strategies = {}

        if self.evolution_prompts_dir.exists():
            # Scan for directories containing strategy.md
            for strategy_dir in self.evolution_prompts_dir.iterdir():
                if not strategy_dir.is_dir():
                    continue

                # Skip special directories
                if strategy_dir.name.startswith('.') or strategy_dir.name.upper() == 'README':
                    continue

                # Check for strategy.md
                strategy_file = strategy_dir / "strategy.md"
                if strategy_file.exists():
                    strategy_name = strategy_dir.name
                    strategies[strategy_name] = strategy_file

        return strategies
    
    def get_strategy_for_iteration(self, 
                                  iteration: int,
                                  strategy_override: Optional[str] = None) -> tuple[str, str]:
        """
        Get the evolution strategy for a given iteration.
        
        Args:
            iteration: Current iteration number (1-based) - for future use
            strategy_override: Optional strategy to use instead of default
            
        Returns:
            Tuple of (strategy_name, strategy_content)
        """
        # Note: iteration parameter preserved for compatibility but not currently used
        # Use override if provided
        if strategy_override:
            if strategy_override == "none":
                # Special case: no evolution
                return "none", ""
            elif strategy_override == "challenger":
                # Special case: no evolution, but with challenger selection
                return "challenger", ""
            elif strategy_override == "greedy":
                # Special case: no evolution, deterministic top-k selection
                return "greedy", ""
            elif strategy_override == "random":
                # Special case: random selection
                strategy_name = random.choice(list(self.available_strategies.keys()))
            elif strategy_override in self.available_strategies:
                strategy_name = strategy_override
            else:
                available = sorted(list(self.available_strategies.keys()))
                raise ValueError(
                    f"Strategy '{strategy_override}' not found.\n"
                    f"Available strategies: {available}\n"
                    f"Special strategies: none, challenger, greedy, random"
                )
        else:
            strategy_name = self.default_strategy
        
        # Load the strategy content
        if strategy_name == "none":
            return "none", ""
        elif strategy_name == "challenger":
            return "challenger", ""
        elif strategy_name == "greedy":
            return "greedy", ""

        strategy_path = self.available_strategies[strategy_name]
        strategy_content = strategy_path.read_text()
        
        return strategy_name, strategy_content
    
    def get_strategy_content(self, strategy_name: str) -> str:
        """
        Get the content of a specific strategy.
        
        Args:
            strategy_name: Name of the strategy to load
            
        Returns:
            Strategy content as string
        """
        if strategy_name == "none":
            return ""
        elif strategy_name == "challenger":
            return ""
        elif strategy_name == "greedy":
            return ""

        if strategy_name == "random":
            # Random selection
            strategy_name = random.choice(list(self.available_strategies.keys()))
        
        if strategy_name not in self.available_strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found. Available: {list(self.available_strategies.keys())}")
        
        return self.available_strategies[strategy_name].read_text()
    
    def list_strategies(self) -> List[str]:
        """List all available evolution strategies."""
        return list(self.available_strategies.keys())
    
    def list_all_strategies(self) -> List[str]:
        """List all available strategies including special ones."""
        strategies = list(self.available_strategies.keys())
        # Add special strategies
        strategies.append("none")
        strategies.append("challenger")
        strategies.append("greedy")
        strategies.append("random")
        return sorted(strategies)


def parse_evolution_schedule(schedule_str: str) -> Dict[int, str]:
    """
    Parse evolution schedule from JSON string.
    
    Args:
        schedule_str: JSON string like '{"2": "none", "3": "cross_pollination_judgment"}'
        
    Returns:
        Dictionary mapping iteration numbers to strategy names
    """
    import json
    
    if not schedule_str:
        return {}
    
    try:
        # Parse JSON
        raw_schedule = json.loads(schedule_str)
        
        # Convert string keys to integers
        schedule = {}
        for key, value in raw_schedule.items():
            try:
                iteration = int(key)
                schedule[iteration] = value
            except ValueError:
                print(f"Warning: Invalid iteration number '{key}' in schedule")
        
        return schedule
        
    except json.JSONDecodeError as e:
        print(f"Error parsing evolution schedule: {e}")
        return {}


if __name__ == "__main__":
    # Test the evolution strategy selector
    print("=" * 60)
    print("Testing RoboPhD Evolution Strategy Selector")
    print("=" * 60)
    
    selector = EvolutionStrategySelector()
    
    print("\nAvailable strategies (including special):")
    for name in selector.list_all_strategies():
        print(f"  - {name}")
    
    print("\nTesting strategy loading:")
    for strategy in selector.list_strategies()[:3]:
        content = selector.get_strategy_content(strategy)
        print(f"\n{strategy}: {len(content)} characters")
        print(f"  First line: {content.split(chr(10))[0][:60]}...")
    
    print("\nTesting schedule parsing:")
    test_schedule = '{"2": "none", "3": "cross_pollination_judgment", "5": "random"}'
    parsed = parse_evolution_schedule(test_schedule)
    print(f"  Input: {test_schedule}")
    print(f"  Parsed: {parsed}")