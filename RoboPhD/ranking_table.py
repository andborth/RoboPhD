"""
Complete Performance Ranking Table module for RoboPhD
Provides the full iteration-by-iteration ranking table as in APE
"""

from typing import Dict, List


def calculate_mean_ranks(records: Dict) -> Dict[str, float]:
    """Calculate mean average rank for each agent across iterations based on overall score."""
    # Group results by iteration
    iteration_results = {}
    for agent_id, record in records.items():
        # Use iteration_results field from performance_records
        for iter_result in record.get('iteration_results', []):
            iteration = iter_result.get('iteration')
            if iteration:
                if iteration not in iteration_results:
                    iteration_results[iteration] = {}

                # Get average_score directly from iteration_result
                if iter_result.get('average_score') is not None:
                    iteration_results[iteration][agent_id] = iter_result.get('average_score')
    
    # Calculate ranks within each iteration with proper tie handling
    agent_ranks = {aid: [] for aid in records.keys()}
    for iteration, scores in iteration_results.items():
        if len(scores) > 1:  # Need at least 2 agents to rank
            # Group agents by score for proper tie handling
            score_groups = {}
            for agent_id, score in scores.items():
                if score not in score_groups:
                    score_groups[score] = []
                score_groups[score].append(agent_id)

            # Assign ranks with proper tie handling
            current_rank = 1
            for score in sorted(score_groups.keys(), reverse=True):
                agents_at_score = score_groups[score]
                for agent_id in agents_at_score:
                    agent_ranks[agent_id].append(current_rank)
                # Skip ranks for ties
                current_rank += len(agents_at_score)
    
    # Calculate mean ranks
    mean_ranks = {}
    for agent_id, ranks in agent_ranks.items():
        if ranks:
            mean_ranks[agent_id] = sum(ranks) / len(ranks)
        else:
            mean_ranks[agent_id] = 999  # Not ranked
    
    return mean_ranks


def generate_ranking_table(test_history: List, performance_records: Dict, for_evolution: bool = False, clone_agent_ids: set = None) -> str:
    """
    Generate comprehensive ranking table for agents across all iterations.

    Args:
        test_history: Complete test history data
        performance_records: Performance records for ELO/rank calculations
        for_evolution: If True, format for evolution prompts (simpler). If False, for final report.
        clone_agent_ids: Set of agent IDs that were detected as exact clones (ELO penalized).
    """
    if clone_agent_ids is None:
        clone_agent_ids = set()
    if not test_history or len(test_history) < 1:
        return ""
    
    # Use provided performance_records or empty dict
    if performance_records is None:
        performance_records = {}
    
    # Collect all unique agents (RoboPhD uses direct agent keys format)
    all_agents = set()
    for iteration_data in test_history:
        for agent_id in iteration_data.keys():
            if isinstance(agent_id, str):
                all_agents.add(agent_id)
    
    # Calculate mean ranks
    mean_ranks = calculate_mean_ranks(performance_records) if performance_records else {}
    
    # Build iteration data for each agent
    agent_iteration_data = {}
    for agent_id in all_agents:
        agent_iteration_data[agent_id] = {
            'iterations': {},
            'elo': performance_records.get(agent_id, {}).get('elo', 1500),
            'mean_rank': mean_ranks.get(agent_id, 999)
        }
    
    # Process each iteration
    for iter_num, iteration_data in enumerate(test_history, 1):
        # Get all agents and their scores for this iteration
        agent_scores = {}

        # RoboPhD format: direct agent keys with average_score field
        for agent_id, agent_data in iteration_data.items():
            if isinstance(agent_data, dict) and 'average_score' in agent_data:
                agent_scores[agent_id] = agent_data['average_score']

        # Rank agents for this iteration with proper tie handling
        score_groups = {}
        for agent_id, score in agent_scores.items():
            if score not in score_groups:
                score_groups[score] = []
            score_groups[score].append(agent_id)

        # Assign ranks with proper tie handling
        current_rank = 1
        for score in sorted(score_groups.keys(), reverse=True):
            agents_at_score = score_groups[score]
            for agent_id in agents_at_score:
                agent_iteration_data[agent_id]['iterations'][iter_num] = {
                    'rank': current_rank,
                    'score': score
                }
            # Skip ranks for ties (e.g., if 2 agents tied at rank 1, next is rank 3)
            current_rank += len(agents_at_score)
    
    # Sort agents by ELO score (highest first)
    sorted_agents = sorted(agent_iteration_data.items(), 
                          key=lambda x: x[1]['elo'], 
                          reverse=True)
    
    # Generate table with appropriate title
    if for_evolution:
        table = ""  # Header already included in evolution prompt template
    else:
        table = "\n## Complete Performance Ranking Table\n\n"
    
    # Get num_iterations
    num_iterations = len(test_history)
    
    # Header
    header = "| Agent |"
    for i in range(1, num_iterations + 1):
        header += f" Iter {i} |"
    header += " Final ELO | Mean Rank |\n"
    
    # Separator
    separator = "|-------|"
    for i in range(1, num_iterations + 1):
        separator += "--------|"
    separator += "-----------|----------|\n"
    
    table += header + separator
    
    # Rows
    for agent_id, data in sorted_agents:
        # Truncate long names
        if len(agent_id) > 35:
            display_id = agent_id[:32] + "..."
        else:
            display_id = agent_id
        
        row = f"| {display_id} |"
        
        # Add iteration data
        for i in range(1, num_iterations + 1):
            if i in data['iterations']:
                rank = data['iterations'][i]['rank']
                score = data['iterations'][i]['score']

                # Format based on rank
                if rank == 1:
                    cell = f" **#1** {score:.3f} |"
                elif rank == 2:
                    cell = f" #2 {score:.3f} |"
                elif rank == 3:
                    cell = f" #3 {score:.3f} |"
                else:
                    cell = f" #{rank} {score:.3f} |"
            else:
                cell = " - |"

            row += cell
        
        # Add ELO and mean rank
        elo = data['elo']
        mean_rank = data['mean_rank']
        clone_marker = "*" if agent_id in clone_agent_ids else ""

        # Highlight best performer
        if sorted_agents and elo == max(d['elo'] for _, d in sorted_agents):
            row += f" **{elo:.0f}**{clone_marker} |"
        else:
            row += f" {elo:.0f}{clone_marker} |"
        
        if mean_rank < 2.0:
            row += f" **{mean_rank:.2f}** |"
        else:
            row += f" {mean_rank:.2f} |" if mean_rank < 999 else " - |"
        
        row += "\n"
        table += row
    
    # Add legend
    table += "\n### Legend:\n"
    table += "- **#1** = 1st place (winner of iteration)\n"
    table += "- #2, #3, etc. = 2nd, 3rd place, etc.\n"
    table += "- Score = Average score (0-1) on that iteration's problems\n"
    table += "- **Bold ELO/Rank** = Top performer\n"
    table += "- `-` = Agent not tested in that iteration\n"
    if clone_agent_ids:
        table += "- \\* *Exact clone: identical per-problem scores to an existing agent on debut. ELO penalized by 200.*\n"

    return table