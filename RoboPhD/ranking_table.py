"""
Complete Performance Ranking Table module for RoboPhD
Provides the full iteration-by-iteration ranking table as in APE
"""

from typing import Dict, List


def calculate_mean_ranks(records: Dict) -> Dict[str, float]:
    """Calculate mean average rank for each agent across iterations based on overall accuracy."""
    # Group results by iteration
    iteration_results = {}
    for agent_id, record in records.items():
        # Use iteration_results field from performance_records
        for iter_result in record.get('iteration_results', []):
            iteration = iter_result.get('iteration')
            if iteration:
                if iteration not in iteration_results:
                    iteration_results[iteration] = {}
                
                # Get accuracy directly from iteration_result
                if iter_result.get('accuracy') is not None:
                    iteration_results[iteration][agent_id] = iter_result.get('accuracy')
    
    # Calculate ranks within each iteration with proper tie handling
    agent_ranks = {aid: [] for aid in records.keys()}
    for iteration, scores in iteration_results.items():
        if len(scores) > 1:  # Need at least 2 agents to rank
            # Group agents by accuracy for proper tie handling
            accuracy_groups = {}
            for agent_id, accuracy in scores.items():
                if accuracy not in accuracy_groups:
                    accuracy_groups[accuracy] = []
                accuracy_groups[accuracy].append(agent_id)
            
            # Assign ranks with proper tie handling
            current_rank = 1
            for accuracy in sorted(accuracy_groups.keys(), reverse=True):
                agents_at_accuracy = accuracy_groups[accuracy]
                for agent_id in agents_at_accuracy:
                    agent_ranks[agent_id].append(current_rank)
                # Skip ranks for ties
                current_rank += len(agents_at_accuracy)
    
    # Calculate mean ranks
    mean_ranks = {}
    for agent_id, ranks in agent_ranks.items():
        if ranks:
            mean_ranks[agent_id] = sum(ranks) / len(ranks)
        else:
            mean_ranks[agent_id] = 999  # Not ranked
    
    return mean_ranks


def generate_ranking_table(test_history: List, performance_records: Dict, for_evolution: bool = False) -> str:
    """
    Generate comprehensive ranking table for agents across all iterations.
    
    Args:
        test_history: Complete test history data
        performance_records: Performance records for ELO/rank calculations
        for_evolution: If True, format for evolution prompts (simpler). If False, for final report.
    """
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
        # Get all agents and their accuracies for this iteration
        agent_accuracies = {}
        
        # RoboPhD format: direct agent keys with accuracy field
        for agent_id, agent_data in iteration_data.items():
            if isinstance(agent_data, dict) and 'accuracy' in agent_data:
                agent_accuracies[agent_id] = agent_data['accuracy']
        
        # Rank agents for this iteration with proper tie handling
        accuracy_groups = {}
        for agent_id, accuracy in agent_accuracies.items():
            if accuracy not in accuracy_groups:
                accuracy_groups[accuracy] = []
            accuracy_groups[accuracy].append(agent_id)
        
        # Assign ranks with proper tie handling
        current_rank = 1
        for accuracy in sorted(accuracy_groups.keys(), reverse=True):
            agents_at_accuracy = accuracy_groups[accuracy]
            for agent_id in agents_at_accuracy:
                agent_iteration_data[agent_id]['iterations'][iter_num] = {
                    'rank': current_rank,
                    'accuracy': accuracy
                }
            # Skip ranks for ties (e.g., if 2 agents tied at rank 1, next is rank 3)
            current_rank += len(agents_at_accuracy)
    
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
                accuracy = data['iterations'][i]['accuracy']
                
                # Format based on rank
                if rank == 1:
                    cell = f" **#1** {accuracy:.1f}% |"
                elif rank == 2:
                    cell = f" #2 {accuracy:.1f}% |"
                elif rank == 3:
                    cell = f" #3 {accuracy:.1f}% |"
                else:
                    cell = f" #{rank} {accuracy:.1f}% |"
            else:
                cell = " - |"
            
            row += cell
        
        # Add ELO and mean rank
        elo = data['elo']
        mean_rank = data['mean_rank']
        
        # Highlight best performer
        if sorted_agents and elo == max(d['elo'] for _, d in sorted_agents):
            row += f" **{elo:.0f}** |"
        else:
            row += f" {elo:.0f} |"
        
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
    table += "- Percentage = Accuracy on that iteration's databases\n"
    table += "- **Bold ELO/Rank** = Top performer\n"
    table += "- `-` = Agent not tested in that iteration\n"
    
    return table