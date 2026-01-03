#!/usr/bin/env python3
"""
Research Paper Selection Tool for Evolution

Simple deterministic paper selection based on checkpoint state.
Papers are shuffled once and saved in checkpoint for consistency.
"""

import json
import random
from pathlib import Path

# List of available papers from papers/bird_methods/
AVAILABLE_PAPERS = [
    "bird_methods/askdata_gpt4o/Shkapenyuk_2025_AskData.pdf",
    "bird_methods/chase_sql_gemini/Pourreza_2024_CHASE_SQL.pdf",
    "bird_methods/xiyan_sql/Liu_2024_XiYan_SQL.pdf",
    "bird_methods/csc_sql/Sheng_2025_CSC_SQL.pdf",
    "bird_methods/reasoning_sql_14b/Pourreza_2025_Reasoning_SQL.pdf",
    "bird_methods/opensearch_sql_gpt4o/Xie_2025_OpenSearch_SQL.pdf",
    "bird_methods/omnisql_32b/Li_2025_OmniSQL.pdf",
    "bird_methods/distillery_gpt4o/Maamari_2024_Distillery.pdf",
    "bird_methods/genasql/Donder_2025_GenaSQL.pdf",
    "bird_methods/chess_stanford/Talaei_2024_CHESS.pdf",
]

# Paper metadata for display
PAPER_METADATA = {
    "bird_methods/askdata_gpt4o/Shkapenyuk_2025_AskData.pdf": {"name": "AskData + GPT-4o", "accuracy": "77.14%"},
    "bird_methods/chase_sql_gemini/Pourreza_2024_CHASE_SQL.pdf": {"name": "CHASE-SQL + Gemini", "accuracy": "76.02%"},
    "bird_methods/xiyan_sql/Liu_2024_XiYan_SQL.pdf": {"name": "XiYan-SQL", "accuracy": "75.63%"},
    "bird_methods/csc_sql/Sheng_2025_CSC_SQL.pdf": {"name": "CSC-SQL", "accuracy": "73.67%"},
    "bird_methods/reasoning_sql_14b/Pourreza_2025_Reasoning_SQL.pdf": {"name": "Reasoning-SQL 14B", "accuracy": "72.78%"},
    "bird_methods/opensearch_sql_gpt4o/Xie_2025_OpenSearch_SQL.pdf": {"name": "OpenSearch-SQL v2 + GPT-4o", "accuracy": "72.28%"},
    "bird_methods/omnisql_32b/Li_2025_OmniSQL.pdf": {"name": "OmniSQL-32B", "accuracy": "72.05%"},
    "bird_methods/distillery_gpt4o/Maamari_2024_Distillery.pdf": {"name": "Distillery + GPT-4o", "accuracy": "71.83%"},
    "bird_methods/genasql/Donder_2025_GenaSQL.pdf": {"name": "GenaSQL", "accuracy": "72.28%"},
    "bird_methods/chess_stanford/Talaei_2024_CHESS.pdf": {"name": "CHESS_IR+CG+UT", "accuracy": "71.10%"},
}


def find_checkpoint():
    """
    Find checkpoint.json in parent directories.
    
    Returns:
        Path to checkpoint.json or None if not found
    """
    current_path = Path.cwd()
    
    # Walk up parent directories looking for checkpoint.json
    search_path = current_path
    for _ in range(10):  # Limit search depth
        checkpoint_path = search_path / 'checkpoint.json'
        if checkpoint_path.exists():
            print(f"Found checkpoint at: {checkpoint_path}")
            return checkpoint_path
        
        # Move up one directory
        if search_path.parent == search_path:
            break
        search_path = search_path.parent
    
    print("No checkpoint.json found in parent directories")
    return None


def select_paper():
    """
    Select a paper based on checkpoint state using deterministic shuffling.
    
    Returns:
        Selected paper path
    """
    # Find and load checkpoint
    checkpoint_path = find_checkpoint()
    if not checkpoint_path:
        # No checkpoint, just select randomly
        selected = random.choice(AVAILABLE_PAPERS)
        print(f"No checkpoint found - randomly selected: {selected}")
        return selected
    
    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)
    
    # Get shuffled papers pool from checkpoint
    exp_config = checkpoint.get('experiment_config', {})
    shuffled_papers = exp_config.get('shuffled_papers_pool', None)
    
    # Fallback to creating a new shuffle if not in checkpoint (shouldn't happen in RoboPhD)
    if not shuffled_papers:
        print("⚠️ No shuffled_papers_pool in checkpoint, creating new shuffle")
        shuffled_papers = AVAILABLE_PAPERS.copy()
        random.shuffle(shuffled_papers)
    else:
        print(f"Using shuffled papers pool from checkpoint ({len(shuffled_papers)} papers)")
    
    # Count ALL research_driven evolutions in history
    # Each time research_driven strategy is used, it should use the next paper
    evolution_history = exp_config.get('evolution_history', [])
    research_driven_count = 0
    
    # Count ALL research_driven entries in history
    for entry in evolution_history:
        if 'research_driven' in entry.get('strategy', ''):
            research_driven_count += 1
    
    print(f"Found {research_driven_count} previous research_driven evolutions in history")
    
    # The next paper to use is based on how many times research_driven has been used
    total_used = research_driven_count
    print(f"Papers used: {total_used} (from research_driven count)")
    
    # Select next paper
    if total_used < len(shuffled_papers):
        selected = shuffled_papers[total_used]
        print(f"Selected paper #{total_used + 1}: {selected}")
    else:
        # Wrapped around - start over
        wrapped_index = total_used % len(shuffled_papers)
        selected = shuffled_papers[wrapped_index]
        print(f"Wrapped around paper list, selected #{wrapped_index + 1}: {selected}")
    
    return selected


def main():
    """Main entry point for the paper selection tool."""
    selected_paper = select_paper()
    
    # Print full paper information
    metadata = PAPER_METADATA.get(selected_paper, {})
    print(f"\n📚 Selected Paper Information:")
    print(f"   Path: papers/{selected_paper}")
    print(f"   Method: {metadata.get('name', 'Unknown')}")
    print(f"   Accuracy: {metadata.get('accuracy', 'N/A')}")
    
    # Output the path for the evolution process to use
    print(f"\n✅ Paper selection complete")
    print(f"SELECTED_PAPER=papers/{selected_paper}")
    
    # Find the evolution_output/iteration_XXX directory
    # We're called from the experiment root directory, so look for the latest iteration
    current_path = Path.cwd()
    evolution_output_dir = current_path / 'evolution_output'
    
    if evolution_output_dir.exists():
        # Find the latest iteration directory
        iter_dirs = sorted([d for d in evolution_output_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('iteration_')])
        if iter_dirs:
            output_dir = iter_dirs[-1]  # Latest iteration
            output_file = output_dir / 'selected_paper.txt'
        else:
            # No iteration directories yet, fallback to root
            output_file = Path('selected_paper.txt')
    else:
        # No evolution_output directory, fallback to root
        output_file = Path('selected_paper.txt')
    
    # Write to the determined output file
    with open(output_file, 'w') as f:
        f.write(f"papers/{selected_paper}")
    print(f"Written to: {output_file}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())