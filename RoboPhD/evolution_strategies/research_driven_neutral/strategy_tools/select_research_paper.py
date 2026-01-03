#!/usr/bin/env python3
"""
Research Paper Selection Tool for Evolution

Manages paper selection from the strategy's local papers pool.
Papers are tracked in papers_pool.json to ensure each paper is used once.
"""

import json
import random
from pathlib import Path


def load_papers_pool():
    """
    Load papers pool from the strategy's tools directory.

    Returns:
        dict: Papers pool with 'papers' and 'used_papers' lists
    """
    # Find papers_pool.json in the same directory as this script
    script_dir = Path(__file__).parent
    pool_path = script_dir / 'papers_pool.json'

    if not pool_path.exists():
        raise FileNotFoundError(f"Papers pool not found at: {pool_path}")

    with open(pool_path, 'r') as f:
        pool = json.load(f)

    return pool, pool_path


def save_papers_pool(pool, pool_path):
    """
    Save papers pool back to disk.

    Args:
        pool: Papers pool dict
        pool_path: Path to papers_pool.json
    """
    with open(pool_path, 'w') as f:
        json.dump(pool, f, indent=2)


def initialize_pool():
    """
    Initialize the papers pool with a random shuffle.

    This should be called once when the strategy is first copied to the
    experiment directory. It shuffles the papers list to provide per-experiment
    randomization.
    """
    pool, pool_path = load_papers_pool()

    # Shuffle the papers list for this experiment
    papers = pool.get('papers', [])
    random.shuffle(papers)
    pool['papers'] = papers

    # Clear any existing used_papers
    pool['used_papers'] = []

    save_papers_pool(pool, pool_path)
    print(f"✓ Initialized papers pool with shuffled order ({len(papers)} papers)")


def select_paper():
    """
    Select next unused paper from the pool.

    Papers are shuffled once per experiment (when pool is first created).
    Each call takes the next unused paper from the shuffled list.
    When all papers are used, the pool is re-shuffled for the next cycle.

    Returns:
        dict: Selected paper with path, title, topics, year
    """
    pool, pool_path = load_papers_pool()

    papers = pool.get('papers', [])
    used_papers = pool.get('used_papers', [])

    if not papers:
        raise ValueError("Papers pool is empty")

    # Find unused papers
    used_paths = set(used_papers)
    unused_papers = [p for p in papers if p['path'] not in used_paths]

    if not unused_papers:
        # All papers used, re-shuffle and start over
        print(f"⚠️  All {len(papers)} papers have been used, re-shuffling pool")
        random.shuffle(papers)
        pool['papers'] = papers
        pool['used_papers'] = []
        save_papers_pool(pool, pool_path)
        unused_papers = papers

    # Select next unused paper (first in shuffled list)
    selected = unused_papers[0]

    print(f"📚 Selected paper {len(used_papers) + 1}/{len(papers)}: {selected['title']} ({selected['year']})")
    print(f"   Topics: {', '.join(selected['topics'])}")

    # Mark as used
    pool['used_papers'].append(selected['path'])
    save_papers_pool(pool, pool_path)

    return selected


def main():
    """Main entry point for the paper selection tool."""
    try:
        selected_paper = select_paper()

        # Print full paper information
        print(f"\n📄 Paper Details:")
        print(f"   Path: papers/{selected_paper['path']}")
        print(f"   Title: {selected_paper['title']}")
        print(f"   Year: {selected_paper['year']}")
        print(f"   Topics: {', '.join(selected_paper['topics'])}")

        # Output the path for the evolution process to use
        print(f"\n✅ Paper selection complete")
        print(f"SELECTED_PAPER=papers/{selected_paper['path']}")

        # Write to output file for evolution agent to read
        # Evolution workspace is typically evolution_output/iteration_XXX/
        # This script is called from that directory
        output_file = Path('selected_paper.txt')
        with open(output_file, 'w') as f:
            f.write(f"papers/{selected_paper['path']}")
        print(f"Written to: {output_file}")

        return 0

    except Exception as e:
        print(f"❌ Error selecting paper: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
