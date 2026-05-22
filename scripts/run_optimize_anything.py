#!/usr/bin/env python3
"""
Demo: Evolve a math-problem-solving prompt using optimize_anything().

Optimizes a system prompt that guides Claude Haiku to solve math word problems.
The seed prompt is deliberately minimal (no chain-of-thought, no formatting).
Evolution discovers better prompting strategies through Elo competition.

Usage:
    # Quick demo (3 iterations, ~$1-2 in API costs)
    python scripts/run_optimize_anything.py --num-iterations 3

    # Minimal test run
    python scripts/run_optimize_anything.py --num-iterations 2 \
        --evaluation-budget 50 --examples-per-iteration 5

    # Longer run for better results
    python scripts/run_optimize_anything.py --num-iterations 10 \
        --evaluation-budget 500

Requires:
    - ANTHROPIC_API_KEY_FOR_ROBOPHD environment variable
    - Claude Code CLI installed (for evolution)
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RoboPhD import optimize_anything, eval_candidate, RoboPhDConfig
from RoboPhD.config import get_api_key
from RoboPhD.llm_providers import get_provider

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset: 20 math word problems (self-contained, no downloads)
# ---------------------------------------------------------------------------

DATASET = [
    # Simple arithmetic
    {"id": "arith_01", "question": "If you have 15 apples and give away 7, how many apples do you have left?", "answer": "8"},
    {"id": "arith_02", "question": "What is 247 + 358?", "answer": "605"},
    {"id": "arith_03", "question": "A baker makes 12 cookies per batch. How many cookies does he make in 8 batches?", "answer": "96"},
    {"id": "arith_04", "question": "If you divide 156 equally among 12 people, how much does each person get?", "answer": "13"},
    {"id": "arith_05", "question": "What is 17 times 23?", "answer": "391"},

    # Multi-step problems
    {"id": "multi_01", "question": "A store sells shirts for $25 each. If you buy 3 shirts and get a 20% discount on the total, how much do you pay?", "answer": "60"},
    {"id": "multi_02", "question": "A car travels at 60 mph for 2 hours, then at 40 mph for 3 hours. What is the total distance traveled in miles?", "answer": "240"},
    {"id": "multi_03", "question": "You have $100. You spend $35 on books and $22 on lunch. Then you find $10 on the ground. How much money do you have now?", "answer": "53"},
    {"id": "multi_04", "question": "A factory produces 150 widgets per hour. If it runs for 8 hours a day and 10% of widgets are defective, how many good widgets are produced per day?", "answer": "1080"},
    {"id": "multi_05", "question": "Three friends split a dinner bill of $87 equally, and each leaves a $5 tip. How much does each person pay in total?", "answer": "34"},

    # Percentages and fractions
    {"id": "pct_01", "question": "What is 15% of 80?", "answer": "12"},
    {"id": "pct_02", "question": "A jacket originally costs $120. It is on sale for 35% off. What is the sale price?", "answer": "78"},
    {"id": "pct_03", "question": "If 3/4 of a class of 32 students passed the exam, how many students passed?", "answer": "24"},
    {"id": "pct_04", "question": "A stock price increased from $40 to $50. What was the percentage increase?", "answer": "25"},

    # Reasoning / word problems
    {"id": "reason_01", "question": "A train leaves Station A at 9:00 AM traveling at 80 km/h. Another train leaves Station B (400 km away) at the same time traveling toward Station A at 120 km/h. At what time do they meet? Give the answer as the number of hours after 9:00 AM.", "answer": "2"},
    {"id": "reason_02", "question": "A rectangular garden is 12 meters long and 8 meters wide. What is its area in square meters?", "answer": "96"},
    {"id": "reason_03", "question": "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?", "answer": "5"},
    {"id": "reason_04", "question": "A pond has lily pads that double in number every day. If the pond is completely covered on day 30, on what day was the pond half covered?", "answer": "29"},
    {"id": "reason_05", "question": "You have a 3-gallon jug and a 5-gallon jug. How many gallons can the 5-gallon jug hold?", "answer": "5"},
    {"id": "reason_06", "question": "A snail climbs 3 feet up a wall during the day but slides back 2 feet at night. How many days does it take to reach the top of a 10-foot wall?", "answer": "8"},
]


# ---------------------------------------------------------------------------
# Seed prompt (deliberately minimal — no CoT, no formatting instructions)
# ---------------------------------------------------------------------------

SEED_CANDIDATE = {"system_prompt": "Solve the following math problem. Give your final answer as a number."}


# ---------------------------------------------------------------------------
# Objective and background for the evolution AI
# ---------------------------------------------------------------------------

OBJECTIVE = (
    "Evolve a system prompt that maximizes accuracy when Claude Haiku 4.5 "
    "solves math word problems. The prompt should guide the model to reason "
    "carefully and present its final answer in a way that can be reliably "
    "extracted."
)

BACKGROUND = """\
## Task: Math Word Problem Prompt Optimization

You are optimizing a **system prompt** (`system_prompt.txt`) that will be sent to
Claude alongside a math word problem. The model's response is then
parsed to extract a numeric answer, which is compared to the ground truth.

### How scoring works

- Score is 1.0 if the extracted answer matches the expected answer, 0.0 otherwise
- Answer extraction looks for: the last standalone number on its own line,
  then falls back to the last number in the response

### Dataset

The dataset contains 20 math word problems spanning:
- Simple arithmetic (addition, multiplication, division)
- Multi-step problems (discounts, distance/time, sequences of operations)
- Percentages and fractions
- Reasoning / trick questions (rate problems, exponential growth, classic puzzles)

"""


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(response: str) -> str | None:
    """
    Extract the final numeric answer from an LLM response.

    Looks for the last standalone number on its own line first,
    then falls back to the last number in the response.
    """
    if not response:
        return None

    # First: look for a final line that is just a number (possibly with decimal)
    lines = response.strip().splitlines()
    for line in reversed(lines):
        line = line.strip().strip("*").strip()  # handle **42** markdown bold
        if re.fullmatch(r'-?\d+\.?\d*', line):
            # Normalize: "60.0" -> "60", "12.5" -> "12.5"
            val = float(line)
            return str(int(val)) if val == int(val) else str(val)

    # Fallback: last standalone number in the response
    matches = re.findall(r'(?<!\w)-?\d+\.?\d*(?!\w)', response)
    if matches:
        val = float(matches[-1])
        return str(int(val)) if val == int(val) else str(val)

    return None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def make_evaluator(solver_model: str = "haiku-4.5"):
    """Create an evaluator function closed over a shared provider."""

    provider = get_provider(solver_model)

    def evaluator(candidate, example, *, problem_dir=None):
        system_prompt = candidate["system_prompt"]
        question = example["question"]
        expected = example["answer"]

        try:
            response = provider.generate(
                system_prompt=system_prompt,
                user_prompt=question,
                max_tokens=1024,
            )
            response_text = response.text
        except Exception as e:
            logger.error(f"LLM call failed for {example['id']}: {e}")
            return 0.0, {"error.md": f"LLM call failed: {e}"}

        extracted = extract_answer(response_text)
        correct = (extracted == expected)
        score = 1.0 if correct else 0.0

        diagnostics = {
            "score": score,
            "predicted_answer": extracted,
            "expected_answer": expected,
            # String values -> written as files in problem_dir for evolution AI
            "question.md": question,
            "system_prompt.md": system_prompt,
            "response.md": response_text,
        }
        return score, diagnostics

    return evaluator


# ---------------------------------------------------------------------------
# Seed baseline
# ---------------------------------------------------------------------------

def run_seed_baseline(evaluator_fn):
    """Run the seed prompt on all examples and print accuracy."""
    eval_result = eval_candidate(evaluator=evaluator_fn, dataset=DATASET, candidate=SEED_CANDIDATE)
    correct = int(eval_result.total_score)
    total = eval_result.num_examples

    for i, (score, ex) in enumerate(zip(eval_result.per_example_scores, DATASET)):
        if score < 1.0:
            logger.debug(f"  MISS {ex['id']}: expected={ex['answer']}")

    pct = eval_result.mean_score * 100
    print(f"\nSeed baseline: {correct}/{total} correct ({pct:.0f}%)")
    print(f"Seed prompt: {SEED_CANDIDATE['system_prompt']!r}\n")
    return eval_result.mean_score


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo: Evolve a math-solving prompt with optimize_anything()",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--num-iterations", type=int, default=3,
                        help="Number of evolution iterations (default: 3)")
    parser.add_argument("--evaluation-budget", type=int, default=200,
                        help="Max evaluator calls across all iterations (default: 200)")
    parser.add_argument("--examples-per-iteration", type=int, default=10,
                        help="Examples sampled per iteration (default: 10)")
    parser.add_argument("--solver-model", type=str, default="haiku-4.5",
                        help="Model for solving math problems (default: haiku-4.5)")
    parser.add_argument("--parent-experiments-dir", type=str, default="../robophd_runs",
                        help="Root directory under which experiment folders are created (default: ../robophd_runs)")
    parser.add_argument("--seed-only", action="store_true",
                        help="Only run seed baseline (no evolution)")
    parser.add_argument("--demo-resume", action="store_true",
                        help="Demo resume/extend: 2 iterations → extend +1 → from_iteration 2 + extend 1")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Resume / extend demo
# ---------------------------------------------------------------------------

def print_step_result(step_name: str, result):
    """Print a compact summary of an optimization step."""
    print(f"\n{'─' * 50}")
    print(f"  {step_name}")
    print(f"{'─' * 50}")
    print(f"  Iterations completed: {result.num_iterations_completed}")
    print(f"  Total evaluations:    {result.total_evaluations}")
    print(f"  Best Elo:             {result.best_score:.1f}")
    print(f"  Agents explored:      {len(result.all_candidates)}")
    print(f"  Experiment dir:       {result.experiment_dir}")


def run_demo_resume(evaluator_fn, parent_experiments_dir: str):
    """Demo: fresh run → extend → from_iteration + extend."""
    # Step 1: Fresh run for 2 iterations
    print("\n" + "=" * 60)
    print("  Step 1: Fresh run (2 iterations)")
    print("=" * 60)
    result = optimize_anything(
        evaluator=evaluator_fn,
        dataset=DATASET,
        seed_candidate=SEED_CANDIDATE,
        objective=OBJECTIVE,
        background=BACKGROUND,
        config=RoboPhDConfig(
            num_iterations=2,
            evaluation_budget=500,
            examples_per_iteration=5,
            parent_experiments_dir=parent_experiments_dir,
        ),
    )
    print_step_result("Step 1 complete", result)
    experiment_dir = result.experiment_dir

    # Step 2: Extend by 1 iteration (continues from iteration 3)
    # objective/background are recovered from checkpoint — no need to pass them
    print("\n" + "=" * 60)
    print("  Step 2: Extend +1 iteration (resume from checkpoint)")
    print("=" * 60)
    result = optimize_anything(
        evaluator=evaluator_fn,
        dataset=DATASET,
        config=RoboPhDConfig(
            experiment_dir=experiment_dir,
            extend_iterations=1,
            evaluation_budget=500,
            examples_per_iteration=5,
            parent_experiments_dir=parent_experiments_dir,
        ),
    )
    print_step_result("Step 2 complete", result)

    # Step 3: Restart from iteration 2 and extend by 1
    # This discards iterations 2-3 and re-runs from iteration 2
    print("\n" + "=" * 60)
    print("  Step 3: Restart from iteration 2, extend +1 (re-does iterations 2-3)")
    print("=" * 60)
    result = optimize_anything(
        evaluator=evaluator_fn,
        dataset=DATASET,
        config=RoboPhDConfig(
            experiment_dir=experiment_dir,
            from_iteration=2,
            extend_iterations=1,
            evaluation_budget=500,
            examples_per_iteration=5,
            parent_experiments_dir=parent_experiments_dir,
        ),
    )
    print_step_result("Step 3 complete", result)

    # Final summary
    print("\n" + "=" * 60)
    print("  Demo Complete")
    print("=" * 60)
    print(f"\n  Best prompt:\n")
    print(result.best_candidate["system_prompt"])

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Check API key
    api_key = get_api_key()
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY_FOR_ROBOPHD (or ANTHROPIC_API_KEY) environment variable.")
        sys.exit(1)

    evaluator_fn = make_evaluator(args.solver_model)

    if args.demo_resume:
        print("=" * 60)
        print("  optimize_anything() Demo: Resume & Extend")
        print("=" * 60)
        print(f"\nSolver model: {args.solver_model}")
        print(f"Dataset: {len(DATASET)} math word problems")

        print("\n--- Seed Baseline ---")
        run_seed_baseline(evaluator_fn)

        run_demo_resume(evaluator_fn, args.parent_experiments_dir)
        os._exit(0)

    print("=" * 60)
    print("  optimize_anything() Demo: Math Problem Prompt Evolution")
    print("=" * 60)
    print(f"\nSolver model: {args.solver_model}")
    print(f"Dataset: {len(DATASET)} math word problems")
    print(f"Evolution: {args.num_iterations} iterations, budget={args.evaluation_budget}")

    # Run seed baseline
    print("\n--- Seed Baseline ---")
    seed_accuracy = run_seed_baseline(evaluator_fn)

    if args.seed_only:
        return

    # Run evolution
    print("\n--- Starting Evolution ---\n")
    result = optimize_anything(
        evaluator=evaluator_fn,
        dataset=DATASET,
        seed_candidate=SEED_CANDIDATE,
        objective=OBJECTIVE,
        background=BACKGROUND,
        config=RoboPhDConfig(
            num_iterations=args.num_iterations,
            evaluation_budget=args.evaluation_budget,
            examples_per_iteration=args.examples_per_iteration,
            parent_experiments_dir=args.parent_experiments_dir,
        ),
    )

    # Print results
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"\nIterations completed: {result.num_iterations_completed}")
    print(f"Total evaluations: {result.total_evaluations}")
    print(f"Best agent Elo: {result.best_score:.1f}")
    print(f"Experiment dir: {result.experiment_dir}")

    print(f"\n--- Best Evolved Prompt ---\n")
    print(result.best_candidate["system_prompt"])

    # Run best candidate on all examples for final accuracy
    print(f"\n--- Final Evaluation (best prompt on all {len(DATASET)} problems) ---\n")
    eval_result = eval_candidate(evaluator=evaluator_fn, dataset=DATASET, candidate=result.best_candidate)
    correct = int(eval_result.total_score)
    final_accuracy = eval_result.mean_score
    print(f"Evolved prompt: {correct}/{len(DATASET)} correct ({final_accuracy * 100:.0f}%)")
    print(f"Seed prompt:    {seed_accuracy * 100:.0f}%")
    improvement = (final_accuracy - seed_accuracy) * 100
    print(f"Improvement:    {'+' if improvement >= 0 else ''}{improvement:.0f}pp")

    os._exit(0)


if __name__ == "__main__":
    main()
