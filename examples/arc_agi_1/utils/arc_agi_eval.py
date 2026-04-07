# =============================================================================
# VENDORED FROM GEPA — keep these functions untouched
#
# Source: https://github.com/gepa-ai/gepa/blob/main/examples/arc_agi/utils.py
# Commit: 831cb70e974da717a0620582aed9f09a21c11e21
# Fetched: 2026-03-06
#
# This example uses only:
#   - load_arc_dataset()
#   - compare_grid()
#   - evaluate_predictions()
#   - evaluate_test()
#
# Removed from vendored original (not used by this example):
#   - TrackedLLM (replaced by evaluator.py's self-contained TrackedLLM)
#   - run_agent (replaced by evaluator.py's run_agent with stdout capture)
#   - evaluate_on_testset (replaced by eval_candidate() API)
#   - BACKGROUND, OBJECTIVE prompts (in background.md / objective.md)
# =============================================================================

"""ARC-AGI utilities: dataset loading and evaluation."""

import random
from typing import Any

import dspy
from datasets import load_dataset


# =============================================================================
# DATASET
# =============================================================================

def load_arc_dataset(seed: int = 0):
    """Load ARC-AGI dataset from HuggingFace.

    Returns (train_set, val_set, test_set) as dspy.Example lists.
    Format matches original: train_in, train_out, test_in, test_out
    """
    ds = load_dataset("dataartist/arc-agi")

    def make_example(ex):
        return dspy.Example(
            problem_id=ex["id"],
            train_in=[t["input"] for t in ex["train"]],
            train_out=[t["output"] for t in ex["train"]],
            test_in=[t["input"] for t in ex["test"]],
            test_out=[t["output"] for t in ex["test"]],
        ).with_inputs("problem_id", "train_in", "train_out", "test_in", "test_out")

    trainset = [make_example(ex) for ex in ds["training"]]
    testset = [make_example(ex) for ex in ds["evaluation"]]

    random.Random(seed).shuffle(trainset)

    val_set = trainset[-200:]
    train_set = trainset[:-200]
    test_set = testset

    print(f"Dataset: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    return train_set, val_set, test_set


# =============================================================================
# EVALUATION
# =============================================================================

def compare_grid(pred, gold) -> tuple[bool, str]:
    """Compare predicted grid to gold. Returns (is_correct, feedback)."""
    if not isinstance(pred, list):
        return (
            False,
            f"The matrix must be a List[List[int]], found {type(pred).__name__}. The correct matrix is {gold}.",
        )

    n = len(pred)
    if n == 0:
        return False, f"The matrix must have at least one row. The correct matrix is {gold}."

    if not isinstance(pred[0], list):
        return False, f"The matrix must be a 2D list. Row 0 is {type(pred[0]).__name__}. The correct matrix is {gold}."

    m = len(pred[0])
    if m == 0:
        return False, f"The matrix must have at least one column. The correct matrix is {gold}."

    # Structural and type checks
    for i in range(n):
        if not isinstance(pred[i], list):
            return False, f"Row {i} must be a list, found {type(pred[i]).__name__}. The correct matrix is {gold}."
        if len(pred[i]) != m:
            return (
                False,
                f"The matrix is staggered. Row 0 has {m} columns, but row {i} has {len(pred[i])} columns. The correct matrix is {gold}.",
            )
        for j in range(m):
            if not isinstance(pred[i][j], (int, float)):
                return (
                    False,
                    f"Element at ({i}, {j}) must be an int, found {type(pred[i][j]).__name__}. The correct matrix is {gold}.",
                )

    # Shape check
    pred_shape = (n, m)
    gold_shape = (len(gold), len(gold[0]))

    if pred_shape != gold_shape:
        return False, f"Shape {pred_shape} != expected {gold_shape}. The correct matrix is {gold}."

    # Value check
    wrong = []
    for i in range(len(gold)):
        for j in range(len(gold[0])):
            if int(pred[i][j]) != gold[i][j]:
                wrong.append((i, j))

    if not wrong:
        return True, "Correct!"

    if len(wrong) < 10:
        return False, f"Incorrect values at indices: {wrong}. The correct matrix is {gold}."
    return False, f"Incorrect values at {len(wrong)} positions. The correct matrix is {gold}."


def evaluate_predictions(preds: list, golds: list) -> tuple[float, list[dict]]:
    """Evaluate single predictions against gold. Returns (score, results)."""
    if not preds:
        return 0.0, [{"idx": i, "correct": False, "feedback": "No prediction"} for i in range(len(golds))]

    results = []
    for i in range(len(golds)):
        if i < len(preds) and preds[i] is not None:
            correct, feedback = compare_grid(preds[i], golds[i])
        else:
            correct, feedback = False, "No prediction"
        results.append({"idx": i, "correct": correct, "feedback": feedback})

    score = sum(1 for r in results if r["correct"]) / len(results) if results else 0.0
    return score, results


def evaluate_test(test_preds: list[list], test_out: list) -> tuple[float, list[dict]]:
    """Evaluate test with up to 2 attempts per example. Pass if ANY attempt correct."""
    if not test_preds:
        return 0.0, [{"idx": i, "correct": False, "feedback": "No prediction"} for i in range(len(test_out))]

    # Normalize: ensure each entry is a list of attempts
    normalized = [a[:2] if isinstance(a, list) else [a] for a in test_preds]

    # Evaluate each attempt using evaluate_predictions
    attempt1 = [attempts[0] if attempts else None for attempts in normalized]
    attempt2 = [attempts[1] if len(attempts) > 1 else None for attempts in normalized]

    _, results1 = evaluate_predictions(attempt1, test_out)
    _, results2 = evaluate_predictions(attempt2, test_out)

    # Aggregate: pass if ANY attempt correct
    results = []
    for i in range(len(test_out)):
        r1, r2 = results1[i], results2[i]
        correct = r1["correct"] or r2["correct"]
        feedback = r1["feedback"] if r1["correct"] else (r2["feedback"] if r2["correct"] else r1["feedback"])
        results.append({"idx": i, "correct": correct, "feedback": feedback})

    # ARC-AGI: must get ALL test examples correct to solve the problem (binary score)
    all_correct = all(r["correct"] for r in results)
    score = 1.0 if all_correct else 0.0
    return score, results
