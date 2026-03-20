"""Shared test evaluation utilities for run_gepa.py and eval_test_set.py."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

logger = logging.getLogger(__name__)


def run_parallel_eval(
    evaluator,
    candidate: dict,
    examples: list,
    max_workers: int | None = None,
    eval_timeout: int = 300,
    progress_interval: int = 10,
    quiet: bool = False,
) -> dict:
    """Run evaluator on examples in parallel with timeout handling.

    Returns dict with:
        "scores": list of raw scores (ordered by example index)
        "test_results": dict ready for JSON (mean_test_score, total_test_score, total_test_problems)
        "timed_out": True if any evaluation timed out (leaked threads still running)

    Timed-out evaluations score 0. Logs progress and final summary.
    """
    timed_out = False
    timed_out_idxs: set[int] = set()

    score_map: dict[int, float] = {}
    diag_map: dict[int, dict] = {}
    idx_to_example = {i: ex for i, ex in enumerate(examples)}
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_to_idx = {
            executor.submit(evaluator, candidate, ex): i
            for i, ex in enumerate(examples)
        }
        remaining = set(future_to_idx.keys())

        while remaining:
            done, not_done = wait(
                remaining, timeout=eval_timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                still_remaining = set()
                for future in not_done:
                    idx = future_to_idx[future]
                    if future.cancel():
                        if idx in timed_out_idxs:
                            # Already resubmitted once — give up
                            logger.warning(
                                f"EVAL TIMEOUT: example {idx} never started after resubmit — scored 0"
                            )
                            score_map[idx] = 0.0
                            diag_map[idx] = {"error": f"timeout after {eval_timeout}s"}
                            timed_out = True
                        else:
                            # Queued (never ran) — resubmit once
                            logger.info(
                                f"EVAL TIMEOUT: example {idx} never started within {eval_timeout}s — resubmitting"
                            )
                            timed_out_idxs.add(idx)
                            new_future = executor.submit(
                                evaluator, candidate, idx_to_example[idx]
                            )
                            future_to_idx[new_future] = idx
                            still_remaining.add(new_future)
                    else:
                        logger.warning(
                            f"EVAL TIMEOUT: example {idx} exceeded {eval_timeout}s — "
                            f"scored 0, thread leaked (will burn CPU until process exit)"
                        )
                        score_map[idx] = 0.0
                        diag_map[idx] = {"error": f"timeout after {eval_timeout}s"}
                        timed_out_idxs.add(idx)
                        timed_out = True
                remaining = still_remaining
                continue

            for future in done:
                idx = future_to_idx[future]
                try:
                    score, diag = future.result()
                except Exception as e:
                    # Re-raise rate limit errors — even one corrupts score comparisons
                    error_str = str(e)
                    if "RateLimitError" in type(e).__name__ or "rate_limit" in error_str:
                        for f in not_done:
                            f.cancel()
                        raise RuntimeError(f"API_RATE_LIMIT: {e}") from e
                    logger.warning(f"Evaluator error on example {idx}: {e}")
                    score, diag = 0.0, {"error": error_str}
                score_map[idx] = score
                diag_map[idx] = diag
            remaining = not_done

            if not quiet and len(score_map) % progress_interval == 0:
                vals = list(score_map.values())
                mean = sum(vals) / len(vals)
                logger.info(f"Test progress: {len(score_map)}/{len(examples)}, running score: {mean:.3f}")
    finally:
        if timed_out:
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True)

    scores = [score_map[i] for i in range(len(examples))]
    diagnostics = [diag_map.get(i, {}) for i in range(len(examples))]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    if not quiet:
        logger.info(f"Test score: {mean_score:.3f} ({len(scores)} problems)")

    test_results = {
        "mean_test_score": mean_score,
        "total_test_score": sum(scores),
        "total_test_problems": len(scores),
    }

    return {"scores": scores, "diagnostics": diagnostics, "test_results": test_results, "timed_out": timed_out}
