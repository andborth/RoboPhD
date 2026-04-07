"""Shared evaluation utilities: rate limit handling, parallel eval loops, cleanup."""

import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

logger = logging.getLogger(__name__)


class EvalRateLimitError(Exception):
    """Raised when an evaluator hits an API rate limit.

    Even one rate-limited evaluation corrupts score comparisons because
    the agent gets an unfair 0 on a problem it might have solved.
    """
    pass


def is_rate_limit_error(exc: BaseException) -> bool:
    """Check whether an exception is a rate limit error.

    Walks the exception's MRO and cause chain to catch rate limit errors
    from any library (openai, litellm, etc.) without importing them.
    """
    # Check class hierarchy names (catches openai.RateLimitError,
    # litellm.exceptions.RateLimitError, and subclasses)
    for cls in type(exc).__mro__:
        if "RateLimitError" in cls.__name__:
            return True
    # Check chained causes — explicit (raise X from Y) and implicit (raise X inside except Y)
    chained = exc.__cause__ or exc.__context__
    if chained and chained is not exc:
        return is_rate_limit_error(chained)
    return False


def retry_on_rate_limit(fn, max_retries=5, base_delay=0.5):
    """Call fn(), retrying with exponential backoff on rate limit errors.

    Use this inside LLM callables (TrackedLLM, make_tracked_llm, etc.)
    to handle transient rate limits before they reach the framework.
    Retry must happen inside the callable because agent code is untrusted
    and may catch exceptions — if swallowed at the agent level, the
    framework's EvalRateLimitError detection never fires.

    If all retries are exhausted, the exception propagates up through
    the evaluator to domain.py, which raises EvalRateLimitError to
    crash the run and prevent corrupted scores.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if is_rate_limit_error(e) and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Rate limit hit, retrying in %.1fs (attempt %d/%d)",
                    delay, attempt + 1, max_retries,
                )
                time.sleep(delay)
                continue
            raise


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
                    if is_rate_limit_error(e):
                        for f in not_done:
                            f.cancel()
                        raise EvalRateLimitError(str(e)) from e
                    logger.warning(f"Evaluator error on example {idx}: {e}")
                    score, diag = 0.0, {"error": str(e)}
                score_map[idx] = score
                diag_map[idx] = diag
            remaining = not_done

            if not quiet and len(score_map) % progress_interval == 0:
                vals = list(score_map.values())
                mean = sum(vals) / len(vals)
                logger.info(f"Test progress: {len(score_map)}/{len(examples)}, running score: {mean:.3f}")
    except EvalRateLimitError:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
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


def force_exit_if_threads_leaked(grace_period: float = 0.5):
    """Force-exit if non-daemon threads are still alive after a grace period.

    Call this at the end of a script's if __name__ == "__main__" block.
    Leaked eval timeout threads and httpx/litellm connection pools can
    block Python's atexit handler indefinitely. A brief grace period
    lets well-behaved threads shut down naturally.

    Usage::

        if __name__ == "__main__":
            try:
                main()
            finally:
                force_exit_if_threads_leaked()
    """
    time.sleep(grace_period)
    alive = [t for t in threading.enumerate()
             if t is not threading.main_thread() and t.is_alive() and not t.daemon]
    if alive:
        names = ", ".join(t.name for t in alive)
        logger.info(f"Force-exiting ({len(alive)} non-daemon thread(s) still running: {names})")
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if sys.exc_info()[0] else 0)
