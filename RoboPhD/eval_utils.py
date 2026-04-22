"""Shared evaluation utilities: rate limit handling, parallel eval loops, stdout capture, cleanup."""

import builtins
import io
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


# Dedupe warning keys for extract_response_cost's final-fallback path.
# Module-global so warnings fire once per (model, error_type) per process
# rather than once per call.
_cost_lookup_warned: set = set()


def extract_response_cost(resp, model: str) -> float:
    """Best-effort cost extraction from a litellm completion/embedding response.

    ``litellm.completion_cost(resp)`` raises ``"This model isn't mapped yet"``
    when the provider returns a dated or versioned model name that isn't in
    litellm's pricing database. OpenRouter is a frequent offender: a request
    for ``openrouter/google/gemini-3.1-flash-lite-preview`` comes back with
    ``resp.model`` like ``google/gemini-3.1-flash-lite-preview-20260303``, and
    the dated suffix isn't indexed even when the undated alias is.

    The actual billed cost is still available in the response — OpenRouter
    populates ``resp.usage.cost`` and litellm mirrors it as
    ``resp._hidden_params["response_cost"]``. Try those provider-direct
    sources first, then fall back to a pricing lookup with the model name the
    caller originally passed (which typically IS in litellm's DB), then 0.

    Terminates on ``is not None`` rather than ``> 0``: a legitimate zero-cost
    call (free tier, zero-token completion) should be reported as $0 rather
    than passed along as "cost unknown" to subsequent sources.

    Warns once per (model, error type) when the final fallback returns 0 so
    pricing-DB regressions surface loudly without per-call log spam.
    """
    usage_cost = getattr(getattr(resp, "usage", None), "cost", None)
    if usage_cost is not None:
        return float(usage_cost)
    hidden = getattr(resp, "_hidden_params", None) or {}
    hp_cost = hidden.get("response_cost")
    if hp_cost is not None:
        return float(hp_cost)
    # Final fallback: explicit-model pricing lookup. Local import so tasks
    # that never hit this helper (e.g. cant_be_late) don't take a litellm
    # import cost.
    try:
        import litellm
        return float(
            litellm.completion_cost(completion_response=resp, model=model) or 0.0
        )
    except Exception as e:
        key = (model, type(e).__name__)
        if key not in _cost_lookup_warned:
            _cost_lookup_warned.add(key)
            logger.warning(
                "extract_response_cost: falling back to $0.00 for model=%r "
                "(%s: %s). Cost tracking will under-report this model until "
                "either the provider populates usage.cost / "
                "_hidden_params.response_cost on the response, or the model "
                "is added to litellm's pricing DB.",
                model, type(e).__name__, str(e)[:200],
            )
        return 0.0


def exec_with_stdout_capture(code: str, extra_namespace: dict = None, then=None) -> tuple:
    """Execute Python code with print() output captured.

    Patches __builtins__ so print() is captured even from nested function
    calls and imported modules within the exec'd code. This is how RoboPhD
    surfaces agent diagnostics to the evolution AI.

    Args:
        code: Python source code to execute.
        extra_namespace: Additional names to inject (e.g., llm, embed callables).
        then: Optional callable(namespace) invoked after exec, still within
            the capture context. Use this to call agent functions (e.g.,
            solve(), answer()) so their print() output is also captured.
            The return value is stored in namespace["_result"].

    Returns:
        (namespace, stdout): The exec namespace (for extracting results)
        and the captured print output as a string.

    On exception, attaches partial stdout to the exception as ``stdout``
    attribute so callers can recover diagnostic output from crashes.
    """
    buf = io.StringIO()

    def _captured_print(*args, **kwargs):
        kwargs.setdefault("file", buf)
        print(*args, **kwargs)

    patched_builtins = dict(vars(builtins))
    patched_builtins["print"] = _captured_print
    namespace = {"print": _captured_print, "__builtins__": patched_builtins}
    if extra_namespace:
        namespace.update(extra_namespace)
    try:
        exec(code, namespace)
        if then is not None:
            namespace["_result"] = then(namespace)
    except Exception as e:
        e.stdout = buf.getvalue()
        raise
    return namespace, buf.getvalue()


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


def _silence_resource_tracker():
    """Silence the multiprocessing resource_tracker before os._exit().

    When os._exit() bypasses normal cleanup, the resource tracker (a child
    process) warns about "leaked semaphore objects". These are harmless —
    the kernel reclaims POSIX semaphores at process exit — but the warning
    is confusing.

    The tracker inherits the parent's stderr at spawn time, so we can't
    redirect it after the fact. Instead we kill the tracker process
    (SIGKILL) and reap it, preventing it from running its cleanup/warning
    code at all. The kernel still reclaims any POSIX semaphores.
    """
    try:
        import signal
        from multiprocessing import resource_tracker
        tracker = resource_tracker._resource_tracker
        if tracker._pid is not None:
            pid = tracker._pid
            os.close(tracker._fd)
            tracker._fd = None
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            tracker._pid = None
    except Exception:
        pass


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
        _silence_resource_tracker()
        os._exit(1 if sys.exc_info()[0] else 0)
