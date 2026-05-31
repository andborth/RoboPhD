"""Shared evaluation utilities: rate limit handling, parallel eval loops, stdout capture, cleanup."""

import builtins
import io
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

logger = logging.getLogger(__name__)


class PeakRSSSampler:
    """Record the peak process RSS over a window, to attribute in-eval memory
    spikes to a specific agent.

    A daemon thread polls the kernel-maintained RSS of the current process via
    psutil every `interval` seconds and keeps the max. It is strictly read-only:
    it never touches the evaluated agent's code, namespace, or builtins — it
    only observes a number the OS already maintains. Because evals parallelize
    across the *examples of a single agent* (one candidate per run), the peak
    seen between start()/stop() is attributable to that one agent.

    Gated behind ROBOPHD_MEMLOG=1 so normal runs pay nothing (no thread, no
    polling). Boundary tracemalloc snapshots only sample the trough between
    evals; this catches the peak *during* exec(agent_code).
    """

    def __init__(self, label: str, interval: float = 0.1):
        self.label = label
        self.interval = interval
        self._enabled = os.environ.get("ROBOPHD_MEMLOG") == "1"
        self.start_rss = 0
        self.peak_rss = 0
        self._proc = None
        self._stop = threading.Event()
        self._thread = None

    def _read_rss(self) -> int:
        try:
            return self._proc.memory_info().rss
        except Exception:
            return 0

    def _run(self):
        while not self._stop.is_set():
            rss = self._read_rss()
            if rss > self.peak_rss:
                self.peak_rss = rss
            self._stop.wait(self.interval)

    def start(self):
        if not self._enabled:
            return self
        try:
            import psutil
            self._proc = psutil.Process(os.getpid())
        except Exception:
            self._enabled = False
            return self
        self.start_rss = self.peak_rss = self._read_rss()
        self._thread = threading.Thread(target=self._run, daemon=True, name="peak-rss-sampler")
        self._thread.start()
        return self

    def stop(self):
        if not self._enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # Capture any last-moment growth the poll loop may have missed.
        rss = self._read_rss()
        if rss > self.peak_rss:
            self.peak_rss = rss
        gb = 1024 ** 3
        print(
            f"🧠 PEAK[{self.label}] start={self.start_rss / gb:.2f}GB "
            f"peak={self.peak_rss / gb:.2f}GB Δ={(self.peak_rss - self.start_rss) / gb:+.2f}GB"
        )


# Process-global eval-health counters, surfaced on the per-iteration MEM line
# so API pressure (timeouts / rate limits) is visible at a glance alongside
# memory. Cumulative across the run; the MEM line also shows the per-iteration
# delta. Incremented from the eval-loop thread; lock-guarded to be safe.
_EVAL_COUNTERS = {"timeouts": 0, "rate_limits": 0}
_EVAL_COUNTERS_LOCK = threading.Lock()


def record_eval_event(kind: str) -> None:
    """Increment a process-global eval-health counter ('timeouts' | 'rate_limits')."""
    with _EVAL_COUNTERS_LOCK:
        if kind in _EVAL_COUNTERS:
            _EVAL_COUNTERS[kind] += 1


def get_eval_counters() -> dict:
    """Snapshot the process-global eval-health counters."""
    with _EVAL_COUNTERS_LOCK:
        return dict(_EVAL_COUNTERS)


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
    for ``openrouter/google/gemini-3.1-flash-lite`` comes back with
    ``resp.model`` like ``google/gemini-3.1-flash-lite-20260507``, and
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
                # `agent_cost_usd` is what we're optimizing on the cost
                # axis (it tracks model spend that the evolution loop
                # can act on). Total eval_cost includes infra/judge
                # spend that's interesting for accounting but isn't an
                # optimization target — keep it out of the running log.
                # Only surface the running mean when at least one example
                # reported a non-zero agent cost, so cost-less domains
                # (Sudoku, Can't Be Late, ARC) log unchanged.
                agent_costs = [
                    (d or {}).get("agent_cost_usd") or 0.0
                    for d in diag_map.values()
                ]
                msg = f"Test progress: {len(score_map)}/{len(examples)}, running score: {mean:.3f}"
                if any(c > 0 for c in agent_costs):
                    mean_cost = sum(agent_costs) / len(agent_costs)
                    msg += f", running mean cost: ${mean_cost:.4f}"
                logger.info(msg)
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

    # Iteration-level aggregate. Evaluators can opt into batch-level
    # scoring by exposing an `aggregate(results) -> (scalar, explanation)`
    # method. Default falls back to simple mean. The test path doesn't
    # go through ExternalEvaluatorDomain, so we duplicate the hook here
    # for architectural symmetry — but the evaluator may behave
    # differently in test mode (e.g. DS-1000 disables its cost penalty
    # via apply_cost_penalty=False, returning the leaderboard-format
    # fraction instead of the scaled training number).
    aggregator = getattr(evaluator, "aggregate", None)
    if aggregator is not None:
        # Build per-example results matching the domain layer's shape so
        # the aggregator sees a consistent input. The test path doesn't
        # normalize agent_cost_usd → eval_cost like the domain does, so
        # we pass diagnostics through verbatim; aggregators that need
        # cost must coalesce the two keys.
        per_example_results = [
            {"score": scores[i], **(diagnostics[i] or {})}
            for i in range(len(scores))
        ]
        mean_score, aggregate_explanation = aggregator(per_example_results)
    else:
        mean_score = sum(scores) / len(scores) if scores else 0.0
        aggregate_explanation = ""

    if not quiet:
        logger.info(f"Test score: {mean_score:.3f} ({len(scores)} problems)")

    test_results = {
        "mean_test_score": mean_score,
        "total_test_score": sum(scores),
        "total_test_problems": len(scores),
        "aggregate_explanation": aggregate_explanation,
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


# ---------------------------------------------------------------------------
# Agent subprocess isolation
#
# Some evolved agents are pathological: they allocate unbounded memory (a
# bomb that OOMs the whole run) or arm signal.alarm() from a worker thread
# where the handler can't install (a stray SIGALRM that terminates the whole
# process). Running the agent in a disposable subprocess contains both: a
# memory cap turns the bomb into a scored-0 child death, and in a subprocess
# the agent runs on the main thread so its signal.alarm() works as intended
# (and any stray signal only kills the child). Pattern mirrors the ASTA
# evaluators' _evaluate_via_subprocess; the memory cap is the addition.
# ---------------------------------------------------------------------------

# Distinct exit code the in-child memory watchdog uses, so the parent can
# tell "agent exceeded the memory ceiling" apart from a generic nonzero exit.
AGENT_OOM_EXIT_CODE = 42


def apply_agent_memory_cap(limit_bytes: int, poll_interval: float = 0.05) -> None:
    """Cap the CURRENT (child) process's memory, called from the eval worker.

    Two mechanisms:
    1. Best-effort RLIMIT_AS soft cap (unreliable on macOS, so belt-and-
       suspenders only — when it does fire, allocations raise MemoryError).
    2. Primary: an always-on RSS watchdog daemon thread (same polling pattern
       as PeakRSSSampler) that os._exit(AGENT_OOM_EXIT_CODE) once RSS exceeds
       the ceiling. Runs as a background daemon so the agent keeps the main
       thread (required for the agent's own signal.signal(SIGALRM) to install).
    """
    if limit_bytes <= 0:
        return
    try:
        import resource
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))
    except (ValueError, OSError, ImportError):
        pass  # macOS frequently won't enforce this; the watchdog is the real guard

    try:
        import psutil
        proc = psutil.Process(os.getpid())
    except Exception:
        return  # no psutil → rely on RLIMIT_AS alone

    def _watch():
        while True:
            try:
                if proc.memory_info().rss > limit_bytes:
                    sys.stderr.write(
                        f"[agent-mem-watchdog] RSS exceeded {limit_bytes} bytes — killing child\n"
                    )
                    sys.stderr.flush()
                    os._exit(AGENT_OOM_EXIT_CODE)
            except Exception:
                pass
            time.sleep(poll_interval)

    threading.Thread(target=_watch, daemon=True, name="agent-mem-watchdog").start()


def run_evaluation_in_subprocess(
    worker_script: str,
    input_params: dict,
    *,
    timeout: float,
    memory_limit_bytes: int,
    drain_timeout: float = 30.0,
) -> tuple:
    """Run one evaluation in a fresh, memory-capped subprocess.

    Spawns ``python <worker_script> <in.json> <out.json>`` in a new session
    (so timeouts can killpg the whole group), passing the memory ceiling via
    the ROBOPHD_AGENT_MEMORY_BYTES env var for the worker to self-apply. The
    worker writes ``{"score": float, "diagnostics": dict}`` JSON.

    Returns ``(score, diagnostics)``. Every failure mode (memory kill,
    timeout, nonzero exit, signal death, bad payload) maps to ``(0.0, {...})``
    with an "error" string and truncated "subprocess_stderr" so evolution sees
    actionable feedback and the parent run never dies.
    """
    gb = memory_limit_bytes / (1024 ** 3)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as inf:
        json.dump(input_params, inf, default=str)
        in_path = inf.name
    out_path = in_path + ".out"
    env = {**os.environ, "ROBOPHD_AGENT_MEMORY_BYTES": str(int(memory_limit_bytes))}
    _t0 = time.monotonic()

    try:
        proc = subprocess.Popen(
            [sys.executable, str(worker_script), in_path, out_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            # New session → killpg on timeout reaches grandchildren too.
            # POSIX-only; ignored on Windows (not supported elsewhere here).
            start_new_session=True,
        )
        try:
            _stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            try:
                _stdout, stderr = proc.communicate(timeout=drain_timeout)
            except subprocess.TimeoutExpired as drain_exc:
                proc.kill()
                stderr = drain_exc.stderr or e.stderr or "<post-kill drain timed out>"
            return 0.0, {
                "error": f"agent subprocess timed out after {timeout:.0f}s",
                "subprocess_stderr": (stderr or "")[-2000:],
            }

        rc = proc.returncode
        if rc != 0:
            stub = (stderr or "")[-2000:]
            if rc == AGENT_OOM_EXIT_CODE or rc == -signal.SIGKILL:
                error = f"Agent ran out of memory: exceeded {gb:.1f} GB ceiling"
            elif rc < 0:
                try:
                    signame = signal.Signals(-rc).name
                except ValueError:
                    signame = str(-rc)
                error = f"agent subprocess killed by {signame}"
            else:
                error = f"agent subprocess failed (exit {rc})"
            return 0.0, {"error": error, "subprocess_stderr": stub}

        try:
            with open(out_path) as f:
                payload = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return 0.0, {
                "error": f"agent subprocess produced no valid output: {type(e).__name__}: {e}",
                "subprocess_stderr": (stderr or "")[-2000:],
            }
        return float(payload.get("score", 0.0)), (payload.get("diagnostics", {}) or {})
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass
