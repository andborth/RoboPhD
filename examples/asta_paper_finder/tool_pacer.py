"""Global launch pacer for Asta MCP tool calls.

The Asta MCP backend enforces ~10 requests/second per endpoint, shared by
every caller using the key at once. Concurrent evaluation (training: one
subprocess per sample, up to max_workers at once; official `astabench eval`:
several samples as asyncio tasks in one process) can aggregate far past that
budget, and the overrun surfaces as windows of HTTP 500s in which every call
to the endpoint fails (run asta_paper_finder_20260724_193339, iterations
9-11: 100+ tool errors in one batch; exact-match queries zeroed). Per-agent
self-throttles cannot fix aggregate contention — no agent can see the other
workers — so launches are paced here, at the harness level.

Algorithm: slot reservation per endpoint (keyed by tool name). Acquiring a
slot takes the earliest permissible launch time (`max(now, next_slot)`),
advances `next_slot` by `1/rate`, then sleeps until its slot. No polling,
O(1) state, FIFO-fair through lock ordering. The rate bounds LAUNCHES, not
in-flight calls, so slow responses still overlap — pacing caps request rate
without giving up latency hiding.

Two state backends, selected by environment:

- ``PF_TOOL_PACER_PATH`` set  -> file-backed state shared across processes
  (flock on a ``.lock`` sidecar, atomic tempfile+rename writes — the
  _safe_cache_rmw pattern from evaluator.py). Wall-clock timestamps, since
  monotonic clocks are not comparable across processes. Used by RoboPhD
  training/test evals, where every sample is its own subprocess; pointing
  concurrent runs at the same path makes them share the budget, which is
  correct because they share the API key.
- unset -> in-process state (dict + threading.Lock; the critical section is
  pure arithmetic, so a thread lock held for microseconds is safe under
  asyncio and works regardless of which event loop calls it). Used by the
  staged leaderboard-submission wrapper, where `astabench eval` runs all
  samples in one process.

``PF_TOOL_LAUNCH_RATE`` (float, launches/second per endpoint) overrides the
default of 8.0 — chosen to leave headroom under the documented 10 req/s.
A rate <= 0 disables pacing.

This module is deliberately self-contained (stdlib + inspect_ai, which is
the runtime in both contexts): it is copied verbatim into leaderboard
submission working directories by scripts/asta_paper_finder_submit.py and
must not import anything from RoboPhD.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import tempfile
import threading
import time

ENV_RATE = "PF_TOOL_LAUNCH_RATE"
ENV_PATH = "PF_TOOL_PACER_PATH"
DEFAULT_RATE = 8.0

_local_slots: dict[str, float] = {}
_local_lock = threading.Lock()


def _effective_rate(rate: float | None) -> float:
    if rate is not None:
        return rate
    raw = os.environ.get(ENV_RATE, "")
    try:
        return float(raw) if raw else DEFAULT_RATE
    except ValueError:
        return DEFAULT_RATE


def _reserve_local(endpoint: str, rate: float) -> float:
    """Reserve the next launch slot in-process; return seconds to wait."""
    with _local_lock:
        now = time.monotonic()
        slot = max(_local_slots.get(endpoint, 0.0), now)
        _local_slots[endpoint] = slot + 1.0 / rate
        return slot - now


def _reserve_filed(path: str, endpoint: str, rate: float) -> float:
    """Reserve the next launch slot in shared file state; return seconds to wait.

    Synchronous and module-level (not a closure) so the multi-process gate
    (_check_pacer_stress.py) can exercise exactly this code, and so pace()
    can push the blocking flock + file I/O off the event loop via
    asyncio.to_thread. Wall-clock (time.time) because the state is shared
    across processes.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".lock", "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            state: dict = {}
            try:
                text = open(path).read()
            except OSError:
                text = ""
            if text:
                try:
                    state = json.loads(text)
                except json.JSONDecodeError:
                    state = {}  # tiny state; a torn file just resets pacing
            now = time.time()
            slot = max(float(state.get(endpoint, 0.0)), now)
            state[endpoint] = slot + 1.0 / rate
            fd, tmp_path = tempfile.mkstemp(
                dir=os.path.dirname(path) or ".", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as tmp_f:
                    json.dump(state, tmp_f)
                os.replace(tmp_path, path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            return slot - now
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


async def pace(endpoint: str, rate: float | None = None) -> None:
    """Wait for this endpoint's next launch slot (no-op when rate <= 0)."""
    eff = _effective_rate(rate)
    if eff <= 0:
        return
    path = os.environ.get(ENV_PATH, "")
    if path:
        wait = await asyncio.to_thread(_reserve_filed, path, endpoint, eff)
    else:
        wait = _reserve_local(endpoint, eff)
    if wait > 0:
        await asyncio.sleep(wait)


def pace_tools(tools: list, rate: float | None = None) -> list:
    """Wrap inspect tools so every invocation acquires a launch slot first.

    ToolDef round-trip preserves name/description/parameters (the schema
    inspect needs) — the same mechanism astabench's retry wrapper and the
    evaluator's provenance wrapper use. The slot is acquired per logical
    call; when the retry ladder lives inside the wrapped tool (as in the
    staged submission wrapper, which wraps astabench's already-retrying
    tools), retries share the paced launch. Training installs pacing inside
    the retry ladder instead (see evaluator._install_tool_transport_hardening).
    """
    from inspect_ai.tool import ToolDef

    wrapped = []
    for t in tools:
        td = ToolDef(t)
        orig = td.tool

        def _make(orig_call, name):
            async def _paced(*args, **kwargs):
                await pace(name, rate)
                return await orig_call(*args, **kwargs)
            return _paced

        td.tool = _make(orig, td.name)
        wrapped.append(td.as_tool())
    return wrapped
