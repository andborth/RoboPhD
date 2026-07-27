"""Unit tests for tool_pacer — the global tool-call launch pacer.

Timing assertions are lower-bound only (a loaded CI box can always be
slower, never faster than the reserved slots), with generous rates so the
whole file runs in ~1s.
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

PFB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PFB_DIR))

import tool_pacer  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    """Isolate every test: clear in-process slots and both env knobs."""
    tool_pacer._local_slots.clear()
    monkeypatch.delenv(tool_pacer.ENV_RATE, raising=False)
    monkeypatch.delenv(tool_pacer.ENV_PATH, raising=False)


def _run_concurrent(n: int, endpoint: str = "snippet_search", rate: float = 50.0):
    async def go():
        t0 = time.monotonic()
        await asyncio.gather(*(tool_pacer.pace(endpoint, rate) for _ in range(n)))
        return time.monotonic() - t0
    return asyncio.run(go())


def test_serializes_one_endpoint_at_rate():
    n, rate = 10, 50.0
    elapsed = _run_concurrent(n, rate=rate)
    assert elapsed >= (n - 1) / rate * 0.95  # slots are 1/rate apart


def test_endpoints_do_not_serialize_each_other():
    async def go():
        t0 = time.monotonic()
        await asyncio.gather(*(
            tool_pacer.pace(f"endpoint_{i}", 5.0) for i in range(8)
        ))
        return time.monotonic() - t0
    # 8 distinct endpoints, first slot each: no waiting.
    assert asyncio.run(go()) < 0.5


def test_rate_zero_disables():
    elapsed = _run_concurrent(50, endpoint="x", rate=0)
    assert elapsed < 0.2


def test_env_rate_used_when_no_explicit_rate(monkeypatch):
    monkeypatch.setenv(tool_pacer.ENV_RATE, "50")
    n = 8
    async def go():
        t0 = time.monotonic()
        await asyncio.gather(*(tool_pacer.pace("e") for _ in range(n)))
        return time.monotonic() - t0
    assert asyncio.run(go()) >= (n - 1) / 50.0 * 0.95


def test_env_rate_garbage_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(tool_pacer.ENV_RATE, "not-a-number")
    assert tool_pacer._effective_rate(None) == tool_pacer.DEFAULT_RATE


def test_file_backend_serializes(tmp_path, monkeypatch):
    state = tmp_path / "pacer" / "launch_slots.json"
    monkeypatch.setenv(tool_pacer.ENV_PATH, str(state))
    n, rate = 8, 50.0
    async def go():
        t0 = time.monotonic()
        await asyncio.gather(*(
            tool_pacer.pace("snippet_search", rate) for _ in range(n)
        ))
        return time.monotonic() - t0
    elapsed = asyncio.run(go())
    assert elapsed >= (n - 1) / rate * 0.90  # wall-clock backend: looser bound
    assert state.exists()


def test_file_backend_survives_corrupt_state(tmp_path, monkeypatch):
    state = tmp_path / "slots.json"
    state.write_text("{torn")
    monkeypatch.setenv(tool_pacer.ENV_PATH, str(state))
    asyncio.run(tool_pacer.pace("e", 50.0))  # must not raise
    import json
    assert "e" in json.loads(state.read_text())


def test_pace_tools_wraps_and_paces():
    async def search_papers(query: str) -> str:
        """Search for papers.

        Args:
            query: The search query.

        Returns:
            Search results.
        """
        return f"results for {query}"

    n, rate = 6, 50.0
    [wrapped] = tool_pacer.pace_tools([search_papers], rate=rate)

    async def go():
        t0 = time.monotonic()
        results = await asyncio.gather(*(wrapped(query="q") for _ in range(n)))
        return time.monotonic() - t0, results

    elapsed, results = asyncio.run(go())
    assert all(r == "results for q" for r in results)
    assert elapsed >= (n - 1) / rate * 0.95


def test_pace_tools_preserves_tool_schema():
    from inspect_ai.tool import ToolDef

    async def search_papers(query: str) -> str:
        """Search for papers.

        Args:
            query: The search query.

        Returns:
            Search results.
        """
        return "ok"

    [wrapped] = tool_pacer.pace_tools([search_papers])
    td = ToolDef(wrapped)
    assert td.name == "search_papers"
    assert "query" in td.parameters.properties
