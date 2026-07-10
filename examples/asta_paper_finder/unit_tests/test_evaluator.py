"""Unit tests for the asta_paper_finder evaluator.

Covers the pieces most likely to regress silently:

  - `aggregate()` math: free zone, breach arithmetic, unbounded-below,
    test-mode fraction, the eval_cost/agent_cost_usd coalesce.
  - Cost-knob validation at construction.
  - The agent-vs-judge cost split on a synthetic inspect log: judge
    usage (gpt-4o-2024-11-20) must land in other_cost_usd and NEVER in
    cost_usd / cost_by_model_usd.
  - `_head_tail_truncate` slice/marker behavior.
  - AST/source guards: `subprocess` is imported (the archived example
    shipped without it and crashed on the first eval), the killpg
    machinery is intact, and no diagnostics key uses bare "error"
    (the framework's failure detection reads "error.md").

Aggregate/split tests build instances via object.__new__ so they run
without provider keys or an astabench-importable environment beyond
what evaluator.py itself needs at module import.
"""
import ast
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PFB_DIR = REPO_ROOT / "examples" / "asta_paper_finder"
EVALUATOR_SRC = (PFB_DIR / "evaluator.py").read_text()


@pytest.fixture(scope="module")
def ev_mod():
    """Import evaluator.py once per module (heavy astabench import).

    PFB_DIR stays on sys.path for the module's lifetime — the
    constructor lazily does `from model_registry import ...`, which
    needs the example dir importable at construction time, not just at
    module-import time. (Run these tests as their own pytest invocation;
    the flat `evaluator` module name collides across examples.)
    """
    if str(PFB_DIR) not in sys.path:
        sys.path.insert(0, str(PFB_DIR))
    import evaluator  # noqa: E402
    return evaluator


def _bare_evaluator(ev_mod, *, apply_cost_penalty=True,
                    min_cost_threshold=0.10, cost_per_error=0.02):
    """Instance without running __init__ (no provider keys needed)."""
    ev = object.__new__(ev_mod.PaperFinderEvaluator)
    ev.apply_cost_penalty = apply_cost_penalty
    ev.min_cost_threshold = min_cost_threshold
    ev.cost_per_error = cost_per_error
    ev.tool_source = "mcp"
    ev.total_eval_cost = 0.0
    ev.total_judge_cost = 0.0
    ev._cost_lock = threading.Lock()
    return ev


# --- aggregate() ------------------------------------------------------------


def test_aggregate_empty_batch(ev_mod):
    ev = _bare_evaluator(ev_mod)
    assert ev.aggregate([]) == (0.0, "")


def test_aggregate_free_zone_scales_to_percentage(ev_mod):
    ev = _bare_evaluator(ev_mod)
    results = [{"score": 0.5, "eval_cost": 0.02}] * 20
    score, explanation = ev.aggregate(results)
    assert score == pytest.approx(50.0)
    assert "free zone" in explanation


def test_aggregate_breach_exact_arithmetic(ev_mod):
    """mean_cost 0.20 → excess 0.10 → 5 error-equivalents → 5 * (100/20)
    = 25 pts off a base of 100 * 0.4 = 40 → 15."""
    ev = _bare_evaluator(ev_mod)
    results = [{"score": 0.4, "eval_cost": 0.20}] * 20
    score, explanation = ev.aggregate(results)
    assert score == pytest.approx(40.0 - 25.0)
    assert "exceeded threshold" in explanation


def test_aggregate_unbounded_below(ev_mod):
    ev = _bare_evaluator(ev_mod)
    results = [{"score": 0.1, "eval_cost": 5.0}] * 10
    score, _ = ev.aggregate(results)
    assert score < -1000


def test_aggregate_test_mode_returns_fraction_no_explanation(ev_mod):
    ev = _bare_evaluator(ev_mod, apply_cost_penalty=False)
    results = [{"score": 0.5, "eval_cost": 99.0}] * 4
    assert ev.aggregate(results) == (0.5, "")


def test_aggregate_coalesces_agent_cost_usd_key(ev_mod):
    """Test-path diagnostics carry agent_cost_usd instead of eval_cost;
    the aggregator must read either."""
    ev = _bare_evaluator(ev_mod)
    results = [{"score": 0.4, "agent_cost_usd": 0.20}] * 20
    score, _ = ev.aggregate(results)
    assert score == pytest.approx(15.0)


def test_aggregate_never_reads_other_cost_usd(ev_mod):
    """Judge spend must not leak into the penalty: a batch whose only
    cost is other_cost_usd stays in the free zone."""
    ev = _bare_evaluator(ev_mod)
    results = [{"score": 0.5, "other_cost_usd": 10.0}] * 20
    score, explanation = ev.aggregate(results)
    assert score == pytest.approx(50.0)
    assert "free zone" in explanation


# --- constructor knob validation ---------------------------------------------


@pytest.mark.parametrize("kwargs,match", [
    ({"min_cost_threshold": -0.01}, "min_cost_threshold"),
    ({"cost_per_error": 0.0}, "cost_per_error"),
    ({"cost_per_error": -1.0}, "cost_per_error"),
])
def test_constructor_rejects_bad_knobs(ev_mod, monkeypatch, kwargs, match):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    with pytest.raises(ValueError, match=match):
        ev_mod.PaperFinderEvaluator(tool_source="search", **kwargs)


def test_constructor_requires_provider_keys(ev_mod, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY_FOR_ROBOPHD", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        ev_mod.PaperFinderEvaluator(tool_source="search")


def test_auto_tool_source_without_key_raises(ev_mod, monkeypatch):
    """No silent fallback: tool_source=None (auto) with no ASTA_TOOL_KEY
    must be a hard startup error, not a warn-and-degrade to the search
    tier (the asta_paper_finder_20260710_081139 failure mode)."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("ASTA_TOOL_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ASTA_TOOL_KEY"):
        ev_mod.PaperFinderEvaluator()


def test_explicit_search_without_key_constructs_with_warning(
    ev_mod, monkeypatch, caplog
):
    """--tool-source search stays available as the explicit dev opt-in
    even with no key at all — but it must warn about unauthenticated S2."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("ASTA_TOOL_KEY", raising=False)
    with caplog.at_level("WARNING"):
        ev = ev_mod.PaperFinderEvaluator(tool_source="search")
    assert ev.tool_source == "search"
    assert any("unauthenticated" in r.message for r in caplog.records)


def test_search_tools_get_retry_wrapper():
    """Retry parity: astabench's MCP factory wraps its tools in
    make_retry_wrapper internally; the search fallback ships bare, so
    _build_tools must apply the same wrapper there (and only there —
    the mcp branch must not double-wrap)."""
    idx_search_branch = EVALUATOR_SRC.index('if tool_source == "search":')
    idx_mcp_branch = EVALUATOR_SRC.index('if tool_source == "mcp":')
    idx_wrap = EVALUATOR_SRC.index("make_retry_wrapper(td)")
    assert EVALUATOR_SRC.count("make_retry_wrapper(td)") == 1
    assert idx_wrap > idx_search_branch > idx_mcp_branch


# --- judge-cost split ---------------------------------------------------------


def _fake_log(model_usage: dict):
    usage_objs = {
        name: SimpleNamespace(
            input_tokens=u[0], output_tokens=u[1],
            total_tokens=u[0] + u[1], reasoning_tokens=0,
        )
        for name, u in model_usage.items()
    }
    sample_log = SimpleNamespace(
        error=None,
        scores={"scorer": SimpleNamespace(value=0.5)},
        output=SimpleNamespace(completion="{}"),
    )
    return SimpleNamespace(
        samples=[sample_log],
        stats=SimpleNamespace(model_usage=usage_objs),
    )


def _fake_sample(ev_mod):
    from inspect_ai.dataset import Sample
    return Sample(
        id="semantic_2",
        input="find papers",
        target="criteria",
        metadata={"score_type": "semantic", "raw_query": "find papers"},
    )


def test_judge_split_routes_costs(ev_mod, monkeypatch):
    """gpt-4o judge usage → other_cost_usd; agent model → cost_usd +
    cost_by_model_usd; judge never in the agent buckets."""
    monkeypatch.setattr(
        ev_mod.PaperFinderEvaluator, "_estimate_cost",
        staticmethod(lambda model_name, counts: 1.0),
    )
    ev = _bare_evaluator(ev_mod)
    log = _fake_log({
        "openai/gpt-4o-2024-11-20": (5000, 500),   # the judge
        "openai/gpt-5.4-mini": (1000, 100),         # the agent
    })
    score, diag = ev._extract_score_and_diagnostics(log, _fake_sample(ev_mod), "")
    assert score == 0.5
    assert diag["other_cost_usd"] == pytest.approx(1.0)
    assert diag["cost_usd"] == pytest.approx(1.0)
    assert diag["agent_cost_usd"] == pytest.approx(1.0)
    assert "openai/gpt-4o-2024-11-20" not in diag["cost_by_model_usd"]
    assert "openai/gpt-5.4-mini" in diag["cost_by_model_usd"]
    # Usage summary keeps BOTH models for the audit trail.
    assert set(diag["usage"]) == {
        "openai/gpt-4o-2024-11-20", "openai/gpt-5.4-mini",
    }
    # Running totals split the same way.
    assert ev.total_eval_cost == pytest.approx(1.0)
    assert ev.total_judge_cost == pytest.approx(1.0)


def test_judge_ids_match_astabench(ev_mod):
    """The module-load assert enforces this, but pin it as a test too so
    a refactor that drops the assert still gets caught."""
    from astabench.evals.paper_finder.relevance import GRADER_MODEL_NAME
    assert GRADER_MODEL_NAME in ev_mod.JUDGE_MODEL_IDS


# --- _head_tail_truncate ------------------------------------------------------


def test_truncate_short_string_passes_through(ev_mod):
    s = "x" * 100
    assert ev_mod._head_tail_truncate(s) == s


def test_truncate_long_string_keeps_head_and_tail(ev_mod):
    s = "H" * 200 + "M" * 5000 + "T" * 1500
    out = ev_mod._head_tail_truncate(s)
    assert out.startswith("H" * 200)
    assert out.endswith("T" * 1500)
    assert "chars truncated" in out
    assert len(out) < len(s)


def test_truncate_marker_reports_count(ev_mod):
    s = "a" * 200 + "b" * 1000 + "c" * 1500
    out = ev_mod._head_tail_truncate(s)
    assert "(1000 chars truncated)" in out


# --- source-level guards ------------------------------------------------------


def test_subprocess_is_imported():
    """Finding A regression guard: the archived example used
    subprocess.run without importing subprocess — first eval crashed
    with NameError."""
    tree = ast.parse(EVALUATOR_SRC)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "subprocess" in imported
    assert "signal" in imported
    assert "time" in imported


def test_killpg_machinery_intact():
    """A refactor that drops start_new_session/killpg/bounded drain
    silently re-introduces the orphan-grandchild hang class."""
    assert "start_new_session=True" in EVALUATOR_SRC
    assert "os.killpg" in EVALUATOR_SRC
    assert "proc.communicate(timeout=30)" in EVALUATOR_SRC


def test_no_bare_error_diagnostics_key():
    """Every failure-path diagnostic must use "error.md" — the framework's
    failure detection (domains/external/domain.py) checks that exact key,
    and bare "error" records crashes as error:false in result.json."""
    for bad in ('"error":', "'error':", '["error"]', "diagnostics[\"error\"]"):
        assert bad not in EVALUATOR_SRC, (
            f"bare {bad} found in evaluator.py — use \"error.md\""
        )
    assert '"error.md"' in EVALUATOR_SRC


# --- _estimate_cost / bundled price map ---------------------------------------
#
# Ported from asta_ds1000's unit_tests/test_evaluator.py (ae1e410): the
# pricing machinery here is a byte-for-byte copy of ds1000's, and the
# repo's self-contained-examples convention means the ds1000 tests do
# NOT protect this copy — these duplicates do. If the two copies are
# ever deliberately diverged, these tests fail loudly on this side,
# which is the alarm we want (see README "Code work" for the
# extract-to-shared-module endpoint). Two invariants under pin:
#
# 1. Reasoning-token fold: Gemini reports reasoning separately and
#    excludes it from output_tokens; billed at the output rate. The
#    strict token-arithmetic guard must fire for the Gemini pattern
#    only — OpenAI (reasoning inside output) and Anthropic (cache
#    tokens inflating total) must NOT fold.
#
# 2. Bundled-map basis: internal costs are priced from litellm's BUNDLED
#    snapshot (what `astabench score` bills under
#    LITELLM_LOCAL_MODEL_COST_MAP=True), NOT the live map. Models absent
#    from the snapshot fall back to the live map WITH a one-shot warning;
#    a snapshot that fails to load warns loudly instead of silently
#    repricing everything on the live map.
#
# All tests inject a fixture price map (and stub litellm.cost_per_token
# for the fallback path) so they are hermetic against the installed
# litellm's actual map contents.

FIXTURE_PRICES = {
    # litellm-1.88.1 bundled rates for the v0_0_6 models (per token)
    "gemini/gemini-3.1-flash-lite": {
        "input_cost_per_token": 4.5e-07, "output_cost_per_token": 2.7e-06,
    },
    "gpt-5.4-mini": {
        "input_cost_per_token": 7.5e-07, "output_cost_per_token": 4.5e-06,
    },
    "claude-haiku-4-5-20251001": {
        "input_cost_per_token": 1e-06, "output_cost_per_token": 5e-06,
    },
    "gpt-5.4-2026-03-05": {
        "input_cost_per_token": 2.5e-06, "output_cost_per_token": 1.5e-05,
    },
}


@pytest.fixture()
def bundled_fixture_map(ev_mod, monkeypatch):
    """Point the bundled-map cache at FIXTURE_PRICES for one test."""
    monkeypatch.setattr(ev_mod, "_BUNDLED_PRICE_MAP", dict(FIXTURE_PRICES))
    return ev_mod


def _est(mod, model, **counts):
    return mod.PaperFinderEvaluator._estimate_cost(model, counts)


def test_gemini_reasoning_tokens_are_billed(bundled_fixture_map):
    """Gemini pattern (input == total - output - reasoning): fold fires."""
    mod = bundled_fixture_map
    cost = _est(
        mod, "google/gemini-3.1-flash-lite",
        input_tokens=1_000_000, output_tokens=100_000,
        total_tokens=1_300_000, reasoning_tokens=200_000,
    )
    expected = 1_000_000 * 4.5e-07 + (100_000 + 200_000) * 2.7e-06
    assert cost == pytest.approx(expected)


def test_openai_reasoning_not_double_billed(bundled_fixture_map):
    """OpenAI pattern (reasoning already inside output): fold must NOT fire."""
    mod = bundled_fixture_map
    with_r = _est(
        mod, "gpt-5.4-mini",
        input_tokens=1000, output_tokens=500, total_tokens=1500,
        reasoning_tokens=300,
    )
    without_r = _est(
        mod, "gpt-5.4-mini",
        input_tokens=1000, output_tokens=500, total_tokens=1500,
        reasoning_tokens=0,
    )
    assert with_r == pytest.approx(without_r)


def test_anthropic_cache_pattern_not_mistaken_for_reasoning(bundled_fixture_map):
    """Cache tokens inflate total (input excludes them); no reasoning fold."""
    mod = bundled_fixture_map
    cost = _est(
        mod, "claude-haiku-4-5-20251001",
        input_tokens=1000, output_tokens=500, total_tokens=2500,
        reasoning_tokens=0,
    )
    expected = 1000 * 1e-06 + 500 * 5e-06
    assert cost == pytest.approx(expected)


def test_bundled_map_wins_over_live_map(bundled_fixture_map, monkeypatch):
    """When the snapshot has the model, litellm.cost_per_token is never hit."""
    mod = bundled_fixture_map
    import litellm

    def _boom(**kwargs):
        raise AssertionError("live map must not be consulted for bundled models")

    monkeypatch.setattr(litellm, "cost_per_token", _boom)
    cost = _est(
        mod, "gpt-5.4-2026-03-05",
        input_tokens=1000, output_tokens=100, total_tokens=1100,
        reasoning_tokens=0,
    )
    assert cost == pytest.approx(1000 * 2.5e-06 + 100 * 1.5e-05)


def test_missing_model_falls_back_to_live_map_with_warning(
    bundled_fixture_map, monkeypatch, caplog
):
    mod = bundled_fixture_map
    import litellm

    monkeypatch.setattr(
        litellm, "cost_per_token", lambda **kw: (0.001, 0.002)
    )
    mod._live_map_fallback_warned.discard("brand-new-model")
    with caplog.at_level("WARNING"):
        cost = _est(
            mod, "brand-new-model",
            input_tokens=1000, output_tokens=100, total_tokens=1100,
            reasoning_tokens=0,
        )
    assert cost == pytest.approx(0.003)
    assert any("NOT on the leaderboard's billing basis" in r.message for r in caplog.records)
    # One-shot: second call must not warn again
    with caplog.at_level("WARNING"):
        caplog.clear()
        _est(
            mod, "brand-new-model",
            input_tokens=1000, output_tokens=100, total_tokens=1100,
            reasoning_tokens=0,
        )
    assert not any("billing basis" in r.message for r in caplog.records)


def test_snapshot_load_failure_warns_loudly(ev_mod, monkeypatch, caplog):
    """A missing/renamed snapshot must not silently reprice on the live map."""
    monkeypatch.setattr(ev_mod, "_BUNDLED_PRICE_MAP", None)

    import litellm
    real_file = litellm.__file__
    monkeypatch.setattr(litellm, "__file__", "/nonexistent/litellm/__init__.py")
    try:
        with caplog.at_level("WARNING"):
            result = ev_mod._bundled_price_map()
    finally:
        monkeypatch.setattr(litellm, "__file__", real_file)
    assert result == {}
    assert any(
        "bundled price snapshot" in r.message for r in caplog.records
    ), "snapshot load failure must warn, not silently drift to the live map"


def test_v0_0_6_official_cost_regression(bundled_fixture_map):
    """Golden regression: the recorded model_usage from asta_ds1000's
    officially-scored v0_0_6 run must reproduce the official astabench
    per-problem cost ($0.004280335) on the frozen fixture rates. The
    fixture is ds1000 provenance, but it pins THIS module's copy of the
    pricing function (reasoning fold + bundled basis, end to end) —
    which is byte-identical and must stay on the same billing basis."""
    mod = bundled_fixture_map
    usage = {
        "google/gemini-3.1-flash-lite": dict(
            input_tokens=1_465_500, output_tokens=163_410,
            total_tokens=1_862_265, reasoning_tokens=233_355,
        ),
        "gpt-5.4-mini": dict(
            input_tokens=1_171_114, output_tokens=79_480,
            total_tokens=1_250_594, reasoning_tokens=0,
        ),
        "claude-haiku-4-5-20251001": dict(
            input_tokens=254_413, output_tokens=42_332,
            total_tokens=296_745, reasoning_tokens=0,
        ),
        "gpt-5.4-2026-03-05": dict(
            input_tokens=141_163, output_tokens=4_439,
            total_tokens=145_602, reasoning_tokens=0,
        ),
    }
    total = sum(_est(mod, m, **c) for m, c in usage.items())
    assert total / 900 == pytest.approx(0.004280335, abs=1e-6)
