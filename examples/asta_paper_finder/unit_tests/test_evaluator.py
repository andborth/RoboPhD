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
