"""Tests for the per-model cost-breakdown render paths in
`_generate_iteration_cost_report` (RoboPhD/researcher.py).

The report has two new surfaces gated on multi-model usage:

  - "## Cost by Model" — iteration-level section, gates on at least one
    agent having >=2 models in its aggregate breakdown. When it fires,
    every agent gets a per-agent block (single-model agents render at
    100% for cross-agent comparison).
  - Inline annotation on Top-5 task lines — per-(agent, task) gate; only
    pairs that hit >=2 models get a "(model $X, model $Y, …)" suffix.

Tests below pin: (a) the iteration-level gate, (b) the per-task gate,
(c) provider-prefix stripping in the inline annotation, (d) descending
sort within blocks/inline lists, (e) full model list inline (no truncation).
"""
import tempfile
from pathlib import Path

from RoboPhD.researcher import ParallelAgentResearcher


class _FakeResearcher:
    """Minimal harness: just expose the report method bound to an
    object with .experiment_dir set. The method only reads
    self.experiment_dir; everything else flows through arguments."""
    _generate_iteration_cost_report = ParallelAgentResearcher._generate_iteration_cost_report

    def __init__(self, expdir):
        self.experiment_dir = Path(expdir)


def _result(agent, ctx, eval_cost, cost_by_model=None):
    return {
        "success": True,
        "context": ctx,
        "agent_id": agent,
        "score": 100,
        "total": 1,
        "error": None,
        "eval_cost": eval_cost,
        "other_cost": 0.0,
        "cost_by_model": cost_by_model or {},
    }


def _render(results_by_agent):
    """Build minimal costs_by_context from results_by_agent and run the
    report; return the rendered text."""
    costs_by_context = {
        a: {r["context"]: {"eval": r["eval_cost"], "other": 0.0} for r in rs}
        for a, rs in results_by_agent.items()
    }
    with tempfile.TemporaryDirectory() as tmp:
        expdir = Path(tmp)
        (expdir / "iteration_001").mkdir()
        _FakeResearcher(expdir)._generate_iteration_cost_report(
            1, results_by_agent, costs_by_context, None
        )
        return (expdir / "iteration_001" / "cost_report.md").read_text()


# ---------------------------------------------------------------------------
# Agent Cost Summary: Avg/Problem column, footnote, no Most-Expensive section
# ---------------------------------------------------------------------------

def test_agent_summary_avg_per_problem_column():
    """Avg/Problem = total_cost / problems_tested, cache-agnostic.

    An agent with 20 problems summing to $0.20 must render `$0.010`
    in the Avg/Problem column. This pins the cache-independence of
    the calculation — adding cache stats to the input must not change
    the answer (regression test for the prior cost/fresh workaround).
    """
    # 20 problems @ $0.01 each = $0.20 total, $0.20 / 20 = $0.010
    rba = {
        "iter1": [_result("iter1", f"p{i:02d}", 0.01) for i in range(20)],
    }
    text = _render(rba)
    # Split on the Cost Insights heading rather than the bare `---`,
    # since the markdown table separator (e.g. `|---|---|`) trips the
    # naive split.
    summary_section = text.split("## Agent Cost Summary", 1)[1].split("## Cost Insights", 1)[0]
    # New Avg/Problem column header
    assert "Avg/Problem" in summary_section
    # Per-agent value
    iter1_row = [line for line in summary_section.splitlines() if line.startswith("| iter1 ")][0]
    assert "$0.010" in iter1_row
    # Total row: single agent so total avg = same
    total_row = [line for line in summary_section.splitlines() if "**Total**" in line][0]
    assert "**$0.010**" in total_row


def test_agent_summary_has_cache_footnote():
    """Footnote clarifies the Avg/Problem semantics and that cache
    plays no role in the calculation. Drops cleanly out of plain
    legacy reports — the footnote always renders when the summary
    table does."""
    rba = {
        "iter1": [_result("iter1", f"p{i:02d}", 0.01) for i in range(20)],
    }
    text = _render(rba)
    summary_section = text.split("## Agent Cost Summary", 1)[1].split("## Cost Insights", 1)[0]
    assert (
        "Avg/Problem is total cost divided by problems tested. "
        "Cache does not affect this calculation." in summary_section
    )


def test_footnote_suppressed_when_no_tests():
    """When zero problems ran (all agents failed before evaluation),
    every Avg/Problem cell is $0.000 — the footnote would describe a
    calculation that didn't happen, so it's suppressed."""
    rba = {"agent_a": []}  # agent in pool, no results
    text = _render(rba)
    assert "Cache does not affect this calculation" not in text


def test_most_expensive_agents_section_removed():
    """The legacy '### Most Expensive Agents' section is subsumed by the
    Avg/Problem column and must not appear in any output. Pins the
    literal heading rather than a generic substring so future prose
    that mentions 'most expensive agents' in a sentence wouldn't
    false-positive the test."""
    rba = {
        "seed": [_result("seed", "p1", 0.02), _result("seed", "p2", 0.03)],
        "iter1": [_result("iter1", "p1", 0.05), _result("iter1", "p2", 0.04)],
    }
    text = _render(rba)
    assert "### Most Expensive Agents" not in text


# ---------------------------------------------------------------------------
# Iteration-level gating: section presence
# ---------------------------------------------------------------------------

def test_no_cost_by_model_anywhere_omits_section():
    """Legacy task / no opt-in: section absent, report unchanged from today."""
    rba = {
        "seed": [_result("seed", "p1", 0.02), _result("seed", "p2", 0.03)],
        "iter1": [_result("iter1", "p1", 0.05), _result("iter1", "p2", 0.04)],
    }
    text = _render(rba)
    assert "## Cost by Model" not in text


def test_only_single_model_agents_omits_section():
    """Every agent uses exactly one model: section gated out."""
    rba = {
        "seed": [
            _result("seed", "p1", 0.02, {"openai/gpt-5.4-mini": 0.02}),
            _result("seed", "p2", 0.03, {"openai/gpt-5.4-mini": 0.03}),
        ],
        "iter1": [
            _result("iter1", "p1", 0.05, {"openai/gpt-5.4-mini": 0.05}),
            _result("iter1", "p2", 0.04, {"openai/gpt-5.4-mini": 0.04}),
        ],
    }
    text = _render(rba)
    assert "## Cost by Model" not in text


def test_at_least_one_multimodel_agent_renders_section():
    """One multi-model agent triggers the section; all agents get a block."""
    rba = {
        "seed": [
            _result("seed", "p1", 0.02, {"openai/gpt-5.4-mini": 0.02}),
            _result("seed", "p2", 0.03, {"openai/gpt-5.4-mini": 0.03}),
        ],
        "iter5_multi": [
            _result("iter5_multi", "p1", 0.075, {
                "anthropic/claude-sonnet-4-5": 0.030,
                "openai/gpt-5.4": 0.025,
                "openai/gpt-5.4-mini": 0.020,
            }),
        ],
    }
    text = _render(rba)
    assert "## Cost by Model" in text
    section = text.split("## Cost by Model", 1)[1].split("---", 1)[0]
    # Both agents present — single-model `seed` shown at 100% for context
    assert "**seed**" in section
    assert "**iter5_multi**" in section
    assert "(100%)" in section


def test_router_pattern_triggers_section():
    """Agent that uses model A on some problems and model B on others —
    no individual problem is multi-model, but aggregate is. Section
    should fire on aggregate >=2 keys."""
    rba = {
        "router": [
            _result("router", "p1", 0.05, {"openai/gpt-5.4": 0.05}),       # hard problem
            _result("router", "p2", 0.01, {"openai/gpt-5.4-mini": 0.01}),  # easy problem
        ],
    }
    text = _render(rba)
    assert "## Cost by Model" in text
    # Both routed-to models appear in the per-agent block
    section = text.split("## Cost by Model", 1)[1]
    assert "openai/gpt-5.4" in section
    assert "openai/gpt-5.4-mini" in section


# ---------------------------------------------------------------------------
# Per-agent block: descending sort + percent share
# ---------------------------------------------------------------------------

def test_per_agent_block_sorted_descending_by_cost():
    """Within a per-agent block, models appear sorted by total cost desc."""
    rba = {
        "multi": [
            _result("multi", "p1", 0.075, {
                "anthropic/claude-sonnet-4-5": 0.030,
                "openai/gpt-5.4": 0.025,
                "openai/gpt-5.4-mini": 0.020,
            }),
        ],
    }
    text = _render(rba)
    section = text.split("## Cost by Model", 1)[1].split("---", 1)[0]
    # First model line under **multi** should be claude-sonnet-4-5
    lines = [l for l in section.splitlines() if l.startswith("- ")]
    assert lines[0].startswith("- anthropic/claude-sonnet-4-5")
    assert lines[1].startswith("- openai/gpt-5.4:")  # exclude -mini via the colon
    assert lines[2].startswith("- openai/gpt-5.4-mini")


def test_aggregation_sums_across_problems():
    """Per-agent rollup correctly sums each model's cost across problems."""
    rba = {
        "multi": [
            _result("multi", "p1", 0.05, {
                "anthropic/claude-sonnet-4-5": 0.030,
                "openai/gpt-5.4-mini": 0.020,
            }),
            _result("multi", "p2", 0.04, {
                "anthropic/claude-sonnet-4-5": 0.020,
                "openai/gpt-5.4-mini": 0.020,
            }),
        ],
    }
    text = _render(rba)
    section = text.split("## Cost by Model", 1)[1].split("---", 1)[0]
    # sonnet total = 0.050, mini total = 0.040; sonnet appears first
    assert "anthropic/claude-sonnet-4-5: $0.050" in section
    assert "openai/gpt-5.4-mini: $0.040" in section
    # Agent header shows the total
    assert "**multi** ($0.090 total)" in section


# ---------------------------------------------------------------------------
# Inline Top-3 annotation: per-task gate + provider-prefix strip + +N more
# ---------------------------------------------------------------------------

def test_inline_annotation_only_when_task_used_multiple_models():
    """A task that only used one model has a bare top-3 line."""
    rba = {
        "iter1": [
            _result("iter1", "p_multi", 0.075, {
                "anthropic/claude-sonnet-4-5": 0.030,
                "openai/gpt-5.4": 0.025,
                "openai/gpt-5.4-mini": 0.020,
            }),
            _result("iter1", "p_solo", 0.020, {
                "openai/gpt-5.4-mini": 0.020,
            }),
        ],
    }
    text = _render(rba)
    top3 = text.split("Top 5 Most Expensive Tasks per Agent", 1)[1]
    # Multi-model task gets the inline breakdown
    assert "p_multi: $0.075 (claude-sonnet-4-5 $0.030" in top3
    # Single-model task stays bare — no parens after the cost
    assert "p_solo: $0.020\n" in top3 or top3.rstrip().endswith("p_solo: $0.020")


def test_inline_annotation_strips_provider_prefix():
    """Inline list uses last `/`-segment for readability."""
    rba = {
        "iter1": [
            _result("iter1", "p1", 0.075, {
                "anthropic/claude-sonnet-4-5": 0.030,
                "openai/gpt-5.4": 0.025,
                "openai/gpt-5.4-mini": 0.020,
            }),
        ],
    }
    text = _render(rba)
    top3 = text.split("Top 5 Most Expensive Tasks per Agent", 1)[1]
    # Provider prefix stripped in inline display
    assert "(claude-sonnet-4-5 $0.030, gpt-5.4 $0.025, gpt-5.4-mini $0.020)" in top3
    # But full prefixed names still in standalone "Cost by Model" section
    section = text.split("## Cost by Model", 1)[1].split("---", 1)[0]
    assert "anthropic/claude-sonnet-4-5" in section
    assert "openai/gpt-5.4" in section


def test_inline_annotation_descending_sort():
    """Inline contributors are listed cheapest-first → most-expensive-first."""
    rba = {
        "iter1": [
            _result("iter1", "p1", 0.06, {
                "openai/gpt-5.4-mini": 0.005,
                "anthropic/claude-sonnet-4-5": 0.045,
                "openai/gpt-5.4": 0.010,
            }),
        ],
    }
    text = _render(rba)
    top3 = text.split("Top 5 Most Expensive Tasks per Agent", 1)[1]
    # Order: sonnet (highest), then gpt-5.4, then mini
    inline = top3.split("(", 1)[1].split(")", 1)[0]
    parts = [p.strip() for p in inline.split(",")]
    assert parts[0].startswith("claude-sonnet-4-5")
    assert parts[1].startswith("gpt-5.4 ")  # space disambiguates from -mini
    assert parts[2].startswith("gpt-5.4-mini")


def test_inline_annotation_shows_all_models_no_truncation():
    """Tasks with many models surface ALL of them inline, in descending
    cost order — no '+N more' truncation. Evolution needs the full
    cost breakdown to make routing decisions."""
    rba = {
        "iter1": [
            _result("iter1", "p1", 0.10, {
                "anthropic/claude-sonnet-4-5": 0.040,
                "openai/gpt-5.4": 0.030,
                "openai/gpt-5.4-mini": 0.020,
                "google/gemini-3-flash-preview": 0.005,
                "anthropic/claude-haiku-4-5": 0.005,
            }),
        ],
    }
    text = _render(rba)
    top = text.split("Top 5 Most Expensive Tasks per Agent", 1)[1]
    # No truncation: all five models render inline
    assert "claude-sonnet-4-5 $0.040" in top
    assert "gpt-5.4 $0.030" in top
    assert "gpt-5.4-mini $0.020" in top
    assert "gemini-3-flash-preview $0.005" in top
    assert "claude-haiku-4-5 $0.005" in top
    # And no overflow marker
    assert "more)" not in top
    assert "+" not in top.split("p1: $0.100", 1)[1].split("\n", 1)[0]
    # Descending order pinned: sonnet first, mini cheaper than gpt-5.4
    inline = top.split("p1: $0.100", 1)[1].split("(", 1)[1].split(")", 1)[0]
    parts = [p.strip() for p in inline.split(",")]
    assert parts[0].startswith("claude-sonnet-4-5")
    assert parts[1].startswith("gpt-5.4 ")
    assert parts[2].startswith("gpt-5.4-mini")


# ---------------------------------------------------------------------------
# Defensive: malformed cost_by_model values
# ---------------------------------------------------------------------------

def test_non_numeric_cost_by_model_value_skipped():
    """A non-coercible value in cost_by_model doesn't crash the report."""
    rba = {
        "iter1": [
            _result("iter1", "p1", 0.05, {
                "openai/gpt-5.4": 0.030,
                "openai/gpt-5.4-mini": "not-a-number",  # pathological input
                "anthropic/claude-sonnet-4-5": 0.020,
            }),
        ],
    }
    # Should render without raising; the bad entry is silently dropped.
    text = _render(rba)
    section = text.split("## Cost by Model", 1)[1].split("---", 1)[0]
    assert "openai/gpt-5.4:" in section
    assert "anthropic/claude-sonnet-4-5" in section
    # "not-a-number" should not appear — it was silently coerced/dropped
    assert "not-a-number" not in text
