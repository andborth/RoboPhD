"""Pin behavior of the iteration-level aggregator hook.

The hook is in two pieces:

1. ``_default_aggregate`` in ``domain.py`` — the framework-level
   fallback that runs when an evaluator doesn't expose ``aggregate``.
   Default behavior must be identical to the prior ``score_sum/total``
   mean so every existing task is numerically unchanged.

2. ``Ds1000Evaluator.aggregate`` — task-specific override that applies
   a cost penalty to the iteration mean. The penalty branch only runs
   on training (``apply_cost_penalty=True``); test mode returns the
   raw mean unchanged for leaderboard parity. Per-example score is
   raw correctness (0.0/1.0) in both modes.

Tests use a SimpleNamespace stub for ``Ds1000Evaluator.aggregate`` so
the heavy ``__init__`` (env-var checks, Docker pre-flight, registry
resolution) is skipped — only the attributes the aggregator actually
reads need to be present.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ASTA_DS1000_DIR = REPO_ROOT / "examples" / "asta_ds1000"


# ---------------------------------------------------------------------------
# _default_aggregate
# ---------------------------------------------------------------------------


def _import_default_aggregate():
    from RoboPhD.domains.external.domain import _default_aggregate
    return _default_aggregate


def test_default_aggregate_empty_list():
    """No per-example results → 0.0 score, empty explanation."""
    agg = _import_default_aggregate()
    score, explanation = agg([])
    assert score == 0.0
    assert explanation == ""


def test_default_aggregate_single_example():
    agg = _import_default_aggregate()
    score, explanation = agg([{"score": 0.75}])
    assert score == 0.75
    assert explanation == ""


def test_default_aggregate_multiple_examples():
    """Simple mean — preserves legacy score_sum/total formula."""
    agg = _import_default_aggregate()
    score, explanation = agg([
        {"score": 1.0},
        {"score": 0.0},
        {"score": 1.0},
        {"score": 1.0},
    ])
    assert score == pytest.approx(0.75)
    assert explanation == ""


def test_default_aggregate_missing_score_key():
    """Resilient to entries missing 'score' — treats as 0."""
    agg = _import_default_aggregate()
    score, explanation = agg([{"score": 1.0}, {"question_id": "x"}])
    assert score == pytest.approx(0.5)
    assert explanation == ""


# ---------------------------------------------------------------------------
# Ds1000Evaluator.aggregate — test path (apply_cost_penalty=False)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds1000_aggregate():
    """Return the unbound aggregate method + SCORE_SCALE.

    Module-scoped because evaluator.py imports astabench / inspect_evals
    (heavy). Each test builds a SimpleNamespace stub with just the
    attrs the aggregator reads — sidesteps the env-var / Docker checks
    in Ds1000Evaluator.__init__.
    """
    sys.path.insert(0, str(ASTA_DS1000_DIR))
    try:
        from evaluator import Ds1000Evaluator, SCORE_SCALE
        return Ds1000Evaluator.aggregate, SCORE_SCALE
    finally:
        sys.path.remove(str(ASTA_DS1000_DIR))


def _stub(apply_cost_penalty, threshold=0.04, saturation=10.0):
    return SimpleNamespace(
        apply_cost_penalty=apply_cost_penalty,
        min_cost_threshold=threshold,
        cost_penalty_saturation=saturation,
    )


def test_ds1000_aggregate_test_path_returns_fraction(ds1000_aggregate):
    """apply_cost_penalty=False → fraction scale, no explanation.

    This is the leaderboard-format mode used by --eval-test-set and
    --eval-only. Per-example score is raw 0/1; the aggregator returns
    the mean of those as a [0, 1] fraction. Pinning this prevents a
    silent scale shift in the test-side outputs.
    """
    aggregate, _ = ds1000_aggregate
    stub = _stub(apply_cost_penalty=False)
    score, explanation = aggregate(stub, [
        {"score": 1.0, "eval_cost": 0.50},  # high cost ignored in test mode
        {"score": 1.0, "eval_cost": 0.50},
        {"score": 0.0, "eval_cost": 0.50},
        {"score": 1.0, "eval_cost": 0.50},
    ])
    assert score == pytest.approx(0.75)
    assert explanation == ""


def test_ds1000_aggregate_test_path_empty(ds1000_aggregate):
    aggregate, _ = ds1000_aggregate
    score, explanation = aggregate(_stub(apply_cost_penalty=False), [])
    assert score == 0.0
    assert explanation == ""


# ---------------------------------------------------------------------------
# Ds1000Evaluator.aggregate — training, free zone
# ---------------------------------------------------------------------------


def test_ds1000_aggregate_training_free_zone_below_threshold(ds1000_aggregate):
    """Mean cost ≤ threshold → SCORE_SCALE × mean_raw, no penalty.

    Critical for the selective-routing case: an agent that runs
    expensive models on some problems but averages cheaply across the
    batch should pay zero penalty.
    """
    aggregate, SCORE_SCALE = ds1000_aggregate
    stub = _stub(apply_cost_penalty=True)  # threshold=0.04, saturation=10
    # 18 cheap examples + 2 expensive — mean is $0.013, deep in free zone
    results = (
        [{"score": 1.0, "eval_cost": 0.005}] * 17  # 17 correct cheap
        + [{"score": 0.0, "eval_cost": 0.005}] * 1  # 1 wrong cheap
        + [{"score": 1.0, "eval_cost": 0.05}] * 2   # 2 correct expensive
    )
    score, explanation = aggregate(stub, results)
    # 19/20 correct; SCORE_SCALE * 0.95 = 95.0
    assert score == pytest.approx(95.0)
    # Explanation surfaces the scale even in free zone (non-empty)
    assert "free zone" in explanation
    assert "no tiebreaker penalty applied" in explanation
    assert "percentage" in explanation
    assert "$0.0" in explanation  # mean cost reported


def test_ds1000_aggregate_training_free_zone_at_threshold(ds1000_aggregate):
    """Mean cost exactly equal to threshold → still free zone (≤ boundary)."""
    aggregate, SCORE_SCALE = ds1000_aggregate
    stub = _stub(apply_cost_penalty=True, threshold=0.04, saturation=10.0)
    results = [{"score": 1.0, "eval_cost": 0.04}] * 10  # mean = $0.04
    score, explanation = aggregate(stub, results)
    assert score == pytest.approx(100.0)
    assert "free zone" in explanation


# ---------------------------------------------------------------------------
# Ds1000Evaluator.aggregate — training, breach branch
# ---------------------------------------------------------------------------


def test_ds1000_aggregate_training_above_threshold(ds1000_aggregate):
    """Mean cost > threshold → SCORE_SCALE × mean_raw - linear penalty.

    Pinning the penalty arithmetic: (mean_cost - threshold) /
    (saturation - threshold). With mean_cost=$0.20, threshold=$0.04,
    saturation=$10.00: penalty = 0.16 / 9.96 ≈ 0.01606.
    """
    aggregate, SCORE_SCALE = ds1000_aggregate
    stub = _stub(apply_cost_penalty=True, threshold=0.04, saturation=10.0)
    results = [{"score": 1.0, "eval_cost": 0.20}] * 10  # all correct, mean=$0.20
    score, explanation = aggregate(stub, results)
    expected_penalty = (0.20 - 0.04) / (10.0 - 0.04)
    expected_score = 100.0 - expected_penalty
    assert score == pytest.approx(expected_score)
    # Explanation surfaces the calculation
    assert "exceeded threshold" in explanation
    assert "tie-breaking penalty" in explanation
    assert "$0.2000" in explanation
    assert "(percentage)" in explanation


def test_ds1000_aggregate_training_at_saturation(ds1000_aggregate):
    """Mean cost ≥ saturation → penalty clamps at 1.0."""
    aggregate, SCORE_SCALE = ds1000_aggregate
    stub = _stub(apply_cost_penalty=True, threshold=0.04, saturation=10.0)
    results = [{"score": 1.0, "eval_cost": 15.0}] * 5  # mean $15 >> saturation
    score, explanation = aggregate(stub, results)
    # 100.0 - 1.0 = 99.0 (penalty capped at 1.0)
    assert score == pytest.approx(99.0)
    assert "exceeded threshold" in explanation


def test_ds1000_aggregate_training_uses_agent_cost_usd_fallback(ds1000_aggregate):
    """Cost coalesces between 'eval_cost' (domain-normalized) and
    'agent_cost_usd' (test-path raw diagnostic). The training path
    normally sees eval_cost, but the fallback keeps the aggregator
    invariant to which caller built the input dict.
    """
    aggregate, SCORE_SCALE = ds1000_aggregate
    stub = _stub(apply_cost_penalty=True, threshold=0.04, saturation=10.0)
    # No eval_cost field — only agent_cost_usd
    results = [{"score": 1.0, "agent_cost_usd": 0.20}] * 10
    score, _ = aggregate(stub, results)
    expected_penalty = (0.20 - 0.04) / (10.0 - 0.04)
    assert score == pytest.approx(100.0 - expected_penalty)


# ---------------------------------------------------------------------------
# Scale asymmetry: training vs test
# ---------------------------------------------------------------------------


def test_ds1000_aggregate_scale_asymmetry(ds1000_aggregate):
    """Same per-example inputs → ~85 in training, ~0.85 in test.

    This is the intentional asymmetry that lets training use a
    percentage-scale objective with cost as a [0,1] tiebreaker, while
    test reports a leaderboard-format fraction.
    """
    aggregate, SCORE_SCALE = ds1000_aggregate
    results = [{"score": 1.0, "eval_cost": 0.005}] * 17 + \
              [{"score": 0.0, "eval_cost": 0.005}] * 3   # 17/20, free zone

    train_score, _ = aggregate(_stub(apply_cost_penalty=True), results)
    test_score, _ = aggregate(_stub(apply_cost_penalty=False), results)

    assert train_score == pytest.approx(85.0)
    assert test_score == pytest.approx(0.85)
    # Ratio is SCORE_SCALE — pin the relationship explicitly
    assert train_score == pytest.approx(test_score * SCORE_SCALE)


# ---------------------------------------------------------------------------
# Dual-column report rendering — pins the conditional layout switch
# ---------------------------------------------------------------------------


def test_score_summary_default_single_column():
    """No explanations → legacy single Mean Score column."""
    from RoboPhD.report_generator import _format_score_summary
    lines = _format_score_summary({"a1": [1.0, 0.0, 1.0]}, ["a1"])
    joined = "\n".join(lines)
    assert "| Agent | Mean Score | Problems |" in joined
    assert "Mean Raw Score" not in joined
    assert "Aggregate notes" not in joined


def test_score_summary_dual_column_when_explanation_present():
    """Any non-empty explanation → Mean Raw + Mean Score columns + notes."""
    from RoboPhD.report_generator import _format_score_summary
    lines = _format_score_summary(
        {"a1": [1.0, 0.0, 1.0], "a2": [1.0, 1.0, 0.0]},
        ["a1", "a2"],
        agent_explanations={"a1": "penalty applied", "a2": ""},
        agent_aggregate_scores={"a1": 65.5, "a2": 100.0},
    )
    joined = "\n".join(lines)
    assert "| Agent | Mean Raw Score | Mean Score | Problems |" in joined
    assert "65.500" in joined  # aggregator output shows up
    assert "**Aggregate notes**" in joined
    assert "**a1**: penalty applied" in joined
    # a2's explanation is empty — should NOT appear in notes block
    assert "**a2**:" not in joined


def test_continuous_table_header_flips_with_explanations():
    """`## Score Comparison` → `## Raw Score Comparison` when any
    agent has a non-empty explanation, to signal the per-problem
    cells are raw (pre-aggregator)."""
    from RoboPhD.report_generator import format_continuous_score_table
    scores = {"q1": {"a1": 0.5, "a2": 1.0}, "q2": {"a1": 1.0, "a2": 0.0}}

    default = "\n".join(format_continuous_score_table(scores, ["a1", "a2"]))
    assert "## Score Comparison" in default
    assert "## Raw Score Comparison" not in default

    with_exp = "\n".join(format_continuous_score_table(
        scores, ["a1", "a2"],
        agent_explanations={"a1": "x", "a2": ""},
        agent_aggregate_scores={"a1": 80.0, "a2": 50.0},
    ))
    assert "## Raw Score Comparison" in with_exp
    assert "## Score Comparison\n" not in with_exp  # exact section not present


def test_binary_report_dual_column_when_explanation_present():
    """format_binary_report_comparative gets the same conditional
    layout (DS-1000 lands here after the per-example penalty move)."""
    from RoboPhD.report_generator import format_binary_report_comparative
    index = {
        "summary": {
            "agents": ["a1", "a2"],
            "agent_accuracies": {"a1": 85.0, "a2": 85.0},
            "agent_aggregate_scores": {"a1": 85.0, "a2": 84.978},
            "agent_explanations": {
                "a1": "free zone",
                "a2": "mean cost exceeded threshold — applied 0.022 penalty",
            },
            "total_questions": 20,
            "consensus_stats": {
                "all_correct": 15, "all_correct_pct": 75,
                "all_failed": 2, "all_failed_pct": 10,
                "split_decisions": 3, "split_decisions_pct": 15,
            },
        },
        "by_agent": {
            "a1": {"total_correct": 17, "total_failed": 3, "total_errors": 0, "accuracy": 85.0},
            "a2": {"total_correct": 17, "total_failed": 3, "total_errors": 0, "accuracy": 85.0},
        },
    }
    joined = "\n".join(format_binary_report_comparative(index))
    assert "Mean Raw Score" in joined
    assert "| Mean Score |" in joined
    assert "85.000" in joined and "84.978" in joined
    assert "**Aggregate notes**" in joined
    assert "**a1**: free zone" in joined
    assert "applied 0.022 penalty" in joined


def test_binary_report_legacy_layout_when_no_explanations():
    """Default tasks (no aggregator) get the original 5-column layout."""
    from RoboPhD.report_generator import format_binary_report_comparative
    index = {
        "summary": {
            "agents": ["a1"],
            "agent_accuracies": {"a1": 85.0},
            "total_questions": 20,
            "consensus_stats": {
                "all_correct": 17, "all_correct_pct": 85,
                "all_failed": 2, "all_failed_pct": 10,
                "split_decisions": 1, "split_decisions_pct": 5,
            },
        },
        "by_agent": {
            "a1": {"total_correct": 17, "total_failed": 3, "total_errors": 0, "accuracy": 85.0},
        },
    }
    joined = "\n".join(format_binary_report_comparative(index))
    assert "| Agent | Correct | Failed | Errors | Accuracy |" in joined
    assert "Mean Raw Score" not in joined
    assert "Aggregate notes" not in joined


# ---------------------------------------------------------------------------
# Index-builder symmetry — pins comparative ⇆ deep-focus parallel
#
# The two index-builder subprocesses (create_comparative_error_index.py and
# create_deep_focus_error_index.py) both have to read summary.aggregate_explanation
# from each agent's evaluation.json and emit agent_explanations +
# agent_aggregate_scores in their respective error_index.json. The original
# commit missed the deep-focus parallel — these tests pin the symmetry so a
# future change to one builder can't silently regress the other.
# ---------------------------------------------------------------------------


def _binary_eval_data(correct: int, total: int, aggregate_score=None, explanation=""):
    """Build an evaluation.json-shaped dict for index-builder tests."""
    results = {}
    for i in range(correct):
        results[f"q{i:03d}"] = {"score": 1, "error": False}
    for i in range(correct, total):
        results[f"q{i:03d}"] = {"score": 0, "error": False}
    summary = {
        "total_problems": total,
        "score_sum": float(correct),
        "average_score": aggregate_score if aggregate_score is not None else correct / total,
    }
    if explanation:
        summary["aggregate_explanation"] = explanation
    else:
        summary["aggregate_explanation"] = ""
    return {"summary": summary, "results": results}


def _write_iteration(tmp_path, eval_data_by_agent):
    """Lay out iteration_001/agent_*/evaluation.json files."""
    import json
    iteration_dir = tmp_path / "iteration_001"
    for agent_name, eval_data in eval_data_by_agent.items():
        agent_dir = iteration_dir / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        with open(agent_dir / "evaluation.json", "w") as f:
            json.dump(eval_data, f)
    return iteration_dir


def test_comparative_index_propagates_aggregate_explanation(tmp_path):
    """When evaluation.json has summary.aggregate_explanation populated,
    the comparative index must emit it under summary.agent_explanations
    (keyed by stripped agent name) and propagate summary.average_score
    under summary.agent_aggregate_scores."""
    from RoboPhD.tools.error_analysis.create_comparative_error_index import (
        build_error_index,
    )
    iteration_dir = _write_iteration(tmp_path, {
        "agent_a1": _binary_eval_data(17, 20, aggregate_score=85.0, explanation="Mean cost $0.025 within free zone"),
        "agent_a2": _binary_eval_data(16, 20, aggregate_score=80.0, explanation="Mean cost $0.10 exceeded threshold"),
    })
    index = build_error_index(iteration_dir)
    explanations = index["summary"].get("agent_explanations", {})
    aggregates = index["summary"].get("agent_aggregate_scores", {})
    assert explanations.get("a1") == "Mean cost $0.025 within free zone"
    assert explanations.get("a2") == "Mean cost $0.10 exceeded threshold"
    assert aggregates.get("a1") == pytest.approx(85.0)
    assert aggregates.get("a2") == pytest.approx(80.0)


def test_comparative_index_empty_explanations_when_default_aggregator(tmp_path):
    """Tasks without a custom aggregator emit empty explanation strings;
    the aggregate score equals the default mean (correct/total)."""
    from RoboPhD.tools.error_analysis.create_comparative_error_index import (
        build_error_index,
    )
    iteration_dir = _write_iteration(tmp_path, {
        "agent_a1": _binary_eval_data(17, 20),  # no aggregate_score override → defaults to mean; no explanation
    })
    index = build_error_index(iteration_dir)
    assert index["summary"].get("agent_explanations", {}).get("a1") == ""
    assert index["summary"].get("agent_aggregate_scores", {}).get("a1") == pytest.approx(0.85)


def _deep_focus_results(eval_data_by_agent):
    """Build the `results` shape that _build_binary_index expects —
    matches what load_evaluation_results would have produced, including
    the agent_summaries block."""
    by_question = {}
    by_agent = {}
    agent_summaries = {}
    for agent_name, eval_data in eval_data_by_agent.items():
        results_for_agent = eval_data["results"]
        by_agent[agent_name] = results_for_agent
        for qid, r in results_for_agent.items():
            by_question.setdefault(qid, {})[agent_name] = r
        summary = eval_data.get("summary") or {}
        agent_summaries[agent_name] = {
            "aggregate_explanation": summary.get("aggregate_explanation", ""),
            "average_score": summary.get("average_score", 0.0),
            "score_sum": summary.get("score_sum", 0.0),
            "total_problems": summary.get("total_problems", 0),
        }
    return {
        "by_question": by_question,
        "by_agent": by_agent,
        "agent_summaries": agent_summaries,
    }


def test_deep_focus_index_propagates_aggregate_explanation():
    """Deep-focus index builder must propagate the same fields as the
    comparative one — without this, evolution_output/iteration_*/
    iteration_*_test/error_analysis_report.md would render the legacy
    layout even when DS-1000 has explanations to surface.

    Calls `_build_binary_index` directly (skipping the
    agents-dir / newest-agent detection that `build_error_index` does)
    so the test focuses on the summary-propagation logic that the
    parallel builders must keep in sync.
    """
    from RoboPhD.tools.error_analysis.create_deep_focus_error_index import (
        _build_binary_index,
    )
    eval_data_by_agent = {
        "agent_new": _binary_eval_data(17, 20, aggregate_score=85.0, explanation="new agent free zone"),
        "agent_baseline": _binary_eval_data(16, 20, aggregate_score=80.0, explanation="baseline free zone"),
    }
    results = _deep_focus_results(eval_data_by_agent)
    scores_by_question = {
        qid: {
            agent.replace("agent_", "", 1): r.get("score", 0)
            for agent, r in agents.items()
        }
        for qid, agents in results["by_question"].items()
    }
    index = _build_binary_index(
        "agent_new", ["agent_baseline"], results, scores_by_question
    )
    explanations = index["summary"].get("agent_explanations", {})
    aggregates = index["summary"].get("agent_aggregate_scores", {})
    assert explanations.get("new") == "new agent free zone"
    assert explanations.get("baseline") == "baseline free zone"
    assert aggregates.get("new") == pytest.approx(85.0)
    assert aggregates.get("baseline") == pytest.approx(80.0)


def test_deep_focus_index_empty_when_no_aggregator():
    """Default-aggregator case: explanations empty, aggregate scores
    fall back to correct/total via _raw_mean_fallback. Pins the
    fallback's behavior so a future change can't silently swap it
    back to the `accuracy / 100` percentage round-trip."""
    from RoboPhD.tools.error_analysis.create_deep_focus_error_index import (
        _build_binary_index,
    )
    eval_data_by_agent = {
        "agent_new": _binary_eval_data(17, 20),
        "agent_baseline": _binary_eval_data(16, 20),
    }
    results = _deep_focus_results(eval_data_by_agent)
    # Wipe the summary blocks → simulate legacy evaluation.json where
    # summary.aggregate_explanation / average_score weren't written.
    for v in results["agent_summaries"].values():
        v["aggregate_explanation"] = ""
        v.pop("average_score")
    scores_by_question = {
        qid: {
            agent.replace("agent_", "", 1): r.get("score", 0)
            for agent, r in agents.items()
        }
        for qid, agents in results["by_question"].items()
    }
    index = _build_binary_index(
        "agent_new", ["agent_baseline"], results, scores_by_question
    )
    explanations = index["summary"].get("agent_explanations", {})
    aggregates = index["summary"].get("agent_aggregate_scores", {})
    assert explanations.get("new") == ""
    assert explanations.get("baseline") == ""
    # Fallback should be correct/total — NOT accuracy/100
    assert aggregates.get("new") == pytest.approx(17 / 20)
    assert aggregates.get("baseline") == pytest.approx(16 / 20)
