"""Tests for the mean-cost-per-example display surfaces: the console
eval-line suffix and the Quick Summary Mean Cost cell. Both use the same
batch basis — cached results included at their replayed costs — matching
mean_score and the cost-penalty aggregator's view of a batch."""

from RoboPhD.report_generator import _mean_cost_cell
from RoboPhD.researcher import format_mean_cost_suffix


# --- Console suffix ----------------------------------------------------------

def test_suffix_formats_batch_mean():
    assert format_mean_cost_suffix(0.476, 14) == " $0.034/ex"


def test_suffix_empty_for_zero_cost_domains():
    assert format_mean_cost_suffix(0.0, 14) == ""


def test_suffix_empty_for_empty_batch():
    assert format_mean_cost_suffix(1.0, 0) == ""


# --- Quick Summary cell ------------------------------------------------------

def test_mean_cost_cell_from_counters():
    perf = {"eval_cost_sum": 0.476, "eval_cost_questions": 14}
    assert _mean_cost_cell(perf) == "$0.034"


def test_mean_cost_cell_dash_when_counters_absent():
    """Records from pre-change checkpoints have no cost counters."""
    assert _mean_cost_cell({"mean_score": 45.8, "test_count": 3}) == "-"


def test_mean_cost_cell_dash_for_zero_cost_domain():
    perf = {"eval_cost_sum": 0.0, "eval_cost_questions": 14}
    assert _mean_cost_cell(perf) == "-"


# --- Basis wiring ------------------------------------------------------------

def test_cached_results_count_in_the_batch_mean():
    """Both surfaces share the batch numerator: eval_cost summed over ALL
    results, cached entries at their replayed costs."""
    results = [
        {"eval_cost": 0.030, "cached": True},
        {"eval_cost": 0.040},
        {"eval_cost": 0.050},
    ]
    batch = sum(r.get("eval_cost", 0.0) for r in results)
    assert format_mean_cost_suffix(batch, 3) == " $0.040/ex"
    assert _mean_cost_cell({"eval_cost_sum": batch, "eval_cost_questions": 3}) == "$0.040"
