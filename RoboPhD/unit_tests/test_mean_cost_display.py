"""Tests for the mean-cost-per-example display surfaces: the console
eval-line suffix (whole-batch basis, cached included) and the Quick
Summary cells (fresh-only / unique-example basis)."""

from RoboPhD.report_generator import _mean_cost_cell, _mean_score_cell
from RoboPhD.researcher import format_mean_cost_suffix


# --- Console suffix: batch basis --------------------------------------------

def test_suffix_formats_batch_mean():
    assert format_mean_cost_suffix(0.476, 14) == " $0.034/ex"


def test_suffix_empty_for_zero_cost_domains():
    assert format_mean_cost_suffix(0.0, 14) == ""


def test_suffix_empty_for_empty_batch():
    assert format_mean_cost_suffix(1.0, 0) == ""


# --- Quick Summary cells: fresh-only basis -----------------------------------

def test_mean_cost_cell_from_fresh_counters():
    perf = {"fresh_eval_cost_sum": 0.476, "fresh_questions": 14}
    assert _mean_cost_cell(perf) == "$0.034"


def test_mean_cost_cell_dash_when_counters_absent():
    """Records from pre-change checkpoints have no fresh counters."""
    assert _mean_cost_cell({"mean_score": 45.8, "test_count": 3}) == "-"


def test_mean_cost_cell_dash_for_zero_cost_domain():
    perf = {"fresh_eval_cost_sum": 0.0, "fresh_questions": 14}
    assert _mean_cost_cell(perf) == "-"


def test_mean_score_cell_prefers_fresh_mean():
    perf = {
        "fresh_aggregate_weighted": 45.0 * 10,
        "fresh_questions": 10,
        "mean_score": 99.9,  # batch-basis legacy value must NOT win
    }
    assert _mean_score_cell(perf) == "45.000"


def test_mean_score_cell_falls_back_to_legacy_mean():
    assert _mean_score_cell({"mean_score": 45.802}) == "45.802"


def test_mean_score_cell_dash_when_nothing_known():
    assert _mean_score_cell({}) == "-"


# --- Basis wiring: cached results count in batch cost, not fresh sums --------

def test_batch_cost_includes_cached_fresh_sums_exclude():
    """Mirror of the researcher/domain wiring: the console numerator sums
    eval_cost over ALL results (cached replay their original cost), while
    metadata['eval_cost'] — the Quick Summary numerator — is fresh-only."""
    results = [
        {"eval_cost": 0.030, "cached": True},
        {"eval_cost": 0.040},
        {"eval_cost": 0.050},
    ]
    batch = sum(r.get("eval_cost", 0.0) for r in results)
    fresh = sum(r.get("eval_cost", 0.0) for r in results if not r.get("cached"))
    assert format_mean_cost_suffix(batch, 3) == " $0.040/ex"
    assert _mean_cost_cell({"fresh_eval_cost_sum": fresh, "fresh_questions": 2}) == "$0.045"
