"""Pin the classification branches of `_check_infra_failures.py`.

The scanner parses EXTERNAL on-disk artifacts — result.json, the bare
`error` file, evaluation.json, and four historical shapes of
test_results_* — across a strict if/elif chain whose branches are
mutually exclusive by construction. That is exactly the shape that rots
silently when an upstream format shifts: a scanner that stops
recognizing a failure family reports PASS, which reads identically to a
healthy catalogue. The 59-run live scan validates today's data; these
pin the behavior against tomorrow's.

Three branches are subtle enough to deserve naming, and each is pinned
by a test that fails under the obvious wrong implementation:

  * `score_ok` is `> 0`, NOT `== 1` and NOT `!= 0`. Six early runs
    record on a 0-100 scale, and four of those apply a per-problem cost
    penalty that pushes raw-zero problems strictly NEGATIVE (35 such
    rows exist). `== 0` silently reclassifies every one as passing.
  * A row with BOTH an error file AND zero cost — the common wedge
    shape — must count ONCE, under the error-file branch. Reordering
    the chain double-counts it.
  * An `error` file that matches no infra token is `unclassified`, not
    `infra`. Two files in the catalogue are genuine agent bugs
    (NameError, IndexError); binning them as infrastructure would
    inflate wedge counts and blame the harness for the agent.

Also pinned: cached (symlinked) problem dirs are never classified but
their scores DO fold into the round mean — a cell whose passes were all
cache hits otherwise reports raw_mean 0.00 against a recorded 45.0 —
and `_collapse` returns (count, text), not Counter's native
(text, count).

Fixtures are synthetic tmp_path trees, deliberately NOT the real run
catalogue: these must keep failing for the right reason after those
runs are archived or pruned.

The module under test is loaded by file path rather than by bare name
on sys.path (the pattern the sibling test files use for evaluator.py),
because it deliberately imports nothing — no sys.path entry is needed,
and a unique module name keeps it out of the cross-example sys.modules
collision that pytest.ini documents.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "examples" / "asta_ds1000" / "_check_infra_failures.py"


@pytest.fixture(scope="module")
def chk():
    """Load _check_infra_failures.py once per module, by path."""
    spec = importlib.util.spec_from_file_location("_ds1000_infra_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Args:
    """Minimal stand-in for the argparse namespace the printers read."""

    def __init__(self, **kw):
        self.include_archived = kw.get("include_archived", False)
        self.min_cluster = kw.get("min_cluster", 3)
        self.test_wedge_rate = kw.get("test_wedge_rate", 0.02)
        self.max_detail = kw.get("max_detail", 12)


def _problem(cell_dir: Path, qid: str, score, cost=0.01, error=None,
             write_result=True):
    """Create one problem dir with the artifacts the scanner reads."""
    pdir = cell_dir / "problems" / qid
    pdir.mkdir(parents=True, exist_ok=True)
    if write_result:
        (pdir / "result.json").write_text(json.dumps(
            {"question_id": qid, "score": score, "eval_cost": cost,
             "error": False}))
    if error is not None:
        (pdir / "error").write_text(error)
    return pdir


def _cell(tmp_path: Path, agent: str = "agent_x", iteration: str = "iteration_001"):
    cell = tmp_path / iteration / agent
    (cell / "problems").mkdir(parents=True, exist_ok=True)
    return cell


def _classify(chk, cell_dir: Path, run_dir: Path, min_cluster: int = 3):
    return chk.classify_cell(chk.scan_cell(cell_dir / "problems"), run_dir, min_cluster)


TIMEOUT_ERR = "subprocess timed out after 1770s"
AGENT_BUG_ERR = "NameError(\"name 'win_cands' is not defined\")"


# --- score_ok: the two score scales ----------------------------------------


def test_score_ok_binary_scale(chk):
    assert chk.score_ok(1.0) is True
    assert chk.score_ok(0.0) is False


def test_score_ok_hundred_scale(chk):
    """Six runs record 0-100; a cost-penalized pass lands at 99.998."""
    assert chk.score_ok(100.0) is True
    assert chk.score_ok(99.998) is True


def test_negative_cost_penalized_score_is_a_failure(chk):
    """35 rows in the catalogue are negative. `== 0` would pass them."""
    assert chk.score_ok(-5.0) is False
    assert chk.score_ok(-0.001) is False


def test_negative_score_rows_are_classified_as_failures(chk, tmp_path):
    cell = _cell(tmp_path)
    _problem(cell, "1", score=-5.0, cost=0.01)
    _problem(cell, "2", score=100.0, cost=0.01)
    rep = _classify(chk, cell, tmp_path)
    assert rep["n_ok"] == 1
    assert rep["n_genuine"] == 1


# --- the bucket chain ------------------------------------------------------


def test_infra_error_file_is_infra(chk, tmp_path):
    cell = _cell(tmp_path)
    _problem(cell, "1", score=0.0, cost=0.0, error=TIMEOUT_ERR)
    rep = _classify(chk, cell, tmp_path)
    assert rep["n_infra"] == 1
    assert rep["n_unclassified"] == 0


def test_error_file_without_infra_token_is_unclassified_not_infra(chk, tmp_path):
    """Genuine agent bugs write the error file too; they are not the harness."""
    cell = _cell(tmp_path)
    _problem(cell, "1", score=0.0, cost=0.01, error=AGENT_BUG_ERR)
    rep = _classify(chk, cell, tmp_path)
    assert rep["n_unclassified"] == 1
    assert rep["n_infra"] == 0


def test_zero_cost_without_error_file_is_infra(chk, tmp_path):
    """The OpenAI 429 quota family writes NO error file at all."""
    cell = _cell(tmp_path)
    _problem(cell, "1", score=0.0, cost=0.0)
    rep = _classify(chk, cell, tmp_path)
    assert rep["n_infra"] == 1
    assert rep["n_genuine"] == 0


def test_error_file_and_zero_cost_counted_once(chk, tmp_path):
    """The common wedge shape has both signals; buckets must not overlap."""
    cell = _cell(tmp_path)
    _problem(cell, "1", score=0.0, cost=0.0, error=TIMEOUT_ERR)
    rep = _classify(chk, cell, tmp_path)
    assert rep["n_infra"] == 1
    assert rep["n_infra"] + rep["n_genuine"] + rep["n_unclassified"] \
        + rep["n_ok"] + rep["n_incomplete"] == rep["n_fresh"]


def test_agent_bug_with_zero_cost_stays_unclassified(chk, tmp_path):
    """The branch ORDER is what protects this row, not the branch contents.

    An agent that crashed before making a paid call has a genuine-bug error
    file AND zero cost. If the zero-cost branch ran first it would be filed
    as infrastructure — blaming the harness for the agent's NameError.
    """
    cell = _cell(tmp_path)
    _problem(cell, "1", score=0.0, cost=0.0, error=AGENT_BUG_ERR)
    rep = _classify(chk, cell, tmp_path)
    assert rep["n_unclassified"] == 1
    assert rep["n_infra"] == 0


def test_spent_cost_without_error_file_is_genuine(chk, tmp_path):
    cell = _cell(tmp_path)
    _problem(cell, "1", score=0.0, cost=0.02)
    rep = _classify(chk, cell, tmp_path)
    assert rep["n_genuine"] == 1
    assert rep["n_infra"] == 0


def test_passing_row_is_never_infra_even_at_zero_cost(chk, tmp_path):
    cell = _cell(tmp_path)
    _problem(cell, "1", score=1.0, cost=0.0)
    rep = _classify(chk, cell, tmp_path)
    assert rep["n_ok"] == 1
    assert rep["n_infra"] == 0


def test_missing_result_json_is_incomplete_not_genuine(chk, tmp_path):
    cell = _cell(tmp_path)
    _problem(cell, "1", score=None, write_result=False)
    rep = _classify(chk, cell, tmp_path)
    assert rep["n_incomplete"] == 1
    assert rep["n_genuine"] == 0


def test_unparseable_result_json_is_incomplete(chk, tmp_path):
    cell = _cell(tmp_path)
    pdir = _problem(cell, "1", score=0.0, write_result=False)
    (pdir / "result.json").write_text("{not json")
    rep = _classify(chk, cell, tmp_path)
    assert rep["n_incomplete"] == 1


def test_wedge_label_fires_at_min_cluster(chk, tmp_path):
    cell = _cell(tmp_path)
    for qid in ("1", "2", "3"):
        _problem(cell, qid, score=0.0, cost=0.0, error=TIMEOUT_ERR)
    assert _classify(chk, cell, tmp_path, min_cluster=3)["wedge"] is True
    assert _classify(chk, cell, tmp_path, min_cluster=4)["wedge"] is False


# --- cached problem dirs (symlinks) ----------------------------------------


def test_symlinked_problem_dir_is_cached_not_classified(chk, tmp_path):
    cell = _cell(tmp_path)
    real = _problem(cell, "1", score=0.0, cost=0.0, error=TIMEOUT_ERR)
    os.symlink(real, cell / "problems" / "2")
    rep = _classify(chk, cell, tmp_path)
    assert rep["n_cached"] == 1
    assert rep["n_fresh"] == 1
    # The symlink points at an infra failure; counting it would double-report
    # one wedge in every later iteration that cached it.
    assert rep["n_infra"] == 1


def test_cached_scores_fold_into_the_round_mean(chk, tmp_path):
    """A cell whose passes were all cache hits must not read raw_mean 0.00.

    This is the 20260610_203253 archived cell: 10 fresh evals all
    infra-zeroed, 10 cached passes, recorded average_score 45.0.
    """
    cell = _cell(tmp_path)
    for qid in ("1", "2"):
        _problem(cell, qid, score=0.0, cost=0.0, error=TIMEOUT_ERR)
    # Cache sources live in an EARLIER iteration, as they do on disk — put
    # them in this cell and they would count as fresh passes as well.
    source = _cell(tmp_path, iteration="iteration_000")
    for qid in ("3", "4"):
        real = _problem(source, qid, score=1.0, cost=0.01)
        os.symlink(real, cell / "problems" / qid)
    rep = _classify(chk, cell, tmp_path)
    assert rep["raw_mean"] == pytest.approx(0.5)          # 2 pass / 4 scored
    assert rep["adjusted_mean"] == pytest.approx(1.0)      # infra removed
    assert rep["n_survivors"] == 2


def test_dangling_symlink_does_not_raise(chk, tmp_path):
    cell = _cell(tmp_path)
    _problem(cell, "1", score=1.0, cost=0.01)
    os.symlink(tmp_path / "gone", cell / "problems" / "2")
    rep = _classify(chk, cell, tmp_path)
    assert rep["n_cached"] == 1
    assert rep["n_fresh"] == 1


def test_cached_score_read_from_evaluation_json(chk, tmp_path):
    """One evaluation.json read covers the cell; no per-symlink follow."""
    cell = _cell(tmp_path)
    _problem(cell, "1", score=0.0, cost=0.0, error=TIMEOUT_ERR)
    real = _problem(cell, "src", score=1.0, cost=0.01)
    os.symlink(real, cell / "problems" / "2")
    (cell / "evaluation.json").write_text(json.dumps({
        "summary": {"average_score": 66.7},
        "results": {"2": {"score": 1.0, "cached": True}},
    }))
    rep = _classify(chk, cell, tmp_path)
    assert rep["recorded"] == 66.7
    assert rep["n_survivors"] == 2      # the fresh pass + the cached pass


# --- token matching and cause collapsing -----------------------------------


def test_infra_token_matches_substring_not_prefix(chk):
    """Test-path errors are head+tail truncated with the middle elided."""
    truncated = ("inspect.eval crashed: PrerequisiteError: ERROR: Docker "
                 "sandbox ...<TRUNCATED>... docker.sock; is the docker daemon running?")
    assert chk._infra_token(truncated) is not None
    assert chk._infra_token("") is None
    assert chk._infra_token(AGENT_BUG_ERR) is None


def test_collapse_returns_count_first(chk):
    """Counter.items() is (text, count); the printers want (count, text)."""
    out = chk._collapse(["a", "a", "b"])
    assert out[0] == (2, "a")
    assert out[1] == (1, "b")


# --- the walk --------------------------------------------------------------


def test_archived_pruned_by_default_and_included_with_flag(chk, tmp_path):
    live = _cell(tmp_path)
    _problem(live, "1", score=1.0)
    arch = _cell(tmp_path / "archived_20260101_000000")
    _problem(arch, "1", score=0.0, cost=0.0, error=TIMEOUT_ERR)
    assert len(chk.find_problem_cells(tmp_path, include_archived=False)) == 1
    assert len(chk.find_problem_cells(tmp_path, include_archived=True)) == 2


def test_walk_finds_deeply_nested_round_two_cells(chk, tmp_path):
    """Deep-focus round-2 evals nest under evolution_output/.../iteration_N_test."""
    nested = _cell(tmp_path / "evolution_output" / "iteration_007"
                   / "iteration_005_test")
    _problem(nested, "1", score=1.0)
    assert len(chk.find_problem_cells(tmp_path, include_archived=False)) == 1


# --- the test path ---------------------------------------------------------


def _write_test_artifact(run_dir: Path, stem: str, records, summary=None):
    (run_dir / f"{stem}.per_problem.json").write_text(json.dumps(records))
    if summary is not None:
        (run_dir / f"{stem}.json").write_text(json.dumps(summary))


def test_error_and_primary_error_is_a_hard_failure(chk, tmp_path):
    _write_test_artifact(tmp_path, "test_results_final", [
        {"sample_id": None, "score": 0.0, "error": TIMEOUT_ERR,
         "primary_error": TIMEOUT_ERR, "fallback_used": True},
        {"sample_id": "a", "score": 1.0, "error": None,
         "primary_error": None, "fallback_used": False},
    ])
    art = chk.scan_test(tmp_path, 0.02)["artifacts"][0]
    assert art["n_infra_hard"] == 1
    assert art["n_fallback_sub"] == 0
    assert art["n_null_sample"] == 1


def test_primary_error_alone_is_a_fallback_substitution(chk, tmp_path):
    """The reported score is the SEED's, not the candidate's."""
    _write_test_artifact(tmp_path, "test_results_final", [
        {"sample_id": "a", "score": 1.0, "error": None,
         "primary_error": "subprocess failed (exit -9)", "fallback_used": True},
        {"sample_id": "b", "score": 0.0, "error": None,
         "primary_error": None, "fallback_used": False},
    ])
    art = chk.scan_test(tmp_path, 0.02)["artifacts"][0]
    assert art["n_fallback_sub"] == 1
    assert art["n_infra_hard"] == 0
    # The substituted row leaves the adjusted mean: 1 survivor, scoring 0.
    assert art["raw_mean"] == pytest.approx(0.5)
    assert art["adjusted_mean"] == pytest.approx(0.0)


def test_verdict_uses_max_of_infra_and_fallback_rates(chk, tmp_path):
    """An artifact can report a healthy mean while a tenth of it is the seed's.

    This is autoresearch/asta_ds1000_20260614_000814 — 0 infra, 10/90
    fallback, mean 0.8222. An infra-only rule reports PASS.
    """
    records = [{"sample_id": str(i), "score": 1.0, "error": None,
                "primary_error": "subprocess failed (exit -9)",
                "fallback_used": True} for i in range(10)]
    records += [{"sample_id": str(i), "score": 1.0, "error": None,
                 "primary_error": None, "fallback_used": False}
                for i in range(10, 90)]
    _write_test_artifact(tmp_path, "test_results_experiment", records)
    art = chk.scan_test(tmp_path, 0.02)["artifacts"][0]
    assert art["n_infra_hard"] == 0
    assert art["verdict"] == "FAIL"


def test_clean_artifact_passes(chk, tmp_path):
    _write_test_artifact(tmp_path, "test_results_final", [
        {"sample_id": "a", "score": 1.0, "error": None,
         "primary_error": None, "fallback_used": False},
        {"sample_id": "b", "score": 0.0, "error": None,
         "primary_error": None, "fallback_used": False},
    ])
    art = chk.scan_test(tmp_path, 0.02)["artifacts"][0]
    assert art["verdict"] == "PASS"
    assert art["raw_mean"] == pytest.approx(0.5)


def test_legacy_schema_without_fallback_fields_parses(chk, tmp_path):
    """5 files predate fallback_used/primary_error; n_fallback_used is None."""
    _write_test_artifact(tmp_path, "test_results_final", [
        {"sample_id": "a", "score": 0.0, "error": TIMEOUT_ERR},
        {"sample_id": "b", "score": 1.0, "error": None},
    ], summary={"mean_test_score": 0.5, "n_fallback_used": None,
                "test_eval_cost_usd": 1.0})
    art = chk.scan_test(tmp_path, 0.02)["artifacts"][0]
    assert art["legacy"] is True
    assert art["n_infra_hard"] == 1
    assert art["verdict"] == "FAIL"


def test_dict_under_a_per_problem_name_is_a_summary(chk, tmp_path):
    """test_results_timeout_rerun.per_problem.summary.json is a DICT.

    Filename-based pairing routes it to the records path, where iterating
    yields str keys and raises AttributeError. Shape decides, not the name.
    """
    (tmp_path / "test_results_timeout_rerun.per_problem.summary.json").write_text(
        json.dumps({"mean_test_score": 0.92, "n_fallback_used": 0}))
    result = chk.scan_test(tmp_path, 0.02)
    assert len(result["artifacts"]) == 1
    art = result["artifacts"][0]
    assert art["records_seen"] is False
    assert art["verdict"] == "PASS"


def test_summary_pairs_with_its_per_problem_sidecar(chk, tmp_path):
    _write_test_artifact(tmp_path, "test_results_final", [
        {"sample_id": "a", "score": 1.0, "error": None,
         "primary_error": None, "fallback_used": False},
    ], summary={"mean_test_score": 1.0, "test_eval_cost_usd": 2.5,
                "n_fallback_used": 0})
    arts = chk.scan_test(tmp_path, 0.02)["artifacts"]
    assert len(arts) == 1                      # paired, not double-counted
    assert arts[0]["cost"] == pytest.approx(2.5)


def test_run_with_no_test_artifacts_is_not_a_failure(chk, tmp_path):
    result = chk.scan_test(tmp_path, 0.02)
    assert result["artifacts"] == []
    assert result["n_fail"] == 0
