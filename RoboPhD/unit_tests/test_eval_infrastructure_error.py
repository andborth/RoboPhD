"""Pin the infrastructure-abort contract.

An evaluator raises EvalInfrastructureError when a sample could not be
evaluated for a reason outside the candidate and the candidate never ran
(first user: asta_paper_finder, when the Asta MCP tool start-up fails
every retry). Scoring such a sample 0 is an unfair loss that can decide a
round, and the 0 would be cached in result.json and replayed. The contract
pinned here: the run aborts, the affected sample leaves NO result.json (so
--resume re-evaluates it), and samples that finished keep theirs (so
--resume replays them from the eval-result cache).
"""
import tempfile
from pathlib import Path

import pytest

from RoboPhD.domains.base import SampledProblems
from RoboPhD.domains.external.domain import ExternalEvaluatorDomain
from RoboPhD.eval_utils import (
    EvalInfrastructureError,
    is_infrastructure_error,
    is_rate_limit_error,
    run_parallel_eval,
)


def test_is_infrastructure_error_matches_class_subclass_and_cause_chain():
    class Sub(EvalInfrastructureError):
        pass

    assert is_infrastructure_error(EvalInfrastructureError("x"))
    assert is_infrastructure_error(Sub("x"))
    try:
        try:
            raise EvalInfrastructureError("inner")
        except EvalInfrastructureError as e:
            raise RuntimeError("outer") from e
    except RuntimeError as outer:
        assert is_infrastructure_error(outer)
    assert not is_infrastructure_error(RuntimeError("plain"))
    assert not is_infrastructure_error(TimeoutError("slow"))


def test_is_infrastructure_error_matches_a_same_named_class_from_another_import():
    """An example may import eval_utils under a different module name than
    the framework; the match is by class name, like is_rate_limit_error."""
    Other = type("EvalInfrastructureError", (Exception,), {})
    assert is_infrastructure_error(Other("x"))


def test_infrastructure_and_rate_limit_errors_are_distinct():
    assert not is_rate_limit_error(EvalInfrastructureError("x"))


def _domain(evaluator):
    data = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    d = ExternalEvaluatorDomain({
        "evaluator_fn": evaluator, "dataset": data,
        "file_mapping": {"agent.py": "agent.py"},
    })
    tmp = Path(tempfile.mkdtemp())
    (tmp / "agent").mkdir()
    (tmp / "agent" / "agent.py").write_text("x")
    sampled = SampledProblems(contexts=d.get_contexts(),
                              problems_by_context=d.load_problems())
    return d, sampled, tmp


def test_domain_aborts_the_batch_and_writes_no_result_for_the_sample():
    seen = []

    def evaluator(candidate, example, *, problem_dir=None):
        seen.append(example["id"])
        if example["id"] == "b":
            raise EvalInfrastructureError("b: tool init failed")
        return 1.0, {}

    d, sampled, tmp = _domain(evaluator)
    with pytest.raises(EvalInfrastructureError, match="tool init failed"):
        d.run_evaluation(sampled, tmp / "agent", tmp / "out",
                         {"agent_id": "ag", "max_workers": 1, "eval_timeout": 30})
    # No evaluation.json: the round has no score for this agent.
    assert not (tmp / "out" / "evaluation.json").exists()
    # The failed sample has a directory but no cached result.
    result_dirs = [p.parent for p in (tmp / "out").rglob("result.json")]
    all_dirs = [p for p in (tmp / "out" / "problems").iterdir()]
    assert len(all_dirs) > len(result_dirs)


def test_domain_still_scores_ordinary_evaluator_errors_zero():
    """Only the named abort escapes; any other evaluator exception keeps
    the existing score-0 behaviour."""
    def evaluator(candidate, example, *, problem_dir=None):
        if example["id"] == "b":
            raise ValueError("agent bug")
        return 1.0, {}

    d, sampled, tmp = _domain(evaluator)
    r = d.run_evaluation(sampled, tmp / "agent", tmp / "out",
                         {"agent_id": "ag", "max_workers": 1, "eval_timeout": 30})
    assert sorted(x["score"] for x in r.results) == [0.0, 1.0, 1.0]


def test_run_parallel_eval_propagates_the_abort():
    """The shared test-path runner must not turn the abort into a 0 either
    (an evaluator that defers instead of raising never reaches this)."""
    def evaluator(candidate, example):
        if example["id"] == "b":
            raise EvalInfrastructureError("b: tool init failed")
        return 1.0, {}

    with pytest.raises(EvalInfrastructureError):
        run_parallel_eval(evaluator, {"agent.py": "x"},
                          [{"id": "a"}, {"id": "b"}], max_workers=1,
                          eval_timeout=30, quiet=True)


def test_run_parallel_eval_still_scores_ordinary_errors_zero():
    def evaluator(candidate, example):
        if example["id"] == "b":
            raise ValueError("agent bug")
        return 1.0, {}

    out = run_parallel_eval(evaluator, {"agent.py": "x"},
                            [{"id": "a"}, {"id": "b"}], max_workers=1,
                            eval_timeout=30, quiet=True)
    assert out["scores"] == [1.0, 0.0]
