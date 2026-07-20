"""Unit tests for _judge_normalize — the lenient judge-output extractor
shared by the training-judge path and the calibration study.

Every rescue class here is one astabench's strict parser would DROP
(scoring the doc Not Relevant): a regression in any ladder rung silently
reintroduces judge-side score noise under an alternate training judge.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PFB_DIR = REPO_ROOT / "examples" / "asta_paper_finder"


@pytest.fixture()
def jn():
    if str(PFB_DIR) not in sys.path:
        sys.path.insert(0, str(PFB_DIR))
    import _judge_normalize
    _judge_normalize.reset()
    yield _judge_normalize
    _judge_normalize.reset()


GOOD = (
    '{"criteria": {"C1": {"relevance": "Perfectly Relevant", '
    '"relevant_snippet": "s"}}, "relevance_summary": "ok"}'
)


def test_strict_json_passes_untouched(jn):
    out = jn._lenient_extract_json(GOOD)
    assert out["criteria"]["C1"]["relevance"] == "Perfectly Relevant"
    assert jn.last_repairs() == {
        "strict_ok": 1, "recovered": 0, "shape_fixed": 0, "unrecoverable": 0,
    }


def test_trailing_comma_recovered(jn):
    broken = GOOD.replace('"relevance_summary": "ok"}', '"relevance_summary": "ok",}')
    out = jn._lenient_extract_json(broken)
    assert out["relevance_summary"] == "ok"
    assert jn.last_repairs()["recovered"] == 1


def test_missing_closing_brace_recovered(jn):
    out = jn._lenient_extract_json(GOOD[:-1])  # drop final }
    assert out["criteria"]["C1"]["relevant_snippet"] == "s"
    assert jn.last_repairs()["recovered"] == 1


def test_criteria_list_shape_fixed(jn):
    listy = (
        '{"criteria": [{"name": "C1", "relevance": "Not Relevant", '
        '"relevant_snippet": null}], "relevance_summary": "x"}'
    )
    out = jn._lenient_extract_json(listy)
    assert out["criteria"]["C1"]["relevance"] == "Not Relevant"
    assert jn.last_repairs()["shape_fixed"] == 1


def test_nested_summary_relocated_and_snippet_filled(jn):
    nested = (
        '{"criteria": {"C1": {"relevance": "Somewhat Relevant"}, '
        '"relevance_summary": "misplaced"}}'
    )
    out = jn._lenient_extract_json(nested)
    assert out["relevance_summary"] == "misplaced"
    assert out["criteria"]["C1"]["relevant_snippet"] is None
    assert jn.last_repairs()["shape_fixed"] == 1


def test_garbage_unrecoverable(jn):
    assert jn._lenient_extract_json("no json here at all") is None
    assert jn.last_repairs()["unrecoverable"] == 1


def test_reset_zeroes_and_copies(jn):
    jn._lenient_extract_json(GOOD)
    snap = jn.last_repairs()
    jn.reset()
    assert jn.last_repairs() == {
        "strict_ok": 0, "recovered": 0, "shape_fixed": 0, "unrecoverable": 0,
    }
    assert snap["strict_ok"] == 1  # last_repairs returned a copy, not the live dict


def test_install_patches_the_binding_relevance_calls(jn, monkeypatch):
    """relevance.py binds extract_json_from_response by from-import — the
    attribute on the relevance module is what the judge actually calls, so
    that binding is what install() must replace."""
    from astabench.evals.paper_finder import relevance
    monkeypatch.setattr(
        relevance, "extract_json_from_response", relevance.extract_json_from_response
    )
    jn.install()
    assert relevance.extract_json_from_response is jn._lenient_extract_json


def test_calibration_script_uses_shared_module():
    """The calibration script must import this machinery, not carry its own
    copy — one implementation so measurement and production can't drift."""
    src = (PFB_DIR / "_check_judge_calibration.py").read_text()
    assert "from _judge_normalize import" in src
    assert "def _lenient_extract_json" not in src
    assert "def _normalize_judgement_shape" not in src
