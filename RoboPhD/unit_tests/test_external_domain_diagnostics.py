"""Pin ExternalEvaluatorDomain's diagnostics persistence contract.

The domain persists STRING diagnostics as per-problem files and consumes
a fixed set of non-string keys (CONSUMED_NONSTRING_DIAGNOSTICS) into
result.json / cost tracking; any other container-valued diagnostic is
dropped. That drop used to be silent — asta_paper_finder's
judge_format_repairs dict vanished without a trace (2026-07-22) — so
write_diagnostic_files now warns once per process per key. These tests
pin both halves: files written for strings, loud warning (not silence)
for the drop class.
"""
import logging

import pytest

from RoboPhD.domains.external import domain as ext_domain


@pytest.fixture(autouse=True)
def _fresh_warn_dedup():
    saved = set(ext_domain._dropped_diagnostics_warned)
    ext_domain._dropped_diagnostics_warned.clear()
    yield
    ext_domain._dropped_diagnostics_warned.clear()
    ext_domain._dropped_diagnostics_warned.update(saved)


def test_string_diagnostics_become_files(tmp_path):
    ext_domain.write_diagnostic_files(
        {"notes.md": "hello", "agent_stdout": "out", "empty": ""}, tmp_path
    )
    assert (tmp_path / "notes.md").read_text() == "hello"
    assert (tmp_path / "agent_stdout").read_text() == "out"
    assert not (tmp_path / "empty").exists()


def test_string_diagnostics_never_clobber(tmp_path):
    (tmp_path / "notes.md").write_text("original")
    ext_domain.write_diagnostic_files({"notes.md": "new"}, tmp_path)
    assert (tmp_path / "notes.md").read_text() == "original"


def test_unconsumed_container_diagnostic_warns_loudly(tmp_path, caplog):
    """A dict-valued diagnostic outside the consumed set is dropped — the
    demonstrated judge_format_repairs failure class — and must warn."""
    with caplog.at_level(logging.WARNING, logger=ext_domain.__name__):
        ext_domain.write_diagnostic_files(
            {"my_new_counters": {"a": 1}}, tmp_path
        )
    assert any(
        "my_new_counters" in r.message and "NOT be persisted" in r.message
        for r in caplog.records
    )
    assert not (tmp_path / "my_new_counters").exists()


def test_consumed_container_diagnostics_stay_silent(tmp_path, caplog):
    """usage / cost_by_model_usd etc. are consumed upstream — no warning."""
    with caplog.at_level(logging.WARNING, logger=ext_domain.__name__):
        ext_domain.write_diagnostic_files(
            {"usage": {"m": {"input_tokens": 1}},
             "cost_by_model_usd": {"m": 0.1},
             "score": 0.5,
             "eval_wall_clock_seconds": 1.2},
            tmp_path,
        )
    assert not caplog.records


def test_drop_warning_fires_once_per_key(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger=ext_domain.__name__):
        ext_domain.write_diagnostic_files({"k": {"a": 1}}, tmp_path)
        ext_domain.write_diagnostic_files({"k": {"a": 2}}, tmp_path)
    assert sum("NOT be persisted" in r.message for r in caplog.records) == 1
