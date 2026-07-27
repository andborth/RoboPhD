"""Tests for tool_probe.py — the session-side corpus probe.

Network-free by construction: the astabench/inspect imports live inside
functions, so importing the module and exercising the CLI guards needs
no MCP handshake.
"""
import importlib.util
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "tool_probe_under_test", HERE / "tool_probe.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_import_is_network_free():
    """Module import must not touch astabench/inspect_ai (no handshake)."""
    before = set(sys.modules)
    _load_probe()
    loaded = set(sys.modules) - before
    assert not any(m.startswith(("astabench", "inspect_ai")) for m in loaded)


def test_parse_kv_types():
    probe = _load_probe()
    kwargs = probe.parse_kv(
        ["limit=5", "fields=title,year", 'ids=["CorpusId:1", "CorpusId:2"]',
         "keyword=sparse attention"]
    )
    assert kwargs == {
        "limit": 5,
        "fields": "title,year",
        "ids": ["CorpusId:1", "CorpusId:2"],
        "keyword": "sparse attention",
    }


def test_parse_kv_rejects_bare_token():
    probe = _load_probe()
    with pytest.raises(SystemExit):
        probe.parse_kv(["limit"])


def test_main_help_and_key_guard(monkeypatch, capsys):
    probe = _load_probe()
    assert probe.main([]) == 0
    assert "Usage" in capsys.readouterr().out

    monkeypatch.delenv("ASTA_TOOL_KEY", raising=False)
    assert probe.main(["--list"]) == 2
    assert "ASTA_TOOL_KEY" in capsys.readouterr().err


def test_shipped_filename_consistent_across_docs():
    """Anti-staleness guard: the filename main.py ships must exist and be
    the same one background.md tells sessions to run."""
    main_src = (HERE / "main.py").read_text()
    match = re.search(r'session_tools=\[str\(HERE / "([^"]+)"\)\]', main_src)
    assert match, "main.py no longer ships session_tools"
    shipped = match.group(1)

    assert (HERE / shipped).is_file()
    background = (HERE / "background.md").read_text()
    assert f"session_tools/{shipped}" in background
