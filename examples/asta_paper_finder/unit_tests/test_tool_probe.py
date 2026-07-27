"""Tests for tool_probe.py — the session-side corpus probe.

Network-free by construction: the astabench/inspect imports live inside
functions, so importing the module and exercising the CLI guards needs
no MCP handshake.
"""
import importlib.util
import re
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
    """tool_probe must defer astabench/inspect_ai imports into functions
    so module import never triggers the MCP handshake. Checked at the
    AST level — a sys.modules delta would pass vacuously whenever an
    earlier test in the process already imported astabench."""
    import ast
    tree = ast.parse((HERE / "tool_probe.py").read_text())
    offenders = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            offenders.append(node.module or "")
    offenders = [m for m in offenders if m.startswith(("astabench", "inspect_ai"))]
    assert not offenders, f"top-level heavyweight imports: {offenders}"


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
    """Anti-staleness guard: the filename shipped via session_tools must
    exist, ship on BOTH shell-bearing engines (RoboPhD + autoresearch),
    and be the one the interpolated session-access note tells sessions
    to run."""
    main_src = (HERE / "main.py").read_text()
    shipped = re.findall(r'session_tools=\[str\(HERE / "([^"]+)"\)\]', main_src)
    assert len(shipped) == 2, "expected RoboPhD + Autoresearch to ship session_tools"
    assert set(shipped) == {"tool_probe.py"}
    assert (HERE / "tool_probe.py").is_file()

    # The note text references the shipped script, and background.md
    # carries the placeholder the note lands in.
    assert "/tool_probe.py" in main_src
    assert "${SESSION_ACCESS_NOTE}" in (HERE / "background.md").read_text()
