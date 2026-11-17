"""--frozen-harness: config-only evolution over a fixed agent source.

The claims under test are the ones the mode's name makes: the code cannot
change, only module-level model slots / efforts / numeric and boolean
constants can, and the starting point is the frozen agent itself.

The reference harness is the archived winner of the $0.279 control run; it
is checked in, so these tests pin the exact schema a reproduction would get.
Nothing here calls a model.
"""
import ast
import json
import os
import re
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = REPO_ROOT / "examples" / "asta_paper_finder"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

import frozen_harness as fh  # noqa: E402
from model_menu import MENU  # noqa: E402

CONTROL_WINNER = (
    REPO_ROOT / "example_runs" / "robophd" / "asta_paper_finder"
    / "v0_1_1_cap_0_279_fable52" / "agents" / "iter20_year_steered_expansion"
)

requires_keys = pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY")
         or os.environ.get("ANTHROPIC_API_KEY_FOR_ROBOPHD")),
    reason="model_registry needs an Anthropic key to construct its handles",
)

SMALL = textwrap.dedent('''\
    """A small harness."""
    from model_registry import (GPT_5_4, GPT_5_4_MINI,
                                CLAUDE_OPUS_4_8)

    PLAN = CLAUDE_OPUS_4_8        # planner
    PANEL = (GPT_5_4, GPT_5_4_MINI)
    DEPTH = 3
    RATIO = 0.5
    BUDGET_S = 21 * 60
    NEG = -2
    ON = False
    NAME = "frozen string"
    FIELDS = ("a", "b")
    PROMPT = f"depth {DEPTH}"


    async def solve():
        limit = 30
        return await GPT_5_4_MINI.generate(PROMPT)
''')


@pytest.fixture(scope="module")
def control():
    return fh.load_harness(CONTROL_WINNER)


@pytest.fixture()
def small():
    return fh.Harness(SMALL)


def _cfg(h):
    return json.loads(h.seed_config_text())


# ---------------------------------------------------------------------------
# What is evolvable
# ---------------------------------------------------------------------------

def test_small_harness_schema(small):
    cfg = small.config()
    assert cfg["models"] == {
        "PLAN": {"model": "CLAUDE_OPUS_4_8", "effort": None},
        "PANEL": [{"model": "GPT_5_4", "effort": None},
                  {"model": "GPT_5_4_MINI", "effort": None}],
    }
    assert cfg["constants"] == {
        "DEPTH": 3, "RATIO": 0.5, "BUDGET_S": 1260, "NEG": -2, "ON": False,
    }
    # Strings, tuples of non-handles, f-strings and function-body literals
    # are not settings.
    for frozen in ("NAME", "FIELDS", "PROMPT", "limit"):
        assert frozen not in cfg["models"] and frozen not in cfg["constants"]


def test_control_winner_schema(control):
    cfg = control.config()
    assert list(cfg["models"]) == [
        "PLAN_MODEL", "ALT_MODELS", "HEAD_MODEL", "PICK_MODEL", "MID_MODEL",
        "GRADE_MODEL", "REGRADE_MODEL",
    ]
    assert len(cfg["models"]["ALT_MODELS"]) == 2
    assert len(cfg["constants"]) == 22
    assert cfg["constants"]["HEAD_CHECK"] is False
    assert cfg["constants"]["QUERY_BUDGET_S"] == 1260
    assert cfg["constants"]["HARD_BUDGET_S"] == 1530.0
    assert isinstance(cfg["constants"]["HARD_BUDGET_S"], float)
    # No prompt or field list is a setting.
    assert not [k for k in cfg["constants"] if k.endswith(("_PROMPT", "_FIELDS"))]


def test_no_string_value_is_ever_a_setting(control):
    tree = ast.parse(control.source)
    strings = {
        s.targets[0].id for s in tree.body
        if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name)
        and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)
    }
    assert strings  # the harness does have prompts
    cfg = control.config()
    assert not strings & (set(cfg["models"]) | set(cfg["constants"]))


# ---------------------------------------------------------------------------
# The seed is the frozen agent
# ---------------------------------------------------------------------------

def test_seed_renders_byte_identical(control):
    seed = control.parse_config(control.seed_config_text())
    assert control.render(seed) == control.source
    assert control.render(seed) == (CONTROL_WINNER / "agent.py").read_text()


def test_seed_renders_byte_identical_small(small):
    assert small.render(small.parse_config(small.seed_config_text())) == SMALL


# ---------------------------------------------------------------------------
# Rendering changes settings and nothing else
# ---------------------------------------------------------------------------

def _strip_settings(source):
    """The source with every evolvable right-hand side and the registry
    import blanked: what must be identical across all renders."""
    h = fh.Harness(source)
    tree = ast.parse(source)
    drop = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "model_registry":
            drop.add(stmt)
        if (isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id in h._kinds):
            drop.add(stmt)
    kept = [s for s in tree.body if s not in drop]
    return ast.dump(ast.Module(body=kept, type_ignores=[]))


def test_render_changes_only_settings(control):
    cfg = _cfg(control)
    for name, value in cfg["models"].items():
        entry = {"model": "GEMINI_3_1_FLASH_LITE", "effort": "high"}
        cfg["models"][name] = [entry] if isinstance(value, list) else entry
    cfg["constants"]["MAX_GRADE"] = 64
    cfg["constants"]["HEAD_CHECK"] = True
    cfg["constants"]["HARD_BUDGET_S"] = 1400
    cfg["constants"]["COUNT_MARGIN"] = 0.65
    parsed = control.parse_config(json.dumps(cfg))
    out = control.render(parsed)

    assert out != control.source
    assert _strip_settings(out) == _strip_settings(control.source)
    assert fh.Harness(out).config() == parsed
    tree = ast.parse(out)
    imp = next(s for s in tree.body
               if isinstance(s, ast.ImportFrom) and s.module == "model_registry")
    assert {a.name for a in imp.names} == {"GEMINI_3_1_FLASH_LITE", "with_effort"}
    assert 'GRADE_MODEL = with_effort(GEMINI_3_1_FLASH_LITE, "high")' in out
    assert "MAX_GRADE = 64 " in out
    assert "HARD_BUDGET_S = 1400.0" in out


@pytest.mark.parametrize("n", [0, 1, 3])
def test_model_list_may_change_length(small, n):
    cfg = _cfg(small)
    cfg["models"]["PANEL"] = [{"model": "GPT_5_5", "effort": None}] * n
    parsed = small.parse_config(json.dumps(cfg))
    out = small.render(parsed)
    ns = {}
    line = next(ln for ln in out.split("\n") if ln.startswith("PANEL = "))
    exec(line, {"GPT_5_5": "H"}, ns)
    assert ns["PANEL"] == ("H",) * n
    assert small._read_back(out)["PANEL"] == parsed["models"]["PANEL"]


def test_round_trip_check_is_type_strict(small, monkeypatch):
    """If rendering ever wrote a value of the wrong type, the read-back
    check must notice: 1260.0 is not the 1260 that was asked for."""
    real = small._splice
    monkeypatch.setattr(
        small, "_splice",
        lambda edits: real(edits).replace("DEPTH = 5", "DEPTH = 5.0"),
    )
    cfg = _cfg(small)
    cfg["constants"]["DEPTH"] = 5
    with pytest.raises(fh.HarnessError, match="read back"):
        small.render(small.parse_config(json.dumps(cfg)))


def test_handle_used_by_frozen_code_stays_imported(small):
    cfg = _cfg(small)
    cfg["models"]["PLAN"] = {"model": "GPT_5_5", "effort": None}
    cfg["models"]["PANEL"] = [{"model": "GPT_5_5", "effort": None}]
    out = small.render(small.parse_config(json.dumps(cfg)))
    imp = next(s for s in ast.parse(out).body if isinstance(s, ast.ImportFrom))
    # solve() calls GPT_5_4_MINI directly; that is frozen code.
    assert {a.name for a in imp.names} == {"GPT_5_4_MINI", "GPT_5_5"}


def test_render_handles_non_ascii_before_a_value():
    src = SMALL.replace("DEPTH = 3", "DEPTH = 3  # — depth —").replace(
        '"""A small harness."""', '"""A småll härness — ünïcode."""'
    )
    h = fh.Harness(src)
    cfg = _cfg(h)
    cfg["constants"]["DEPTH"] = 9
    out = h.render(h.parse_config(json.dumps(cfg)))
    assert "DEPTH = 9  # — depth —" in out
    assert out.replace("DEPTH = 9", "DEPTH = 3") == src


# ---------------------------------------------------------------------------
# Validation: a candidate is data, with the harness's own shape
# ---------------------------------------------------------------------------

def _mutations():
    def m(fn):
        return fn

    def drop_key(c): del c["constants"]["DEPTH"]
    def add_key(c): c["constants"]["EXTRA"] = 1
    def add_model(c): c["models"]["NEW_SLOT"] = {"model": "GPT_5_4", "effort": None}
    def add_section(c): c["code"] = "import os"
    def bool_for_int(c): c["constants"]["DEPTH"] = True
    def float_for_int(c): c["constants"]["DEPTH"] = 3.5
    def int_for_bool(c): c["constants"]["ON"] = 1
    def str_for_number(c): c["constants"]["RATIO"] = "0.5"
    def unknown_model(c): c["models"]["PLAN"]["model"] = "GPT_6"
    def code_as_model(c): c["models"]["PLAN"]["model"] = "__import__('os')"
    def model_not_string(c): c["models"]["PLAN"]["model"] = ["GPT_5_4"]
    def bad_effort(c): c["models"]["PLAN"]["effort"] = "max"
    def slot_extra_field(c): c["models"]["PLAN"]["prompt"] = "x"
    def slot_missing_effort(c): del c["models"]["PLAN"]["effort"]
    def list_for_slot(c): c["models"]["PLAN"] = [c["models"]["PLAN"]]
    def slot_for_list(c): c["models"]["PANEL"] = c["models"]["PANEL"][0]
    def gemini_medium(c):
        c["models"]["PLAN"] = {"model": "GEMINI_3_5_FLASH", "effort": "medium"}
    return [v for k, v in locals().items() if callable(v) and k != "m"]


@pytest.mark.parametrize("mutate", _mutations(), ids=lambda f: f.__name__)
def test_invalid_configs_are_rejected(small, mutate):
    cfg = _cfg(small)
    mutate(cfg)
    with pytest.raises(fh.ConfigError):
        small.parse_config(json.dumps(cfg))


@pytest.mark.parametrize("text", [
    "", "not json", "[]", '{"models": {}}',
    '{"models": {}, "constants": {}, "constants": {}}',
])
def test_malformed_documents_are_rejected(small, text):
    with pytest.raises(fh.ConfigError):
        small.parse_config(text)


def test_non_finite_numbers_are_rejected(small):
    text = small.seed_config_text().replace('"RATIO": 0.5', '"RATIO": NaN')
    assert "NaN" in text
    with pytest.raises(fh.ConfigError):
        small.parse_config(text)


def test_integer_accepted_for_a_float_constant(small):
    cfg = _cfg(small)
    cfg["constants"]["RATIO"] = 1
    parsed = small.parse_config(json.dumps(cfg))
    assert parsed["constants"]["RATIO"] == 1.0
    assert isinstance(parsed["constants"]["RATIO"], float)
    assert "RATIO = 1.0" in small.render(parsed)


def test_all_problems_are_reported_together(small):
    cfg = _cfg(small)
    cfg["constants"]["DEPTH"] = "x"
    cfg["models"]["PLAN"]["model"] = "NOPE"
    with pytest.raises(fh.ConfigError) as e:
        small.parse_config(json.dumps(cfg))
    assert "DEPTH" in str(e.value) and "PLAN" in str(e.value)


def test_menu_narrows_the_allowed_models(small):
    seed = small.seed_config_text()
    small.parse_config(seed, MENU)
    with pytest.raises(fh.ConfigError, match="not on the menu"):
        small.parse_config(seed, ("GPT_5_4",))


@pytest.mark.parametrize("effort", ["low", "medium", "high", None])
def test_efforts_accepted_per_family(small, effort):
    cfg = _cfg(small)
    cfg["models"]["PLAN"] = {"model": "CLAUDE_SONNET_4_6", "effort": effort}
    assert small.parse_config(json.dumps(cfg))["models"]["PLAN"]["effort"] == effort


def test_efforts_match_background_table():
    """EFFORTS restates background.md's 'available overrides' column."""
    rows = re.findall(
        r"^\| `([A-Z0-9_]+)` \|[^|]*\|[^|]*\|[^|]*\|([^|]*)\|$",
        (TASK_DIR / "background.md").read_text(), re.M,
    )
    table = {h: tuple(re.findall(r'"(\w+)"', opts)) for h, opts in rows if h in MENU}
    assert set(table) == set(MENU)
    for handle, options in table.items():
        assert fh.EFFORTS[fh.family(handle)] == options, handle


# ---------------------------------------------------------------------------
# Sources that cannot be frozen
# ---------------------------------------------------------------------------

def test_twice_bound_name_is_frozen():
    src = SMALL + "\nDEPTH = 4\n"
    assert "DEPTH" not in fh.Harness(src).config()["constants"]


@pytest.mark.parametrize("src, match", [
    ("X = 1\n", "exactly one"),
    ("from model_registry import GPT_5_4 as G\nM = G\n", "alias"),
    ("from model_registry import GPT_5_4\nN = 3\n", "no module-level model slot"),
    ("from model_registry import GPT_5_4\nM = GPT_5_4\n"
     "def with_effort(a, b):\n    return a\n", "with_effort"),
    ("def broken(:\n", "does not parse"),
])
def test_unusable_sources_are_refused(src, match):
    with pytest.raises(fh.HarnessError, match=match):
        fh.Harness(src)


# ---------------------------------------------------------------------------
# Evaluator wrapper
# ---------------------------------------------------------------------------

class _Inner:
    total_eval_cost = 1.25

    def __init__(self, tag="train"):
        self.tag = tag
        self.calls = []

    def __call__(self, candidate, example, *, problem_dir=None):
        self.calls.append((candidate, example, problem_dir))
        return 0.5, {"ok": True}

    def aggregate(self, results):
        return 42.0, "agg"

    def with_overrides(self, **kw):
        return _Inner(tag=kw)


def test_evaluator_renders_then_delegates(small):
    inner = _Inner()
    ev = fh.FrozenHarnessEvaluator(inner, small)
    cfg = _cfg(small)
    cfg["constants"]["DEPTH"] = 7
    score, diag = ev({fh.CONFIG_FILENAME: json.dumps(cfg)}, {"q": 1}, problem_dir="d")
    assert (score, diag) == (0.5, {"ok": True})
    (candidate, example, problem_dir), = inner.calls
    assert list(candidate) == [fh.AGENT_FILENAME]
    assert "DEPTH = 7" in candidate[fh.AGENT_FILENAME]
    assert (example, problem_dir) == ({"q": 1}, "d")


def test_evaluator_scores_invalid_config_zero_without_running(small):
    inner = _Inner()
    ev = fh.FrozenHarnessEvaluator(inner, small)
    for candidate in ({}, {fh.CONFIG_FILENAME: "{"},
                      {fh.AGENT_FILENAME: "print('mine')"}):
        score, diag = ev(candidate, {})
        assert score == 0.0 and "error.md" in diag
    assert inner.calls == []


def test_evaluator_explains_a_render_fault(small, monkeypatch):
    inner = _Inner()
    ev = fh.FrozenHarnessEvaluator(inner, small)

    def broken(config):
        raise fh.HarnessError("does not read back")
    monkeypatch.setattr(small, "render", broken)
    score, diag = ev({fh.CONFIG_FILENAME: small.seed_config_text()}, {})
    assert score == 0.0
    assert "renderer" in diag["error.md"] and "does not read back" in diag["error.md"]
    assert inner.calls == []


def test_candidate_supplied_agent_source_is_never_run(small):
    inner = _Inner()
    ev = fh.FrozenHarnessEvaluator(inner, small)
    ev({fh.CONFIG_FILENAME: small.seed_config_text(),
        fh.AGENT_FILENAME: "raise SystemExit('smuggled')"}, {})
    assert inner.calls[0][0] == {fh.AGENT_FILENAME: SMALL}


def test_evaluator_exposes_the_wrapped_evaluator(small):
    ev = fh.FrozenHarnessEvaluator(_Inner(), small, ("GPT_5_4",))
    assert ev.total_eval_cost == 1.25
    assert ev.aggregate([]) == (42.0, "agg")
    sibling = ev.with_overrides(apply_cost_penalty=False)
    assert isinstance(sibling, fh.FrozenHarnessEvaluator)
    assert sibling.tag == {"apply_cost_penalty": False}
    # The menu travels with it.
    score, _ = sibling({fh.CONFIG_FILENAME: small.seed_config_text()}, {})
    assert score == 0.0


# ---------------------------------------------------------------------------
# Run-level resolution
# ---------------------------------------------------------------------------

def _agent_dir(tmp_path, name, source=SMALL):
    d = tmp_path / name
    d.mkdir()
    (d / fh.AGENT_FILENAME).write_text(source)
    return d


def test_fresh_run_resolution(tmp_path):
    assert fh.resolve_run_harness(None, {}) == (None, None)
    d = _agent_dir(tmp_path, "a")
    harness, stored = fh.resolve_run_harness(str(d), {})
    assert harness.source == SMALL
    assert stored == {"path": str(d), "sha256": harness.sha256}


def test_resume_uses_the_runs_own_copy(tmp_path):
    d = _agent_dir(tmp_path, "a")
    _, stored = fh.resolve_run_harness(str(d), {})
    run = tmp_path / "run"
    (run / "session_tools").mkdir(parents=True)
    (run / "session_tools" / fh.HARNESS_SESSION_FILENAME).write_text(SMALL)
    (d / fh.AGENT_FILENAME).write_text(SMALL + "\n# edited after the run began\n")
    harness, again = fh.resolve_run_harness(
        None, {fh.RUNTIME_KEY: stored}, resume_dir=run
    )
    assert harness.source == SMALL and again == stored


def test_resume_falls_back_to_the_stored_path(tmp_path):
    d = _agent_dir(tmp_path, "a")
    _, stored = fh.resolve_run_harness(str(d), {})
    harness, _ = fh.resolve_run_harness(
        None, {fh.RUNTIME_KEY: stored}, resume_dir=tmp_path / "run"
    )
    assert harness.source == SMALL


def test_resume_rejects_a_different_harness(tmp_path):
    d = _agent_dir(tmp_path, "a")
    other = _agent_dir(tmp_path, "b", SMALL.replace("DEPTH = 3", "DEPTH = 4"))
    _, stored = fh.resolve_run_harness(str(d), {})
    with pytest.raises(SystemExit, match="cannot be changed"):
        fh.resolve_run_harness(str(other), {fh.RUNTIME_KEY: stored},
                               resume_dir=tmp_path / "run")


def test_resume_refuses_when_the_harness_is_gone_or_altered(tmp_path):
    d = _agent_dir(tmp_path, "a")
    _, stored = fh.resolve_run_harness(str(d), {})
    (d / fh.AGENT_FILENAME).write_text(SMALL + "\n# changed\n")
    with pytest.raises(SystemExit, match="Cannot resume"):
        fh.resolve_run_harness(None, {fh.RUNTIME_KEY: stored},
                               resume_dir=tmp_path / "run")


def test_ordinary_run_cannot_be_frozen_on_resume(tmp_path):
    d = _agent_dir(tmp_path, "a")
    with pytest.raises(SystemExit, match="cannot be added"):
        fh.resolve_run_harness(str(d), {"model_menu": []}, resume_dir=tmp_path)
    assert fh.resolve_run_harness(None, {}, resume_dir=tmp_path) == (None, None)


def test_session_files(tmp_path, small):
    paths = [Path(p) for p in fh.materialize_session_files(small, tmp_path)]
    names = [p.name for p in paths]
    assert names == [fh.HARNESS_SESSION_FILENAME, "frozen_harness.py", "model_menu.py"]
    assert all(p.is_file() for p in paths)
    assert paths[0].read_text() == SMALL
    assert paths[0].parent.name == small.sha256[:16]
    # Idempotent: a resume re-materializes the same file.
    assert fh.materialize_session_files(small, tmp_path) == [str(p) for p in paths]


def test_cli_runs_from_a_bare_session_tools_directory(tmp_path, small):
    """Sessions run the tool from session_tools/, with only the shipped
    files beside it and no evaluator dependencies."""
    import shutil
    import subprocess
    for p in fh.materialize_session_files(small, tmp_path / "cache"):
        shutil.copy2(p, tmp_path / Path(p).name)
    cfg = tmp_path / fh.CONFIG_FILENAME
    cfg.write_text(small.seed_config_text())

    def run(*args):
        return subprocess.run(
            [sys.executable, str(tmp_path / "frozen_harness.py"), *args],
            capture_output=True, text=True, cwd="/",
            env={"PATH": os.environ.get("PATH", "")},
        )

    ok = run("validate", str(cfg))
    assert ok.returncode == 0 and ok.stdout.strip() == "OK", ok.stderr
    assert run("render", str(cfg)).stdout == SMALL
    assert run("seed").stdout == small.seed_config_text()
    narrowed = run("--menu", "GPT_5_4", "validate", str(cfg))
    assert narrowed.returncode == 1 and "not on the menu" in narrowed.stderr
    cfg.write_text("{")
    bad = run("validate", str(cfg))
    assert bad.returncode == 1 and "INVALID" in bad.stderr


# ---------------------------------------------------------------------------
# Evolution-facing documents and main.py wiring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("doc", [
    "objective_frozen_harness.md", "background_frozen_harness.md",
])
def test_every_placeholder_in_the_frozen_docs_is_interpolated(doc):
    main_src = (TASK_DIR / "main.py").read_text()
    placeholders = set(re.findall(r"\$\{[A-Z_]+\}", (TASK_DIR / doc).read_text()))
    assert placeholders
    missing = [p for p in placeholders if f'"{p}"' not in main_src]
    assert not missing, f"main.py never replaces {missing}"


def test_frozen_docs_name_no_file_literally():
    """File names reach the documents through placeholders only, so a rename
    cannot leave the instructions pointing at a file that is not there."""
    for doc in ("objective_frozen_harness.md", "background_frozen_harness.md"):
        text = (TASK_DIR / doc).read_text()
        for name in (fh.CONFIG_FILENAME, fh.HARNESS_SESSION_FILENAME,
                     "frozen_harness.py"):
            assert name not in text, f"{doc} hardcodes {name}"


def test_objective_has_a_replaceable_opening_paragraph():
    """main.py swaps objective.md's first paragraph for the frozen one and
    keeps the rest; that needs a first paragraph that only says what is
    being evolved."""
    first, sep, rest = (TASK_DIR / "objective.md").read_text().strip().partition("\n\n")
    assert sep and rest.strip()
    assert first.startswith("Evolve a PaperFindingBench agent")
    assert "${COST_THRESHOLD}" not in first and "${COST_THRESHOLD}" in rest


# ---------------------------------------------------------------------------
# model_registry.with_effort, and a rendered agent against the real registry
# ---------------------------------------------------------------------------

@requires_keys
def test_with_effort_siblings():
    import model_registry as mr
    for name in ("CLAUDE_OPUS_4_8", "GPT_5_4_MINI", "GEMINI_3_1_FLASH_LITE"):
        base = getattr(mr, name)
        sibling = mr.with_effort(base, "high")
        assert sibling is not base
        assert sibling.config.reasoning_effort == "high"
        assert str(sibling) == str(base)  # same model id => same billing bucket
        assert mr.with_effort(base, "high") is sibling
        assert mr.with_effort(base, None) is base
    with pytest.raises(ValueError):
        mr.with_effort(object(), "low")


def test_pinned_effort_replaces_a_per_call_effort():
    import asyncio
    from inspect_ai.model import GenerateConfig
    import model_registry as mr
    seen = []

    async def generate(input, tools, tool_choice, config, **kwargs):
        seen.append((config.reasoning_effort, config.max_tokens, kwargs))

    pinned = mr._pin_effort(generate, "high")
    asyncio.run(pinned("q"))
    asyncio.run(pinned("q", config=GenerateConfig(reasoning_effort="low",
                                                  max_tokens=64), cache=True))
    assert seen == [("high", None, {}), ("high", 64, {"cache": True})]


@requires_keys
def test_rendered_control_winner_imports_against_the_registry(control):
    import model_registry as mr
    cfg = _cfg(control)
    cfg["models"]["GRADE_MODEL"] = {"model": "GEMINI_3_1_FLASH_LITE", "effort": "high"}
    cfg["models"]["ALT_MODELS"] = []
    cfg["constants"]["MAX_GRADE"] = 100
    out = control.render(control.parse_config(json.dumps(cfg)))
    ns = {"__name__": "rendered_agent"}
    exec(compile(out, "rendered_agent.py", "exec"), ns)
    assert ns["GRADE_MODEL"] is mr.with_effort(mr.GEMINI_3_1_FLASH_LITE, "high")
    assert ns["GRADE_MODEL"].config.reasoning_effort == "high"
    assert ns["PLAN_MODEL"] is mr.CLAUDE_OPUS_4_8
    assert ns["ALT_MODELS"] == ()
    assert ns["MAX_GRADE"] == 100
    assert callable(ns["make_solver"])
