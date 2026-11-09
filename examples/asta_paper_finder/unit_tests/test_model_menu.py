"""--model-menu: a run's solver menu can only narrow the nine registry
handles, and the restriction is enforced rather than merely documented.

Covers the dependency-free helpers in model_menu.py (parsing, resume
resolution, the background.md and seed rewrites), the registry's pruning and
the evaluator's off-menu scoring rule. Registry and evaluator construction
run in subprocesses: the registry reads the menu once at import, so each
menu needs a fresh interpreter -- exactly as each production worker gets.
Fake provider keys suffice; nothing here calls a model.

This file is kept identical in asta_ds1000 and asta_paper_finder apart from
the TASK block below.
"""
import ast
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# --- TASK block -------------------------------------------------------------
TASK = "asta_paper_finder"
EVALUATOR_CLASS = "PaperFinderEvaluator"
EVALUATOR_KWARGS = "subprocess_isolation=False"
FAKE_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "ASTA_TOOL_KEY")
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = REPO_ROOT / "examples" / TASK
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

import model_menu as mm  # noqa: E402

BACKGROUND = (TASK_DIR / "background.md").read_text()
SEED_SRC = (TASK_DIR / "seeds" / "baseline" / "agent.py").read_text()
REGISTRY_TREE = ast.parse((TASK_DIR / "model_registry.py").read_text())
MAIN_TREE = ast.parse((TASK_DIR / "main.py").read_text())
EVALUATOR_TREE = ast.parse((TASK_DIR / "evaluator.py").read_text())
WORKER_TREE = ast.parse((TASK_DIR / "_eval_worker.py").read_text())


def _registry_assign(name: str) -> ast.expr:
    return next(
        n.value for n in REGISTRY_TREE.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", "") == name for t in n.targets)
    )


def _python_examples(text: str) -> list[str]:
    return [m.group(2) for m in mm._FENCE_RE.finditer(text)
            if mm._IMPORT_LINE_RE.search(m.group(2))]


def _model_section(text: str) -> tuple[str, str, str]:
    start = text.index("## LLM calls")
    end = text.index("\n## ", start + 1)
    return text[:start], text[start:end], text[end:]


def _run(script: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    # A minimal environment, not a copy of this one: other tests in the suite
    # set process env vars (judge selection and the like) that would leak into
    # the child and change what the evaluator does.
    env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LANG")
           if k in os.environ}
    env.update({k: "sk-test" for k in FAKE_KEYS})
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=TASK_DIR, env=env, capture_output=True, text=True, timeout=300,
    )


# --- the two copies of the handle list agree --------------------------------

def test_registry_menu_copy_matches_model_menu():
    """model_registry keeps its own copy (it is shipped alone in submission
    directories); a drift would let the two disagree on what a menu means."""
    assert ast.literal_eval(_registry_assign("_MENU")) == mm.MENU
    assert ast.literal_eval(_registry_assign("_MODEL_MENU_ENV")) == mm.MODEL_MENU_ENV


def test_menu_matches_background_table():
    rows = [m.group(1) for ln in BACKGROUND.splitlines()
            if (m := mm._TABLE_ROW_RE.match(ln))]
    assert tuple(rows) == mm.MENU


def test_registry_maps_every_menu_handle_to_an_id():
    keys = [k.value for k in _registry_assign("_HANDLE_IDS").keys]
    assert set(mm.MENU) <= set(keys)


# --- parsing ------------------------------------------------------------------

def test_parse_single_and_list():
    assert mm.parse_model_menu("GPT_5_4") == ("GPT_5_4",)
    assert mm.parse_model_menu(" GPT_5_4 , GPT_5_4_MINI ,GPT_5_4") == (
        "GPT_5_4", "GPT_5_4_MINI"
    )
    assert mm.parse_model_menu(None) is None


@pytest.mark.parametrize("spec", ["", ",", " , "])
def test_parse_rejects_empty(spec):
    with pytest.raises(ValueError, match="empty"):
        mm.parse_model_menu(spec)


@pytest.mark.parametrize("name", ["GPT_6", "gpt_5_4", "CLAUDE_OPUS_4_7",
                                  "GEMINI_3_FLASH_PREVIEW", "_GPT_5_4_ID"])
def test_parse_rejects_anything_off_the_nine(name):
    """Narrow only: archive-only handles and invented names both fail."""
    with pytest.raises(ValueError, match="Valid handles"):
        mm.parse_model_menu(f"GPT_5_4,{name}")


# --- resume resolution --------------------------------------------------------

def _resolve(spec, store=None, *, resume=False, eval_only=False):
    return mm.resolve_run_menu(spec, store or {}, resume=resume, eval_only=eval_only)


def test_fresh_run():
    assert _resolve(None) is None
    assert _resolve("GPT_5_4") == ("GPT_5_4",)
    assert _resolve(",".join(mm.MENU)) is None  # all nine IS unrestricted


def test_fresh_run_bad_name_exits():
    with pytest.raises(SystemExit, match="--model-menu"):
        _resolve("GPT_6")


def test_stored_value_round_trips():
    for menu in (None, ("GPT_5_4",), ("GPT_5_4_MINI", "CLAUDE_SONNET_4_6")):
        store = {"model_menu": mm.stored_menu_value(menu)}
        assert _resolve(None, store, resume=True) == menu
    assert mm.stored_menu_value(None) == list(mm.MENU)


def test_resume_locks_the_menu():
    store = {"model_menu": ["GPT_5_4"]}
    assert _resolve("GPT_5_4", store, resume=True) == ("GPT_5_4",)
    with pytest.raises(SystemExit, match="cannot be changed"):
        _resolve("GPT_5_4_MINI", store, resume=True)
    unrestricted = {"model_menu": list(mm.MENU)}
    with pytest.raises(SystemExit, match="cannot be changed"):
        _resolve("GPT_5_4", unrestricted, resume=True)
    with pytest.raises(SystemExit, match="cannot be changed"):
        _resolve("GPT_5_4_MINI", store, resume=True, eval_only=True)


def test_resume_of_a_run_that_predates_the_flag():
    assert _resolve(None, {}, resume=True) is None
    with pytest.raises(SystemExit, match="predates"):
        _resolve("GPT_5_4", {}, resume=True)
    # An eval-only resume of such a run (or of a GEPA/Autoresearch run,
    # which persists no task config) takes the flag.
    assert _resolve("GPT_5_4", {}, resume=True, eval_only=True) == ("GPT_5_4",)


# --- background.md rewrite ----------------------------------------------------

def test_full_menu_leaves_background_untouched():
    assert mm.restrict_background(BACKGROUND, mm.MENU) == BACKGROUND


@pytest.mark.parametrize("handle", mm.MENU)
def test_single_handle_background(handle):
    out = mm.restrict_background(BACKGROUND, (handle,))
    before, section, after = _model_section(out)
    b_before, _, b_after = _model_section(BACKGROUND)
    assert (before, after) == (b_before, b_after), "only the model section may change"

    rows = [m.group(1) for ln in section.splitlines()
            if (m := mm._TABLE_ROW_RE.match(ln))]
    assert rows == [handle]

    examples = _python_examples(section)
    assert examples, "the model section must keep its code example"
    for body in examples:
        assert set(mm._HANDLE_RE.findall(body)) == {handle}
        assert f"from model_registry import {handle}\n" in body
        assert ".generate(" in body
        compile(body, "<example>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def test_menu_covering_an_example_leaves_it_alone():
    for body in _python_examples(BACKGROUND):
        used = tuple(dict.fromkeys(mm._HANDLE_RE.findall(body)))
        out = mm.restrict_background(BACKGROUND, used)
        assert body in out


def test_family_specific_stanzas_are_not_cross_pointed():
    """A stanza whose comment documents one provider family's behaviour is
    dropped, not re-pointed at another family's handle."""
    for handle in ("GPT_5_4", "GPT_5_4_MINI"):
        for body in _python_examples(mm.restrict_background(BACKGROUND, (handle,))):
            for stanza in body.split("\n\n"):
                if mm._PROVIDER_WORDS_RE.search(stanza):
                    assert "OpenAI" in stanza, stanza


# --- seed rewrite and seed checks ---------------------------------------------

@pytest.mark.parametrize("handle", mm.MENU)
def test_seed_is_repointed_onto_any_single_handle_menu(handle):
    out = mm.restrict_seed_source(SEED_SRC, (handle,))
    assert mm.off_menu_handles(out, (handle,)) == []
    assert mm.registry_names_used(out) == {handle}
    ast.parse(out)


def test_seed_on_the_menu_is_unchanged():
    used = tuple(mm.registry_names_used(SEED_SRC))
    assert mm.restrict_seed_source(SEED_SRC, used) == SEED_SRC


def test_registry_names_used_sees_every_route():
    src = textwrap.dedent("""
        import model_registry as mr
        import model_registry
        from model_registry import GPT_5_4, _allowed_model_ids
        a = mr.CLAUDE_OPUS_4_8
        b = model_registry.GEMINI_3_5_FLASH
        c = mr.ACTIVE_MENU
    """)
    assert mm.registry_names_used(src) == {
        "GPT_5_4", "CLAUDE_OPUS_4_8", "GEMINI_3_5_FLASH", "ACTIVE_MENU",
    }
    assert mm.off_menu_handles(src, ("GPT_5_4", "ACTIVE_MENU")) == [
        "CLAUDE_OPUS_4_8", "GEMINI_3_5_FLASH",
    ]


def test_check_seed_agents(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    (good / "agent.py").write_text("from model_registry import GPT_5_4\n")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "agent.py").write_text("from model_registry import GPT_5_4, GPT_5_5\n")
    mm.check_seed_agents({"seed_good": good}, ("GPT_5_4",))
    mm.check_seed_agents({"seed_mem": {"agent.py": "x = 1\n"}}, ("GPT_5_4",))
    with pytest.raises(SystemExit, match=r"seed_bad/agent\.py: GPT_5_5"):
        mm.check_seed_agents({"seed_good": good, "seed_bad": bad}, ("GPT_5_4",))


# --- the evaluator's off-menu rule ----------------------------------------------

@pytest.fixture(scope="module")
def ev_mod():
    import evaluator
    return evaluator


def test_off_menu_rule_flags_unlisted_models(ev_mod):
    diag: dict = {}
    ev_mod._apply_off_menu_rule(
        frozenset({"openai/gpt-5.4-2026-03-05"}),
        {"openai/gpt-5.4-2026-03-05": 0.01, "openai/gpt-5.5": 0.02,
         "anthropic/x": 0.0},
        diag,
    )
    assert diag["off_menu_models"] == ["anthropic/x", "openai/gpt-5.5"]
    assert "openai/gpt-5.5" in diag["off_menu.md"]
    # Never an error: in DS-1000 that key would trigger the test-time seed
    # fallback and replace the violating answer.
    assert not any("error" in k for k in diag)


def test_off_menu_rule_is_silent_when_clean(ev_mod):
    diag: dict = {}
    ev_mod._apply_off_menu_rule(frozenset({"m"}), {"m": 0.5}, diag)
    ev_mod._apply_off_menu_rule(frozenset({"m"}), {}, diag)
    assert diag == {}


def test_extraction_applies_the_rule_before_scoring():
    """The rule must run on the billed models and zero the score."""
    fn = next(
        n for n in ast.walk(EVALUATOR_TREE)
        if isinstance(n, ast.FunctionDef) and n.name == "_extract_score_and_diagnostics"
    )
    src = ast.unparse(fn)
    rule = src.index("_apply_off_menu_rule(self._allowed_model_ids, cost_by_model_usd")
    assert src.index("cost_by_model_usd[model_name] = model_cost") < rule
    assert rule < src.index("diagnostics['score'] = score_value")
    assert "score_value = 0.0" in src[rule:src.index("diagnostics['score'] = score_value")]


# --- plumbing: main.py -> evaluator -> worker -------------------------------------

def _dict_keys_in(tree: ast.AST, func_name: str) -> set[str]:
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == func_name)
    return {k.value for n in ast.walk(fn) if isinstance(n, ast.Dict)
            for k in n.keys if isinstance(k, ast.Constant)}


def test_menu_rides_with_overrides_and_into_the_worker():
    assert "model_menu" in _dict_keys_in(EVALUATOR_TREE, "with_overrides")
    assert "model_menu" in _dict_keys_in(EVALUATOR_TREE, "_evaluate_via_subprocess")
    call = next(n for n in ast.walk(WORKER_TREE)
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == EVALUATOR_CLASS)
    kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert kw.get("model_menu") == "params['model_menu']"


def test_main_wires_the_menu():
    evaluator_calls = [n for n in ast.walk(MAIN_TREE)
                       if isinstance(n, ast.Call) and getattr(n.func, "id", "") == EVALUATOR_CLASS]
    assert evaluator_calls and all(
        any(k.arg == "model_menu" and getattr(k.value, "id", "") == "model_menu"
            for k in c.keywords)
        for c in evaluator_calls
    )
    runtime = next(
        n.value for n in ast.walk(MAIN_TREE)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", "") == "resolved_runtime" for t in n.targets)
    )
    assert "model_menu" in [k.value for k in runtime.keys]
    # The background rewrite only ever runs under a menu.
    for n in ast.walk(MAIN_TREE):
        if isinstance(n, ast.If) and any(
            isinstance(c, ast.Call) and getattr(c.func, "id", "") == "restrict_background"
            for c in ast.walk(ast.Module(body=n.body, type_ignores=[]))
        ):
            assert ast.unparse(n.test) == "model_menu"
            break
    else:
        pytest.fail("restrict_background must be guarded by `if model_menu:`")


# --- registry pruning and evaluator construction (subprocesses) -------------------

def test_registry_prunes_to_the_menu():
    r = _run("""
        import json, model_registry as m
        print(json.dumps({
            "active": m.ACTIVE_MENU,
            "exported": sorted(n for n in m._HANDLE_IDS if hasattr(m, n)),
            "allowed": sorted(m._allowed_model_ids(m.ACTIVE_MENU)),
        }))
        try:
            from model_registry import GPT_5_4_MINI
            print("IMPORTED")
        except ImportError:
            print("BLOCKED")
    """, {mm.MODEL_MENU_ENV: "GPT_5_4"})
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    info = json.loads(lines[-2])
    assert info["active"] == ["GPT_5_4"]
    assert info["exported"] == ["GPT_5_4"]
    assert info["allowed"] == ["openai/gpt-5.4-2026-03-05"]
    assert lines[-1] == "BLOCKED"


def test_registry_rejects_an_unknown_menu_at_import():
    r = _run("import model_registry", {mm.MODEL_MENU_ENV: "GPT_5_4,GPT_6"})
    assert r.returncode != 0
    assert "GPT_6" in r.stderr and "not on the model menu" in r.stderr


def test_unrestricted_registry_lists_every_handle():
    """Every Model the registry exports must be in _HANDLE_IDS, or a menu
    could not prune it and the off-menu check would not know its id."""
    r = _run("""
        import json, model_registry as m
        from inspect_ai.model import Model
        models = sorted(n for n, v in vars(m).items() if isinstance(v, Model))
        print(json.dumps({
            "active": m.ACTIVE_MENU,
            "models": models,
            "listed": sorted(m._HANDLE_IDS),
            "allowed": sorted(m._allowed_model_ids()),
            "ids": sorted(set(m._HANDLE_IDS.values())),
        }))
    """)
    assert r.returncode == 0, r.stderr
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["active"] is None
    assert info["models"] == info["listed"]
    assert info["allowed"] == info["ids"]


def test_evaluator_owns_the_menu():
    """The evaluator publishes its menu before importing the registry, picks
    an on-menu default model, and refuses a second menu in one process."""
    r = _run(f"""
        import json, os
        from evaluator import {EVALUATOR_CLASS} as E
        ev = E({EVALUATOR_KWARGS}, model_menu=("GPT_5_4",))
        import model_registry as m
        out = {{
            "env": os.environ.get({mm.MODEL_MENU_ENV!r}),
            "default_is_menu": ev._default_model is m.GPT_5_4,
            "allowed": sorted(ev._allowed_model_ids),
            "derived": list(ev.with_overrides(apply_cost_penalty=False).model_menu),
        }}
        try:
            E({EVALUATOR_KWARGS})
            out["second"] = "constructed"
        except RuntimeError as e:
            out["second"] = "refused"
        print(json.dumps(out))
    """)
    assert r.returncode == 0, r.stderr
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info == {
        "env": "GPT_5_4",
        "default_is_menu": True,
        "allowed": ["openai/gpt-5.4-2026-03-05"],
        "derived": ["GPT_5_4"],
        "second": "refused",
    }


def test_unrestricted_evaluator_clears_a_stray_env_var():
    """A menu exported in the launching shell must not restrict a run that
    did not ask for one."""
    r = _run(f"""
        import json, os
        from evaluator import {EVALUATOR_CLASS} as E
        ev = E({EVALUATOR_KWARGS})
        import model_registry as m
        print(json.dumps({{
            "env": os.environ.get({mm.MODEL_MENU_ENV!r}),
            "active": m.ACTIVE_MENU,
            "n_allowed": len(ev._allowed_model_ids),
        }}))
    """, {mm.MODEL_MENU_ENV: "GPT_5_4"})
    assert r.returncode == 0, r.stderr
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["env"] is None and info["active"] is None
    assert info["n_allowed"] >= len(mm.MENU)
