"""Pin the resilience-wrapper machinery in
`scripts/asta_paper_finder_submit.py`.

Same guard class as asta_ds1000's test_submit.py: the wrapper inserted at
stage time gives submissions per-sample crash tolerance matching the
internal subprocess-isolated evaluator. If a refactor drops the seed
import, forgets to copy `seed_agent.py`, or regresses the last-resort
output to an empty string (which would trigger the scorer's LLM re-parse
instead of a clean 0), we'd discover it only after spending API money on
a real submission run. These tests catch the structural regressions
cheaply, plus paper-finder-specific pins: the empty submission must be
schema-valid, smoke runs must be log-isolated and untarred, and the
staged snapshot + priced-models registry must exist/agree.
"""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMIT_PATH = REPO_ROOT / "scripts" / "asta_paper_finder_submit.py"


@pytest.fixture(scope="module")
def submit_mod():
    """Import the submit script as a module (module level is just
    constants and defs — no side effects)."""
    spec = importlib.util.spec_from_file_location("pf_submit", SUBMIT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def submit_tree() -> ast.AST:
    return ast.parse(SUBMIT_PATH.read_text())


@pytest.fixture(scope="module")
def wrapper_tree(submit_mod) -> ast.AST:
    """The wrapper template is itself valid Python — parse it as a module."""
    return ast.parse(submit_mod.WRAPPER_TEMPLATE)


def test_wrapper_template_imports_seed_agent(wrapper_tree: ast.AST) -> None:
    """Without the seed import, the second fallback tier can't fire and
    submissions silently revert to empty-submission-on-error."""
    seed_imports = [
        node for node in ast.walk(wrapper_tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "seed_agent"
        and any(alias.name == "make_solver" for alias in node.names)
    ]
    assert seed_imports


def test_wrapper_template_has_two_tier_fallback(wrapper_tree: ast.AST) -> None:
    try_nodes = [n for n in ast.walk(wrapper_tree) if isinstance(n, ast.Try)]
    assert len(try_nodes) >= 2, (
        f"WRAPPER_TEMPLATE must contain at least two try statements "
        f"(primary + seed-fallback tier); found {len(try_nodes)}"
    )


def test_wrapper_template_has_per_tier_timeouts(wrapper_tree: ast.AST) -> None:
    """Both tiers must be bounded by asyncio.wait_for so a hung primary
    (rate-limit retry storm, snippet_search stall) can't wedge the eval.
    Floor at 300s: legitimate paper-finder queries run many minutes."""
    wait_for_calls = [
        n for n in ast.walk(wrapper_tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "wait_for"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "asyncio"
    ]
    assert len(wait_for_calls) >= 2
    consts = {
        stmt.targets[0].id: stmt.value.value
        for stmt in wrapper_tree.body
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, (int, float))
    }
    for call in wait_for_calls:
        timeout_kw = next((kw for kw in call.keywords if kw.arg == "timeout"), None)
        assert timeout_kw is not None, (
            f"asyncio.wait_for at line {call.lineno} must pass timeout=N"
        )
        if isinstance(timeout_kw.value, ast.Constant):
            val = timeout_kw.value.value
        else:
            assert isinstance(timeout_kw.value, ast.Name)
            val = consts.get(timeout_kw.value.id)
            assert val is not None
        assert val >= 300


def test_wrapper_empty_submission_is_schema_valid(submit_mod) -> None:
    """The last-resort output must parse as the task's expected JSON shape
    — an empty STRING would route through the scorer's GPT-4o-mini
    re-parse fallback instead of scoring a clean 0. Executes the
    template's actual _empty_submission function."""
    wrapper_tree = ast.parse(submit_mod.WRAPPER_TEMPLATE)
    fn = next(
        n for n in wrapper_tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_empty_submission"
    )
    ns: dict = {"json": json, "TaskState": object}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<wrapper>", "exec"), ns)
    completion = ns["_empty_submission"](SimpleNamespace(sample_id="semantic_1"))
    payload = json.loads(completion)  # must be directly parseable
    assert payload["output"]["query_id"] == "semantic_1"
    assert payload["output"]["results"] == []
    # And astabench's own datamodel accepts it (results list of SingleResult).
    from astabench.evals.paper_finder.datamodel import ExpectedAgentOutput
    ExpectedAgentOutput(**payload["output"])


def test_stage_copies_seed_file(submit_tree: ast.AST) -> None:
    """stage() must copy the seed to seed_agent.py or the wrapper's import
    fails at eval start — after `astabench eval` startup time is spent."""
    stage_fn = next(
        n for n in ast.walk(submit_tree)
        if isinstance(n, ast.FunctionDef) and n.name == "stage"
    )
    found = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "copy"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "shutil"
        and len(n.args) >= 2
        and "seed_agent.py" in ast.unparse(n.args[1])
        for n in ast.walk(stage_fn)
    )
    assert found


def test_task_name_matches_astabench_config(submit_mod) -> None:
    """The CLI's --task filter matches config task NAMES, not task paths —
    the first smoke run failed with 'Filtered to 0 of 11 tasks' because
    the script passed the path (paper_finder_test). Pin TASK_NAME against
    the installed astabench default config's test-split names."""
    import astabench
    import yaml
    cfg_dir = Path(astabench.__file__).parent / "config"
    cfg_path = max(cfg_dir.glob("v*.yml"))  # the CLI's default config
    cfg = yaml.safe_load(cfg_path.read_text())
    test_split = next(s for s in cfg["splits"] if s["name"] == "test")
    names = {t["name"] for t in test_split["tasks"]}
    assert submit_mod.TASK_NAME in names, (
        f"TASK_NAME {submit_mod.TASK_NAME!r} is not a test-split task name "
        f"in {cfg_path.name}; available: {sorted(names)}"
    )


def test_registry_snapshots_exist(submit_mod) -> None:
    """Every SUBMISSIONS entry must point at a committed snapshot — a
    typo'd name/path would otherwise fail at stage() time."""
    for s in submit_mod.SUBMISSIONS:
        src = submit_mod.SOURCE_BASE / s.name / s.agent_rel_path
        assert src.exists(), f"snapshot missing for {s.name}: {src}"


def test_agent_models_cover_snapshot_imports(submit_mod) -> None:
    """AGENT_MODELS (the pricing preflight's checklist) must cover every
    model_registry handle the staged agents import — a handle added to a
    future snapshot without a pricing entry would revive the DS-1000
    v0_0_3 cost=null failure after the money is spent."""
    registry_src = (submit_mod.EXAMPLES_DIR / "model_registry.py").read_text()
    for s in submit_mod.SUBMISSIONS:
        agent_src = (submit_mod.SOURCE_BASE / s.name / s.agent_rel_path).read_text()
        tree = ast.parse(agent_src)
        handles = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "model_registry"
            for alias in node.names
        ]
        assert handles, f"{s.name}: agent imports no model_registry handles?"
        for h in handles:
            # The registry defines _<HANDLE>_ID = "provider/<model>"; every
            # imported handle's model must be listed in AGENT_MODELS.
            import re
            m = re.search(rf'_{h}_ID\s*=\s*"[^/"]+/([^"]+)"', registry_src)
            assert m, f"{s.name}: cannot resolve model id for handle {h}"
            assert m.group(1) in submit_mod.AGENT_MODELS, (
                f"{s.name}: handle {h} resolves to {m.group(1)!r}, which is "
                f"not in AGENT_MODELS {submit_mod.AGENT_MODELS} — the pricing "
                f"preflight would not check it"
            )


def test_interrupted_log_resumes_instead_of_skipping(submit_mod, monkeypatch, tmp_path) -> None:
    """An interrupted run leaves a cancelled/errored .eval containing only
    the finished samples. Skipping on mere file existence would score+tar
    that PARTIAL log as a submission; only status=='success' may skip,
    anything else must re-run (inspect eval-set reuses completed samples)."""
    s = submit_mod.SUBMISSIONS[0]
    calls: list = []
    monkeypatch.setattr(submit_mod, "run",
                        lambda cmd, *, cwd, extra_env=None: (calls.append(cmd), 0)[1])

    # Cancelled log → re-run; post-run status still not success → False.
    monkeypatch.setattr(submit_mod, "_log_status", lambda d: ("cancelled", 42))
    assert submit_mod.eval_submission(s, tmp_path, None) is False
    assert calls, "cancelled log must trigger a re-run, not a skip"

    # Successful log → skip without running anything.
    calls.clear()
    monkeypatch.setattr(submit_mod, "_log_status", lambda d: ("success", 267))
    assert submit_mod.eval_submission(s, tmp_path, None) is True
    assert not calls, "successful log must skip the eval"


def test_smoke_runs_are_log_isolated_and_untarred(submit_mod, submit_tree) -> None:
    """--limit runs must log outside logs/full_test (else a partial run
    trips the full run's idempotency skip) and must never be tarred
    (else a partial run masquerades as a submission)."""
    assert submit_mod.log_subdir(None) == submit_mod.FULL_LOG_SUBDIR
    assert submit_mod.log_subdir(3) != submit_mod.FULL_LOG_SUBDIR
    # tar_submission must only be called under a `limit is None` guard.
    main_fn = next(
        n for n in ast.walk(submit_tree)
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    tar_calls_guarded = [
        node for node in ast.walk(main_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "tar_submission"
    ]
    assert tar_calls_guarded, "main() must call tar_submission"
    src = ast.unparse(main_fn)  # note: unparse parenthesizes `not`
    assert "args.limit is None and (not tar_submission" in src, (
        "tar_submission must be gated on `args.limit is None`"
    )
