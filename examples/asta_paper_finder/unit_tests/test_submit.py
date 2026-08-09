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


def _isolated_loads_of(tree: ast.AST, stem: str) -> list[ast.Call]:
    """Calls of the form _isolated("<stem>") anywhere in the template."""
    return [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "_isolated"
        and n.args and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == stem
    ]


def test_wrapper_template_can_reach_the_seed_tier(wrapper_tree: ast.AST) -> None:
    """Without a way to load the seed, the second fallback tier can't fire and
    submissions silently revert to empty-submission-on-error. The seed is now
    loaded per-sample via _isolated() rather than imported at module scope."""
    legacy_import = [
        node for node in ast.walk(wrapper_tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "seed_agent"
        and any(alias.name == "make_solver" for alias in node.names)
    ]
    assert _isolated_loads_of(wrapper_tree, "seed_agent") or legacy_import, (
        "wrapper must load seed_agent, via _isolated() or a module-scope import"
    )


def test_wrapper_template_isolates_agent_inner_per_sample(
        wrapper_tree: ast.AST) -> None:
    """`astabench eval --max-samples 6` runs six samples as asyncio tasks in
    ONE process, while RoboPhD training runs one sample per subprocess. A
    module-scope `from agent_inner import ...` therefore turns the agent's
    per-sample state global: v0_0_9_cap_0_063_fable kept its deadline clock and
    tool semaphore at module scope, _remaining() never counted down, and 30% of
    samples were guillotined."""
    module_scope_import = [
        n for n in ast.walk(wrapper_tree)
        if isinstance(n, ast.ImportFrom) and n.module == "agent_inner"
    ]
    assert not module_scope_import, (
        "agent_inner must NOT be imported at module scope — that is exactly "
        "the shared-state bug this wrapper exists to prevent"
    )
    assert _isolated_loads_of(wrapper_tree, "agent_inner"), (
        "agent_inner must be loaded per-sample via _isolated()"
    )


def test_wrapper_template_keeps_registry_and_pacer_shared(
        wrapper_tree: ast.AST) -> None:
    """model_registry holds the model handles (one connection pool) and
    tool_pacer holds the global launch budget — isolating either would give
    every sample its own and blow the Asta rate limit.

    The model_registry import is load-bearing for a second reason: it runs
    inside inspect's chdir_python() window, pinning the module into sys.modules
    so agent_inner's own `from model_registry import ...` resolves at solve()
    time, long after chdir_python.__exit__ restored sys.path."""
    imported = {
        a.name for n in wrapper_tree.body if isinstance(n, ast.Import)
        for a in n.names
    }
    for mod in ("tool_pacer", "model_registry"):
        assert mod in imported, f"{mod} must stay a plain module-scope import"
        assert not _isolated_loads_of(wrapper_tree, mod), (
            f"{mod} must stay shared across samples, not isolated"
        )


def test_wrapper_template_warms_up_above_the_solver(wrapper_tree: ast.AST) -> None:
    """The warm-up load must sit at module scope (so it runs inside
    chdir_python's window, pinning model_registry and compiling the pyc) and
    ABOVE the @solver def, so the wrapper is the last thing registered under
    the bare name "make_solver" — inspect's resume path resolves by that name."""
    warm = [
        n for n in wrapper_tree.body
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name) and n.value.func.id == "_isolated"
    ]
    assert warm, "template must warm-load agent_inner at module scope"
    solver_def = next(
        n for n in wrapper_tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "make_solver"
    )
    assert warm[0].lineno < solver_def.lineno, (
        "warm-up must precede the @solver definition"
    )


def test_wrapper_template_loads_inner_inside_the_try(wrapper_tree: ast.AST) -> None:
    """A per-sample load failure (transient OSError, MemoryError) must degrade
    to the seed tier, not propagate out of solve() and abort the whole eval —
    which would defeat the wrapper's entire purpose."""
    solve_fn = next(
        n for n in ast.walk(wrapper_tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "solve"
    )
    guarded = any(
        isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        and c.func.id == "_isolated" and c.args
        and getattr(c.args[0], "value", None) == "agent_inner"
        for t in ast.walk(solve_fn) if isinstance(t, ast.Try)
        for stmt in t.body for c in ast.walk(stmt)
    )
    assert guarded, "_isolated('agent_inner') must be inside solve()'s try block"


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


def test_stage_freezes_a_submission_with_a_successful_full_log(
        submit_mod, monkeypatch, tmp_path) -> None:
    """stage() runs BEFORE eval_submission's skip-on-success check, so without
    a freeze any re-run — including a bare invocation with no --only — silently
    rewrites the staged source of an already-posted submission. The eval is
    skipped either way, so the loss is invisible. v0_0_7_soft_cap_0_06_fable
    still carries its original 3702-byte, pre-tool_pacer wrapper; that is the
    artifact this guard exists to preserve."""
    s = submit_mod.SUBMISSIONS[0]
    monkeypatch.setattr(submit_mod, "WORKING_BASE", tmp_path)
    monkeypatch.setattr(submit_mod, "_log_status", lambda p: ("success", 267))
    dst = tmp_path / s.name
    dst.mkdir(parents=True)
    (dst / "agent.py").write_text("ORIGINAL WRAPPER")

    assert submit_mod.stage(s) == dst
    assert (dst / "agent.py").read_text() == "ORIGINAL WRAPPER", (
        "frozen working dir was overwritten"
    )
    # --restage is the deliberate escape hatch.
    submit_mod.stage(s, restage=True)
    assert (dst / "agent.py").read_text() == submit_mod.WRAPPER_TEMPLATE


def test_stage_proceeds_without_a_successful_log(
        submit_mod, monkeypatch, tmp_path) -> None:
    """The freeze must not block a first run, nor a resume after an
    interrupted one (status cancelled/error)."""
    s = submit_mod.SUBMISSIONS[0]
    monkeypatch.setattr(submit_mod, "WORKING_BASE", tmp_path)
    for status in (None, "cancelled", "error"):
        monkeypatch.setattr(submit_mod, "_log_status", lambda p, _s=status: (_s, 10))
        dst = submit_mod.stage(s)
        assert (dst / "agent.py").read_text() == submit_mod.WRAPPER_TEMPLATE
        (dst / "agent.py").write_text("clobbered")


def test_work_suffix_gives_ab_arms_their_own_dir(
        submit_mod, monkeypatch, tmp_path) -> None:
    """A/B arms must not stage into the real (frozen) submission dir."""
    s = submit_mod.SUBMISSIONS[0]
    monkeypatch.setattr(submit_mod, "WORKING_BASE", tmp_path)
    # Only the real dir has a completed run; a fresh arm dir has no log.
    monkeypatch.setattr(
        submit_mod, "_log_status",
        lambda p: (None, 0) if "__ab_" in str(p) else ("success", 267))
    real, arm = submit_mod.work_dir(s), submit_mod.work_dir(s, "__ab_new")
    assert real != arm and arm.name.endswith("__ab_new")
    # The real dir is frozen; the arm is not, so it stages normally.
    assert submit_mod.stage(s, "__ab_new") == arm
    assert (arm / "agent.py").read_text() == submit_mod.WRAPPER_TEMPLATE
    assert not real.exists(), "A/B staging must not touch the submission dir"


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
    # Structural, not a substring match on ast.unparse: the guard grows
    # conjuncts over time (--work-suffix added one), and a literal match
    # fails on a gate that got STRICTER, which is the wrong direction to
    # break in.
    tar_if = next(
        n for n in ast.walk(main_fn)
        if isinstance(n, ast.If)
        and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                and c.func.id == "tar_submission" for c in ast.walk(n.test))
    )
    conds = (tar_if.test.values if isinstance(tar_if.test, ast.BoolOp)
             else [tar_if.test])
    rendered = [ast.unparse(c) for c in conds]
    assert any("args.limit is None" in c for c in rendered), (
        f"tar_submission must be gated on `args.limit is None`; got {rendered}"
    )
    # A/B arms stage into <name><suffix>/ but tar_submission derives the
    # archive name from s.name, so an untarred-arm guard is required too.
    assert any("args.work_suffix" in c for c in rendered), (
        f"tar_submission must be gated on an empty --work-suffix; got {rendered}"
    )


def test_backup_stock_judge_cache_moves_file_aside(submit_mod, monkeypatch,
                                                   tmp_path) -> None:
    """A fresh full run must judge cold: the accumulated stock verdict cache
    is moved to a timestamped sibling (never deleted), because cache hits
    permute the nDCG verdict ordering (../robophd_runs/docs/astabench_judge_ordering_issue.md)."""
    from astabench.evals.paper_finder import paper_finder_utils as pfu

    cache = tmp_path / "detailed_reference.json"
    cache.write_text('{"semantic_1": {"123": "perfectly_relevant_papers"}}')
    monkeypatch.setattr(pfu, "detailed_reference_path", str(cache))
    submit_mod.backup_stock_judge_cache()
    assert not cache.exists()
    baks = list(tmp_path.glob("detailed_reference.json.bak-*"))
    assert len(baks) == 1
    assert "semantic_1" in baks[0].read_text()  # content preserved, not deleted

    # Missing file (fresh install / already cold): silent no-op.
    submit_mod.backup_stock_judge_cache()
    assert len(list(tmp_path.glob("detailed_reference.json.bak-*"))) == 1


def test_eval_submission_backs_up_cache_only_on_fresh_full_run(
        submit_tree) -> None:
    """The backup must fire for a fresh FULL run only: smoke runs don't
    need it (the full run clears their warmth anyway) and a resume keeps
    the cache — its completed samples' ordering is already baked into the
    log."""
    fn = next(
        n for n in ast.walk(submit_tree)
        if isinstance(n, ast.FunctionDef) and n.name == "eval_submission"
    )
    src = ast.unparse(fn)
    assert "backup_stock_judge_cache()" in src
    assert "elif limit is None:" in src, (
        "backup must be in the fresh-start (status is None) full-run branch"
    )
