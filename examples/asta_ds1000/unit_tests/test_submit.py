"""Pin the two-tier resilience wrapper machinery in
`scripts/asta_ds1000_submit.py`.

The wrapper inserted at stage time gives submissions the same
fallback-to-seed-on-error semantics that internal inference gets via
`Ds1000Evaluator(fallback_candidate=...)`. If a refactor drops the seed
import from the template or forgets to copy `seed_agent.py` in `stage()`,
the submission falls back to empty completion (score 0) instead of
seed-recovered scores — and we discover this only by reviewing a real
submission log after spending API money.

These AST-level tests catch the structural regression cheaply.
"""
import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMIT_PATH = REPO_ROOT / "scripts" / "asta_ds1000_submit.py"


@pytest.fixture(scope="module")
def submit_source() -> str:
    return SUBMIT_PATH.read_text()


@pytest.fixture(scope="module")
def submit_tree(submit_source: str) -> ast.AST:
    return ast.parse(submit_source)


@pytest.fixture(scope="module")
def wrapper_template_str(submit_tree: ast.AST) -> str:
    """Extract the WRAPPER_TEMPLATE string constant from the parsed
    submit script. It's defined as a module-level assignment to a
    triple-quoted string."""
    for node in submit_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "WRAPPER_TEMPLATE":
                assert isinstance(node.value, ast.Constant), (
                    "WRAPPER_TEMPLATE must be a literal string constant"
                )
                return node.value.value
    raise AssertionError("WRAPPER_TEMPLATE assignment not found in submit script")


@pytest.fixture(scope="module")
def wrapper_tree(wrapper_template_str: str) -> ast.AST:
    """The wrapper template is itself valid Python — parse it as a
    module so the tests can walk its AST."""
    return ast.parse(wrapper_template_str)


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
    """The wrapper must be able to load `seed_agent` (the bundled seed file
    `stage()` copies in). Without it the seed fallback can't fire and
    submissions silently revert to empty-completion-on-error. The seed is now
    loaded per-sample via _isolated() rather than imported at module scope."""
    legacy_import = [
        node for node in ast.walk(wrapper_tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "seed_agent"
        and any(alias.name == "make_solver" for alias in node.names)
    ]
    assert _isolated_loads_of(wrapper_tree, "seed_agent") or legacy_import, (
        "WRAPPER_TEMPLATE must load seed_agent — via _isolated('seed_agent') "
        "or a module-scope import — so the second fallback tier can fire"
    )


def test_wrapper_template_isolates_agent_inner_per_sample(
        wrapper_tree: ast.AST) -> None:
    """`astabench eval --max-samples N` runs samples as asyncio tasks in ONE
    process, while RoboPhD's evaluator runs one sample per subprocess. A
    module-scope `from agent_inner import ...` therefore turns any per-sample
    state the agent keeps at module scope into shared global state. No DS-1000
    agent has hit this yet; the paper_finder submission did, losing pacing on
    30% of samples."""
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


def test_wrapper_template_keeps_registry_shared(wrapper_tree: ast.AST) -> None:
    """model_registry holds the model handles (one connection pool); isolating
    it would give every sample its own. The import is also load-bearing: it
    runs inside inspect's chdir_python() window, pinning the module into
    sys.modules so agent_inner's own `from model_registry import ...` resolves
    at solve() time, long after chdir_python.__exit__ restored sys.path."""
    imported = {
        a.name for n in wrapper_tree.body if isinstance(n, ast.Import)
        for a in n.names
    }
    assert "model_registry" in imported, (
        "model_registry must stay a plain module-scope import"
    )
    assert not _isolated_loads_of(wrapper_tree, "model_registry"), (
        "model_registry must stay shared across samples, not isolated"
    )


def test_wrapper_template_has_two_tier_fallback(wrapper_tree: ast.AST) -> None:
    """The wrapper must have at least two `try` statements: one wrapping
    the primary `_inner_make_solver` call and a nested one wrapping the
    seed fallback. A single try/except would collapse the two tiers and
    miss the seed-recovery step entirely."""
    try_nodes = [n for n in ast.walk(wrapper_tree) if isinstance(n, ast.Try)]
    assert len(try_nodes) >= 2, (
        f"WRAPPER_TEMPLATE must contain at least two try statements "
        f"(primary + seed-fallback tier); found {len(try_nodes)}"
    )


def test_stage_copies_seed_file(submit_tree: ast.AST) -> None:
    """The `stage()` function must call `shutil.copy(...)` with
    `seed_agent.py` as the destination basename, so the bundled seed
    is present at submission-eval time. Catches: someone updated the
    template but forgot to copy the file (would surface as ImportError
    at eval start, but only AFTER `astabench eval` spends startup time)."""
    stage_fn = None
    for node in ast.walk(submit_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "stage":
            stage_fn = node
            break
    assert stage_fn is not None, "stage() function not found"

    found_seed_copy = False
    for node in ast.walk(stage_fn):
        if not isinstance(node, ast.Call):
            continue
        # Match shutil.copy(...)
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr == "copy"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "shutil"):
            continue
        if len(node.args) < 2:
            continue
        dst_src = ast.unparse(node.args[1])
        if "seed_agent.py" in dst_src:
            found_seed_copy = True
            break
    assert found_seed_copy, (
        "stage() must call `shutil.copy(<seed_src>, dst_dir / \"seed_agent.py\")` "
        "so the wrapper's seed import resolves at submission-eval time"
    )


def test_wrapper_template_has_per_tier_timeouts(wrapper_tree: ast.AST) -> None:
    """Both tiers of the wrapper must be wrapped in `asyncio.wait_for`
    so a hung primary or seed can't wedge the eval indefinitely. The
    first iter10 submission attempt wedged for 1:45+ on a stuck sample
    because the wrapper's outer `except Exception` only catches raised
    exceptions, not duration; this test pins the regression."""
    wait_for_calls = [
        n for n in ast.walk(wrapper_tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "wait_for"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "asyncio"
    ]
    assert len(wait_for_calls) >= 2, (
        f"WRAPPER_TEMPLATE must wrap BOTH tiers in asyncio.wait_for; "
        f"found {len(wait_for_calls)}"
    )
    # Every wait_for must have a timeout kwarg (positional second arg
    # would also work, but we use kwargs in the template — pin that).
    # Floor at 300s: legitimate iter10 samples can take several minutes
    # (Opus reasoning + LOOP-SCRUB verify). A future "we don't need 20
    # minutes" reduction to e.g. 10s would defeat the wrapper's purpose
    # and false-timeout most samples; this test flags that class of
    # accidental tightening without preventing reasonable retuning
    # (anywhere from 5min to 30min stays within the floor).
    MIN_TIMEOUT_S = 300
    for call in wait_for_calls:
        kw_names = {kw.arg for kw in call.keywords}
        assert "timeout" in kw_names, (
            f"asyncio.wait_for at line {call.lineno} must pass a "
            f"timeout=N kwarg"
        )
        # Resolve the timeout to a numeric value. The template uses a
        # module-level constant (PRIMARY_TIMEOUT_S / SEED_TIMEOUT_S) as
        # the kwarg value, so we look it up in the wrapper module's
        # top-level Assign nodes.
        timeout_kw = next(kw for kw in call.keywords if kw.arg == "timeout")
        if isinstance(timeout_kw.value, ast.Constant) and isinstance(timeout_kw.value.value, (int, float)):
            timeout_val = timeout_kw.value.value
        elif isinstance(timeout_kw.value, ast.Name):
            const_name = timeout_kw.value.id
            timeout_val = None
            for stmt in wrapper_tree.body:
                if (isinstance(stmt, ast.Assign)
                        and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                        and stmt.targets[0].id == const_name
                        and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, (int, float))):
                    timeout_val = stmt.value.value
                    break
            assert timeout_val is not None, (
                f"asyncio.wait_for at line {call.lineno} references "
                f"unresolvable constant `{const_name}`"
            )
        else:
            raise AssertionError(
                f"asyncio.wait_for at line {call.lineno} has unrecognized "
                f"timeout value type: {ast.dump(timeout_kw.value)}"
            )
        assert timeout_val >= MIN_TIMEOUT_S, (
            f"asyncio.wait_for at line {call.lineno} has timeout="
            f"{timeout_val}, below the {MIN_TIMEOUT_S}s floor. Legitimate "
            f"iter10 samples can take several minutes; too-short timeouts "
            f"would false-trip on slow-but-correct runs."
        )
