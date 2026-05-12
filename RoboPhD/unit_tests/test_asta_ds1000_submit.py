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

REPO_ROOT = Path(__file__).resolve().parents[2]
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


def test_wrapper_template_imports_seed_agent(wrapper_tree: ast.AST) -> None:
    """The wrapper must import `make_solver` from `seed_agent` (the
    bundled seed file `stage()` copies in). Without this import, the
    seed fallback can't fire and submissions silently revert to
    empty-completion-on-error."""
    seed_imports = [
        node for node in ast.walk(wrapper_tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "seed_agent"
        and any(alias.name == "make_solver" for alias in node.names)
    ]
    assert seed_imports, (
        "WRAPPER_TEMPLATE must contain `from seed_agent import make_solver "
        "as _seed_make_solver` (or equivalent) so the second tier of the "
        "fallback can fire"
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
