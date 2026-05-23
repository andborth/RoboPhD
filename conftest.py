"""Root-level pytest configuration.

`testpaths = examples` (in pytest.ini) walks the entire examples/ tree
looking for `test_*.py`. Most of what's there is genuine pytest under
`examples/<task>/unit_tests/`, but some example directories carry CLI
scripts whose filenames start with `test_` for historical reasons (e.g.
`examples/sudoku/test_eval_candidate.py` is a test-set evaluation CLI,
not a pytest module). Importing those at collection time triggers
sys.path manipulation and unrelated import side effects we don't want
during routine test runs.

We can't filter these out with `collect_ignore_glob` because pytest
applies it via `fnmatch`, where `*` matches across `/` — a pattern like
`examples/*/test_*.py` would also exclude legitimate test files nested
under `examples/<task>/unit_tests/`. Use a path-shape predicate instead:
ignore `examples/<task>/test_*.py` (exactly three path components under
the repo root) but keep anything deeper.

Scope: this filter only fires during *recursive* discovery from
`testpaths`. If someone explicitly names a file on the CLI
(`pytest examples/sudoku/test_eval_candidate.py`), pytest will still
collect it — that's standard "explicit beats implicit" behavior. The
hook is a guard against accidental discovery during `pytest` with no
args, not a hard quarantine.
"""

from pathlib import Path


def pytest_ignore_collect(collection_path: Path, config):
    """Skip misnamed CLI scripts at example roots while keeping
    `examples/<task>/unit_tests/test_*.py` discoverable."""
    try:
        rel = collection_path.relative_to(config.rootpath)
    except ValueError:
        return False
    parts = rel.parts
    return (
        len(parts) == 3
        and parts[0] == "examples"
        and parts[2].startswith("test_")
        and parts[2].endswith(".py")
    )
