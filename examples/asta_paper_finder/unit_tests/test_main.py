"""Unit tests for asta_paper_finder's main.py wiring.

Source/AST-level guards for the modernization invariants:

  - every eval_candidate/eval_run call passes config= (finding E: the
    archived example ran test evals at the framework's default worker
    count, ignoring --max-workers);
  - the doc files carry every ${...} placeholder main.py interpolates,
    and main.py interpolates every placeholder the docs carry;
  - task_config_extras persistence is wired (finding B);
  - --model is gone (the nine-handle registry replaced it) and the
    cost/tool knobs exist;
  - the background.md price table matches litellm's registry for the
    underlying model IDs (menu-vs-billed-reality drift).
"""
import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PFB_DIR = REPO_ROOT / "examples" / "asta_paper_finder"
MAIN_SRC = (PFB_DIR / "main.py").read_text()
MAIN_TREE = ast.parse(MAIN_SRC)


# --- eval calls pass config= --------------------------------------------------


def test_every_eval_call_passes_config():
    offenders = []
    for node in ast.walk(MAIN_TREE):
        if isinstance(node, ast.Call):
            func_name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if func_name in ("eval_candidate", "eval_run"):
                kwargs = {kw.arg for kw in node.keywords}
                if "config" not in kwargs:
                    offenders.append(f"line {node.lineno}: {func_name}")
    assert not offenders, (
        "eval calls without config= (test eval ignores --max-workers/"
        "eval_timeout): " + "; ".join(offenders)
    )


# --- interpolation coverage -----------------------------------------------------


EXPECTED_PLACEHOLDERS = {
    "${COST_PENALTY_TABLE}",
    "${COST_THRESHOLD}",
    "${COST_PER_ERROR}",
    "${EVAL_TIMEOUT_MIN}",
}


def _doc_placeholders() -> set[str]:
    found = set()
    for doc in ("background.md", "objective.md"):
        text = (PFB_DIR / doc).read_text()
        found |= set(re.findall(r"\$\{[A-Z_]+\}", text))
    return found


def test_docs_use_only_interpolated_placeholders():
    """Any ${...} in the docs that main.py doesn't substitute would ship
    to the evolution AI as literal template text."""
    unknown = _doc_placeholders() - EXPECTED_PLACEHOLDERS
    assert not unknown, f"placeholders with no _interpolate substitution: {unknown}"


def test_main_interpolates_every_expected_placeholder():
    for ph in EXPECTED_PLACEHOLDERS:
        assert f'.replace("{ph}"' in MAIN_SRC, (
            f"main.py's _interpolate no longer substitutes {ph}"
        )


def test_docs_carry_the_cost_placeholders():
    """The cost framing must actually reach the evolution AI: the table
    in background.md and the threshold/slope in objective.md."""
    background = (PFB_DIR / "background.md").read_text()
    objective = (PFB_DIR / "objective.md").read_text()
    assert "${COST_PENALTY_TABLE}" in background
    assert "${COST_THRESHOLD}" in background
    assert "${EVAL_TIMEOUT_MIN}" in background
    assert "${COST_THRESHOLD}" in objective
    assert "${COST_PER_ERROR}" in objective


# --- task_config_extras / CLI surface -------------------------------------------


def test_task_config_extras_wired():
    assert "task_config_extras={PFB_TASK_CONFIG_KEY: resolved_runtime}" in MAIN_SRC


def _argparse_flags() -> set[str]:
    flags = set()
    for node in ast.walk(MAIN_TREE):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            flags.add(node.args[0].value)
    return flags


def test_model_flag_removed():
    """--model was replaced by the model_registry menu; its presence
    would mean a solver-model bypass around the registry."""
    assert "--model" not in _argparse_flags()


def test_cost_and_tool_knobs_present():
    flags = _argparse_flags()
    for required in ("--cost-threshold", "--cost-per-error", "--tool-source",
                     "--max-workers", "--eval-only", "--eval-agent"):
        assert required in flags, f"missing CLI flag: {required}"


def test_max_workers_defaults_none():
    """argparse default must be None so the resolution ladder can
    distinguish user-explicit from default (finding E: a concrete
    default silently clobbers the checkpoint's value on resume)."""
    for node in ast.walk(MAIN_TREE):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--max-workers"
        ):
            defaults = [kw.value for kw in node.keywords if kw.arg == "default"]
            assert defaults and isinstance(defaults[0], ast.Constant) \
                and defaults[0].value is None
            return
    pytest.fail("--max-workers flag not found")


def test_fresh_run_packs_task_defaults():
    """PFB's task defaults that differ from the framework's must be
    stamped into engine_overrides on fresh runs: new_agent_test_rounds=0
    (framework default 1) and examples_per_iteration=ceil(train/5)=14
    for the 66-sample pool (framework default 20 — too much per-example
    reuse for a pool this small). Both live under the fresh-run-only
    branch so --resume inherits the checkpoint instead."""
    assert 'engine_overrides["new_agent_test_rounds"] = 0' in MAIN_SRC
    assert 'engine_overrides["examples_per_iteration"] = -(-len(train) // 5)' in MAIN_SRC
    # ceiling-division sanity for the current pool size
    assert -(-66 // 5) == 14


def test_gepa_and_autoresearch_apply_engine_config():
    """Finding D: --engine-config must not be a silent no-op on the
    non-default engines."""
    assert MAIN_SRC.count("apply_engine_config(cfg, parsed_engine_config)") >= 2


# --- price-table consistency -----------------------------------------------------


def test_background_md_prices_match_litellm_registry():
    """The model-handle table's advertised $/MTok rates must match
    litellm's price registry for the underlying model IDs.

    The table is hand-maintained, and providers reprice or re-alias
    models upstream. The cost *penalty* is unaffected by table drift
    (it uses measured rates keyed by the response model name), but
    evolution makes routing decisions against the menu — an incident
    class asta_ds1000 hit when Flash repriced 3x.

    Handle -> model ID comes from AST-parsing model_registry.py's
    `_<HANDLE>_ID` constants rather than importing the module (import
    requires an Anthropic key at construction time).

    Failure-class split: a price MISMATCH always hard-fails — that's
    the drift signal. A model MISSING from litellm's registry skips
    instead: offline runs and stale bundled maps lack recent models,
    which is registry staleness, not table drift.

    "Billed reality" is defined by evaluator._estimate_cost: litellm's
    BUNDLED price snapshot first (the leaderboard's billing basis),
    falling back to the live map only for models the snapshot lacks.
    The lookup order here mirrors the evaluator exactly.
    """
    import json

    import litellm

    from RoboPhD.runner_utils import register_supported_model_pricing

    register_supported_model_pricing()

    try:
        bundled_map = json.loads(
            (Path(litellm.__file__).parent
             / "model_prices_and_context_window_backup.json").read_text()
        )
    except Exception:
        bundled_map = {}

    registry_src = (PFB_DIR / "model_registry.py").read_text()
    handle_to_id = {}
    for node in ast.walk(ast.parse(registry_src)):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("_")
            and node.targets[0].id.endswith("_ID")
            and isinstance(node.value, ast.Constant)
        ):
            handle_to_id[node.targets[0].id[1:-3]] = node.value.value

    background_md = (PFB_DIR / "background.md").read_text()
    handle_rows = [
        line for line in background_md.split("\n")
        if (
            line.endswith(" |")
            and any(
                line.startswith(f"| `{prefix}")
                for prefix in ("GPT_", "CLAUDE_", "GEMINI_")
            )
        )
    ]
    assert len(handle_rows) == 9, (
        f"expected 9 model-handle rows in background.md, found {len(handle_rows)}"
    )

    mismatches = []     # genuine drift — always a hard failure
    unverifiable = []   # registry staleness / offline — skip, don't cry wolf
    broken_rows = []    # repo-internal inconsistency — hard failure
    for row in handle_rows:
        cells = [c.strip() for c in row.split("|")]
        # ['', '`HANDLE`', input, output, default, overrides, '']
        handle = cells[1].strip("`")
        advertised_in, advertised_out = float(cells[2]), float(cells[3])

        model_id = handle_to_id.get(handle)
        if model_id is None:
            broken_rows.append(f"{handle}: no _<HANDLE>_ID in model_registry.py")
            continue

        bare = model_id.split("/", 1)[1]
        candidates = (model_id, bare, f"gemini/{bare}")
        key = next(
            (k for k in candidates
             if bundled_map.get(k, {}).get("input_cost_per_token") is not None),
            None,
        )
        source = bundled_map
        if key is None:
            key = next(
                (k for k in candidates if k in litellm.model_cost), None
            )
            source = litellm.model_cost
        if key is None:
            unverifiable.append(f"{handle} ({model_id})")
            continue

        mc = source[key]
        actual_in = mc["input_cost_per_token"] * 1e6
        actual_out = mc["output_cost_per_token"] * 1e6
        if advertised_in != pytest.approx(actual_in, abs=1e-6) or (
            advertised_out != pytest.approx(actual_out, abs=1e-6)
        ):
            mismatches.append(
                f"{handle}: background.md advertises "
                f"${advertised_in}/{advertised_out} per MTok but litellm "
                f"({key}) says ${actual_in:.2f}/{actual_out:.2f} — the "
                f"agent-facing menu has drifted from billed reality"
            )

    assert not broken_rows, (
        "table rows without a model_registry constant:\n  "
        + "\n  ".join(broken_rows)
    )
    assert not mismatches, "price-table drift:\n  " + "\n  ".join(mismatches)
    if unverifiable:
        pytest.skip(
            "rows unverifiable — models absent from litellm's price "
            "registry (staleness, not drift): " + ", ".join(unverifiable)
        )
