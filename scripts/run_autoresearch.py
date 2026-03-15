#!/usr/bin/env python3
"""
Autoresearch engine runner: a single continuous Claude Code session that
autonomously runs experiments with greedy hill-climbing.

Inspired by Karpathy's autoresearch. Instead of training neural networks,
the agent optimizes text artifacts (prompts, code, agent architectures)
and evaluates via the same task evaluators used by GEPA and RoboPhD.

Usage:
    # Smoke test
    python scripts/run_autoresearch.py --task aime \
        --engine-config '{"evaluation_budget": 20}'

    # Full run
    python scripts/run_autoresearch.py --task codegen \
        --task-config '{"seed_agent": "RoboPhD/codegen_agents/naive_critic"}' \
        --engine-config '{"evaluation_budget": 200}'

    # With test-set evaluation
    python scripts/run_autoresearch.py --task aime \
        --engine-config '{"evaluation_budget": 100}' --eval-test-set

Config merge order: task defaults -> --task-config -> --engine-config
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RoboPhD.adapters.candidate_utils import extract_candidate, materialize_candidate
from RoboPhD.adapters.runner_utils import parse_config_arg, print_task_params
from RoboPhD.autoresearch.evaluate_cli import __file__ as _evaluate_cli_source
from RoboPhD.autoresearch.program import generate_program
from RoboPhD.autoresearch.server import EvalServer
from RoboPhD.config import CLAUDE_CLI_MODEL_MAP
from RoboPhD.eval_utils import run_parallel_eval
from RoboPhD.tasks import get_task, list_tasks
from utilities.claude_cli import _is_rate_limit_error, _parse_rate_limit_reset_time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Engine defaults
_AUTORESEARCH_DEFAULTS = {
    "evaluation_budget": (100, "Total evaluator calls (train + val)"),
    "val_ratio": (0.2, "Fraction held out for validation"),
    "val_size": (None, "Exact val size (mutually exclusive with val_ratio)"),
    "seed": (0, "Random seed"),
    "model": ("opus-4.6", "Claude Code model for the session"),
    "overall_timeout": (None, "Max wall-clock seconds for the entire run (None = no limit)"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run autoresearch optimization on a registered task"
    )

    parser.add_argument(
        "--task",
        required=True,
        help=f"Task to optimize. Available: {', '.join(list_tasks())}",
    )
    parser.add_argument(
        "--list-params",
        action="store_true",
        help="List all valid config parameters, then exit",
    )
    parser.add_argument(
        "--task-config",
        default=None,
        help="Task config: JSON file path or inline JSON string",
    )
    parser.add_argument(
        "--engine-config",
        default=None,
        help="Autoresearch engine config: JSON file path or inline JSON string",
    )
    parser.add_argument(
        "--eval-test-set",
        action="store_true",
        help="Evaluate best candidate on the held-out test set after optimization",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (overrides --runs-dir)",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Root directory for outputs (default: ../robophd_runs)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to experiment directory to resume after a crash",
    )

    args = parser.parse_args()
    if args.runs_dir is None:
        args.runs_dir = Path("../robophd_runs")
    return args


def _list_params(task):
    """Print all valid parameters and exit."""
    print("=" * 70)
    print("VALID CONFIGURATION PARAMETERS — run_autoresearch.py")
    print("=" * 70)
    print("\nConfig merge order: task defaults -> --task-config -> --engine-config")

    print_task_params(task)

    print("Autoresearch engine parameters (--engine-config):")
    for k, (default, desc) in _AUTORESEARCH_DEFAULTS.items():
        print(f"  - {k}: {default!r}  — {desc}")
    print()

    print("CLI-only arguments (not in config):")
    print("  --eval-test-set        Evaluate best candidate on held-out test set")
    print("  --output-dir PATH      Output directory (overrides --runs-dir)")
    print("  --runs-dir PATH        Root directory for outputs (default: ../robophd_runs)")
    print("  --resume PATH          Resume a crashed run from its experiment directory")
    print()


def _setup_workspace(
    workspace: Path,
    seed_candidate: dict,
    file_mapping: dict,
    task,
    train_example_ids: list,
    val_size: int,
    evaluation_budget: int,
):
    """Initialize the workspace as a git repo with seed candidate and infrastructure."""
    workspace.mkdir(parents=True, exist_ok=True)

    # Write seed candidate files
    for key, filepath in file_mapping.items():
        if key in seed_candidate:
            dest = workspace / filepath
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(seed_candidate[key])

    # Generate and write program.md
    program = generate_program(task, file_mapping, train_example_ids, val_size, evaluation_budget)
    (workspace / "program.md").write_text(program)

    # Copy evaluate CLI
    source = Path(_evaluate_cli_source)
    shutil.copy2(source, workspace / "_evaluate.py")

    # Write training example IDs
    (workspace / "_train_examples.json").write_text(json.dumps(train_example_ids, indent=2))

    # Write CLAUDE.md with task background
    claude_md = f"# {task.name}\n\n"
    claude_md += f"**Objective**: {task.objective}\n\n"
    if task.background:
        claude_md += task.background
    claude_md += "\n\n---\n\nRead `program.md` for the full autoresearch protocol.\n"
    (workspace / "CLAUDE.md").write_text(claude_md)

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=workspace, capture_output=True, check=True)
    subprocess.run(["git", "add", "-A"], cwd=workspace, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Seed candidate"],
        cwd=workspace, capture_output=True, check=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "autoresearch", "GIT_AUTHOR_EMAIL": "autoresearch@robophd",
             "GIT_COMMITTER_NAME": "autoresearch", "GIT_COMMITTER_EMAIL": "autoresearch@robophd"},
    )

    logger.info(f"Workspace initialized at {workspace}")


def _read_candidate_from_workspace(workspace: Path, file_mapping: dict) -> dict:
    """Read the current candidate files from workspace."""
    candidate = {}
    for key, filepath in file_mapping.items():
        src = workspace / filepath
        candidate[key] = src.read_text() if src.exists() else ""
    return candidate


def _wait_for_rate_limit(result: subprocess.CompletedProcess):
    """Parse rate limit reset time and sleep until then."""
    import time

    try:
        output = json.loads(result.stdout)
        error_msg = output.get("result", "")
    except (json.JSONDecodeError, TypeError):
        error_msg = ""

    reset_time = _parse_rate_limit_reset_time(error_msg)
    if reset_time:
        if reset_time.tzinfo is not None:
            now = datetime.now(reset_time.tzinfo)
        else:
            now = datetime.now()
        wait_seconds = max(0, (reset_time - now).total_seconds()) + 30  # 30s buffer
        reset_str = reset_time.strftime('%I:%M%p').lstrip('0').lower()
        logger.info(f"Rate limit — waiting {int(wait_seconds)}s until {reset_str}")
        time.sleep(wait_seconds)
    else:
        logger.warning("Rate limit hit but couldn't parse reset time — waiting 5 minutes")
        time.sleep(300)


def _find_claude_cli():
    """Find Claude CLI, reusing the utility if available."""
    try:
        from utilities.claude_cli import find_claude_cli
        cli = find_claude_cli()
        if cli:
            return cli
    except ImportError:
        pass

    # Fallback
    for path in [
        os.path.expanduser("~/.claude/local/claude"),
        "/usr/local/bin/claude",
    ]:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path

    result = subprocess.run(["which", "claude"], capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()

    return None


def _run_session(
    claude_cli: str,
    cli_model: str,
    workspace: Path,
    session_id: str,
    overall_timeout,
    evaluation_budget: int,
    prompt: str,
    is_resume: bool = False,
) -> tuple:
    """Run a single Claude Code session with rate-limit retry.

    Returns (total_cost, tokens_in, tokens_out).
    """
    total_cost = 0.0
    tokens_in = 0
    tokens_out = 0

    env = {**os.environ}

    while True:
        if is_resume:
            cmd = [
                claude_cli,
                "--model", cli_model,
                "--permission-mode", "bypassPermissions",
                "--output-format", "json",
                "--resume", session_id,
                "--print", "You were interrupted. Please continue.",
            ]
        else:
            cmd = [
                claude_cli,
                "--model", cli_model,
                "--permission-mode", "bypassPermissions",
                "--output-format", "json",
                "--session-id", session_id,
                "--print", prompt,
            ]

        cmd.extend(["--settings", json.dumps({"autoCompact": True})])

        logger.info(f"Launching Claude Code session (session_id={session_id[:8]}..., resume={is_resume})")

        try:
            result = subprocess.run(
                cmd, cwd=workspace, capture_output=True, text=True,
                timeout=overall_timeout, env=env,
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"Claude Code session timed out after {overall_timeout}s")
            break
        except Exception as e:
            logger.error(f"Claude Code session error: {e}")
            break

        # Parse output and accumulate cost
        try:
            output = json.loads(result.stdout)
            result_text = output.get("result", "")
            result_preview = result_text[:200] + "..." if len(result_text) > 200 else result_text
            logger.info(f"Session exited (code={result.returncode}): {result_preview}")

            call_cost = output.get("total_cost_usd", 0.0)
            if call_cost:
                total_cost += call_cost
            usage = output.get("usage", {})
            tokens_in += usage.get("input_tokens", 0)
            tokens_out += usage.get("output_tokens", 0)
        except (json.JSONDecodeError, TypeError):
            logger.info(f"Session exited (code={result.returncode}), stdout={result.stdout[:200]}")

        # Check for rate limit — wait and retry with --resume
        if _is_rate_limit_error(result):
            _wait_for_rate_limit(result)
            is_resume = True
            continue

        if result.returncode != 0:
            logger.warning(f"Claude Code failed (code={result.returncode})")
            if result.stderr:
                logger.error(f"stderr: {result.stderr[:500]}")

        break  # Done (success or non-rate-limit error)

    return total_cost, tokens_in, tokens_out


def _resume_run(args):
    """Resume a crashed autoresearch run from its experiment directory."""
    resume_dir = args.resume.resolve()

    # Check if run already completed
    if (resume_dir / "optimization_summary.json").exists():
        logger.error(f"Run already completed. Cannot resume a finished run: {resume_dir}")
        sys.exit(1)

    workspace = resume_dir / "workspace"
    if not workspace.exists():
        logger.error(f"No workspace found at {workspace}")
        sys.exit(1)

    # Read session_id
    session_id_path = workspace / "_session_id"
    if not session_id_path.exists():
        logger.error(f"No session ID found at {session_id_path} — cannot resume")
        sys.exit(1)
    session_id = session_id_path.read_text().strip()

    # Read budget state
    budget_path = workspace / "_budget.json"
    if not budget_path.exists():
        logger.error(f"No budget state found at {budget_path} — cannot resume")
        sys.exit(1)
    budget_state = json.loads(budget_path.read_text())
    remaining_budget = budget_state.get("remaining", 0)
    if remaining_budget <= 0:
        logger.error(f"No budget remaining ({remaining_budget}) — nothing to resume")
        sys.exit(1)

    # Read seed config to reconstruct task
    seed_config_path = resume_dir / "seed_candidate.json"
    if not seed_config_path.exists():
        logger.error(f"No seed_candidate.json found at {resume_dir}")
        sys.exit(1)

    # Load task
    task = get_task(args.task)
    task_config = parse_config_arg(args.task_config)
    engine_config = parse_config_arg(args.engine_config)
    config = {**task.config_defaults, **task_config, **engine_config}

    evaluation_budget = budget_state.get("total", config.get("evaluation_budget", _AUTORESEARCH_DEFAULTS["evaluation_budget"][0]))
    model = config.get("model", _AUTORESEARCH_DEFAULTS["model"][0])
    overall_timeout = config.get("overall_timeout", _AUTORESEARCH_DEFAULTS["overall_timeout"][0])
    seed = config.get("seed", _AUTORESEARCH_DEFAULTS["seed"][0])
    val_ratio = config.get("val_ratio", _AUTORESEARCH_DEFAULTS["val_ratio"][0])
    val_size_cfg = config.get("val_size", _AUTORESEARCH_DEFAULTS["val_size"][0])
    seed_agent = Path(config.get("seed_agent", task.default_seed_agent))

    # Rebuild datasets (deterministic with same seed)
    import random
    rng = random.Random(seed)

    if task.gepa_datasets_builder is not None:
        trainset, valset = task.gepa_datasets_builder(config)
    else:
        dataset = task.dataset_builder(config)
        shuffled = list(dataset)
        rng.shuffle(shuffled)

        if val_size_cfg is not None:
            split_idx = len(shuffled) - val_size_cfg
        else:
            split_idx = max(1, int(len(shuffled) * (1 - val_ratio)))

        trainset = shuffled[:split_idx]
        valset = shuffled[split_idx:]

    # Assign IDs (same logic as fresh run)
    _ID_FIELDS = ("id", "question_id", "problem_id", "task_id")

    def _find_id_field(examples):
        if not examples:
            return None
        for field in _ID_FIELDS:
            if field in examples[0]:
                return field
        return None

    def _assign_ids(examples, prefix=""):
        id_field = _find_id_field(examples)
        for i, ex in enumerate(examples):
            if id_field and id_field != "id":
                ex["id"] = str(ex[id_field])
            elif "id" in ex:
                ex["id"] = str(ex["id"])
            else:
                ex["id"] = f"{prefix}{i}"

    _assign_ids(trainset)
    _assign_ids(valset, prefix="val_")

    # Create evaluator
    config["work_dir"] = str(resume_dir / "work")
    config["output_dir"] = str(resume_dir)
    evaluator = task.evaluator_factory(config)

    # Start eval server with remaining budget
    def get_candidate():
        return _read_candidate_from_workspace(workspace, task.file_mapping)

    used = budget_state.get("used", 0)
    eval_server = EvalServer(
        evaluator=evaluator,
        candidate_fn=get_candidate,
        train_examples=trainset,
        val_examples=valset,
        evaluation_budget=evaluation_budget,
        workspace=workspace,
    )
    eval_server.start()
    # Adjust budget to account for already-used evals
    eval_server.budget_remaining = remaining_budget

    # Find Claude CLI
    claude_cli = _find_claude_cli()
    if not claude_cli:
        logger.error("Claude CLI not found.")
        eval_server.stop()
        sys.exit(1)

    cli_model = CLAUDE_CLI_MODEL_MAP.get(model, model)

    logger.info(
        f"Resuming autoresearch session (session_id={session_id[:8]}..., "
        f"budget_remaining={remaining_budget}, model={cli_model})"
    )

    total_cost, tokens_in, tokens_out = _run_session(
        claude_cli=claude_cli,
        cli_model=cli_model,
        workspace=workspace,
        session_id=session_id,
        overall_timeout=overall_timeout,
        evaluation_budget=evaluation_budget,
        prompt="You were interrupted. Please continue.",
        is_resume=True,
    )

    eval_server.stop()

    # Run the same post-session steps as a fresh run
    args.output_dir = resume_dir
    _post_session(
        args=args,
        task=task,
        config=config,
        workspace=workspace,
        session_id=session_id,
        claude_cli=claude_cli,
        cli_model=cli_model,
        seed_agent=seed_agent,
        evaluation_budget=evaluation_budget,
        val_ratio=val_ratio,
        val_size_cfg=val_size_cfg,
        valset=valset,
        trainset=trainset,
        overall_timeout=overall_timeout,
        total_session_cost=total_cost,
        total_session_tokens_in=tokens_in,
        total_session_tokens_out=tokens_out,
    )


def main():
    args = parse_args()

    # --- Handle --list-params ---
    if args.list_params:
        task = get_task(args.task)
        _list_params(task)
        sys.exit(0)

    # --- Handle --resume ---
    if args.resume:
        return _resume_run(args)

    # --- 1. Load task and merge config ---
    task = get_task(args.task)
    task_config = parse_config_arg(args.task_config)
    engine_config = parse_config_arg(args.engine_config)
    config = {**task.config_defaults, **task_config, **engine_config}

    logger.info(f"Task: {task.name} — {task.description}")

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = args.runs_dir / "autoresearch" / f"{task.name}_{timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(args.output_dir)

    logger.info(f"Output directory: {args.output_dir}")

    # --- 2. Extract engine config values ---
    evaluation_budget = config.get("evaluation_budget", _AUTORESEARCH_DEFAULTS["evaluation_budget"][0])
    val_ratio = config.get("val_ratio", _AUTORESEARCH_DEFAULTS["val_ratio"][0])
    val_size_cfg = config.get("val_size", _AUTORESEARCH_DEFAULTS["val_size"][0])
    seed = config.get("seed", _AUTORESEARCH_DEFAULTS["seed"][0])
    model = config.get("model", _AUTORESEARCH_DEFAULTS["model"][0])
    overall_timeout = config.get("overall_timeout", _AUTORESEARCH_DEFAULTS["overall_timeout"][0])

    # --- 3. Extract seed candidate ---
    seed_agent = Path(config.get("seed_agent", task.default_seed_agent))
    logger.info(f"Extracting seed candidate from {seed_agent}")
    seed_candidate = extract_candidate(seed_agent, task.file_mapping)
    logger.info(
        f"Seed candidate keys: {list(seed_candidate.keys())}, "
        f"sizes: { {k: len(v) for k, v in seed_candidate.items()} }"
    )

    with open(args.output_dir / "seed_candidate.json", "w") as f:
        json.dump(seed_candidate, f, indent=2)

    # --- 4. Build datasets ---
    import random

    rng = random.Random(seed)

    if task.gepa_datasets_builder is not None:
        trainset, valset = task.gepa_datasets_builder(config)
        logger.info(f"Dataset (pre-split): {len(trainset)} train, {len(valset)} val")
    else:
        dataset = task.dataset_builder(config)
        logger.info(f"Dataset: {len(dataset)} examples")

        if not dataset:
            logger.error("Empty dataset — check cache directory and task configuration.")
            sys.exit(1)

        if val_size_cfg is not None and val_ratio != _AUTORESEARCH_DEFAULTS["val_ratio"][0]:
            logger.error("Cannot specify both val_ratio and val_size")
            sys.exit(1)

        shuffled = list(dataset)
        rng.shuffle(shuffled)

        if val_size_cfg is not None:
            if val_size_cfg < 1:
                logger.error(f"val_size ({val_size_cfg}) must be >= 1")
                sys.exit(1)
            if val_size_cfg >= len(shuffled):
                logger.error(f"val_size ({val_size_cfg}) must be less than dataset size ({len(shuffled)})")
                sys.exit(1)
            split_idx = len(shuffled) - val_size_cfg
        else:
            split_idx = max(1, int(len(shuffled) * (1 - val_ratio)))

        trainset = shuffled[:split_idx]
        valset = shuffled[split_idx:]
        logger.info(f"Training set: {len(trainset)}, Validation set: {len(valset)}")

    # Assign IDs to examples. Tasks use different ID fields (question_id,
    # problem_id, etc.) — discover the right one from the first example.
    _ID_FIELDS = ("id", "question_id", "problem_id", "task_id")

    def _find_id_field(examples):
        if not examples:
            return None
        for field in _ID_FIELDS:
            if field in examples[0]:
                return field
        return None

    def _assign_ids(examples, prefix=""):
        id_field = _find_id_field(examples)
        for i, ex in enumerate(examples):
            if id_field and id_field != "id":
                ex["id"] = str(ex[id_field])
            elif "id" in ex:
                ex["id"] = str(ex["id"])
            else:
                ex["id"] = f"{prefix}{i}"

    _assign_ids(trainset)
    _assign_ids(valset, prefix="val_")

    train_example_ids = [ex["id"] for ex in trainset]

    # --- 5. Create evaluator ---
    config["work_dir"] = str(args.output_dir / "work")
    evaluator = task.evaluator_factory(config)

    # --- 6. Setup workspace ---
    workspace = args.output_dir / "workspace"
    _setup_workspace(
        workspace, seed_candidate, task.file_mapping, task,
        train_example_ids, len(valset), evaluation_budget,
    )

    # --- 7. Start eval server ---
    def get_candidate():
        return _read_candidate_from_workspace(workspace, task.file_mapping)

    eval_server = EvalServer(
        evaluator=evaluator,
        candidate_fn=get_candidate,
        train_examples=trainset,
        val_examples=valset,
        evaluation_budget=evaluation_budget,
        workspace=workspace,
    )
    eval_server.start()

    # --- 8. Find Claude CLI ---
    claude_cli = _find_claude_cli()
    if not claude_cli:
        logger.error("Claude CLI not found. Install from: https://docs.anthropic.com/en/docs/claude-code")
        eval_server.stop()
        sys.exit(1)

    # Map model name to CLI model
    cli_model = CLAUDE_CLI_MODEL_MAP.get(model, model)

    # --- 9. Run Claude Code session (single call + rate-limit retry) ---
    session_id = str(uuid.uuid4())

    # Persist session_id for crash recovery
    (workspace / "_session_id").write_text(session_id)

    total_session_cost, total_session_tokens_in, total_session_tokens_out = _run_session(
        claude_cli=claude_cli,
        cli_model=cli_model,
        workspace=workspace,
        session_id=session_id,
        overall_timeout=overall_timeout,
        evaluation_budget=evaluation_budget,
        prompt="Read program.md and begin the autoresearch protocol.",
    )

    eval_server.stop()

    _post_session(
        args=args,
        task=task,
        config=config,
        workspace=workspace,
        session_id=session_id,
        claude_cli=claude_cli,
        cli_model=cli_model,
        seed_agent=seed_agent,
        evaluation_budget=evaluation_budget,
        val_ratio=val_ratio,
        val_size_cfg=val_size_cfg,
        valset=valset,
        trainset=trainset,
        overall_timeout=overall_timeout,
        total_session_cost=total_session_cost,
        total_session_tokens_in=total_session_tokens_in,
        total_session_tokens_out=total_session_tokens_out,
    )


def _post_session(
    *,
    args,
    task,
    config,
    workspace,
    session_id,
    claude_cli,
    cli_model,
    seed_agent,
    evaluation_budget,
    val_ratio,
    val_size_cfg,
    valset,
    trainset,
    overall_timeout,
    total_session_cost,
    total_session_tokens_in,
    total_session_tokens_out,
):
    """Post-session steps: reflection, results collection, optional test-set eval."""
    model = config.get("model", _AUTORESEARCH_DEFAULTS["model"][0])

    # --- Request reflection ---
    reflection_prompt = (
        "Thanks for your work on this. Looking back at the entire session, "
        "please write a brief reflection to `_reflection.md`. Consider:\n\n"
        "- What approaches worked well? What didn't?\n"
        "- What was surprising or unexpected about the problem?\n"
        "- What would you do differently with a fresh budget?\n"
        "- Any insights about the task itself that a future researcher should know?\n"
        "- Any suggestions for improving the instructions you were given?\n"
        "- Is there any tool or library that you wish you had access to?\n\n"
        "Keep it concise — a few paragraphs, not an essay."
    )
    reflection_cmd = [
        claude_cli,
        "--model", cli_model,
        "--permission-mode", "bypassPermissions",
        "--output-format", "json",
        "--resume", session_id,
        "--print", reflection_prompt,
    ]
    reflection_cmd.extend(["--settings", json.dumps({"autoCompact": True})])

    logger.info("Requesting agent reflection...")
    try:
        refl_result = subprocess.run(reflection_cmd, cwd=workspace, capture_output=True, text=True, timeout=300)
        try:
            refl_output = json.loads(refl_result.stdout)
            refl_cost = refl_output.get("total_cost_usd", 0.0)
            if refl_cost:
                total_session_cost += refl_cost
            refl_usage = refl_output.get("usage", {})
            total_session_tokens_in += refl_usage.get("input_tokens", 0)
            total_session_tokens_out += refl_usage.get("output_tokens", 0)
        except (json.JSONDecodeError, TypeError):
            pass
    except Exception as e:
        logger.warning(f"Reflection request failed: {type(e).__name__}: {e}")

    # --- Collect results ---
    logger.info("Collecting results...")

    # Best candidate = current workspace files
    best_candidate = _read_candidate_from_workspace(workspace, task.file_mapping)

    with open(args.output_dir / "best_candidate.json", "w") as f:
        json.dump(best_candidate, f, indent=2)

    best_agent_dir = args.output_dir / "best_agent"
    materialize_candidate(best_candidate, best_agent_dir, task.file_mapping, name="autoresearch_best")
    logger.info(f"Best agent materialized at: {best_agent_dir}")

    # Read experiment log if it exists
    experiment_log = []
    log_path = workspace / "_experiment_log.jsonl"
    if log_path.exists():
        for line in log_path.read_text().strip().split("\n"):
            if line.strip():
                try:
                    experiment_log.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Get git log for candidate history
    git_log_result = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=workspace, capture_output=True, text=True,
    )
    git_history = git_log_result.stdout.strip().split("\n") if git_log_result.stdout.strip() else []

    # Read budget state
    budget_state = {}
    budget_path = workspace / "_budget.json"
    if budget_path.exists():
        budget_state = json.loads(budget_path.read_text())

    # Copy reflection if the agent wrote one
    reflection_path = workspace / "_reflection.md"
    if reflection_path.exists():
        shutil.copy2(reflection_path, args.output_dir / "reflection.md")
        logger.info(f"Agent reflection saved to {args.output_dir / 'reflection.md'}")

    # Best val score from experiment log
    best_val_score = 0.0
    if experiment_log:
        kept_experiments = [e for e in experiment_log if e.get("kept")]
        if kept_experiments:
            best_val_score = max(e.get("val_score") or 0.0 for e in kept_experiments)

    summary = {
        "task": task.name,
        "objective": task.objective,
        "engine": "autoresearch",
        "seed_agent": str(seed_agent),
        "evaluation_budget": evaluation_budget,
        "val_ratio": val_ratio if val_size_cfg is None else None,
        "val_size": len(valset),
        "seed": config.get("seed", _AUTORESEARCH_DEFAULTS["seed"][0]),
        "model": model,
        "overall_timeout": overall_timeout,
        "train_size": len(trainset),
        "num_experiments": len(experiment_log),
        "num_commits": len(git_history),
        "best_val_score": best_val_score,
        "budget_state": budget_state,
        "experiment_log": experiment_log,
        "git_history": git_history,
        "config": {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool, type(None)))},
        "cost": {
            "evolution_cost_usd": round(total_session_cost, 2),
            "tokens_in": total_session_tokens_in,
            "tokens_out": total_session_tokens_out,
        },
    }
    with open(args.output_dir / "optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Optimization complete. {len(experiment_log)} experiments, "
        f"best val score: {best_val_score:.3f}, "
        f"evolution cost: ${total_session_cost:.2f}"
    )

    # --- Optional test-set evaluation ---
    if args.eval_test_set:
        logger.info("Evaluating best candidate on test set...")
        test_config = {**config, **task.test_overrides}
        test_examples = task.dataset_builder(test_config)
        test_repeats = test_config.get("test_repeats", 1)
        test_examples = test_examples * test_repeats
        logger.info(f"Test set: {len(test_examples)} problems ({len(test_examples) // test_repeats} unique × {test_repeats})")

        test_config["work_dir"] = str(args.output_dir / "test_work")
        test_evaluator = task.evaluator_factory(test_config)

        max_workers = config.get("max_workers") or max(1, min(32, (os.cpu_count() or 4) + 4) // 2)
        eval_timeout = test_config.get("eval_timeout", 300)

        logger.info(f"Test evaluation: {len(test_examples)} problems, {max_workers} workers")

        result = run_parallel_eval(
            test_evaluator, best_candidate, test_examples,
            max_workers=max_workers, eval_timeout=eval_timeout,
        )

        test_eval_cost = getattr(test_evaluator, "total_eval_cost", 0.0)
        test_results = {**result["test_results"], "test_eval_cost_usd": test_eval_cost}
        with open(args.output_dir / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)

    logger.info(f"Done. Results saved to {args.output_dir}")

    # Force-exit if any non-daemon threads are still alive
    alive = [t for t in threading.enumerate()
             if t is not threading.main_thread() and t.is_alive() and not t.daemon]
    if alive:
        names = ", ".join(t.name for t in alive)
        logger.info(f"Force-exiting ({len(alive)} non-daemon thread(s) still running: {names})")
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
