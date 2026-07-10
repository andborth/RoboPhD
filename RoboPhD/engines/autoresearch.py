"""Autoresearch engine for optimize_anything().

Wraps a single continuous Claude Code session with greedy hill-climbing
behind the optimize_anything() API, returning an OptimizeResult.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from RoboPhD.engines import build_val_split

if TYPE_CHECKING:
    from RoboPhD.api import AutoresearchConfig, OptimizeResult

logger = logging.getLogger(__name__)


def _assign_ids(examples: List[Dict], prefix: str = "") -> None:
    """Assign string IDs to examples if missing."""
    _ID_FIELDS = ("id", "question_id", "problem_id", "task_id")

    id_field = None
    if examples:
        for field in _ID_FIELDS:
            if field in examples[0]:
                id_field = field
                break

    for i, ex in enumerate(examples):
        if id_field and id_field != "id":
            ex["id"] = str(ex[id_field])
        elif "id" in ex:
            ex["id"] = str(ex["id"])
        else:
            ex["id"] = f"{prefix}{i}"


def _setup_workspace(
    workspace: Path,
    seed_candidate: dict,
    file_mapping: dict,
    objective: str,
    background: str,
    train_example_ids: list,
    val_size: int,
    evaluation_budget: int,
) -> None:
    """Initialize the workspace as a git repo with seed candidate and infrastructure."""
    from RoboPhD.autoresearch.evaluate_cli import __file__ as _evaluate_cli_source
    from RoboPhD.autoresearch.program import generate_program

    workspace.mkdir(parents=True, exist_ok=True)

    # Write seed candidate files
    for key, filepath in file_mapping.items():
        if key in seed_candidate:
            dest = workspace / filepath
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(seed_candidate[key])

    # Generate and write program.md
    program = generate_program(
        task=None, file_mapping=file_mapping,
        train_example_ids=train_example_ids,
        val_size=val_size, evaluation_budget=evaluation_budget,
    )
    (workspace / "program.md").write_text(program)

    # Copy evaluate CLI
    source = Path(_evaluate_cli_source)
    shutil.copy2(source, workspace / "_evaluate.py")

    # Write training example IDs
    (workspace / "_train_examples.json").write_text(json.dumps(train_example_ids, indent=2))

    # Write CLAUDE.md
    sections = []
    if background:
        sections.append(f"# Domain Background\n\n{background}")
    if objective:
        sections.append(f"# Domain Objective\n\n{objective}")
    sections.append("---\n\nRead `program.md` for the full autoresearch protocol.")
    claude_md = "\n\n".join(sections) + "\n"
    (workspace / "CLAUDE.md").write_text(claude_md)

    # Initialize git repo
    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "autoresearch",
        "GIT_AUTHOR_EMAIL": "autoresearch@robophd",
        "GIT_COMMITTER_NAME": "autoresearch",
        "GIT_COMMITTER_EMAIL": "autoresearch@robophd",
    }
    subprocess.run(["git", "init"], cwd=workspace, capture_output=True, check=True)
    subprocess.run(["git", "add", "-A"], cwd=workspace, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Seed candidate"],
        cwd=workspace, capture_output=True, check=True, env=git_env,
    )

    logger.info(f"Workspace initialized at {workspace}")


def _read_candidate_from_workspace(workspace: Path, file_mapping: dict) -> dict:
    """Read the current candidate files from workspace."""
    candidate = {}
    for key, filepath in file_mapping.items():
        src = workspace / filepath
        candidate[key] = src.read_text() if src.exists() else ""
    return candidate


def _find_claude_cli() -> Optional[str]:
    """Find Claude CLI executable."""
    try:
        from utilities.claude_cli import find_claude_cli
        cli = find_claude_cli()
        if cli:
            return cli
    except ImportError:
        pass

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


def run_autoresearch(
    evaluator: Callable,
    dataset: List[Dict],
    seed_candidate: Optional[Dict[str, str]],
    objective: str,
    background: str,
    cfg: AutoresearchConfig,
    task_name: str,
) -> OptimizeResult:
    """Run Autoresearch optimization and return an OptimizeResult."""
    from RoboPhD.api import OptimizeResult
    from RoboPhD.autoresearch.server import EvalServer
    from RoboPhD.config import CLAUDE_CLI_MODEL_MAP
    from RoboPhD.candidate_utils import materialize_candidate

    if not seed_candidate:
        raise ValueError("seed_candidate is required for Autoresearch")

    # Output directory
    run_dir = Path(cfg.parent_experiments_dir) if cfg.parent_experiments_dir else Path("../robophd_runs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = run_dir / "autoresearch" / f"{task_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build train/val split
    trainset, valset = build_val_split(
        dataset, cfg.val_dataset, cfg.val_size, cfg.seed,
    )

    # Assign IDs
    _assign_ids(trainset)
    _assign_ids(valset, prefix="val_")
    train_example_ids = [ex["id"] for ex in trainset]

    # File mapping
    file_mapping = {key: key for key in seed_candidate}

    # Save seed candidate
    with open(output_dir / "seed_candidate.json", "w") as f:
        json.dump(seed_candidate, f, indent=2)

    # Setup workspace
    workspace = output_dir / "workspace"
    _setup_workspace(
        workspace, seed_candidate, file_mapping, objective, background,
        train_example_ids, len(valset), cfg.evaluation_budget,
    )

    # Start eval server
    def get_candidate():
        return _read_candidate_from_workspace(workspace, file_mapping)

    eval_server = EvalServer(
        evaluator=evaluator,
        candidate_fn=get_candidate,
        train_examples=trainset,
        val_examples=valset,
        evaluation_budget=cfg.evaluation_budget,
        workspace=workspace,
        max_workers=cfg.max_workers,
        eval_timeout=cfg.eval_timeout,
    )
    eval_server.start()

    # Install the read-scope sandbox (parity with the RoboPhD engine). Without
    # this the session runs under bypassPermissions with unrestricted
    # filesystem access, letting it read the source repo, the task dataset, and
    # the evaluator/simulator source, then reconstruct the evaluator locally and
    # run unlimited free evaluations that bypass the evaluation budget entirely
    # (see the paper's Ethics Statement). The PreToolUse hook confines reads to
    # the workspace and writes to the workspace, fail-closed, and tails
    # <workspace>/sandbox_denials.jsonl into the run log.
    from RoboPhD.researcher import install_evolution_sandbox
    from RoboPhD.config import build_evolution_env
    from utilities.claude_cli import claude_cli_settings

    install_evolution_sandbox(workspace)
    # Read scope and write scope are both the workspace: the agent edits the
    # candidate files in place there and evaluates via the localhost EvalServer,
    # so it never legitimately needs to read outside the workspace.
    sandbox_env = build_evolution_env(cfg.model, workspace, workspace)
    # autoCompact + Read(/<repo>/**) deny, matching the RoboPhD evolution
    # session. The hook above is the robust layer (covers Bash); this closes the
    # agent's Read tool as defense in depth.
    session_settings = claude_cli_settings()

    # Find Claude CLI
    claude_cli = _find_claude_cli()
    if not claude_cli:
        eval_server.stop()
        raise RuntimeError(
            "Claude CLI not found. Install from: https://docs.anthropic.com/en/docs/claude-code"
        )

    cli_model = CLAUDE_CLI_MODEL_MAP.get(cfg.model, cfg.model)
    session_id = str(uuid.uuid4())

    logger.info(
        f"Starting autoresearch session (model={cli_model}, "
        f"budget={cfg.evaluation_budget}, timeout={cfg.overall_timeout})"
    )

    total_session_cost = 0.0
    total_session_tokens_in = 0
    total_session_tokens_out = 0

    # Run Claude Code session
    cmd = [
        claude_cli,
        "--model", cli_model,
        "--permission-mode", "bypassPermissions",
        "--output-format", "json",
        "--session-id", session_id,
        "--print", "Read program.md and begin the autoresearch protocol.",
        "--settings", session_settings,
    ]

    try:
        try:
            from utilities.claude_cli import call_claude_cli
        except ImportError:
            call_claude_cli = None

        if call_claude_cli:
            result = call_claude_cli(cmd, cwd=workspace, timeout=cfg.overall_timeout,
                                     logger=logger, extra_env=sandbox_env)
        else:
            result = subprocess.run(
                cmd, cwd=workspace, capture_output=True, text=True,
                timeout=cfg.overall_timeout, env={**os.environ, **sandbox_env},
            )
    except subprocess.TimeoutExpired:
        logger.warning(f"Claude Code session timed out after {cfg.overall_timeout}s")
        result = None
    except Exception as e:
        logger.error(f"Claude Code session error: {e}")
        result = None

    if result is not None:
        try:
            output = json.loads(result.stdout)
            call_cost = output.get("total_cost_usd", 0.0)
            if call_cost:
                total_session_cost += call_cost
            usage = output.get("usage", {})
            total_session_tokens_in += usage.get("input_tokens", 0)
            total_session_tokens_out += usage.get("output_tokens", 0)
        except (json.JSONDecodeError, TypeError):
            pass

    session_ok = result is not None and result.returncode == 0

    # Request reflection
    if session_ok:
        reflection_prompt = (
            "Thanks for your work on this. Looking back at the entire session, "
            "please write a brief reflection to `_reflection.md`. Consider:\n\n"
            "- What approaches worked well? What didn't?\n"
            "- What was surprising or unexpected about the problem?\n"
            "- What would you do differently with a fresh budget?\n\n"
            "Keep it concise — a few paragraphs, not an essay."
        )
        reflection_cmd = [
            claude_cli,
            "--model", cli_model,
            "--permission-mode", "bypassPermissions",
            "--output-format", "json",
            "--resume", session_id,
            "--print", reflection_prompt,
            "--settings", session_settings,
        ]
        try:
            if call_claude_cli:
                refl_result = call_claude_cli(reflection_cmd, cwd=workspace, timeout=300,
                                              logger=logger, extra_env=sandbox_env)
            else:
                refl_result = subprocess.run(
                    reflection_cmd, cwd=workspace, capture_output=True, text=True, timeout=300,
                    env={**os.environ, **sandbox_env},
                )
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

    eval_server.stop()

    # Collect results
    best_candidate = _read_candidate_from_workspace(workspace, file_mapping)

    with open(output_dir / "best_candidate.json", "w") as f:
        json.dump(best_candidate, f, indent=2)

    best_agent_dir = output_dir / "best_agent"
    materialize_candidate(best_candidate, best_agent_dir, file_mapping, name="autoresearch_best")

    # Read experiment log
    experiment_log = []
    log_path = workspace / "_experiment_log.jsonl"
    if log_path.exists():
        for line in log_path.read_text().strip().split("\n"):
            if line.strip():
                try:
                    experiment_log.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Copy reflection
    reflection_path = workspace / "_reflection.md"
    if reflection_path.exists():
        shutil.copy2(reflection_path, output_dir / "reflection.md")

    # Best val score
    best_val_score = 0.0
    if experiment_log:
        kept = [e for e in experiment_log if e.get("kept")]
        if kept:
            best_val_score = max(e.get("val_score") or 0.0 for e in kept)

    eval_cost = getattr(eval_server, "eval_cost", 0.0)

    # Save summary
    summary = {
        "task": task_name,
        "engine": "autoresearch",
        "objective": objective,
        "session_ok": session_ok,
        "evaluation_budget": cfg.evaluation_budget,
        "val_size": len(valset),
        "seed": cfg.seed,
        "model": cfg.model,
        "overall_timeout": cfg.overall_timeout,
        "train_size": len(trainset),
        "num_experiments": len(experiment_log),
        "best_val_score": best_val_score,
        "cost": {
            "eval_cost_usd": round(eval_cost, 2),
            "evolution_cost_usd": round(total_session_cost, 2),
            "total_cost_usd": round(eval_cost + total_session_cost, 2),
        },
    }
    with open(output_dir / "optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Optimization complete. {len(experiment_log)} experiments, "
        f"best val score: {best_val_score:.3f}"
    )

    return OptimizeResult(
        best_candidate=best_candidate,
        best_score=best_val_score,
        experiment_dir=output_dir,
        all_candidates=[{
            "name": "autoresearch_best",
            "candidate": best_candidate,
            "elo": 0,
            "mean_score": best_val_score,
            "test_count": 0,
        }],
        num_iterations_completed=len(experiment_log),
        total_evaluations=getattr(evaluator, "total_evaluations", 0),
        completed_normally=session_ok,
    )
