#!/usr/bin/env python3
"""
Run critic evaluation on preprocessed LiveCodeBench problems.

Evaluates a critic agent by:
1. Running critic on solution.py → feedback.md
2. Running coder revision with feedback → solution_v2.py
3. Comparing Pass@1 before/after critic

Usage:
    python RoboPhD/tools/run_critic_evaluation.py \
        --coder-model opus-4.5 \
        --critic-model haiku-4.5 \
        --critic-agent RoboPhD/codegen_agents/naive_critic \
        --test-set \
        --limit 10

    # Same model for coder and critic:
    python RoboPhD/tools/run_critic_evaluation.py \
        --coder-model opus-4.5 \
        --critic-agent RoboPhD/codegen_agents/naive_critic \
        --test-set
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
# Add grandparent directory to path for utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Reuse evaluation logic from evaluate_livecodebench
from evaluate_livecodebench import evaluate_single, load_dataset
from utilities.claude_cli import call_claude_cli, RateLimitExceeded
from RoboPhD.config import CLAUDE_CLI_MODEL_MAP, LMSTUDIO_DEFAULT_BASE_URL

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Date cutoff for test set
TEST_SET_CUTOFF = "2024-11-01"

# Prompts for coder regeneration
CALL_1_PROMPT = """Read problem.md and create solution.py to solve it.

Requirements:
- If examples show "Input: variable = value" format and ask you to "Return" something:
  Create a Solution class with the appropriate method (e.g., `class Solution: def solve(self, nums): ...`)
- If the problem has "Input" and "Output" sections describing line-by-line format:
  Create a standalone script that reads from stdin and writes to stdout
- Code must run in under 6 seconds on every test case
- Include brief comments explaining your approach"""

# Alias for local use — canonical map lives in RoboPhD.config
MODEL_MAP = CLAUDE_CLI_MODEL_MAP


def get_cli_model(model_version: str) -> str:
    """Map versioned model name to Claude CLI model name."""
    return CLAUDE_CLI_MODEL_MAP.get(model_version, model_version)


def validate_model_version(model: str, param_name: str) -> None:
    """
    Validate that model name is a versioned name from MODEL_MAP.

    Raises ValueError if an Anthropic-looking model name (starts with haiku,
    sonnet, opus, or claude) is not in MODEL_MAP. Non-Anthropic models
    (e.g., 'qwen/qwen3-coder-30b') are accepted as-is since they already
    have unique names for cache isolation.
    """
    _ANTHROPIC_PREFIXES = ('haiku', 'sonnet', 'opus', 'claude')
    # Non-Anthropic models skip validation
    if not any(model.startswith(p) for p in _ANTHROPIC_PREFIXES):
        return
    # Anthropic model - must be a known versioned name
    if model not in MODEL_MAP:
        valid_names = ", ".join(sorted(MODEL_MAP.keys()))
        raise ValueError(
            f"Invalid {param_name}: '{model}'. "
            f"Must use versioned model name for cache isolation. "
            f"Valid names: {valid_names}"
        )


def parse_verdict(feedback_path: Path) -> Optional[bool]:
    """
    Parse verdict from feedback.md.

    Returns:
        True if VERDICT: CORRECT
        False if VERDICT: INCORRECT
        None if invalid/missing format
    """
    if not feedback_path.exists():
        return None

    content = feedback_path.read_text().strip()
    if not content:
        return None

    first_line = content.split('\n')[0].strip().upper()
    if first_line == "VERDICT: CORRECT":
        return True
    elif first_line == "VERDICT: INCORRECT":
        return False
    return None


def parse_acceptance(acceptance_path: Path) -> Tuple[str, str]:
    """
    Parse acceptance.md for category and explanation.

    Scans the first 5 lines for the category keyword to handle
    cases where the model adds headers before the category.

    Returns:
        (category, explanation) where category is one of:
        "accepted_all", "accepted_some", "rejected_all", "invalid"
    """
    if not acceptance_path.exists():
        return "invalid", ""

    content = acceptance_path.read_text().strip()
    if not content:
        return "invalid", ""

    lines = content.split('\n')
    category_map = {
        "ACCEPTED_ALL": "accepted_all",
        "ACCEPTED_SOME": "accepted_some",
        "REJECTED_ALL": "rejected_all",
    }

    # Scan first 5 lines for category keyword (anywhere in line)
    for i, line in enumerate(lines[:5]):
        line_upper = line.strip().upper()
        for keyword, category in category_map.items():
            if keyword in line_upper:
                explanation = '\n'.join(lines[i+1:]).strip()
                return category, explanation

    return "invalid", content


class CriticEvaluator:
    """Evaluates critic agents on preprocessed problems."""

    def __init__(
        self,
        cache_dir: Path,
        critic_agent_dir: Path,
        output_dir: Path,
        coder_model: str,
        critic_model: str,
        codegen_timeout: int = 1200,
        critic_timeout: int = 600,
        lmstudio_base_url: str = LMSTUDIO_DEFAULT_BASE_URL,
        revision_mode: str = "fork",
    ):
        self.cache_dir = cache_dir
        self.critic_agent_dir = critic_agent_dir
        self.output_dir = output_dir
        self.coder_model = get_cli_model(coder_model)
        self.critic_model = get_cli_model(critic_model)
        self.codegen_timeout = codegen_timeout
        self.critic_timeout = critic_timeout
        self.lmstudio_base_url = lmstudio_base_url
        self.revision_mode = revision_mode

        # Precompute LM Studio env overrides for coder and critic models
        # None means the model uses standard Anthropic API (no overrides needed)
        from RoboPhD.config import get_lmstudio_env
        self.coder_extra_env = get_lmstudio_env(coder_model, lmstudio_base_url)
        self.critic_extra_env = get_lmstudio_env(critic_model, lmstudio_base_url)

        # Find Claude CLI
        self.claude_path = self._find_claude_cli()

        # Load critic agent artifacts
        self.agent_md = self._load_agent_md()
        self.eval_instructions = self._load_eval_instructions()

    def _find_claude_cli(self) -> str:
        """Find Claude CLI path."""
        local_claude = Path.home() / ".claude" / "local" / "claude"
        if local_claude.exists():
            return str(local_claude)
        # Try PATH
        result = subprocess.run(["which", "claude"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        raise RuntimeError("Claude CLI not found")

    def _load_agent_md(self) -> str:
        """Load agent.md content."""
        agent_md_path = self.critic_agent_dir / "agent.md"
        if agent_md_path.exists():
            return agent_md_path.read_text()
        return ""

    def _load_eval_instructions(self) -> str:
        """Load eval_instructions.md content."""
        eval_path = self.critic_agent_dir / "eval_instructions.md"
        if eval_path.exists():
            return eval_path.read_text()
        return ""

    def _parse_agent_config(self) -> Dict:
        """Parse YAML frontmatter from agent.md."""
        import re
        match = re.match(r'^---\n(.*?)\n---', self.agent_md, re.DOTALL)
        if match:
            import yaml
            return yaml.safe_load(match.group(1)) or {}
        return {}

    def _run_claude_code(
        self,
        prompt: str,
        working_dir: Path,
        model: str,
        timeout: Optional[int] = None,
        session_id: Optional[str] = None,
        fork_session: bool = False,
        extra_dirs: Optional[List[Path]] = None,
        deny_edit_paths: Optional[List[Path]] = None,
        context: str = "",
        extra_env: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict, str, Dict]:
        """
        Run Claude Code CLI.

        Args:
            working_dir: Directory to run from (determines session lookup).
            fork_session: If True and session_id provided, creates a new session
                         branching from the original (preserves original session).
            extra_dirs: Additional directories to add via --add-dir (for file access).
            deny_edit_paths: Directories to deny Edit tool access to (protects cache).
            extra_env: Optional environment variable overrides for the subprocess
                      (e.g., ANTHROPIC_BASE_URL for LM Studio models).

        Returns (result_dict, session_id, cost_info).
            cost_info contains: cost_usd, input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens
        """
        ctx = f" [{context}]" if context else ""

        # Build settings with optional deny permissions
        settings = {"env": {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000"}}
        if deny_edit_paths:
            # Use /** glob to deny entire directories
            # // prefix means absolute path from filesystem root (//tmp = /tmp)
            # Must strip leading / from resolved path to avoid ///path
            settings["permissions"] = {
                "deny": [f"Edit(//{str(p.resolve()).lstrip('/')}/**)" for p in deny_edit_paths]
            }

        # Use acceptEdits (not bypassPermissions) so deny rules are respected
        permission_mode = "acceptEdits" if deny_edit_paths else "bypassPermissions"

        cmd = [
            self.claude_path, "--print", "--output-format", "json",
            "--append-system-prompt", "Do not read or write MEMORY.md files.",
            "--permission-mode", permission_mode,
            "--settings", json.dumps(settings),
            "--add-dir", str(working_dir.resolve()),
            "--model", model,
        ]

        # Add extra directories for file access
        if extra_dirs:
            for d in extra_dirs:
                cmd.extend(["--add-dir", str(d.resolve())])

        if session_id:
            cmd.extend(["--resume", session_id])
            if fork_session:
                cmd.append("--fork-session")

        cmd.extend(["-p", prompt])

        effective_timeout = timeout if timeout is not None else self.codegen_timeout

        try:
            result = call_claude_cli(
                cmd=cmd,
                cwd=working_dir,
                timeout=effective_timeout,
                logger=logger,
                extra_env=extra_env
            )

            # Try to parse JSON even on non-zero return code
            # Claude CLI may return useful info (like session_id) even on errors
            output = {}
            if result.stdout:
                try:
                    output = json.loads(result.stdout)
                except json.JSONDecodeError:
                    pass

            # Extract cost info from parsed JSON
            usage = output.get('usage', {})
            cost_info = {
                'cost_usd': output.get('total_cost_usd', 0.0),
                'input_tokens': usage.get('input_tokens', 0),
                'output_tokens': usage.get('output_tokens', 0),
                'cache_creation_tokens': usage.get('cache_creation_input_tokens', 0),
                'cache_read_tokens': usage.get('cache_read_input_tokens', 0),
            }

            if result.returncode != 0:
                logger.warning(f"Claude CLI exited with code {result.returncode}{ctx}")
                if result.stderr:
                    truncated = "..." if len(result.stderr) > 500 else ""
                    logger.warning(f"stderr{ctx}: {result.stderr[:500]}{truncated}")
                # Still return parsed output, session_id, and cost_info if available
                return output, output.get("session_id", ""), cost_info

            return output, output.get("session_id", ""), cost_info

        except subprocess.TimeoutExpired:
            logger.warning(f"Claude CLI timed out after {effective_timeout}s{ctx}")
            return {}, "", {}
        except RateLimitExceeded:
            # Let rate limit exceeded propagate for checkpoint/exit handling
            raise
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Claude output{ctx}: {e}")
            return {}, "", {}
        except Exception as e:
            logger.warning(f"Claude CLI error{ctx}: {e}")
            return {}, "", {}

    def check_session_valid(self, session_id: str, working_dir: Path) -> bool:
        """Check if a session is still valid by attempting a minimal resume.

        Must use the same working_dir where the session was created,
        since Claude CLI stores session data in .claude/ within the working directory.
        """
        if not session_id:
            return False

        # Try to resume with a no-op prompt, using fork to avoid polluting original
        result, _, _ = self._run_claude_code(
            "Say 'ok'",  # Minimal prompt
            working_dir=working_dir,
            model=self.coder_model,
            timeout=self.codegen_timeout,
            session_id=session_id,
            fork_session=True,  # Don't pollute original session
            context="session-check",
            extra_env=self.coder_extra_env,
        )

        # If we got an empty result, check if it's due to session expiry
        # The _run_claude_code logs the error and returns {}, "", {}
        return bool(result)

    def regenerate_coder(self, workspace: Path, question_id: str) -> Tuple[str, Dict]:
        """
        Regenerate solution.py with fresh session.

        Workspace must already have problem.md symlinked.
        Returns (session_id, codegen_timing). Raises RuntimeError on failure.
        """
        start = time.time()

        logger.info(f"  [regen-call1:{question_id}] generating solution.py")

        # Call 1: Generate solution
        result, session_id, cost1 = self._run_claude_code(
            CALL_1_PROMPT,
            working_dir=workspace,
            model=self.coder_model,
            timeout=self.codegen_timeout,
            context=f"regen-call1:{question_id}",
            extra_env=self.coder_extra_env,
        )

        if not session_id:
            raise RuntimeError(f"Call 1 failed for {question_id}: no session_id returned")

        # Verify files were created
        if not (workspace / "solution.py").exists():
            raise RuntimeError(f"solution.py not created for {question_id}")

        elapsed = time.time() - start
        codegen_timing = {
            'elapsed_seconds': elapsed,
            'cost_usd': cost1.get('cost_usd', 0),
            'input_tokens': cost1.get('input_tokens', 0),
            'output_tokens': cost1.get('output_tokens', 0),
            'cache_creation_tokens': cost1.get('cache_creation_tokens', 0),
            'cache_read_tokens': cost1.get('cache_read_tokens', 0),
        }

        return session_id, codegen_timing

    def run_critic(
        self,
        cache_problem_dir: Path,
        output_problem_dir: Path,
        question_id: str,
    ) -> Tuple[Optional[Path], Dict]:
        """
        Run critic agent on a problem.

        Runs directly from output_problem_dir (no workspace subdirectory).
        Expects problem.md and solution.py already present.

        Returns (path to feedback.md, timing_info), or (None, timing_info) on failure.
        """
        start = time.time()
        total_cost = {}

        output_problem_dir.mkdir(parents=True, exist_ok=True)
        feedback_path = output_problem_dir / "feedback.md"

        # Copy critic tools if they exist (and not already copied)
        critic_tools = self.critic_agent_dir / "tools"
        output_tools = output_problem_dir / "tools"
        if critic_tools.exists() and not output_tools.exists():
            shutil.copytree(critic_tools, output_tools)

        # Run tool-only mode if configured
        config = self._parse_agent_config()
        agent_analysis = ""

        if config.get("execution_mode") == "tool_only":
            tool_cmd = config.get("tool_command", "")
            if tool_cmd:
                try:
                    # Note: shell=True is safe here since tool_cmd comes from
                    # trusted agent.md configs, not user input
                    result = subprocess.run(
                        tool_cmd,
                        shell=True,
                        cwd=output_problem_dir,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    # Read tool output
                    tool_output_file = config.get("tool_output_file", "tool_output/analysis.txt")
                    tool_output_path = output_problem_dir / tool_output_file
                    if tool_output_path.exists():
                        agent_analysis = tool_output_path.read_text()
                except Exception as e:
                    logger.warning(f"Tool execution failed for {question_id}: {e}")

        # Build critic prompt
        prompt = self._build_critic_prompt(agent_analysis)

        # Save prompt for debugging/analysis
        prompt_path = output_problem_dir / "critic_prompt.md"
        prompt_path.write_text(prompt)

        # Run Claude to generate feedback directly in output_problem_dir
        result, session_id, cost_info = self._run_claude_code(
            prompt,
            working_dir=output_problem_dir,
            model=self.critic_model,
            timeout=self.critic_timeout,
            context=f"critic:{question_id}",
            extra_env=self.critic_extra_env,
        )
        total_cost = cost_info.copy()

        # Validate verdict format, re-prompt once if invalid
        verdict = parse_verdict(feedback_path)
        if verdict is None and session_id:
            if not feedback_path.exists():
                fix_prompt = (
                    f"feedback.md was NOT created in the working directory. "
                    f"The file must be at: {feedback_path} "
                    f"Please create feedback.md in the current working directory (not the scratchpad) "
                    f"starting with exactly 'VERDICT: CORRECT' or 'VERDICT: INCORRECT' on the first line."
                )
            else:
                fix_prompt = (
                    "feedback.md exists but is missing the required verdict line. "
                    "Please rewrite feedback.md starting with exactly 'VERDICT: CORRECT' or "
                    "'VERDICT: INCORRECT' on the first line, followed by your analysis."
                )
            _, _, retry_cost = self._run_claude_code(
                fix_prompt,
                working_dir=output_problem_dir,
                model=self.critic_model,
                timeout=self.critic_timeout,
                session_id=session_id,
                context=f"critic-retry:{question_id}",
                extra_env=self.critic_extra_env,
            )
            # Accumulate retry cost
            for key in ['cost_usd', 'input_tokens', 'output_tokens', 'cache_creation_tokens', 'cache_read_tokens']:
                total_cost[key] = total_cost.get(key, 0) + retry_cost.get(key, 0)

            verdict = parse_verdict(feedback_path)
            if verdict is None:
                logger.warning(
                    f"Invalid verdict format after retry for {question_id}, treating as INCORRECT"
                )

        elapsed = time.time() - start
        timing_info = {
            'elapsed_seconds': elapsed,
            **total_cost
        }

        return feedback_path, timing_info

    def _build_critic_prompt(self, agent_analysis: str) -> str:
        """Build the prompt for the critic.

        Instructions come first so they form a cacheable prefix (Anthropic's
        prompt cache is prefix-based).  The variable per-problem analysis
        follows as the suffix.
        """
        parts = []

        parts.append(f"## Instructions\n\n{self.eval_instructions}")

        if agent_analysis:
            parts.append(f"## Analysis\n\n{agent_analysis}")

        parts.append(
            "## Task\n\n"
            "Review solution.py against problem.md. "
            "Create feedback.md in the current working directory (not the scratchpad) "
            "starting with VERDICT: CORRECT or VERDICT: INCORRECT "
            "on the first line, followed by your analysis."
        )

        return "\n\n".join(parts)

    def run_revision(
        self,
        cache_problem_dir: Path,
        output_problem_dir: Path,
        session_id: str,
        question_id: str,
    ) -> Tuple[Path, str, Dict]:
        """
        Run coder revision with feedback.

        Runs from cache_problem_dir (for session lookup) with deny_edit_paths
        to protect the cache. Writes directly to output_problem_dir/solution_v2.py.
        Uses fork_session to preserve original cached session.

        Returns (path to solution_v2.py, forked_session_id, timing_info).
        """
        start = time.time()

        feedback_path = output_problem_dir / "feedback.md"
        solution_v2_path = output_problem_dir / "solution_v2.py"

        # Read feedback for revision prompt
        feedback = feedback_path.read_text().strip() if feedback_path.exists() else ""

        # Build revision prompt - Claude writes directly to solution_v2.py
        # deny_edit_paths prevents modification of cache files
        prompt = f"""Here is feedback from a code reviewer:

<feedback>
{feedback}
</feedback>

You may accept all, some, or none of this feedback.
If you make changes, write your revised solution to: {solution_v2_path.resolve()}
If no changes needed, confirm the code is correct."""

        # Save prompt for debugging/analysis
        prompt_path = output_problem_dir / "revision_prompt.md"
        prompt_path.write_text(prompt)

        # Run revision from cache_problem_dir (for session) with cache protected
        # Uses coder_model (required: resumes coder's session) with critic_timeout
        # (intentional: revision responsiveness reflects critic quality)
        result, forked_session_id, cost_info = self._run_claude_code(
            prompt,
            working_dir=cache_problem_dir,
            model=self.coder_model,
            timeout=self.critic_timeout,
            session_id=session_id,
            fork_session=True,  # Fork to preserve original cached session
            extra_dirs=[output_problem_dir],  # Allow writes to output dir
            deny_edit_paths=[cache_problem_dir],  # Protect entire cache dir
            context=f"revision:{question_id}",
            extra_env=self.coder_extra_env,
        )

        # If Claude completed but no solution_v2.py was created (no changes needed), copy original
        # On timeout (empty forked_session_id), do NOT fallback — let it be an error
        if forked_session_id and not solution_v2_path.exists():
            cache_solution = cache_problem_dir / "solution.py"
            if cache_solution.exists():
                shutil.copy(cache_solution, solution_v2_path)
                logger.info(f"No revision made, using original for {question_id}")

        elapsed = time.time() - start
        timing_info = {
            'elapsed_seconds': elapsed,
            **cost_info
        }

        return solution_v2_path, forked_session_id, timing_info

    def run_revision_fresh(
        self,
        output_problem_dir: Path,
        question_id: str,
    ) -> Tuple[Path, str, Dict]:
        """
        Run coder revision in a fresh session (no session forking).

        Two sequential calls in one fresh session:
        1. Read problem.md, solution.py, reflection.md — summarize understanding
        2. Present critic feedback — optionally revise to solution_v2.py

        The coder internalizes the solution before seeing feedback, making it
        less likely to blindly accept critic suggestions.

        Returns (path to solution_v2.py, session_id, timing_info).
        """
        start = time.time()

        feedback_path = output_problem_dir / "feedback.md"
        solution_v2_path = output_problem_dir / "solution_v2.py"

        # Call 1: Establish understanding of the solution
        call1_prompt = (
            "Read problem.md, solution.py, and reflection.md. "
            "Summarize your understanding of the problem and the solution approach."
        )

        _, session_id, cost1 = self._run_claude_code(
            call1_prompt,
            working_dir=output_problem_dir,
            model=self.coder_model,
            timeout=self.critic_timeout,
            context=f"fresh-understand:{question_id}",
            extra_env=self.coder_extra_env,
        )

        if not session_id:
            logger.warning(f"Fresh revision call 1 failed for {question_id}: no session_id")
            elapsed = time.time() - start
            return solution_v2_path, "", {"elapsed_seconds": elapsed, **cost1}

        # Call 2: Present feedback and request optional revision
        feedback = feedback_path.read_text().strip() if feedback_path.exists() else ""

        call2_prompt = f"""Here is feedback from a code reviewer:

<feedback>
{feedback}
</feedback>

You may accept all, some, or none of this feedback.
If you make changes, write your revised solution to solution_v2.py.
If no changes needed, confirm the code is correct."""

        _, session_id_2, cost2 = self._run_claude_code(
            call2_prompt,
            working_dir=output_problem_dir,
            model=self.coder_model,
            timeout=self.critic_timeout,
            session_id=session_id,
            context=f"fresh-revise:{question_id}",
            extra_env=self.coder_extra_env,
        )

        # If completed but no solution_v2.py, copy original (no changes needed)
        effective_session = session_id_2 or session_id
        if effective_session and not solution_v2_path.exists():
            original = output_problem_dir / "solution.py"
            if original.exists():
                shutil.copy(original, solution_v2_path)
                logger.info(f"No revision made, using original for {question_id}")

        elapsed = time.time() - start
        # Accumulate costs from both calls
        timing_info = {
            "elapsed_seconds": elapsed,
            "cost_usd": cost1.get("cost_usd", 0) + cost2.get("cost_usd", 0),
            "input_tokens": cost1.get("input_tokens", 0) + cost2.get("input_tokens", 0),
            "output_tokens": cost1.get("output_tokens", 0) + cost2.get("output_tokens", 0),
            "cache_creation_tokens": cost1.get("cache_creation_tokens", 0) + cost2.get("cache_creation_tokens", 0),
            "cache_read_tokens": cost1.get("cache_read_tokens", 0) + cost2.get("cache_read_tokens", 0),
        }

        return solution_v2_path, effective_session, timing_info

    def query_acceptance(
        self,
        cache_problem_dir: Path,
        output_problem_dir: Path,
        session_id: str,
        question_id: str,
        working_dir: Optional[Path] = None,
    ) -> Tuple[Dict, Dict]:
        """
        Post-hoc query (Call 3): which suggestions did you accept?

        Runs from cache_problem_dir (for session) with deny_edit_paths to protect cache.
        When working_dir is provided, runs from there instead (fresh mode).
        Writes acceptance.md to output_problem_dir.

        Returns (acceptance_dict, timing_info) where acceptance_dict contains:
            category: "accepted_all"/"accepted_some"/"rejected_all"/"invalid"
            explanation: free-form text explaining decisions
        """
        start = time.time()

        logger.info(f"Querying acceptance for {question_id}")

        wd = working_dir or cache_problem_dir
        acceptance_path = output_problem_dir / "acceptance.md"

        # Build prompt - write directly to output dir
        if working_dir:
            # Fresh mode: session lives in output_problem_dir, write locally
            prompt = f"""Reflect on the feedback you received and your revision.

Create acceptance.md with the first line being exactly one of:
ACCEPTED_ALL
ACCEPTED_SOME
REJECTED_ALL

Then explain briefly what you accepted or rejected and why."""
        else:
            prompt = f"""Reflect on the feedback you received and your revision.

Create {acceptance_path.resolve()} with the first line being exactly one of:
ACCEPTED_ALL
ACCEPTED_SOME
REJECTED_ALL

Then explain briefly what you accepted or rejected and why."""

        # Save prompt for debugging/analysis
        prompt_path = output_problem_dir / "acceptance_prompt.md"
        prompt_path.write_text(prompt)

        extra_dirs = [output_problem_dir] if not working_dir else None
        deny_edit = [cache_problem_dir] if not working_dir else None

        result, _, cost_info = self._run_claude_code(
            prompt,
            working_dir=wd,
            model=self.coder_model,
            timeout=self.critic_timeout,
            session_id=session_id,
            extra_dirs=extra_dirs,
            deny_edit_paths=deny_edit,
            context=f"acceptance:{question_id}",
            extra_env=self.coder_extra_env,
        )

        category, explanation = parse_acceptance(acceptance_path)

        elapsed = time.time() - start
        timing_info = {
            'elapsed_seconds': elapsed,
            **cost_info
        }

        return {"category": category, "explanation": explanation}, timing_info

    def evaluate_problem(
        self,
        cache_problem_dir: Path,
        output_problem_dir: Path,
        problem_data: Dict,
        question_id: str,
    ) -> Dict:
        """
        Run full critic evaluation on a single problem.

        - Critic runs directly from output_problem_dir (writes feedback.md)
        - Revision runs from cache (session), denies edits to cache/**, writes to output/solution_v2.py
        - Acceptance runs from cache (session), denies edits to cache/**, writes to output/acceptance.md

        Returns result dict with v1 and v2 pass status.
        """
        output_problem_dir.mkdir(parents=True, exist_ok=True)

        # Load meta
        meta_path = cache_problem_dir / "meta.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        regenerated = False
        codegen_timing = None

        logger.info(f"Evaluating {question_id}...")

        if self.revision_mode == "fresh":
            # Fresh mode: skip Phase 1 entirely (no session check, no regeneration)
            session_id = ""
        else:
            # Step 1: Ensure valid session (regenerate if needed)
            session_id = meta.get("session_id", "")

            if not self.check_session_valid(session_id, cache_problem_dir):
                logger.info(f"Session expired for {question_id}, regenerating...")
                session_id, codegen_timing = self.regenerate_coder(cache_problem_dir, question_id)
                regenerated = True

                # Update meta.json with new session_id and codegen_timing
                meta["session_id"] = session_id
                meta["codegen_timing"] = codegen_timing
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
            else:
                # Load codegen_timing from meta if available (from prior regeneration)
                codegen_timing = meta.get("codegen_timing")

        # Symlink problem.md (static, from dataset)
        problem_dest = output_problem_dir / "problem.md"
        problem_src = cache_problem_dir / "problem.md"
        if problem_src.exists() and not problem_dest.exists():
            problem_dest.symlink_to(problem_src.resolve())

        # Copy solution.py (generated, may change if cache rebuilt)
        solution_v1_path = output_problem_dir / "solution.py"
        cache_solution = cache_problem_dir / "solution.py"
        if cache_solution.exists():
            shutil.copy(cache_solution, solution_v1_path)

        # Copy reflection.md from cache (needed for fresh mode)
        if self.revision_mode == "fresh":
            cache_reflection = cache_problem_dir / "reflection.md"
            reflection_dest = output_problem_dir / "reflection.md"
            if cache_reflection.exists() and not reflection_dest.exists():
                shutil.copy(cache_reflection, reflection_dest)

        # Step 2: Run critic
        logger.info(f"Running critic on {question_id}")
        feedback_path, critic_timing = self.run_critic(cache_problem_dir, output_problem_dir, question_id)

        # Check if feedback.md was actually created (handles timeout/failure cases)
        if feedback_path is None or not feedback_path.exists():
            return {
                "question_id": question_id,
                "error": "critic_failed",
                "regenerated": regenerated,
                "timing": {
                    "codegen": codegen_timing,
                    "critic": critic_timing,
                    "revision": None,
                    "acceptance": None,
                }
            }

        # Parse verdict
        verdict = parse_verdict(feedback_path)

        # Step 3: Run revision if needed (session guaranteed valid)
        solution_v2_path = output_problem_dir / "solution_v2.py"

        # Initialize tracking variables
        revision_attempted = False
        revision_failed_reason = None
        revision_timing = None
        acceptance_timing = None

        if verdict is True:
            # CORRECT - no revision needed, symlink v2 to v1
            logger.info(f"Verdict CORRECT for {question_id}, skipping revision")
            if solution_v2_path.exists() or solution_v2_path.is_symlink():
                solution_v2_path.unlink()
            solution_v2_path.symlink_to("solution.py")
            acceptance = None  # No revision, skip Call 3
        else:
            # INCORRECT - apply revision
            logger.info(f"Running revision on {question_id}")
            revision_attempted = True
            revision_failed_reason = None

            try:
                if self.revision_mode == "fresh":
                    # Fresh mode: two-call session in output_problem_dir
                    _, fresh_session_id, revision_timing = self.run_revision_fresh(
                        output_problem_dir, question_id
                    )

                    if not fresh_session_id:
                        logger.warning(f"Fresh revision failed for {question_id}: no session")
                        revision_failed_reason = "timeout_or_no_session"
                        acceptance = None
                    else:
                        acceptance, acceptance_timing = self.query_acceptance(
                            cache_problem_dir, output_problem_dir, fresh_session_id, question_id,
                            working_dir=output_problem_dir,
                        )
                else:
                    # Fork mode: resume coder's cached session
                    _, forked_session_id, revision_timing = self.run_revision(
                        cache_problem_dir, output_problem_dir, session_id, question_id
                    )

                    if not forked_session_id:
                        logger.warning(f"Revision failed for {question_id}: timeout or no session")
                        revision_failed_reason = "timeout_or_no_session"
                        acceptance = None
                    else:
                        # Step 3.5: Query acceptance (Call 3) on forked session
                        acceptance, acceptance_timing = self.query_acceptance(
                            cache_problem_dir, output_problem_dir, forked_session_id, question_id
                        )
            except Exception as e:
                logger.warning(f"Revision failed for {question_id}: {e}")
                revision_failed_reason = str(e)
                acceptance = None
                # Don't create v2 - missing v2 signals revision failure for --resume

        # Step 4: Evaluate v1 and v2 (using explicit paths, no file swapping)
        v1_result = evaluate_single(output_problem_dir, problem_data, timeout=6, solution_path=solution_v1_path)

        # Retry once on timeout to handle transient issues
        if v1_result.get("reason", "").startswith("TIMEOUT"):
            v1_retry = evaluate_single(output_problem_dir, problem_data, timeout=6, solution_path=solution_v1_path)
            if not v1_retry.get("reason", "").startswith("TIMEOUT"):
                v1_result = v1_retry  # Use successful retry

        # If v2 is a symlink to v1, reuse v1's result (no revision happened)
        if solution_v2_path.is_symlink() and solution_v2_path.resolve() == solution_v1_path.resolve():
            v2_result = v1_result
        elif solution_v2_path.exists():
            v2_result = evaluate_single(output_problem_dir, problem_data, timeout=6, solution_path=solution_v2_path)

            # Retry once on timeout to handle transient issues
            if v2_result.get("reason", "").startswith("TIMEOUT"):
                v2_retry = evaluate_single(output_problem_dir, problem_data, timeout=6, solution_path=solution_v2_path)
                if not v2_retry.get("reason", "").startswith("TIMEOUT"):
                    v2_result = v2_retry  # Use successful retry
        else:
            # Revision failed - no v2 to evaluate
            v2_result = {"passed": False, "reason": "revision_failed"}

        # Calculate timing totals
        def get_timing_value(timing_dict, key, default=0):
            """Safely get timing value, returning default if timing_dict is None or key missing."""
            if timing_dict is None:
                return default
            return timing_dict.get(key, default) or default

        total_seconds = (
            get_timing_value(codegen_timing, 'elapsed_seconds') +
            get_timing_value(critic_timing, 'elapsed_seconds') +
            get_timing_value(revision_timing, 'elapsed_seconds') +
            get_timing_value(acceptance_timing, 'elapsed_seconds')
        )
        total_cost_usd = (
            get_timing_value(codegen_timing, 'cost_usd') +
            get_timing_value(critic_timing, 'cost_usd') +
            get_timing_value(revision_timing, 'cost_usd') +
            get_timing_value(acceptance_timing, 'cost_usd')
        )

        result = {
            "question_id": question_id,
            "v1_passed": v1_result["passed"],
            "v2_passed": v2_result["passed"],
            "v1_reason": v1_result.get("reason", ""),
            "v2_reason": v2_result.get("reason", ""),
            "verdict_correct": verdict is True,  # VERDICT: CORRECT
            "regenerated": regenerated,
            "improved": not v1_result["passed"] and v2_result["passed"],
            "regressed": v1_result["passed"] and not v2_result["passed"],
            "acceptance_category": acceptance["category"] if acceptance else None,
            "acceptance_explanation": acceptance["explanation"] if acceptance else None,
            # Revision tracking
            "revision_attempted": revision_attempted,
            "revision_completed": acceptance is not None,  # True if acceptance query ran
            "revision_failed_reason": revision_failed_reason,
            # Timing and cost
            "timing": {
                "codegen": codegen_timing,
                "critic": critic_timing,
                "revision": revision_timing,
                "acceptance": acceptance_timing,
                "total_seconds": total_seconds,
                "total_cost_usd": total_cost_usd,
            }
        }

        # Write result.json immediately (crash-safe)
        result_path = output_problem_dir / "result.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        return result


def load_problems_from_cache(cache_dir: Path) -> List[Tuple[Path, Dict]]:
    """Load problem directories and their metadata from cache."""
    problems = []
    for problem_dir in sorted(cache_dir.iterdir()):
        if not problem_dir.is_dir():
            continue
        meta_path = problem_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            problems.append((problem_dir, meta))
        except (json.JSONDecodeError, IOError):
            continue
    return problems


def ensure_cache_entry(cache_dir: Path, question_id: str, problem_data: Dict) -> Path:
    """
    Ensure cache directory exists for a problem with problem.md and meta.json.

    Creates the directory and files if they don't exist.
    Returns the problem directory path.
    """
    problem_dir = cache_dir / question_id
    problem_dir.mkdir(parents=True, exist_ok=True)

    # Create problem.md if it doesn't exist
    problem_md = problem_dir / "problem.md"
    if not problem_md.exists():
        question_content = problem_data.get("question_content", "")
        starter_code = problem_data.get("starter_code", "")

        # Include starter code section for LeetCode functional problems
        if starter_code:
            content = f"{question_content}\n\n## Starter Code\n\n```python\n{starter_code}\n```"
        else:
            content = question_content

        problem_md.write_text(content)

    # Create meta.json if it doesn't exist
    meta_path = problem_dir / "meta.json"
    if not meta_path.exists():
        meta = {
            "question_id": question_id,
            "contest_date": problem_data.get("contest_date", ""),
            "question_title": problem_data.get("question_title", ""),
            "difficulty": problem_data.get("difficulty", ""),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    return problem_dir


def load_problems_from_dataset(
    dataset: Dict[str, Dict],
    cache_dir: Path,
    test_set: bool = False,
    evolution_set: bool = False,
) -> List[Tuple[Path, Dict]]:
    """
    Load problems from dataset, creating cache entries as needed.

    Filters by date and ensures cache directories exist.
    Returns list of (problem_dir, meta) tuples.
    """
    problems = []

    for question_id, problem_data in sorted(dataset.items()):
        contest_date = problem_data.get("contest_date", "")

        # Apply date filter
        if test_set and contest_date < TEST_SET_CUTOFF:
            continue
        if evolution_set and contest_date >= TEST_SET_CUTOFF:
            continue

        # Ensure cache entry exists
        problem_dir = ensure_cache_entry(cache_dir, question_id, problem_data)

        # Load meta for consistency
        meta_path = problem_dir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        problems.append((problem_dir, meta))

    return problems


def filter_by_date(
    problems: List[Tuple[Path, Dict]],
    test_set: bool = False,
    evolution_set: bool = False,
) -> List[Tuple[Path, Dict]]:
    """Filter problems by contest_date."""
    if not test_set and not evolution_set:
        return problems

    filtered = []
    for problem_dir, meta in problems:
        contest_date = meta.get("contest_date", "")
        if not contest_date:
            continue  # Skip problems without date

        if test_set and contest_date >= TEST_SET_CUTOFF:
            filtered.append((problem_dir, meta))
        elif evolution_set and contest_date < TEST_SET_CUTOFF:
            filtered.append((problem_dir, meta))

    return filtered


def get_problems_dir(run_dir: Path) -> Path:
    """Get the problems subdirectory for a run.

    All problem task directories are stored under run_dir/problems/
    to keep metadata files (config.json, evaluation.json, report.md)
    visible at the top level.
    """
    return run_dir / "problems"


def load_problem_results(run_dir: Path) -> Tuple[List[Dict], Set[str]]:
    """Load results and identify incomplete problems in one pass.

    A problem is considered incomplete if:
    1. result.json doesn't exist, OR
    2. result.json exists but is corrupt/unreadable, OR
    3. result.json exists with acceptance_category="invalid" AND
       acceptance.md doesn't exist (indicating a failed acceptance call,
       likely due to rate limiting)

    Args:
        run_dir: The run directory to scan

    Returns:
        (completed_results, incomplete_ids) tuple
    """
    problems_dir = get_problems_dir(run_dir)
    if not problems_dir.exists():
        return [], set()

    completed = []
    incomplete = set()

    for item in problems_dir.iterdir():
        if not item.is_dir() or item.name.startswith('.'):
            continue

        result_file = item / "result.json"
        if not result_file.exists():
            # No result at all - incomplete
            incomplete.add(item.name)
            continue

        # Try to read result.json
        try:
            with open(result_file) as f:
                result = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # Can't read result.json - treat as incomplete
            logger.warning(f"Failed to load {result_file}: {e}")
            incomplete.add(item.name)
            continue

        # Check for failed acceptance (invalid + no acceptance.md)
        if (result.get("acceptance_category") == "invalid" and
                not (item / "acceptance.md").exists()):
            incomplete.add(item.name)
            continue

        # Fully complete
        completed.append(result)

    return completed, incomplete


def find_incomplete_problems(run_dir: Path) -> Set[str]:
    """Find problems that need to be completed or retried.

    Args:
        run_dir: The run directory to scan

    Returns set of question_ids that need to be completed.
    """
    _, incomplete = load_problem_results(run_dir)
    return incomplete


def load_completed_results(run_dir: Path) -> List[Dict]:
    """Load all result.json files from fully completed problems.

    Args:
        run_dir: The run directory to scan

    Returns list of result dicts from completed problems.
    """
    completed, _ = load_problem_results(run_dir)
    return completed


def aggregate_results(results: List[Dict]) -> Dict:
    """
    Compute all metrics from evaluation results.

    Returns dict with all computed metrics for use by logging and reports.
    """
    valid_results = [r for r in results if not r.get("error")]
    error_results = [r for r in results if r.get("error")]

    total = len(results)
    valid_count = len(valid_results)
    error_count = len(error_results)

    # Core metrics
    v1_passed = sum(1 for r in valid_results if r.get("v1_passed"))
    v2_passed = sum(1 for r in valid_results if r.get("v2_passed"))
    improved = sum(1 for r in valid_results if r.get("improved"))
    regressed = sum(1 for r in valid_results if r.get("regressed"))
    regenerated_count = sum(1 for r in valid_results if r.get("regenerated"))

    # Rates
    v1_rate = 100 * v1_passed / valid_count if valid_count else 0
    v2_rate = 100 * v2_passed / valid_count if valid_count else 0
    regenerated_rate = 100 * regenerated_count / valid_count if valid_count else 0

    # Verdict breakdown
    verdict_incorrect_count = sum(1 for r in valid_results if not r.get("verdict_correct", False))
    verdict_correct_v1_right = sum(1 for r in valid_results if r.get("verdict_correct") and r.get("v1_passed"))
    verdict_correct_v1_wrong = sum(1 for r in valid_results if r.get("verdict_correct") and not r.get("v1_passed"))
    verdict_incorrect_v1_right = sum(1 for r in valid_results if not r.get("verdict_correct", False) and r.get("v1_passed"))
    verdict_incorrect_v1_wrong = sum(1 for r in valid_results if not r.get("verdict_correct", False) and not r.get("v1_passed"))
    verdict_incorrect_rate = 100 * verdict_incorrect_count / valid_count if valid_count else 0

    # Acceptance breakdown
    accepted_all = sum(1 for r in valid_results if r.get("acceptance_category") == "accepted_all")
    accepted_some = sum(1 for r in valid_results if r.get("acceptance_category") == "accepted_some")
    rejected_all = sum(1 for r in valid_results if r.get("acceptance_category") == "rejected_all")
    acceptance_invalid = sum(1 for r in valid_results if r.get("acceptance_category") == "invalid")
    revised_total = accepted_all + accepted_some + rejected_all + acceptance_invalid

    # Acceptance effectiveness helper
    def count_acceptance_correctness(category: str, v1_p: bool, v2_p: bool) -> int:
        return sum(1 for r in valid_results
                   if r.get("acceptance_category") == category
                   and r.get("v1_passed") == v1_p
                   and r.get("v2_passed") == v2_p)

    # Acceptance x correctness for each category
    acceptance_effectiveness = {}
    for cat in ["accepted_all", "accepted_some", "rejected_all", "invalid"]:
        acceptance_effectiveness[cat] = {
            "improved": count_acceptance_correctness(cat, False, True),
            "no_help": count_acceptance_correctness(cat, False, False),
            "no_harm": count_acceptance_correctness(cat, True, True),
            "regressed": count_acceptance_correctness(cat, True, False),
        }

    # Revision tracking
    revision_attempted = sum(1 for r in valid_results if r.get("revision_attempted"))
    revision_completed = sum(1 for r in valid_results if r.get("revision_completed"))
    revision_failed = sum(1 for r in valid_results if r.get("revision_failed_reason"))

    # Timing helper
    def get_timing_val(r, phase, key, default=0):
        timing = r.get("timing", {})
        if timing is None:
            return default
        phase_timing = timing.get(phase, {})
        if phase_timing is None:
            return default
        return phase_timing.get(key, default) or default

    # Timing calculations
    codegen_time = sum(get_timing_val(r, "codegen", "elapsed_seconds") for r in valid_results)
    codegen_cost = sum(get_timing_val(r, "codegen", "cost_usd") for r in valid_results)
    codegen_from_cache = sum(1 for r in valid_results if not r.get("regenerated"))
    codegen_regenerated = regenerated_count

    eval_time = sum(
        get_timing_val(r, "critic", "elapsed_seconds") +
        get_timing_val(r, "revision", "elapsed_seconds") +
        get_timing_val(r, "acceptance", "elapsed_seconds")
        for r in valid_results
    )
    eval_cost = sum(
        get_timing_val(r, "critic", "cost_usd") +
        get_timing_val(r, "revision", "cost_usd") +
        get_timing_val(r, "acceptance", "cost_usd")
        for r in valid_results
    )

    # Token counts
    eval_input_tokens = sum(
        get_timing_val(r, "critic", "input_tokens") +
        get_timing_val(r, "critic", "cache_read_tokens") +
        get_timing_val(r, "revision", "input_tokens") +
        get_timing_val(r, "revision", "cache_read_tokens") +
        get_timing_val(r, "acceptance", "input_tokens") +
        get_timing_val(r, "acceptance", "cache_read_tokens")
        for r in valid_results
    )
    eval_cache_read_tokens = sum(
        get_timing_val(r, "critic", "cache_read_tokens") +
        get_timing_val(r, "revision", "cache_read_tokens") +
        get_timing_val(r, "acceptance", "cache_read_tokens")
        for r in valid_results
    )
    eval_output_tokens = sum(
        get_timing_val(r, "critic", "output_tokens") +
        get_timing_val(r, "revision", "output_tokens") +
        get_timing_val(r, "acceptance", "output_tokens")
        for r in valid_results
    )
    cache_pct = 100 * eval_cache_read_tokens / eval_input_tokens if eval_input_tokens else 0

    return {
        # Raw data references
        "valid_results": valid_results,
        "error_results": error_results,

        # Counts
        "total": total,
        "valid_count": valid_count,
        "error_count": error_count,
        "v1_passed": v1_passed,
        "v2_passed": v2_passed,
        "improved": improved,
        "regressed": regressed,
        "regenerated_count": regenerated_count,

        # Rates
        "v1_rate": v1_rate,
        "v2_rate": v2_rate,
        "regenerated_rate": regenerated_rate,

        # Verdict breakdown
        "verdict_incorrect_count": verdict_incorrect_count,
        "verdict_incorrect_rate": verdict_incorrect_rate,
        "verdict_correct_v1_right": verdict_correct_v1_right,
        "verdict_correct_v1_wrong": verdict_correct_v1_wrong,
        "verdict_incorrect_v1_right": verdict_incorrect_v1_right,
        "verdict_incorrect_v1_wrong": verdict_incorrect_v1_wrong,

        # Acceptance breakdown
        "accepted_all": accepted_all,
        "accepted_some": accepted_some,
        "rejected_all": rejected_all,
        "acceptance_invalid": acceptance_invalid,
        "revised_total": revised_total,
        "acceptance_effectiveness": acceptance_effectiveness,

        # Revision tracking
        "revision_attempted": revision_attempted,
        "revision_completed": revision_completed,
        "revision_failed": revision_failed,

        # Timing
        "codegen_time": codegen_time,
        "codegen_cost": codegen_cost,
        "codegen_from_cache": codegen_from_cache,
        "codegen_regenerated": codegen_regenerated,
        "eval_time": eval_time,
        "eval_cost": eval_cost,
        "eval_input_tokens": eval_input_tokens,
        "eval_cache_read_tokens": eval_cache_read_tokens,
        "eval_output_tokens": eval_output_tokens,
        "cache_pct": cache_pct,
    }


def generate_markdown_report(
    run_dir: Path,
    metrics: Dict,
    config: Dict,
    resume_info: Optional[Dict] = None,
) -> None:
    """Generate report.md with full evaluation summary.

    Args:
        run_dir: Output directory for the report
        metrics: Pre-computed metrics from aggregate_results()
        config: Run configuration
        resume_info: Optional dict with 'total_expected', 'failed_before', 'failed_after'
    """
    # Extract metrics
    valid_results = metrics["valid_results"]
    error_results = metrics["error_results"]
    total = metrics["total"]
    valid_count = metrics["valid_count"]
    error_count = metrics["error_count"]
    v1_passed = metrics["v1_passed"]
    v2_passed = metrics["v2_passed"]
    improved = metrics["improved"]
    regressed = metrics["regressed"]
    regenerated_count = metrics["regenerated_count"]
    v1_rate = metrics["v1_rate"]
    v2_rate = metrics["v2_rate"]
    verdict_correct_v1_right = metrics["verdict_correct_v1_right"]
    verdict_correct_v1_wrong = metrics["verdict_correct_v1_wrong"]
    verdict_incorrect_v1_right = metrics["verdict_incorrect_v1_right"]
    verdict_incorrect_v1_wrong = metrics["verdict_incorrect_v1_wrong"]
    accepted_all = metrics["accepted_all"]
    accepted_some = metrics["accepted_some"]
    rejected_all = metrics["rejected_all"]
    acceptance_invalid = metrics["acceptance_invalid"]
    revised_total = metrics["revised_total"]
    acceptance_effectiveness = metrics["acceptance_effectiveness"]
    codegen_time = metrics["codegen_time"]
    codegen_cost = metrics["codegen_cost"]
    codegen_regenerated = metrics["codegen_regenerated"]
    eval_time = metrics["eval_time"]
    eval_cost = metrics["eval_cost"]

    # Build report
    lines = []
    lines.append("# Critic Evaluation Report")
    lines.append("")
    lines.append(f"**Run:** {run_dir.name}")
    lines.append(f"**Coder Model:** {config.get('coder_model', 'unknown')}")
    lines.append(f"**Critic Model:** {config.get('critic_model', 'unknown')}")
    lines.append(f"**Critic Agent:** {config.get('critic_agent_name', 'unknown')}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Resume summary
    if resume_info:
        lines.append("## Resume Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total problems | {resume_info['total_expected']} |")
        lines.append(f"| Incomplete before | {resume_info['failed_before']} |")
        lines.append(f"| Incomplete after | {resume_info['failed_after']} |")
        fixed = resume_info['failed_before'] - resume_info['failed_after']
        lines.append(f"| Fixed this session | {fixed} |")
        lines.append("")

    # Results summary
    lines.append("## Results")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Problems | {total} ({valid_count} valid, {error_count} errors) |")
    lines.append(f"| Regenerated | {regenerated_count} |")
    lines.append(f"| V1 Pass@1 | {v1_passed}/{valid_count} ({v1_rate:.1f}%) |")
    lines.append(f"| V2 Pass@1 | {v2_passed}/{valid_count} ({v2_rate:.1f}%) |")
    lines.append(f"| Improvement | {v2_rate - v1_rate:+.1f}% |")
    lines.append(f"| Improved | {improved} |")
    lines.append(f"| Regressed | {regressed} |")
    lines.append("")

    # Verdict breakdown
    lines.append("## Verdict Breakdown")
    lines.append("")
    lines.append("| Category | Count | Description |")
    lines.append("|----------|-------|-------------|")
    lines.append(f"| INCORRECT + v1_wrong | {verdict_incorrect_v1_wrong} | True positive |")
    lines.append(f"| INCORRECT + v1_right | {verdict_incorrect_v1_right} | False positive |")
    lines.append(f"| CORRECT + v1_right | {verdict_correct_v1_right} | True negative |")
    lines.append(f"| CORRECT + v1_wrong | {verdict_correct_v1_wrong} | False negative |")
    lines.append("")

    # Acceptance breakdown
    if revised_total > 0:
        lines.append("## Acceptance (revised problems only)")
        lines.append("")
        lines.append("| Category | Count | Pct |")
        lines.append("|----------|-------|-----|")
        lines.append(f"| accepted_all | {accepted_all} | {100*accepted_all/revised_total:.1f}% |")
        lines.append(f"| accepted_some | {accepted_some} | {100*accepted_some/revised_total:.1f}% |")
        lines.append(f"| rejected_all | {rejected_all} | {100*rejected_all/revised_total:.1f}% |")
        lines.append(f"| invalid | {acceptance_invalid} | {100*acceptance_invalid/revised_total:.1f}% |")
        lines.append("")

        # Acceptance effectiveness
        lines.append("## Acceptance Effectiveness")
        lines.append("")
        lines.append("| Category | Improved | No Help | No Harm | Regressed |")
        lines.append("|----------|----------|---------|---------|-----------|")
        if accepted_all > 0:
            eff = acceptance_effectiveness["accepted_all"]
            lines.append(f"| accepted_all ({accepted_all}) | {eff['improved']} | {eff['no_help']} | {eff['no_harm']} | {eff['regressed']} |")
        if accepted_some > 0:
            eff = acceptance_effectiveness["accepted_some"]
            lines.append(f"| accepted_some ({accepted_some}) | {eff['improved']} | {eff['no_help']} | {eff['no_harm']} | {eff['regressed']} |")
        if rejected_all > 0:
            eff = acceptance_effectiveness["rejected_all"]
            lines.append(f"| rejected_all ({rejected_all}) | {eff['improved']} | {eff['no_help']} | {eff['no_harm']} | {eff['regressed']} |")
        if acceptance_invalid > 0:
            eff = acceptance_effectiveness["invalid"]
            lines.append(f"| invalid ({acceptance_invalid}) | {eff['improved']} | {eff['no_help']} | {eff['no_harm']} | {eff['regressed']} |")
        lines.append("")

    # Cost and timing
    lines.append("## Cost and Timing")
    lines.append("")
    lines.append("| Phase | Time | Cost |")
    lines.append("|-------|------|------|")
    if codegen_regenerated > 0:
        lines.append(f"| Codegen (regen only) | {codegen_time/60:.1f}m | ${codegen_cost:.2f} |")
    lines.append(f"| Evaluation | {eval_time/60:.1f}m | ${eval_cost:.2f} |")
    total_time = codegen_time + eval_time
    total_cost = codegen_cost + eval_cost
    lines.append(f"| **Total** | **{total_time/60:.1f}m** | **${total_cost:.2f}** |")
    lines.append("")

    # Errors table
    if error_count > 0:
        lines.append(f"## Errors ({error_count})")
        lines.append("")
        lines.append("| Question ID | Error |")
        lines.append("|-------------|-------|")
        for r in error_results[:20]:  # Limit to first 20
            lines.append(f"| {r['question_id']} | {r.get('error', 'unknown')} |")
        if error_count > 20:
            lines.append(f"| ... | ({error_count - 20} more) |")
        lines.append("")

    # Write report
    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines))
    logger.info(f"Report saved to {report_path}")


def reeval_solutions(run_dir: Path) -> int:
    """
    Re-run evaluation only on existing solution files.

    This is useful to check for flaky results without re-running expensive
    Claude calls (critic, revision, acceptance).
    """
    eval_file = run_dir / "evaluation.json"
    if not eval_file.exists():
        logger.error(f"evaluation.json not found in {run_dir}")
        return 1

    with open(eval_file) as f:
        data = json.load(f)

    results = data.get("results", [])
    logger.info(f"Re-evaluating {len(results)} problems in {run_dir}")

    # Load dataset for problem metadata
    config_file = run_dir / "config.json"
    if not config_file.exists():
        logger.error(f"config.json not found in {run_dir}")
        return 1

    with open(config_file) as f:
        config = json.load(f)

    # Determine dataset - load_dataset returns dict keyed by question_id
    test_set = config.get("test_set", False)
    evolution_set = config.get("evolution_set", False)
    if test_set:
        problem_lookup = load_dataset("test")
    elif evolution_set:
        problem_lookup = load_dataset("evolution")
    else:
        problem_lookup = load_dataset("train")

    changed = 0
    processed = 0
    problems_dir = get_problems_dir(run_dir)
    total_to_process = len([r for r in results if not r.get("error")])
    for r in results:
        if r.get("error"):
            continue  # Skip error results

        qid = r["question_id"]
        problem_dir = problems_dir / qid

        if not problem_dir.exists():
            logger.warning(f"Problem directory not found: {problem_dir}")
            continue

        solution_v1 = problem_dir / "solution.py"
        solution_v2 = problem_dir / "solution_v2.py"

        if not solution_v1.exists():
            logger.warning(f"solution.py not found for {qid}")
            continue

        # Get problem data
        problem_data = problem_lookup.get(qid)
        if not problem_data:
            logger.warning(f"Problem data not found for {qid}")
            continue

        # Re-evaluate v1
        old_v1 = r.get("v1_passed", False)
        v1_result = evaluate_single(problem_dir, problem_data, timeout=6, solution_path=solution_v1)
        new_v1 = v1_result["passed"]

        # Re-evaluate v2 if it exists
        old_v2 = r.get("v2_passed", False)
        if solution_v2.exists():
            v2_result = evaluate_single(problem_dir, problem_data, timeout=6, solution_path=solution_v2)
            new_v2 = v2_result["passed"]
        else:
            v2_result = {"passed": False, "reason": "missing"}
            new_v2 = False

        # Check for changes
        if old_v1 != new_v1 or old_v2 != new_v2:
            logger.info(f"{qid}: v1 {old_v1}->{new_v1}, v2 {old_v2}->{new_v2}")
            changed += 1

        # Update result
        r["v1_passed"] = new_v1
        r["v2_passed"] = new_v2
        r["v1_reason"] = v1_result.get("reason", "")
        r["v2_reason"] = v2_result.get("reason", "")
        r["improved"] = not new_v1 and new_v2
        r["regressed"] = new_v1 and not new_v2

        # Progress logging
        processed += 1
        if processed % 20 == 0 or processed == total_to_process:
            logger.info(f"Progress: {processed}/{total_to_process}")

    # Save updated results
    with open(eval_file, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Re-evaluation complete. {changed} results changed.")
    logger.info(f"Updated: {eval_file}")

    # Print summary stats
    valid_results = [r for r in results if not r.get("error")]
    v1_passed = sum(1 for r in valid_results if r.get("v1_passed"))
    v2_passed = sum(1 for r in valid_results if r.get("v2_passed"))
    improved = sum(1 for r in valid_results if r.get("improved"))
    regressed = sum(1 for r in valid_results if r.get("regressed"))

    logger.info(f"\nSummary:")
    logger.info(f"  V1 Pass@1: {v1_passed}/{len(valid_results)}")
    logger.info(f"  V2 Pass@1: {v2_passed}/{len(valid_results)}")
    logger.info(f"  Improved: {improved}")
    logger.info(f"  Regressed: {regressed}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run critic evaluation on LiveCodeBench problems"
    )
    parser.add_argument(
        "--coder-model",
        type=str,
        help="Model for coder (code gen, revision, acceptance). Cache dir derived from this. Required unless --resume.",
    )
    parser.add_argument(
        "--critic-model",
        type=str,
        default=None,
        help="Model for critic feedback. Defaults to coder-model if not specified.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override cache directory (default: ../robophd_runs/codegen_cache/{coder_model}_v6)",
    )
    parser.add_argument(
        "--critic-agent",
        type=str,
        help="Path to critic agent directory (e.g., RoboPhD/codegen_agents/naive_critic). Required unless --resume.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: critic_evaluations/run_{timestamp})",
    )
    parser.add_argument(
        "--test-set",
        action="store_true",
        help=f"Only evaluate test set (contest_date >= {TEST_SET_CUTOFF})",
    )
    parser.add_argument(
        "--evolution-set",
        action="store_true",
        help=f"Only evaluate evolution set (contest_date < {TEST_SET_CUTOFF})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of problems to evaluate",
    )
    parser.add_argument(
        "--codegen-timeout",
        type=int,
        default=None,
        help="Timeout per codegen Claude call in seconds (default: 1200, or from config when resuming)",
    )
    parser.add_argument(
        "--critic-timeout",
        type=int,
        default=None,
        help="Timeout per critic/revision/acceptance Claude call in seconds (default: 600, or from config when resuming)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum concurrent evaluations (default: 8, or from config when resuming)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume a previous run by providing its directory path. Completes unevaluated problems and retries failures.",
    )
    parser.add_argument(
        "--reeval",
        type=str,
        help="Re-run evaluation only on existing solution files. Provide the run directory path.",
    )
    parser.add_argument(
        "--problem-ids",
        type=str,
        help="Comma-separated list of problem IDs to evaluate. Only these problems will be processed.",
    )
    parser.add_argument(
        "--lmstudio-base-url",
        type=str,
        default=LMSTUDIO_DEFAULT_BASE_URL,
        help=f"LM Studio server URL for non-Anthropic models (default: {LMSTUDIO_DEFAULT_BASE_URL})",
    )

    args = parser.parse_args()

    # Handle --reeval mode (re-run evaluation only, no critic/revision)
    if args.reeval:
        return reeval_solutions(Path(args.reeval))

    # Track if we're resuming and which problems failed
    resume_mode = False
    failed_question_ids = set()
    existing_results = []

    if args.resume:
        # Resume mode: load config from existing run
        resume_mode = True
        resume_dir = Path(args.resume)

        if not resume_dir.exists():
            logger.error(f"Resume directory not found: {resume_dir}")
            return 1

        config_file = resume_dir / "config.json"
        if not config_file.exists():
            logger.error(f"Config file not found: {config_file}")
            return 1

        with open(config_file) as f:
            config = json.load(f)

        # Extract config values
        coder_model = config["coder_model"]
        critic_model = config["critic_model"]
        cache_dir = Path(config["cache_dir"])
        critic_agent_dir = Path(config["critic_agent"])
        output_dir = resume_dir
        test_set = config.get("test_set", False)
        evolution_set = config.get("evolution_set", False)
        # Use explicit CLI args if provided, otherwise fall back to config (then defaults)
        codegen_timeout = args.codegen_timeout if args.codegen_timeout is not None else config.get("codegen_timeout", 1200)
        critic_timeout = args.critic_timeout if args.critic_timeout is not None else config.get("critic_timeout", 600)
        max_concurrent = args.max_concurrent if args.max_concurrent is not None else config.get("max_concurrent", 8)
        # lmstudio_base_url: CLI arg takes precedence, then config, then default
        args.lmstudio_base_url = args.lmstudio_base_url or config.get("lmstudio_base_url", LMSTUDIO_DEFAULT_BASE_URL)

        logger.info(f"Resuming run: {resume_dir}")

    else:
        # Normal mode: require coder-model and critic-agent
        if not args.coder_model:
            logger.error("--coder-model is required (unless using --resume)")
            return 1
        if not args.critic_agent:
            logger.error("--critic-agent is required (unless using --resume)")
            return 1

        # Validate versioned model names (required for cache isolation)
        try:
            validate_model_version(args.coder_model, "--coder-model")
            if args.critic_model:
                validate_model_version(args.critic_model, "--critic-model")
        except ValueError as e:
            logger.error(str(e))
            return 1

        # Set up models
        coder_model = args.coder_model
        critic_model = args.critic_model or coder_model
        test_set = args.test_set
        evolution_set = args.evolution_set
        codegen_timeout = args.codegen_timeout if args.codegen_timeout is not None else 1200
        critic_timeout = args.critic_timeout if args.critic_timeout is not None else 600
        max_concurrent = args.max_concurrent if args.max_concurrent is not None else 8

        # Derive cache dir from coder model if not specified
        if args.cache_dir:
            cache_dir = Path(args.cache_dir)
        else:
            cache_dir = Path(f"../robophd_runs/codegen_cache/{coder_model}_v6")

        critic_agent_dir = Path(args.critic_agent)

        # Determine output directory (timestamped like RoboPhD runs)
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("../robophd_runs/critic_evaluations") / f"run_{timestamp}"

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Write config.json with all run parameters
        config = {
            "cache_dir": str(cache_dir),
            "critic_agent": str(critic_agent_dir),
            "critic_agent_name": critic_agent_dir.name,
            "coder_model": coder_model,
            "critic_model": critic_model,
            "test_set": test_set,
            "evolution_set": evolution_set,
            "limit": args.limit,
            "codegen_timeout": codegen_timeout,
            "critic_timeout": critic_timeout,
            "max_concurrent": max_concurrent,
            "lmstudio_base_url": args.lmstudio_base_url,
            "timestamp": datetime.now().isoformat(),
        }
        config_file = output_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config saved to: {config_file}")

    # Create cache dir if it doesn't exist (solutions will be generated on demand)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not critic_agent_dir.exists():
        logger.error(f"Critic agent not found: {critic_agent_dir}")
        return 1

    # Load dataset (source of truth for problems)
    dataset = load_dataset()

    # Load problems from dataset, creating cache entries as needed
    problems = load_problems_from_dataset(
        dataset, cache_dir, test_set=test_set, evolution_set=evolution_set
    )
    if test_set:
        logger.info(f"Found {len(problems)} test set problems")
    elif evolution_set:
        logger.info(f"Found {len(problems)} evolution set problems")
    else:
        logger.info(f"Found {len(problems)} problems")

    # Filter to specific problem IDs if provided
    if args.problem_ids:
        target_ids = set(args.problem_ids.split(','))
        problems = [
            (problem_dir, meta) for problem_dir, meta in problems
            if meta.get("question_id", problem_dir.name) in target_ids
        ]
        logger.info(f"Filtered to {len(problems)} problems by --problem-ids")

    # Track resume info for summary
    resume_info = None

    # Apply limit or filter to incomplete problems
    if resume_mode:
        # Apply current CLI's limit first (allows extending a limited run)
        if args.limit:
            problems = problems[:args.limit]
            logger.info(f"Resume scope limited to first {args.limit} problems")

        # Load completed results directly from result.json files
        existing_results = load_completed_results(resume_dir)
        completed_ids = {r["question_id"] for r in existing_results}

        # Find incomplete problems
        incomplete_ids = find_incomplete_problems(resume_dir)

        # Also check for problems that never started (no directory created)
        expected_ids = {meta.get("question_id", problem_dir.name) for problem_dir, meta in problems}
        never_started = expected_ids - completed_ids - incomplete_ids
        incomplete_ids.update(never_started)

        # Track for resume summary
        failed_before = len(incomplete_ids)
        total_expected = len(completed_ids) + len(incomplete_ids)

        if not incomplete_ids:
            logger.info("No incomplete problems found - nothing to resume")
            return 0

        logger.info(f"Resume: {len(existing_results)} complete, {len(incomplete_ids)} to retry")

        # Filter to only incomplete problems
        problems = [
            (problem_dir, meta) for problem_dir, meta in problems
            if meta.get("question_id", problem_dir.name) in incomplete_ids
        ]

        # Store resume info for later summary
        resume_info = {
            "total_expected": total_expected,
            "failed_before": failed_before,
        }
    elif args.limit:
        # Apply limit in normal mode
        problems = problems[:args.limit]
        logger.info(f"Limited to {len(problems)} problems")

    if not problems:
        logger.error("No problems to evaluate")
        return 1

    # Create evaluator
    lmstudio_base_url = args.lmstudio_base_url
    evaluator = CriticEvaluator(
        cache_dir=cache_dir,
        critic_agent_dir=critic_agent_dir,
        output_dir=output_dir,
        coder_model=coder_model,
        critic_model=critic_model,
        codegen_timeout=codegen_timeout,
        critic_timeout=critic_timeout,
        lmstudio_base_url=lmstudio_base_url,
    )

    # Run evaluations
    max_workers = max_concurrent
    logger.info(f"Running with max_concurrent={max_workers}")

    # Create problems subdirectory for task outputs
    problems_dir = get_problems_dir(output_dir)
    problems_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_problem = {}
        for problem_dir, meta in problems:
            question_id = meta.get("question_id", problem_dir.name)
            problem_data = dataset.get(question_id, {})
            output_problem_dir = problems_dir / problem_dir.name

            future = executor.submit(
                evaluator.evaluate_problem,
                problem_dir,
                output_problem_dir,
                problem_data,
                question_id,
            )
            future_to_problem[future] = question_id

        # Collect results as they complete
        results = []
        for future in as_completed(future_to_problem):
            question_id = future_to_problem[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Evaluation failed for {question_id}: {e}")
                results.append({"question_id": question_id, "score": 0.0, "error": str(e)})

            # Progress - log at reasonable intervals (~10 updates total)
            log_interval = max(1, len(problems) // 10)
            if len(results) % log_interval == 0 or len(results) == len(problems):
                logger.info(f"{'='*60}")
                logger.info(f">>> PROGRESS: {len(results)}/{len(problems)} completed ({len(results)*100//len(problems)}%) - last: {question_id}")
                logger.info(f"{'='*60}")

    logger.info(f"{'='*60}")
    logger.info(f">>> EVALUATION COMPLETE: {len(results)}/{len(problems)} problems")
    logger.info(f"{'='*60}")

    # Track new results before merging (for resume summary)
    new_results = results

    # Merge with existing results when resuming
    if resume_mode and existing_results:
        results = existing_results + new_results
        logger.info(f"Merged results: {len(existing_results)} existing + {len(new_results)} new = {len(results)} total")

    # Update resume_info with final failed count
    if resume_mode and resume_info:
        new_errors = sum(1 for r in new_results if r.get("error"))
        resume_info["failed_after"] = new_errors

    # Aggregate all metrics
    metrics = aggregate_results(results)

    # Extract commonly used values for convenience
    valid_results = metrics["valid_results"]
    error_results = metrics["error_results"]
    total = metrics["total"]
    valid_count = metrics["valid_count"]
    error_count = metrics["error_count"]
    v1_passed = metrics["v1_passed"]
    v2_passed = metrics["v2_passed"]
    improved = metrics["improved"]
    regressed = metrics["regressed"]
    regenerated_count = metrics["regenerated_count"]
    v1_rate = metrics["v1_rate"]
    v2_rate = metrics["v2_rate"]
    regenerated_rate = metrics["regenerated_rate"]
    verdict_incorrect_count = metrics["verdict_incorrect_count"]
    verdict_incorrect_rate = metrics["verdict_incorrect_rate"]
    verdict_correct_v1_right = metrics["verdict_correct_v1_right"]
    verdict_correct_v1_wrong = metrics["verdict_correct_v1_wrong"]
    verdict_incorrect_v1_right = metrics["verdict_incorrect_v1_right"]
    verdict_incorrect_v1_wrong = metrics["verdict_incorrect_v1_wrong"]
    accepted_all = metrics["accepted_all"]
    accepted_some = metrics["accepted_some"]
    rejected_all = metrics["rejected_all"]
    acceptance_invalid = metrics["acceptance_invalid"]
    revised_total = metrics["revised_total"]
    acceptance_effectiveness = metrics["acceptance_effectiveness"]
    revision_attempted = metrics["revision_attempted"]
    revision_completed = metrics["revision_completed"]
    revision_failed = metrics["revision_failed"]
    codegen_time = metrics["codegen_time"]
    codegen_cost = metrics["codegen_cost"]
    codegen_from_cache = metrics["codegen_from_cache"]
    codegen_regenerated = metrics["codegen_regenerated"]
    eval_time = metrics["eval_time"]
    eval_cost = metrics["eval_cost"]
    eval_input_tokens = metrics["eval_input_tokens"]
    eval_cache_read_tokens = metrics["eval_cache_read_tokens"]
    eval_output_tokens = metrics["eval_output_tokens"]
    cache_pct = metrics["cache_pct"]

    # Resume summary (if applicable)
    if resume_mode and resume_info:
        logger.info(f"\n{'='*50}")
        logger.info("RESUME SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"  Total problems: {resume_info['total_expected']}")
        logger.info(f"  Incomplete before: {resume_info['failed_before']}")
        logger.info(f"  Incomplete after: {resume_info['failed_after']}")
        fixed = resume_info['failed_before'] - resume_info['failed_after']
        logger.info(f"  Fixed this session: {fixed}")

    logger.info(f"\n{'='*50}")
    logger.info("RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"  Problems: {total} ({valid_count} valid, {error_count} errors)")
    logger.info(f"  Regenerated: {regenerated_count}/{valid_count} ({regenerated_rate:.1f}%)")
    logger.info(f"  V1 Pass@1: {v1_passed}/{valid_count} ({v1_rate:.1f}%)")
    logger.info(f"  V2 Pass@1: {v2_passed}/{valid_count} ({v2_rate:.1f}%)")
    logger.info(f"  Improvement: {v2_rate - v1_rate:+.1f}%")
    logger.info(f"  Problems improved: {improved}")
    logger.info(f"  Problems regressed: {regressed}")
    if error_count > 0:
        logger.info("")
        logger.info("ERRORS")
        logger.info(f"  Total: {error_count}")
        # Group by error type
        error_types = Counter(r['error'] for r in error_results)
        for error_type, count in error_types.most_common():
            logger.info(f"    {error_type}: {count}")
    logger.info("")
    logger.info("VERDICT BREAKDOWN")
    logger.info(f"  VERDICT: INCORRECT: {verdict_incorrect_count}/{valid_count} ({verdict_incorrect_rate:.1f}%)")
    logger.info(f"  2x2 Matrix (verdict × v1_correctness):")
    logger.info(f"    INCORRECT + v1_wrong  = {verdict_incorrect_v1_wrong:3d} (true positive)")
    logger.info(f"    INCORRECT + v1_right  = {verdict_incorrect_v1_right:3d} (false positive)")
    logger.info(f"    CORRECT + v1_right    = {verdict_correct_v1_right:3d} (true negative)")
    logger.info(f"    CORRECT + v1_wrong    = {verdict_correct_v1_wrong:3d} (false negative)")
    logger.info("")
    logger.info("ACCEPTANCE (revised problems only)")
    logger.info(f"  Revised: {revised_total}/{valid_count}")
    if revised_total > 0:
        logger.info(f"    accepted_all:  {accepted_all:3d} ({100*accepted_all/revised_total:.1f}%)")
        logger.info(f"    accepted_some: {accepted_some:3d} ({100*accepted_some/revised_total:.1f}%)")
        logger.info(f"    rejected_all:  {rejected_all:3d} ({100*rejected_all/revised_total:.1f}%)")
        logger.info(f"    invalid:       {acceptance_invalid:3d} ({100*acceptance_invalid/revised_total:.1f}%)")
    logger.info("")
    if accepted_all > 0 or accepted_some > 0 or rejected_all > 0 or acceptance_invalid > 0:
        logger.info("ACCEPTANCE EFFECTIVENESS")
        if accepted_all > 0:
            eff = acceptance_effectiveness["accepted_all"]
            logger.info(f"  accepted_all ({accepted_all}):")
            logger.info(f"    v1_wrong → v2_right: {eff['improved']:3d}  (improved)")
            logger.info(f"    v1_wrong → v2_wrong: {eff['no_help']:3d}  (no help)")
            logger.info(f"    v1_right → v2_right: {eff['no_harm']:3d}  (false positive, no harm)")
            logger.info(f"    v1_right → v2_wrong: {eff['regressed']:3d}  (REGRESSED)")
        if accepted_some > 0:
            eff = acceptance_effectiveness["accepted_some"]
            logger.info(f"  accepted_some ({accepted_some}):")
            logger.info(f"    v1_wrong → v2_right: {eff['improved']:3d}  (improved)")
            logger.info(f"    v1_wrong → v2_wrong: {eff['no_help']:3d}  (no help)")
            logger.info(f"    v1_right → v2_right: {eff['no_harm']:3d}  (false positive, no harm)")
            logger.info(f"    v1_right → v2_wrong: {eff['regressed']:3d}  (REGRESSED)")
        if rejected_all > 0:
            eff = acceptance_effectiveness["rejected_all"]
            logger.info(f"  rejected_all ({rejected_all}):")
            logger.info(f"    v1_wrong → v2_right: {eff['improved']:3d}  (improved independently)")
            logger.info(f"    v1_wrong → v2_wrong: {eff['no_help']:3d}  (no change or no help)")
            logger.info(f"    v1_right → v2_right: {eff['no_harm']:3d}  (correctly rejected)")
            logger.info(f"    v1_right → v2_wrong: {eff['regressed']:3d}  (REGRESSED independently)")
        if acceptance_invalid > 0:
            eff = acceptance_effectiveness["invalid"]
            logger.info(f"  invalid ({acceptance_invalid}):")
            logger.info(f"    v1_wrong → v2_right: {eff['improved']:3d}  (improved)")
            logger.info(f"    v1_wrong → v2_wrong: {eff['no_help']:3d}  (no help)")
            logger.info(f"    v1_right → v2_right: {eff['no_harm']:3d}  (no harm)")
            logger.info(f"    v1_right → v2_wrong: {eff['regressed']:3d}  (REGRESSED)")
            # Log the invalid question IDs for debugging
            invalid_ids = [r["question_id"] for r in valid_results if r.get("acceptance_category") == "invalid"]
            logger.info(f"    Invalid IDs: {invalid_ids[:10]}{'...' if len(invalid_ids) > 10 else ''}")
        logger.info("")
    logger.info("REVISION TRACKING")
    logger.info(f"  Attempted: {revision_attempted}")
    logger.info(f"  Completed: {revision_completed}")
    logger.info(f"  Failed: {revision_failed}")
    if revision_failed > 0:
        # Log the failed question IDs for debugging
        failed_ids = [r["question_id"] for r in results if r.get("revision_failed_reason")]
        logger.info(f"  Failed IDs: {failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}")

    logger.info("")
    logger.info("COST AND TIMING")
    logger.info("  Codegen (cache building, one-time cost):")
    logger.info(f"    From cache: {codegen_from_cache}, Regenerated: {codegen_regenerated}")
    if codegen_regenerated > 0:
        logger.info(f"    Regen time: {codegen_time:.1f}s ({codegen_time/60:.1f}m), Cost: ${codegen_cost:.4f}")
    logger.info("  Evaluation (critic system cost):")
    logger.info(f"    Total time: {eval_time:.1f}s ({eval_time/60:.1f}m)")
    if valid_count > 0:
        logger.info(f"    Avg per problem: {eval_time/valid_count:.1f}s")
    logger.info(f"    Total cost: ${eval_cost:.4f}")
    logger.info(f"    Tokens: {eval_input_tokens:,} in ({cache_pct:.0f}% cached) / {eval_output_tokens:,} out")
    logger.info(f"{'='*50}")

    # Save results
    results_file = output_dir / "evaluation.json"
    with open(results_file, "w") as f:
        json.dump({
            "critic_agent": critic_agent_dir.name,
            "cache_dir": str(cache_dir),
            "coder_model": coder_model,
            "critic_model": critic_model,
            "test_set": test_set,
            "evolution_set": evolution_set,
            "summary": {
                "total_problems": total,
                "regenerated_count": regenerated_count,
                "regenerated_rate": regenerated_rate,
                "v1_passed": v1_passed,
                "v2_passed": v2_passed,
                "v1_pass_rate": v1_rate,
                "v2_pass_rate": v2_rate,
                "improvement": v2_rate - v1_rate,
                "problems_improved": improved,
                "problems_regressed": regressed,
                # Verdict breakdown (2x2 matrix)
                "verdict_incorrect_count": verdict_incorrect_count,
                "verdict_incorrect_rate": verdict_incorrect_rate,
                "verdict_incorrect_v1_wrong": verdict_incorrect_v1_wrong,   # true positive
                "verdict_incorrect_v1_right": verdict_incorrect_v1_right,   # false positive
                "verdict_correct_v1_right": verdict_correct_v1_right,       # true negative
                "verdict_correct_v1_wrong": verdict_correct_v1_wrong,       # false negative
                # Acceptance breakdown (revised problems only)
                "accepted_all": accepted_all,
                "accepted_some": accepted_some,
                "rejected_all": rejected_all,
                "acceptance_invalid": acceptance_invalid,
                # Acceptance × correctness (effectiveness)
                "accepted_improved": acceptance_effectiveness["accepted_all"]["improved"],
                "accepted_no_help": acceptance_effectiveness["accepted_all"]["no_help"],
                "accepted_no_harm": acceptance_effectiveness["accepted_all"]["no_harm"],
                "accepted_regressed": acceptance_effectiveness["accepted_all"]["regressed"],
                "some_improved": acceptance_effectiveness["accepted_some"]["improved"],
                "some_no_help": acceptance_effectiveness["accepted_some"]["no_help"],
                "some_no_harm": acceptance_effectiveness["accepted_some"]["no_harm"],
                "some_regressed": acceptance_effectiveness["accepted_some"]["regressed"],
                "rejected_improved": acceptance_effectiveness["rejected_all"]["improved"],
                "rejected_no_help": acceptance_effectiveness["rejected_all"]["no_help"],
                "rejected_no_harm": acceptance_effectiveness["rejected_all"]["no_harm"],
                "rejected_regressed": acceptance_effectiveness["rejected_all"]["regressed"],
                "invalid_improved": acceptance_effectiveness["invalid"]["improved"],
                "invalid_no_help": acceptance_effectiveness["invalid"]["no_help"],
                "invalid_no_harm": acceptance_effectiveness["invalid"]["no_harm"],
                "invalid_regressed": acceptance_effectiveness["invalid"]["regressed"],
                # Timing and cost
                "codegen_timing": {
                    "from_cache": codegen_from_cache,
                    "regenerated": codegen_regenerated,
                    "total_seconds": codegen_time,
                    "total_cost_usd": codegen_cost,
                },
                "eval_timing": {
                    "total_seconds": eval_time,
                    "average_seconds": eval_time / valid_count if valid_count else 0,
                    "total_cost_usd": eval_cost,
                    "input_tokens": eval_input_tokens,
                    "cache_read_tokens": eval_cache_read_tokens,
                    "cache_hit_rate": cache_pct,
                    "output_tokens": eval_output_tokens,
                },
            },
            # Results as dict keyed by question_id for cross-domain compatibility
            "results": {r["question_id"]: r for r in results},
        }, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")

    # Generate markdown report
    generate_markdown_report(
        run_dir=output_dir,
        metrics=metrics,
        config=config,
        resume_info=resume_info,
    )

    return 0


if __name__ == "__main__":
    exit(main())
