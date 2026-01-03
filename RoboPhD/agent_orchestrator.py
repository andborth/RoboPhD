#!/usr/bin/env python3
"""
AgentOrchestrator: Manages agent invocations for RoboPhD system.
Works exclusively with three-artifact agents (agent.md, eval_instructions.md, tools/).
"""

import json
import subprocess
import shutil
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from config import MODEL_FALLBACKS, CLAUDE_CLI_MODEL_MAP
# Add grandparent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utilities.claude_cli import find_claude_cli

class AgentOrchestrator:
    """Orchestrates database analysis agents in RoboPhD system."""
    
    def __init__(self,
                 base_experiment_dir: Path,
                 analysis_model: str = 'haiku-4.5',
                 claude_path: Optional[str] = None,
                 timeout_phase1: int = 900):  # 15 minutes default
        """
        Initialize the orchestrator.

        Args:
            base_experiment_dir: Base directory for the experiment
            analysis_model: Model to use for analysis (opus-4.5, sonnet-4.5, haiku-4.5)
            claude_path: Path to Claude CLI
            timeout_phase1: Timeout for Phase 1 in seconds
        """
        self.base_dir = Path(base_experiment_dir)
        self.analysis_model = analysis_model
        self.claude_path = claude_path or find_claude_cli()
        self.timeout_phase1 = timeout_phase1
        self.performance_log = []
        self.active_agent_name = None
    
    def _log_with_context(self, message: str, agent_id: str, database_name: str):
        """Log message with agent and database context for consistency."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"    [{timestamp}] {agent_id} | {database_name}: {message}")
        
    def setup_workspace(self,
                       iteration: int,
                       database_name: str,
                       database_path: Path,
                       package_dir: Path,
                       agent_id: str) -> Path:
        """
        Create workspace with specific agent.

        Args:
            iteration: Iteration number
            database_name: Name of the database
            database_path: Path to the SQLite database
            package_dir: Path to three-artifact package directory
            agent_id: ID for tracking this agent

        Returns:
            Path to the configured workspace
        """
        # Validate package_dir exists (catch checkpoint restoration issues early)
        if package_dir is None:
            raise ValueError(
                f"package_dir is None for agent {agent_id}\n"
                f"This indicates an issue with checkpoint restoration or agent pool initialization."
            )

        if not package_dir.exists():
            raise FileNotFoundError(
                f"Agent package directory does not exist: {package_dir}\n"
                f"Agent: {agent_id}\n"
                f"Database: {database_name}\n"
                f"This may indicate a checkpoint restoration issue or missing agent files."
            )

        # Create workspace directory
        workspace = self.base_dir / f"iteration_{iteration:03d}" / f"agent_{agent_id}" / database_name
        workspace.mkdir(parents=True, exist_ok=True)
        
        # Set up three-artifact workspace
        self._setup_three_artifact_workspace(workspace, package_dir, database_path)
        
        # Store agent name from the package
        self.active_agent_name = package_dir.name

        # Create symbolic link to database instead of copying (saves disk space)
        db_dest = workspace / "database.sqlite"
        if db_dest.exists():
            db_dest.unlink()
        # Use symlink to avoid copying large database files
        db_dest.symlink_to(database_path.absolute())
        
        # Create required directories
        (workspace / "output").mkdir(exist_ok=True)
        
        return workspace
    
    def _setup_three_artifact_workspace(self, workspace: Path, package_dir: Path, database_path: Path):
        """
        Set up workspace for three-artifact package.
        
        Args:
            workspace: The workspace directory
            package_dir: Directory containing the package
            database_path: Path to the database
        """
        # Copy agent to .claude/agents directory
        agents_dir = workspace / ".claude" / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        
        agent_src = package_dir / "agent.md"
        agent_dest = agents_dir / agent_src.name
        if not agent_src.exists():
            raise FileNotFoundError(f"agent.md not found: {agent_src}")
        shutil.copy2(agent_src, agent_dest)
        
        # Copy eval instructions to workspace
        eval_src = package_dir / "eval_instructions.md"
        eval_dest = workspace / "eval_instructions.md"
        if not eval_src.exists():
            raise FileNotFoundError(f"eval_instructions.md not found: {eval_src}")
        shutil.copy2(eval_src, eval_dest)
        
        # Copy tools if present
        tools_src = package_dir / "tools"
        if tools_src.exists():
            tools_dest = workspace / "tools"
            if tools_dest.exists():
                shutil.rmtree(tools_dest)
            shutil.copytree(tools_src, tools_dest)
        
        # Create tool_output directory
        (workspace / "tool_output").mkdir(exist_ok=True)
    
    def _extract_agent_name(self, workspace: Path) -> str:
        """Extract agent name from YAML frontmatter in agent.md."""
        agent_file = workspace / ".claude" / "agents" / "agent.md"
        if not agent_file.exists():
            return "database analysis"  # fallback name

        try:
            content = agent_file.read_text()
            # Look for YAML frontmatter
            if content.startswith('---'):
                yaml_end = content.find('---', 3)
                if yaml_end > 0:
                    yaml_content = content[3:yaml_end].strip()
                    for line in yaml_content.split('\n'):
                        if line.startswith('name:'):
                            name = line.split(':', 1)[1].strip()
                            # Remove quotes if present
                            if name.startswith('"') and name.endswith('"'):
                                name = name[1:-1]
                            elif name.startswith("'") and name.endswith("'"):
                                name = name[1:-1]
                            return name
        except Exception:
            pass

        return "database analysis"  # fallback name

    def _parse_yaml_frontmatter(self, workspace: Path) -> dict:
        """
        Parse YAML frontmatter from agent.md for tool-only execution support.

        Returns dict with keys:
        - name: str (agent name)
        - description: str (agent description)
        - execution_mode: str or None ("tool_only" or None)
        - tool_command: str or None (shell command to run)
        - tool_output_file: str or None (path to output file)
        """
        agent_file = workspace / ".claude" / "agents" / "agent.md"
        result = {
            'name': None,
            'description': None,
            'execution_mode': None,
            'tool_command': None,
            'tool_output_file': None
        }

        if not agent_file.exists():
            return result

        try:
            content = agent_file.read_text()
            # Look for YAML frontmatter
            if content.startswith('---'):
                yaml_end = content.find('---', 3)
                if yaml_end > 0:
                    yaml_content = content[3:yaml_end].strip()
                    for line in yaml_content.split('\n'):
                        line = line.strip()
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip()

                            # Remove quotes if present
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            elif value.startswith("'") and value.endswith("'"):
                                value = value[1:-1]

                            # Map to result dict
                            if key in result:
                                result[key] = value
        except Exception as e:
            # Log but don't fail - we'll fall back to normal execution
            print(f"  ⚠️  Warning: Failed to parse YAML frontmatter: {e}")

        return result

    def _try_tool_only_execution(self,
                                 workspace: Path,
                                 tool_command: str,
                                 output_file_path: str,
                                 agent_id: str,
                                 database_name: str,
                                 timeout: int = 600) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Attempt tool-only execution by running the tool command directly.

        Args:
            workspace: Workspace directory
            tool_command: Shell command to execute
            output_file_path: Expected output file path (relative to workspace)
            agent_id: Agent identifier for logging
            database_name: Database name for logging
            timeout: Timeout in seconds (default: 600)

        Returns:
            Tuple of (success, content, error_message)
            - success: True if tool execution succeeded and produced valid output
            - content: File content if successful, None otherwise
            - error_message: Error description if failed, None otherwise
        """
        self._log_with_context(f"Attempting tool-only execution: {tool_command}", agent_id, database_name)

        tool_start = time.time()

        try:
            # Run the tool command
            result = subprocess.run(
                tool_command,
                shell=True,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=timeout
            )

            tool_time = time.time() - tool_start

            # Check exit code
            if result.returncode != 0:
                error_msg = f"Tool command exited with code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nStderr: {result.stderr[:500]}"
                if result.stdout:
                    error_msg += f"\nStdout: {result.stdout[:500]}"

                self._log_with_context(
                    f"⚠️  Tool-only execution failed (exit code {result.returncode}), clearing tool_output/ and calling agent",
                    agent_id,
                    database_name
                )
                if result.stderr:
                    error_preview = result.stderr[:200]
                    self._log_with_context(f"Tool error: {error_preview}", agent_id, database_name)

                # Clear tool_output directory
                tool_output_dir = workspace / "tool_output"
                if tool_output_dir.exists():
                    shutil.rmtree(tool_output_dir)
                    tool_output_dir.mkdir()

                return False, None, error_msg

            # Check output file exists
            output_file = workspace / output_file_path
            if not output_file.exists():
                error_msg = f"Output file not found: {output_file_path}"
                if result.stdout:
                    error_msg += f"\nStdout: {result.stdout[:500]}"

                self._log_with_context(
                    f"⚠️  Tool-only execution completed but output file not found: {output_file_path}, calling agent",
                    agent_id,
                    database_name
                )

                # Clear tool_output directory
                tool_output_dir = workspace / "tool_output"
                if tool_output_dir.exists():
                    shutil.rmtree(tool_output_dir)
                    tool_output_dir.mkdir()

                return False, None, error_msg

            # Check file size (minimum 200 bytes)
            file_size = output_file.stat().st_size
            if file_size < 200:
                error_msg = f"Output file too small: {file_size} bytes (minimum 200 bytes required)"

                self._log_with_context(
                    f"⚠️  Tool-only output file too small ({file_size} bytes < 200), calling agent",
                    agent_id,
                    database_name
                )

                # Clear tool_output directory
                tool_output_dir = workspace / "tool_output"
                if tool_output_dir.exists():
                    shutil.rmtree(tool_output_dir)
                    tool_output_dir.mkdir()

                return False, None, error_msg

            # Success! Read the file
            content = output_file.read_text()

            # Copy to agent_output.txt
            agent_output_file = workspace / "output" / "agent_output.txt"
            agent_output_file.write_text(content)

            # Log performance
            self.performance_log.append({
                'phase': 'phase1_tool_only',
                'agent': self.active_agent_name,
                'database': database_name,
                'time': tool_time,
                'success': True,
                'cost': 0.0  # Tool execution has no API cost
            })

            self._log_with_context(f"✅ Tool-only execution succeeded ({tool_time:.1f}s, $0.00)", agent_id, database_name)
            return True, content, None

        except subprocess.TimeoutExpired:
            tool_time = time.time() - tool_start
            error_msg = f"Tool execution timeout after {tool_time:.1f}s (limit: {timeout}s)"

            self._log_with_context(
                f"⏱️  Tool-only execution timeout after {tool_time:.1f}s, calling agent",
                agent_id,
                database_name
            )

            # Clear tool_output directory
            tool_output_dir = workspace / "tool_output"
            if tool_output_dir.exists():
                shutil.rmtree(tool_output_dir)
                tool_output_dir.mkdir()

            return False, None, error_msg

        except Exception as e:
            tool_time = time.time() - tool_start
            error_msg = f"Tool execution exception: {str(e)}"

            self._log_with_context(
                f"⚠️  Tool-only execution error: {str(e)}, calling agent",
                agent_id,
                database_name
            )

            # Clear tool_output directory
            tool_output_dir = workspace / "tool_output"
            if tool_output_dir.exists():
                shutil.rmtree(tool_output_dir)
                tool_output_dir.mkdir()

            return False, None, error_msg

    def run_phase1(self,
                   workspace: Path,
                   agent_id: str,
                   database_name: str,
                   cache_key: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[dict], Optional[str]]:
        """
        Phase 1: Agent analyzes database and generates analysis.

        Args:
            workspace: Workspace directory
            agent_id: Agent identifier for logging
            database_name: Database name for logging
            cache_key: Optional cache key to check for existing output

        Returns:
            Tuple of (success, output_content, cost_info, tool_error)
            - success: bool indicating if Phase 1 succeeded
            - output_content: str with combined agent output + eval instructions (or None on failure)
            - cost_info: dict with cost and token usage (or None on cache hit/failure)
            - tool_error: str with tool-only error message if tool failed but agent succeeded (or None)
        """
        # Track tool-only errors for later reporting
        tool_error = None

        # Check cache first
        output_file = workspace / "output" / "agent_output.txt"
        if cache_key and output_file.exists():
            print(f"  ℹ️  Using cached output for {workspace.name}")
            # Cache hit - no Claude CLI call made, so no cost, no tool error
            return True, output_file.read_text(), None, None

        # Check for tool-only execution mode
        yaml_config = self._parse_yaml_frontmatter(workspace)
        if yaml_config.get('execution_mode') == 'tool_only':
            tool_command = yaml_config.get('tool_command')
            output_file_path = yaml_config.get('tool_output_file')

            if tool_command and output_file_path:
                # Attempt tool-only execution
                success, tool_output, tool_error = self._try_tool_only_execution(
                    workspace=workspace,
                    tool_command=tool_command,
                    output_file_path=output_file_path,
                    agent_id=agent_id,
                    database_name=database_name
                )

                if success:
                    # Tool-only execution succeeded - combine with eval instructions
                    eval_instructions_file = workspace / "eval_instructions.md"

                    if eval_instructions_file.exists():
                        eval_instructions = eval_instructions_file.read_text()
                        combined_prompt = f"{tool_output}\n\n---\n\n{eval_instructions}"

                        # Save combined prompt
                        system_prompt_file = workspace / "output" / "system_prompt.txt"
                        system_prompt_file.write_text(combined_prompt)

                        # Return success with no cost and no tool error (tool execution is free)
                        return True, combined_prompt, None, None
                    else:
                        # Shouldn't happen but handle gracefully
                        self._log_with_context("⚠️  No eval_instructions.md found for tool-only agent", agent_id, database_name)
                        return False, None, None, None

                # Tool-only execution failed - fall through to normal agent execution
                # tool_output/ has been cleared, agent will handle the failure
                # Store the tool error to return if agent succeeds (captured above)
            else:
                self._log_with_context(
                    "⚠️  Tool-only mode specified but missing tool_command or tool_output_file",
                    agent_id,
                    database_name
                )

        # Extract agent name from YAML frontmatter
        agent_name = self._extract_agent_name(workspace)

        # Three-artifact: agent generates database-specific output with agent name
        # Use Task tool with @agent- prefix to ensure Claude Code properly invokes the agent
        prompt = f"""Use the Task tool with your @agent-{agent_name} agent to analyze the database at ./database.sqlite.

After the agent completes, verify that it has generated output at ./output/agent_output.txt.

If the Task tool invocation fails, read and follow the instructions in .claude/agents/agent.md to analyze the database manually. Follow those instructions to analyze ./database.sqlite and save your output to ./output/agent_output.txt
"""
        
        # Build command with model
        cli_model = CLAUDE_CLI_MODEL_MAP.get(self.analysis_model, self.analysis_model)
        cmd = [
            self.claude_path,
            "--model", cli_model,
        ]

        # Add the prompt
        cmd.extend(["--print", prompt])

        # Add JSON output format for cost tracking
        cmd.extend(["--output-format", "json"])

        # Add permission bypass for automation
        cmd.extend(["--permission-mode", "bypassPermissions"])
        
        # Execute Phase 1 with timeout
        phase1_start = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=self.timeout_phase1
            )

            phase1_time = time.time() - phase1_start

            # Parse JSON output for cost tracking
            cost_info = None
            if result.returncode == 0 and result.stdout:
                try:
                    import json
                    json_output = json.loads(result.stdout)
                    usage = json_output.get('usage', {})
                    cost_info = {
                        'cost': json_output.get('total_cost_usd', 0.0),
                        'tokens_in': usage.get('input_tokens', 0),
                        'tokens_out': usage.get('output_tokens', 0),
                        'cache_created': usage.get('cache_creation_input_tokens', 0),
                        'cache_read': usage.get('cache_read_input_tokens', 0)
                    }
                except (json.JSONDecodeError, KeyError) as e:
                    self._log_with_context(f"⚠️  Failed to parse cost data: {e}", agent_id, database_name)
                    # Continue anyway - cost tracking is not critical

            # Log performance
            self.performance_log.append({
                'phase': 'phase1',
                'agent': self.active_agent_name,
                'database': database_name,
                'time': phase1_time,
                'success': result.returncode == 0,
                'cost': cost_info.get('cost', 0.0) if cost_info else 0.0
            })

            if result.returncode != 0:
                self._log_with_context(f"⚠️  Phase 1 failed (code {result.returncode})", agent_id, database_name)
                if result.stderr:
                    error_preview = result.stderr[:500]
                    self._log_with_context(f"Error: {error_preview}", agent_id, database_name)

                # Check for fallback model
                if self.analysis_model in MODEL_FALLBACKS:
                    fallback = MODEL_FALLBACKS[self.analysis_model]
                    if fallback:
                        self._log_with_context(f"Retrying with fallback model: {fallback}", agent_id, database_name)
                        # Pass tool_error through fallback
                        return self._run_phase1_with_fallback(workspace, fallback, prompt, agent_id, database_name, tool_error)

                return False, None, None, tool_error

            # Check output file was created
            if not output_file.exists():
                self._log_with_context("⚠️  Phase 1 completed but no output file created", agent_id, database_name)
                return False, None, cost_info, tool_error

            # Three-artifact: Combine agent output with eval instructions
            agent_output = output_file.read_text()
            eval_instructions_file = workspace / "eval_instructions.md"

            if eval_instructions_file.exists():
                eval_instructions = eval_instructions_file.read_text()
                # Combine for the final system prompt
                combined_prompt = f"{agent_output}\n\n---\n\n{eval_instructions}"

                # Save combined prompt
                system_prompt_file = workspace / "output" / "system_prompt.txt"
                system_prompt_file.write_text(combined_prompt)

                self._log_with_context(f"✅ Phase 1 complete ({phase1_time:.1f}s)", agent_id, database_name)
                # Return success with tool_error if tool failed but agent succeeded
                return True, combined_prompt, cost_info, tool_error
            else:
                # Shouldn't happen in three-artifact mode
                self._log_with_context("⚠️  No eval_instructions.md found", agent_id, database_name)
                return False, None, cost_info, tool_error
            
        except subprocess.TimeoutExpired:
            phase1_time = time.time() - phase1_start
            self._log_with_context(f"⏱️  Phase 1 timeout after {phase1_time:.1f}s", agent_id, database_name)

            self.performance_log.append({
                'phase': 'phase1',
                'agent': self.active_agent_name,
                'database': database_name,
                'time': phase1_time,
                'success': False,
                'error': 'timeout'
            })

            return False, None, None, tool_error
        except Exception as e:
            self._log_with_context(f"❌ Phase 1 error: {e}", agent_id, database_name)
            return False, None, None, tool_error
    
    def _run_phase1_with_fallback(self, workspace: Path, fallback_model: str,
                                  prompt: str, agent_id: str, database_name: str, tool_error: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[dict], Optional[str]]:
        """Run phase 1 with fallback model."""
        cli_model = CLAUDE_CLI_MODEL_MAP.get(fallback_model, fallback_model)
        cmd = [
            self.claude_path,
            "--model", cli_model,
            "--print", prompt,
            "--output-format", "json",
            "--permission-mode", "bypassPermissions"
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=self.timeout_phase1
            )

            # Parse JSON output for cost tracking
            cost_info = None
            if result.returncode == 0 and result.stdout:
                try:
                    import json
                    json_output = json.loads(result.stdout)
                    usage = json_output.get('usage', {})
                    cost_info = {
                        'cost': json_output.get('total_cost_usd', 0.0),
                        'tokens_in': usage.get('input_tokens', 0),
                        'tokens_out': usage.get('output_tokens', 0),
                        'cache_created': usage.get('cache_creation_input_tokens', 0),
                        'cache_read': usage.get('cache_read_input_tokens', 0)
                    }
                except (json.JSONDecodeError, KeyError) as e:
                    self._log_with_context(f"⚠️  Failed to parse cost data from fallback: {e}", agent_id, database_name)

            if result.returncode == 0:
                output_file = workspace / "output" / "agent_output.txt"
                if output_file.exists():
                    agent_output = output_file.read_text()
                    eval_instructions_file = workspace / "eval_instructions.md"

                    if eval_instructions_file.exists():
                        eval_instructions = eval_instructions_file.read_text()
                        combined_prompt = f"{agent_output}\n\n---\n\n{eval_instructions}"

                        system_prompt_file = workspace / "output" / "system_prompt.txt"
                        system_prompt_file.write_text(combined_prompt)

                        self._log_with_context(f"✅ Phase 1 complete with fallback", agent_id, database_name)
                        # Pass through tool_error if present
                        return True, combined_prompt, cost_info, tool_error

            return False, None, cost_info, tool_error

        except (subprocess.TimeoutExpired, Exception):
            return False, None, None, tool_error
    
    def validate_agent_output(self, workspace: Path, agent_id: str, database_name: str) -> bool:
        """
        Validate that agent produced expected output.

        Args:
            workspace: Workspace directory
            agent_id: Agent identifier for logging
            database_name: Database name for logging

        Returns:
            True if output is valid
        """
        output_file = workspace / "output" / "system_prompt.txt"

        if not output_file.exists():
            self._log_with_context("❌ No system prompt generated", agent_id, database_name)
            return False

        content = output_file.read_text()
        if len(content) < 100:
            self._log_with_context(f"⚠️  System prompt too short ({len(content)} chars)", agent_id, database_name)
            return False

        return True
    
    def get_performance_summary(self) -> Dict:
        """Get summary of performance metrics."""
        if not self.performance_log:
            return {}
        
        total_time = sum(entry['time'] for entry in self.performance_log)
        success_count = sum(1 for entry in self.performance_log if entry['success'])
        
        return {
            'total_runs': len(self.performance_log),
            'successful_runs': success_count,
            'success_rate': success_count / len(self.performance_log) if self.performance_log else 0,
            'total_time': total_time,
            'average_time': total_time / len(self.performance_log) if self.performance_log else 0,
            'logs': self.performance_log
        }