"""
Text2SQL domain implementation.

This adapter wraps existing RoboPhD Text2SQL code (core.py classes)
to implement the DomainInterface. Zero modifications to existing code.
"""

import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import DomainInterface, SampledProblems, EvaluationResult
from RoboPhD.config import DATASET_PATHS


class Text2SQLDomain(DomainInterface):
    """
    Text2SQL domain adapter.

    Wraps existing SQLGenerator, Evaluator, and DatabaseManager classes
    via composition - no modifications to core.py required.

    Phase 1 Input: Database file (symlink to .sqlite)
    Phase 1 Output: Database analysis (schema, relationships, patterns)
    Phase 2 Output: SQL query
    Evaluation: Set-based result comparison
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Text2SQL domain.

        Args:
            config: Configuration dictionary with:
                - dataset: Dataset name ('dev', 'train-filtered', etc.)
                - questions_file: Path to questions JSON
                - db_root: Path to database root directory
                - eval_model: Model for SQL generation
                - And other SQLGenerator/Evaluator options
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Resolve paths
        self._resolve_paths()

        # Load questions grouped by database
        self._questions_by_db: Optional[Dict[str, List[Dict]]] = None

    def _resolve_paths(self):
        """Resolve dataset paths based on config."""
        dataset = self.config.get('dataset', 'train-filtered')

        if dataset in DATASET_PATHS:
            paths = DATASET_PATHS[dataset]
            self.questions_file = Path(self.config.get('questions_file', paths['questions']))
            self.db_root = Path(self.config.get('db_root', paths['db_root']))
        else:
            # Custom paths must be provided
            self.questions_file = Path(self.config['questions_file'])
            self.db_root = Path(self.config['db_root'])

        self.dataset = dataset

    def _load_questions(self) -> Dict[str, List[Dict]]:
        """Load and cache questions grouped by database."""
        if self._questions_by_db is not None:
            return self._questions_by_db

        with open(self.questions_file, 'r') as f:
            questions = json.load(f)

        # Group by db_id
        self._questions_by_db = {}
        for i, q in enumerate(questions):
            db_id = q.get('db_id', '')
            if db_id:
                if db_id not in self._questions_by_db:
                    self._questions_by_db[db_id] = []
                # Ensure question_id exists
                if 'question_id' not in q:
                    q['question_id'] = i
                self._questions_by_db[db_id].append(q)

        return self._questions_by_db

    def prepare_phase1_input(self, workspace: Path, context: str, problem: Optional[Dict] = None) -> Path:
        """
        Prepare database for Phase 1 analysis.

        Creates a symlink to the database.sqlite file in the workspace.
        Raises FileNotFoundError if the database doesn't exist.

        Args:
            workspace: Workspace directory for this analysis
            context: Database name (e.g., 'california_schools')
            problem: Not used for Text2SQL (database is the context)

        Returns:
            Path to database.sqlite symlink

        Raises:
            FileNotFoundError: If the database doesn't exist
        """
        # Import here to avoid circular imports
        from RoboPhD.core import DatabaseManager

        # Get actual database path
        db_path = DatabaseManager.get_database_path(self.db_root, context)

        # Create symlink in workspace
        dest = workspace / "database.sqlite"
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(db_path.absolute())

        return dest

    def evaluate(
        self,
        solution: str,
        problem: Dict,
        context: str,
        predictions_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Evaluate SQL solution against ground truth.

        Args:
            solution: Generated SQL query
            problem: Problem dict with ground truth SQL
            context: Database name
            predictions_path: Optional path to save predictions

        Returns:
            Evaluation result with 'correct' bool and details
        """
        # Import here to avoid circular imports
        from RoboPhD.core import Evaluator

        # Create evaluator
        evaluator = Evaluator(
            questions_file=self.questions_file,
            db_root=self.db_root,
            logger=self.logger
        )

        # Format prediction for evaluator
        question_id = str(problem.get('question_id', ''))
        db_id = problem.get('db_id', context)

        # Add BIRD marker if not present
        if '\t----- bird -----\t' not in solution:
            solution = f"{solution}\t----- bird -----\t{db_id}"

        predictions = {
            'predictions': [{
                'question_id': question_id,
                'db_id': db_id,
                'SQL': solution
            }]
        }

        # Run evaluation
        eval_result = evaluator.evaluate(
            predictions=predictions,
            db_name=context,
            save_to=predictions_path
        )

        # Extract result for this specific question
        results = eval_result.get('results', {})
        question_result = results.get(question_id, {})

        return {
            'correct': question_result.get('matches', False),
            'matches': question_result.get('matches', False),
            'predicted_results': question_result.get('predicted_results'),
            'ground_truth_results': question_result.get('ground_truth_results'),
            'predicted_error': question_result.get('predicted_error'),
            'ground_truth_error': question_result.get('ground_truth_error'),
            'details': eval_result
        }

    def load_problems(self) -> Dict[str, List[Dict]]:
        """
        Load all problems grouped by database.

        Returns:
            Dict mapping database_name -> list of questions
        """
        return self._load_questions()

    def get_contexts(self) -> List[str]:
        """
        Get list of all available databases.

        Scans for database directories containing .sqlite files.

        NOTE: Returns databases in filesystem order (not sorted) to match
        original researcher.py behavior for reproducibility with existing seeds.

        Returns:
            List of database names in filesystem iteration order
        """
        # Import here to avoid circular imports
        from RoboPhD.core import DatabaseManager

        excluded_dbs = DatabaseManager.get_blacklisted_databases(self.dataset)
        databases = []

        if self.db_root.exists():
            for db_dir in self.db_root.iterdir():
                if db_dir.is_dir() and db_dir.name not in excluded_dbs:
                    db_file = db_dir / f"{db_dir.name}.sqlite"
                    if db_file.exists():
                        databases.append(db_dir.name)

        return databases

    @property
    def phase1_input_name(self) -> str:
        """Human-readable name for Phase 1 input."""
        return "database"

    @property
    def solution_name(self) -> str:
        """Human-readable name for generated solutions."""
        return "SQL"

    @property
    def evolution_strategies_dir(self) -> str:
        """Directory name for Text2SQL evolution strategies."""
        return "evolution_strategies"

    @property
    def phase1_display_name(self) -> str:
        """Display name for Phase 1: DB Analysis."""
        return "DB Analysis"

    @property
    def phase2_display_name(self) -> str:
        """Display name for Phase 2: SQL Generation."""
        return "SQL Generation"

    @property
    def context_label(self) -> str:
        """Human-readable label for contexts."""
        return "Database"

    @property
    def supports_comparison_report(self) -> bool:
        """Text2SQL supports per-database comparison reports."""
        return True

    @property
    def phase1_short_label(self) -> str:
        """Short label for Phase 1 context."""
        return "DB"

    @property
    def is_hierarchical(self) -> bool:
        """Text2SQL is hierarchical: each database contains multiple questions."""
        return True

    def load_agent_results(self, agent_dir: Path, contexts: List[str]) -> Dict[str, Any]:
        """
        Load evaluation results from an agent's output directory.

        For Text2SQL, results are stored in:
        agent_dir/db_name/results/evaluation.json

        Args:
            agent_dir: Path to agent's output directory
            contexts: List of database names to load results for

        Returns:
            Dict with overall_accuracy, total_questions, correct, and by_context
        """
        results = {
            'overall_accuracy': 0.0,
            'total_questions': 0,
            'correct': 0,
            'by_context': {},
        }

        total_questions = 0
        total_correct = 0

        for db_name in contexts:
            eval_file = agent_dir / db_name / "results" / "evaluation.json"

            if eval_file.exists():
                try:
                    with open(eval_file) as f:
                        db_data = json.load(f)

                    questions = db_data.get('total_questions', 0)
                    correct = db_data.get('correct', 0)
                    accuracy = (correct / questions * 100) if questions > 0 else 0.0

                    results['by_context'][db_name] = {
                        'accuracy': accuracy,
                        'correct': correct,
                        'total': questions,
                    }
                    total_questions += questions
                    total_correct += correct

                except (json.JSONDecodeError, IOError) as e:
                    self.logger.warning(f"Could not load {eval_file}: {e}")
                    results['by_context'][db_name] = {
                        'accuracy': 0.0,
                        'correct': 0,
                        'total': 0,
                        'error': str(e),
                    }
            else:
                results['by_context'][db_name] = {
                    'accuracy': 0.0,
                    'correct': 0,
                    'total': 0,
                    'error': 'evaluation.json not found',
                }

        results['total_questions'] = total_questions
        results['correct'] = total_correct
        results['overall_accuracy'] = (total_correct / total_questions * 100) if total_questions > 0 else 0.0

        return results

    def setup_context_workspace(
        self,
        workspace: Path,
        context_name: str,
        agent_path: Path,
        problem: Optional[Dict] = None
    ) -> None:
        """
        Set up workspace for evaluating a single database.

        Creates:
        - .claude/agents/agent.md
        - eval_instructions.md
        - tools/ (if exists in agent_path)
        - database.sqlite symlink
        - output/, tool_output/, results/ directories
        """
        # Copy agent.md to .claude/agents/
        agents_dir = workspace / ".claude" / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(agent_path / "agent.md", agents_dir / "agent.md")

        # Copy eval_instructions.md to workspace root
        shutil.copy2(agent_path / "eval_instructions.md", workspace / "eval_instructions.md")

        # Copy tools if present
        tools_src = agent_path / "tools"
        if tools_src.exists() and tools_src.is_dir():
            workspace_tools = workspace / "tools"
            if workspace_tools.exists():
                shutil.rmtree(workspace_tools)
            shutil.copytree(tools_src, workspace_tools)

        # Create database symlink (Phase 1 input)
        self.prepare_phase1_input(workspace, context_name, problem)

        # Create required directories
        (workspace / "output").mkdir(exist_ok=True)
        (workspace / "tool_output").mkdir(exist_ok=True)
        (workspace / "results").mkdir(exist_ok=True)

    def run_evaluation_in_workspace(
        self,
        workspace: Path,
        problems: List[Dict],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run Phase 1 + Phase 2 + evaluation for a single database.

        The workspace must already be set up via setup_context_workspace().

        This method owns Phase 1 caching internally (Text2SQL has 1:N relationship
        between Phase 1 analysis and Phase 2 SQL generations - caching saves work).

        Args:
            workspace: Workspace directory (already set up)
            problems: List of problem dicts for this database
            config: Configuration dict with:
                - eval_model: Model for Phase 2 SQL generation
                - analysis_model: Model for Phase 1 analysis
                - timeout: Timeout for operations
                - cache_manager: Optional CacheManager for Phase 1 caching
                - api_key: API key for model calls
                - experiment_dir: Experiment directory for run_dir
                - Additional Text2SQL-specific options

        Returns:
            Dict with:
            - accuracy: Accuracy percentage (0-100)
            - correct: Number of correct solutions
            - total: Total number of problems
            - phase1_cost_info: Cost info from Phase 1 (if cache miss)
            - phase2_cost: API cost for Phase 2
            - evaluation: Full evaluation dict for saving
            - error: Optional error message if evaluation failed
        """
        import tempfile
        import os
        from RoboPhD.agent_orchestrator import AgentOrchestrator
        from RoboPhD.core import SQLGenerator, Evaluator
        from RoboPhD.config import API_KEY_ENV_VAR
        from RoboPhD.utilities.cached_sql_executor import close_database_connections

        # Extract config
        eval_model = config.get('eval_model', 'haiku-4.5')
        analysis_model = config.get('analysis_model', 'haiku-4.5')
        timeout = config.get('timeout', 1800)
        verification_retries = config.get('verification_retries', 2)
        temperature_strategy = config.get('temperature_strategy', 'progressive')
        debug_log_probability = config.get('debug_log_probability', 0.02)
        llm_call_timeout = config.get('llm_call_timeout', 120)
        agent_id = config.get('agent_id', 'agent')
        cache_manager = config.get('cache_manager')
        api_key = config.get('api_key') or os.getenv(API_KEY_ENV_VAR)
        experiment_dir = config.get('experiment_dir')

        # Get database name from workspace (assumes workspace path ends with db_name)
        db_name = workspace.name
        db_path = self.db_root / db_name / f"{db_name}.sqlite"

        result = {
            'accuracy': 0.0,
            'correct': 0,
            'total': len(problems),
            'phase2_cost': 0.0,
            'phase1_cost_info': None,
            'evaluation': None,
        }

        if not problems:
            return result

        try:
            # Phase 1: Database analysis (with caching)
            phase1_cost_info = None
            prompt_content = None
            success = False

            # Check Phase 1 cache before running analysis
            cache_key = None
            if cache_manager:
                try:
                    cache_key = cache_manager.get_phase1_cache_key(agent_id, db_name, db_path)
                    cached_output = cache_manager.get_phase1_cache(cache_key)

                    if cached_output:
                        # Cache hit - use cached output
                        self.logger.debug(f"Phase 1 cache hit for {agent_id}/{db_name}")

                        # Create output directory and write cached content
                        output_dir = workspace / "output"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        (output_dir / "agent_output.txt").write_text(cached_output)

                        # Combine with eval_instructions for Phase 2
                        eval_instructions_file = workspace / "eval_instructions.md"
                        if eval_instructions_file.exists():
                            eval_instructions = eval_instructions_file.read_text()
                            prompt_content = f"{cached_output}\n\n---\n\n{eval_instructions}"
                            (output_dir / "system_prompt.txt").write_text(prompt_content)
                            success = True
                        # phase1_cost_info remains None (no CLI call made)
                except Exception as cache_error:
                    self.logger.warning(f"Cache check failed for {db_name}: {cache_error}")
                    # Continue without cache on error

            if not success:
                # Cache miss - run Phase 1 normally
                orchestrator = AgentOrchestrator(
                    base_experiment_dir=workspace.parent,
                    analysis_model=analysis_model,
                    timeout_phase1=timeout,
                    domain=self
                )

                success, prompt_content, phase1_cost_info, tool_error = orchestrator.run_phase1(
                    workspace=workspace,
                    agent_id=agent_id,
                    database_name=db_name
                )

                if tool_error:
                    result['tool_only_error'] = tool_error

                # Save to cache if successful
                if success and cache_manager and cache_key:
                    try:
                        output_file = workspace / "output" / "agent_output.txt"
                        if output_file.exists():
                            output_content = output_file.read_text()
                            cache_manager.save_phase1_cache(
                                cache_key,
                                output_content,
                                agent_name=agent_id,
                                db_name=db_name
                            )
                    except Exception as cache_error:
                        self.logger.warning(f"Cache save failed for {db_name}: {cache_error}")

            result['phase1_cost_info'] = phase1_cost_info

            # Close connections after Phase 1 analysis
            # Phase 1 may have opened connections to analyze schema
            close_database_connections(str(db_path))

            if not success:
                result['error'] = 'Phase 1 failed'
                result['evaluation'] = {
                    'database': db_name,
                    'total_questions': len(problems),
                    'correct': 0,
                    'accuracy': 0.0,
                    'error': 'Phase 1 failed',
                    'results': {}
                }
                return result

            # Get system prompt file
            prompt_file = workspace / "output" / "system_prompt.txt"
            if not prompt_file.exists():
                error_msg = f"System prompt file not found: {prompt_file}"
                result['error'] = error_msg
                result['evaluation'] = {
                    'database': db_name,
                    'total_questions': len(problems),
                    'correct': 0,
                    'accuracy': 0.0,
                    'error': error_msg,
                    'results': {}
                }
                return result

            # Phase 2: SQL generation
            # Use TemporaryDirectory for automatic cleanup
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                temp_questions_path = temp_path / "questions.json"
                with open(temp_questions_path, 'w') as f:
                    json.dump(problems, f)

                if not api_key:
                    error_msg = f"API key not found in {API_KEY_ENV_VAR}"
                    result['error'] = error_msg
                    result['evaluation'] = {
                        'database': db_name,
                        'total_questions': len(problems),
                        'correct': 0,
                        'accuracy': 0.0,
                        'error': error_msg,
                        'results': {}
                    }
                    return result

                sql_generator = SQLGenerator(
                    eval_model=eval_model,
                    questions_file=temp_questions_path,
                    timeout=timeout,
                    use_evidence=self.config.get('use_evidence', True),
                    api_key=api_key,
                    verification_retries=verification_retries,
                    temperature_strategy=temperature_strategy,
                    debug_log_probability=debug_log_probability,
                    llm_call_timeout=llm_call_timeout,
                    run_dir=experiment_dir,
                    agent_id=agent_id
                )

                results_dir = workspace / "results"
                results_dir.mkdir(exist_ok=True)
                predictions_path = results_dir / "predictions.json"

                try:
                    gen_result, phase2_cost = sql_generator.generate(
                        prompt_file=prompt_file,
                        db_name=db_name,
                        db_path=db_path,
                        output_path=predictions_path,
                        agent_id=agent_id
                    )
                finally:
                    # Close connections after Phase 2 SQL generation
                    close_database_connections(str(db_path))

                result['phase2_cost'] = phase2_cost

                # Extract Phase 2 cache stats if available
                if gen_result and 'metadata' in gen_result and 'phase2_cache_stats' in gen_result['metadata']:
                    result['phase2_cache_stats'] = gen_result['metadata']['phase2_cache_stats']

                # Extract validation stats if available
                if gen_result and 'metadata' in gen_result and 'validation_stats' in gen_result['metadata']:
                    result['validation_stats'] = gen_result['metadata']['validation_stats']

                # Evaluate
                evaluator = Evaluator(
                    questions_file=temp_questions_path,
                    db_root=self.db_root
                )

                evaluation = evaluator.evaluate(
                    predictions=gen_result,
                    db_name=db_name,
                    save_to=results_dir / "evaluation.json"
                )

                result['accuracy'] = evaluation.get('accuracy', 0.0)
                result['correct'] = evaluation.get('correct', 0)
                result['total'] = evaluation.get('total_questions', len(problems))
                result['evaluation'] = evaluation

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Evaluation failed for {db_name}: {e}")
            # Ensure connections are closed on exception
            try:
                close_database_connections(str(db_path))
            except Exception:
                pass

        return result

    def sample_problems(
        self,
        config: Dict[str, Any],
        rng: random.Random,
        available_contexts: Optional[List[str]] = None
    ) -> SampledProblems:
        """
        Sample databases and questions for an iteration.

        Uses two-level hierarchy:
        1. Sample contexts_per_iteration databases
        2. Sample problems_per_context questions from each

        Args:
            config: Configuration with contexts_per_iteration, problems_per_context
            rng: Random number generator for reproducibility
            available_contexts: Optional list of database names to sample from

        Returns:
            SampledProblems with sampled databases and questions
        """
        contexts_per_iteration = config.get("contexts_per_iteration", 5)
        problems_per_context = config.get("problems_per_context", 30)

        # Get available databases
        if available_contexts is None:
            available_contexts = self.get_contexts()

        # Sample databases
        if len(available_contexts) <= contexts_per_iteration:
            sampled_databases = list(available_contexts)
        else:
            sampled_databases = rng.sample(available_contexts, contexts_per_iteration)

        # Sort alphabetically for reproducibility (matching researcher.py behavior)
        sampled_databases = sorted(sampled_databases)

        # Load all questions and sample for each database
        all_questions_by_db = self._load_questions()
        questions_by_database: Dict[str, List[Dict]] = {}

        for db_name in sampled_databases:
            db_questions = all_questions_by_db.get(db_name, [])
            if len(db_questions) <= problems_per_context:
                questions_by_database[db_name] = list(db_questions)
            else:
                questions_by_database[db_name] = rng.sample(db_questions, problems_per_context)

        return SampledProblems(
            contexts=sampled_databases,
            problems_by_context=questions_by_database
        )

    # === Additional Text2SQL-specific methods ===
    # These are used by existing code and preserved for backward compatibility

    def get_sql_generator(self, **kwargs) -> 'SQLGenerator':
        """
        Get an SQLGenerator instance with domain config.

        This is a convenience method for existing code that needs
        direct access to SQLGenerator.
        """
        from RoboPhD.core import SQLGenerator

        return SQLGenerator(
            eval_model=kwargs.get('eval_model', self.config.get('eval_model')),
            questions_file=self.questions_file,
            timeout=kwargs.get('timeout', self.config.get('timeout', 3600)),
            use_evidence=kwargs.get('use_evidence', self.config.get('use_evidence', True)),
            sql_validation_timeout=kwargs.get('sql_validation_timeout', self.config.get('sql_validation_timeout', 30)),
            verification_retries=kwargs.get('verification_retries', self.config.get('verification_retries', 2)),
            temperature_strategy=kwargs.get('temperature_strategy', self.config.get('temperature_strategy', 'progressive')),
            debug_log_probability=kwargs.get('debug_log_probability', self.config.get('debug_log_probability', 0.02)),
            logger=self.logger,
            api_key=kwargs.get('api_key', self.config.get('api_key')),
            run_dir=kwargs.get('run_dir', self.config.get('run_dir')),
            agent_id=kwargs.get('agent_id', self.config.get('agent_id')),
            llm_call_timeout=kwargs.get('llm_call_timeout', self.config.get('llm_call_timeout', 120))
        )

    def get_evaluator(self) -> 'Evaluator':
        """
        Get an Evaluator instance with domain config.

        This is a convenience method for existing code that needs
        direct access to Evaluator.
        """
        from RoboPhD.core import Evaluator

        return Evaluator(
            questions_file=self.questions_file,
            db_root=self.db_root,
            logger=self.logger
        )

    def run_evaluation(
        self,
        sampled: SampledProblems,
        agent_path: Path,
        output_dir: Path,
        config: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Run evaluation on sampled problems with given agent.

        Orchestrates the full Text2SQL evaluation pipeline with parallelism:
        1. Set up workspaces for each database
        2. Run Phase 1 (analysis) + Phase 2 (SQL generation) in parallel
        3. Aggregate results and track costs

        Args:
            sampled: SampledProblems with databases and questions
            agent_path: Path to agent directory (agent.md, eval_instructions.md, tools/)
            output_dir: Directory to write evaluation outputs
            config: Configuration with eval_model, analysis_model, timeouts, etc.
                Required keys:
                - eval_model: Model for SQL generation
                - analysis_model: Model for database analysis
                - phase2_timeout: Timeout for Phase 2 generation
                - max_concurrent: Maximum parallel database evaluations
                - agent_id: Agent identifier
                - api_key: API key for model calls
                - cache_manager: Optional CacheManager for Phase 1 caching
                - experiment_dir: Experiment directory

        Returns:
            EvaluationResult with accuracy, per-question results, and cost metadata
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Extract config
        analysis_model = config.get('analysis_model', 'haiku-4.5')
        max_concurrent = config.get('max_concurrent', 8)
        agent_id = config.get('agent_id', agent_path.name)

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Track results and costs
        all_results: List[Dict[str, Any]] = []
        total_correct = 0
        total_questions = 0
        total_phase1_cost = 0.0
        total_phase2_cost = 0.0
        phase1_cache_hits = 0
        phase1_cache_misses = 0
        phase1_failures = []
        # Token-level tracking
        total_phase1_tokens_in = 0
        total_phase1_tokens_out = 0
        total_phase1_cache_created = 0
        total_phase1_cache_read = 0

        def process_database(db_name: str) -> Dict[str, Any]:
            """Process a single database - runs in thread pool."""
            questions = sampled.problems_by_context.get(db_name, [])
            if not questions:
                return {
                    'db_name': db_name,
                    'results': [],
                    'correct': 0,
                    'total': 0,
                    'phase1_cost': 0.0,
                    'phase2_cost': 0.0,
                    'cache_hit': False,
                    'error': None
                }

            db_path = self.db_root / db_name / f"{db_name}.sqlite"

            # Create workspace directly under output_dir (simplified structure)
            # Output: output_dir / db_name / results / evaluation.json
            workspace = output_dir / db_name
            workspace.mkdir(parents=True, exist_ok=True)

            # Set up workspace artifacts
            self.setup_context_workspace(
                workspace=workspace,
                context_name=db_name,
                agent_path=agent_path,
                problem=None  # Text2SQL doesn't use problem dict for setup
            )

            # Build config for run_evaluation_in_workspace
            workspace_config = {
                'eval_model': config.get('eval_model', 'haiku-4.5'),
                'analysis_model': analysis_model,
                'timeout': config.get('phase2_timeout', 1800),
                'agent_id': agent_id,
                'cache_manager': config.get('cache_manager'),
                'verification_retries': config.get('verification_retries', 2),
                'temperature_strategy': config.get('temperature_strategy', 'progressive'),
                'debug_log_probability': config.get('debug_log_probability', 0.02),
                'llm_call_timeout': config.get('llm_call_timeout', 120),
                'api_key': config.get('api_key'),
                'experiment_dir': config.get('experiment_dir'),
            }

            # Run evaluation in workspace (handles Phase 1 + Phase 2 + evaluation)
            result = self.run_evaluation_in_workspace(
                workspace=workspace,
                problems=questions,
                config=workspace_config
            )

            # Convert evaluation results to per-question format
            db_results = []
            evaluation = result.get('evaluation', {})
            eval_results = evaluation.get('results', {}) if evaluation else {}

            for q in questions:
                qid = str(q['question_id'])
                q_result = eval_results.get(qid, {})
                correct = q_result.get('matches', False)

                db_results.append({
                    'question_id': q.get('question_id'),
                    'db_id': db_name,
                    'correct': correct,
                    'predicted_sql': q_result.get('predicted_sql', ''),
                    'ground_truth_sql': q.get('SQL', ''),
                    'error': result.get('error') if not correct else None
                })

            # Determine if this was a cache hit (no phase1_cost_info means cache hit)
            cache_hit = result.get('phase1_cost_info') is None and result.get('error') is None

            # Extract token details from phase1_cost_info
            phase1_info = result.get('phase1_cost_info') or {}

            return {
                'db_name': db_name,
                'results': db_results,
                'correct': result.get('correct', 0),
                'total': result.get('total', len(questions)),
                'phase1_cost': phase1_info.get('cost', 0.0),
                'phase1_tokens_in': phase1_info.get('tokens_in', 0),
                'phase1_tokens_out': phase1_info.get('tokens_out', 0),
                'phase1_cache_created': phase1_info.get('cache_created', 0),
                'phase1_cache_read': phase1_info.get('cache_read', 0),
                'phase2_cost': result.get('phase2_cost', 0.0),
                'cache_hit': cache_hit,
                'error': result.get('error')
            }

        # Process databases in parallel
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {
                executor.submit(process_database, db_name): db_name
                for db_name in sampled.contexts
            }

            for future in as_completed(futures):
                db_name = futures[future]
                try:
                    result = future.result()

                    # Aggregate results
                    all_results.extend(result['results'])
                    total_correct += result['correct']
                    total_questions += result['total']
                    total_phase1_cost += result['phase1_cost']
                    total_phase2_cost += result['phase2_cost']
                    # Aggregate token counts
                    total_phase1_tokens_in += result['phase1_tokens_in']
                    total_phase1_tokens_out += result['phase1_tokens_out']
                    total_phase1_cache_created += result['phase1_cache_created']
                    total_phase1_cache_read += result['phase1_cache_read']

                    if result['cache_hit']:
                        phase1_cache_hits += 1
                    else:
                        phase1_cache_misses += 1

                    if result['error']:
                        phase1_failures.append(db_name)

                except Exception as e:
                    self.logger.error(f"Failed to process database {db_name}: {e}")
                    # Add failed results for all questions in this database
                    questions = sampled.problems_by_context.get(db_name, [])
                    for q in questions:
                        all_results.append({
                            'question_id': q.get('question_id'),
                            'db_id': db_name,
                            'correct': False,
                            'error': str(e)
                        })
                    total_questions += len(questions)
                    phase1_failures.append(db_name)

        # Calculate accuracy
        accuracy = (100.0 * total_correct / total_questions) if total_questions > 0 else 0.0

        return EvaluationResult(
            accuracy=accuracy,
            total=total_questions,
            correct=total_correct,
            results=all_results,
            metadata={
                'databases': sampled.contexts,
                'agent_path': str(agent_path),
                'phase1_cost': total_phase1_cost,
                'phase2_cost': total_phase2_cost,
                'phase1_cache_hits': phase1_cache_hits,
                'phase1_cache_misses': phase1_cache_misses,
                'phase1_failures': phase1_failures,
                # Token-level tracking
                'phase1_tokens_in': total_phase1_tokens_in,
                'phase1_tokens_out': total_phase1_tokens_out,
                'phase1_cache_created': total_phase1_cache_created,
                'phase1_cache_read': total_phase1_cache_read,
            }
        )
