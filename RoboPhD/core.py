"""
Core utilities and shared classes for RoboPhD system.

This module contains shared functionality across the RoboPhD research system,
including SQL generation, evaluation, and database management.
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime

# Import evaluation utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from RoboPhD.config import API_KEY_ENV_VAR
from utilities.cached_sql_executor import CachedSQLExecutor
from evaluation.evaluate_predictions import evaluate_predictions as eval_predictions_fn


class SQLGenerator:
    """
    Shared SQL generation logic for RoboPhD system.
    
    This class encapsulates the common pattern of:
    1. Building subprocess command for questions_to_sql_prompt_based.py
    2. Managing temporary/permanent output files
    3. Handling timeouts and errors consistently
    4. Extracting cost information
    5. Supporting evidence toggle for BIRD evaluation modes
    """
    
    def __init__(self,
                 eval_model: str,
                 questions_file: Path,
                 timeout: int = 3600,
                 use_evidence: bool = True,
                 sql_validation_timeout: int = 30,
                 verification_retries: int = 2,
                 temperature_strategy: str = "progressive",
                 debug_log_probability: float = 0.02,
                 logger: Optional[logging.Logger] = None,
                 api_key: str = None,
                 run_dir: Optional[Path] = None,
                 agent_id: Optional[str] = None,
                 llm_call_timeout: int = 120):
        """
        Initialize SQL generator.

        Args:
            eval_model: Model to use for SQL generation (e.g., 'sonnet-4.5', 'haiku-4.5')
            questions_file: Path to questions JSON file
            timeout: Timeout in seconds for SQL generation
            use_evidence: Whether to include evidence in prompts (default True)
            sql_validation_timeout: Timeout in seconds for SQL validation (default 30)
            verification_retries: Number of verification attempts (default 2, 0 = current behavior)
            temperature_strategy: Temperature strategy for verification retries (default: progressive)
            debug_log_probability: Probability (0.0-1.0) of logging API calls for debugging (default 0.02)
            logger: Logger instance (creates one if not provided)
            api_key: API key for Anthropic API
            run_dir: Research run directory for Phase 2 caching (optional)
            agent_id: Agent identifier for Phase 2 caching (optional)
            llm_call_timeout: Per-call LLM timeout in seconds (default 120, for local models)
        """
        self.eval_model = eval_model
        self.questions_file = Path(questions_file)
        self.timeout = timeout
        self.use_evidence = use_evidence
        self.sql_validation_timeout = sql_validation_timeout
        self.verification_retries = verification_retries
        self.temperature_strategy = temperature_strategy
        self.debug_log_probability = debug_log_probability
        self.logger = logger or logging.getLogger(__name__)
        self.api_key = api_key
        self.run_dir = Path(run_dir) if run_dir else None
        self.agent_id = agent_id
        self.llm_call_timeout = llm_call_timeout
        self.script_path = self._find_script_path()

        if not api_key:
            raise Exception("must pass the api key in to use for sql generation")
    
    def _find_script_path(self) -> Path:
        """Find questions_to_sql_prompt_based.py script."""
        # Try same directory as this module
        script_dir = Path(__file__).parent
        script_path = script_dir / 'questions_to_sql_prompt_based.py'
        
        # Fallback to current directory if not found
        if not script_path.exists():
            script_path = Path('questions_to_sql_prompt_based.py')
            if not script_path.exists():
                # Try in RoboPhD directory during transition
                script_path = Path('RoboPhD/questions_to_sql_prompt_based.py')
                if not script_path.exists():
                    raise FileNotFoundError(
                        "Could not find questions_to_sql_prompt_based.py in "
                        f"{script_dir}, current directory, or RoboPhD/"
                    )
        
        return script_path
    
    def generate(self,
                 prompt_file: Path,
                 db_name: str,
                 db_path: Path,
                 output_path: Optional[Path] = None,
                 limit: Optional[int] = None,
                 use_evidence: Optional[bool] = None,
                 agent_id: Optional[str] = None) -> Tuple[Dict, float]:
        """
        Generate SQL predictions using prompt.
        
        Args:
            prompt_file: Path to system prompt file
            db_name: Database name to filter questions
            db_path: Path to database file for SQL validation
            output_path: Where to save predictions (uses temp file if None)
            limit: Limit number of questions to process
            use_evidence: Override instance setting for evidence usage
            
        Returns:
            Tuple of (predictions dict, cost float)
        """
        # Allow per-call override of evidence setting
        use_evidence = self.use_evidence if use_evidence is None else use_evidence

        # Use temp file if no output specified
        cleanup = False
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                suffix=f'_{db_name}_predictions.json',
                delete=False
            )
            output_path = Path(temp_file.name)
            temp_file.close()
            cleanup = True
        else:
            output_path = Path(output_path)

        try:
            if not db_path.exists():
                raise RuntimeError(f"Can't find db path ${db_path} to generate sql")

            # Create debug directory if debug logging enabled
            debug_log_dir = None
            if self.debug_log_probability > 0:
                # Determine workspace directory from output_path
                if output_path.parent:
                    workspace_dir = output_path.parent
                else:
                    workspace_dir = Path.cwd()

                debug_log_dir = workspace_dir / "debug"
                debug_log_dir.mkdir(parents=True, exist_ok=True)

            # Build command
            cmd = [
                'python', str(self.script_path),
                '--prompt', str(prompt_file),
                '--questions', str(self.questions_file),
                '--db_name', db_name,
                '--db_path', str(db_path.resolve()),
                '--output', str(output_path),
                '--model', self.eval_model,
                '--api_key', self.api_key,
                '--sql_validation_timeout', str(self.sql_validation_timeout),
                '--verification_retries', str(self.verification_retries),
                '--temperature_strategy', self.temperature_strategy,
                '--llm-call-timeout', str(self.llm_call_timeout)
            ]

            # Add run_dir for Phase 2 caching if available
            if self.run_dir:
                cmd.extend(['--run-dir', str(self.run_dir)])

            # Add agent_id for Phase 2 caching if available
            if self.agent_id:
                cmd.extend(['--agent-id', self.agent_id])

            if self.debug_log_probability > 0:
                cmd.extend(['--debug-log-probability', str(self.debug_log_probability)])
                if debug_log_dir:
                    cmd.extend(['--debug-log-dir', str(debug_log_dir)])

            if not use_evidence:
                cmd.append('--no-evidence')

            if limit:
                cmd.extend(['--limit', str(limit)])

            env = os.environ.copy()

            # Run generation (progress messages printed by questions_to_sql_prompt_based.py)
            start_time = datetime.now()
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=env
                )
                
                if result.returncode != 0:
                    error_msg = result.stderr
                    # Check for rate limit errors - these should abort the entire run
                    if "API_RATE_LIMIT" in error_msg:
                        raise RuntimeError(f"API_RATE_LIMIT: Rate limit exceeded, cannot continue. {error_msg}")
                    raise RuntimeError(f"SQL generation failed: {error_msg}")
                
            except subprocess.TimeoutExpired:
                self.logger.error(f"SQL generation timed out after {self.timeout}s")
                return {}, 0.0

            # Print only rate limit related messages from SQL generation
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if 'rate limit' in line.lower() or '⏸️' in line or '⏯️' in line:
                        print(line)

            # Load predictions
            if not output_path.exists():
                self.logger.error(f"No predictions file created at {output_path}")
                return {}, 0.0

            with open(output_path, 'r') as f:
                predictions = json.load(f)

            # Extract cost if present
            cost = 0.0
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if 'Total cost:' in line:
                        try:
                            cost = float(line.split('$')[-1].strip())
                        except:
                            pass
            
            # Log timing with cache stats if available
            elapsed = (datetime.now() - start_time).total_seconds()
            timestamp = datetime.now().strftime("%H:%M:%S")
            agent_prefix = f"{agent_id} | " if agent_id else ""

            # Extract cache stats from metadata
            cache_info = ""
            if 'metadata' in predictions and 'phase2_cache_stats' in predictions['metadata']:
                stats = predictions['metadata']['phase2_cache_stats']
                hits = stats.get('hits', 0)
                if hits > 0:
                    cache_info = f", {hits} cache hits"

            print(f"    [{timestamp}] {agent_prefix}{db_name}: Generated {len(predictions.get('predictions', []))} queries "
                  f"in {elapsed:.1f}s (cost: ${cost:.2f}{cache_info})")

            return predictions, cost
            
        finally:
            # Clean up temporary file if needed
            if cleanup and output_path.exists():
                try:
                    output_path.unlink()
                except:
                    pass  # Ignore cleanup errors


class Evaluator:
    """
    Shared evaluation logic for RoboPhD system.
    
    This class encapsulates:
    1. Running predictions against ground truth
    2. Calculating accuracy metrics
    3. Handling evaluation errors and timeouts
    4. Generating consistent evaluation reports
    """
    
    def __init__(self,
                 questions_file: Path,
                 db_root: Path,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize evaluator.
        
        Args:
            questions_file: Path to questions JSON with ground truth
            db_root: Root directory containing database files
            logger: Logger instance
        """
        self.questions_file = Path(questions_file)
        self.db_root = Path(db_root)
        self.logger = logger or logging.getLogger(__name__)
        
        # Load questions once
        with open(self.questions_file, 'r') as f:
            self.questions = json.load(f)
    
    def evaluate(self,
                 predictions: Dict,
                 db_name: str,
                 save_to: Optional[Path] = None) -> Dict:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: Predictions dictionary from SQLGenerator
            db_name: Database name
            save_to: Optional path to save evaluation results
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not predictions or 'predictions' not in predictions:
            self.logger.warning(f"No predictions to evaluate for {db_name}")
            return {
                'database': db_name,
                'total_questions': 0,
                'correct': 0,
                'accuracy': 0.0,
                'results': {},
                'metadata': {}
            }
        
        # Use the shared evaluation function
        try:
            # Convert predictions to the format expected by evaluate_predictions
            # It expects a dict with question_id as keys, not a list
            if 'predictions' in predictions:
                pred_data = predictions['predictions']
                if isinstance(pred_data, list):
                    # Convert list to dict with question_id as key
                    # Add BIRD format marker if not present
                    predictions_dict = {}
                    for pred in pred_data:
                        qid = str(pred.get('question_id', ''))
                        sql = pred.get('SQL', '')
                        if qid and sql:
                            # Add BIRD marker if not present
                            if '\t----- bird -----\t' not in sql:
                                # Need to get the database name from the prediction
                                db = pred.get('db_id', db_name)  
                                sql = f"{sql}\t----- bird -----\t{db}"
                            predictions_dict[qid] = sql
                    predictions_to_save = {'predictions': predictions_dict}
                elif isinstance(pred_data, dict):
                    predictions_to_save = predictions
                else:
                    predictions_to_save = {'predictions': {}}
            else:
                predictions_to_save = predictions
            
            # Save predictions to temp file for evaluation
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                temp_pred_file = f.name
                json.dump(predictions_to_save, f)
            
            # Run evaluation (suppress verbose output)
            eval_results = eval_predictions_fn(
                predictions_file=temp_pred_file,
                dev_data_file=str(self.questions_file),
                db_root=str(self.db_root),
                timeout_seconds=300,
                gt_timeout_seconds=300,
                verbose=False
            )
            
            # Clean up temp file
            try:
                Path(temp_pred_file).unlink()
            except:
                pass
            
            # The evaluation results should already be for this database only
            # since we passed predictions for just this database
            # Handle both dict and list formats for results
            if 'results' in eval_results:
                # Check if results is a dict or list
                results = eval_results['results']
                if isinstance(results, dict):
                    # Already a dict, use as-is
                    pass
                elif isinstance(results, list):
                    # If it's a list, convert to dict with index as key
                    eval_results['results'] = {str(idx): r for idx, r in enumerate(results)}
                else:
                    # Neither dict nor list, create empty dict
                    eval_results['results'] = {}
                
                # Check if we have detailed_results with verification info
                detailed_results = predictions.get('detailed_results', [])
                verification_info_map = {}
                for detail in detailed_results:
                    detail_qid = str(detail.get('question_id', ''))
                    if detail_qid and 'verification_info' in detail:
                        verification_info_map[detail_qid] = detail['verification_info']

                # Note: question/evidence/difficulty are now added by evaluate_predictions.py
                # We only add verification_info here if available
                for qid, result in eval_results['results'].items():
                    # Add verification info if available
                    if qid in verification_info_map:
                        result['verification_info'] = verification_info_map[qid]
            
            # Calculate summary metrics
            # At this point, results should be a dict (we converted it above if it was a list)
            results_dict = eval_results.get('results', {})
            total = len(results_dict)
            correct = sum(1 for r in results_dict.values() if r.get('matches', False))

            # Count different types of errors (APE format)
            prediction_errors = 0
            prediction_timeouts = 0
            ground_truth_errors = 0
            ground_truth_timeouts = 0

            for qid, result in results_dict.items():
                # Check prediction errors
                pred_error = result.get('predicted_error')
                if pred_error:
                    if 'timeout' in str(pred_error).lower():
                        prediction_timeouts += 1
                    else:
                        prediction_errors += 1

                # Check ground truth errors
                gt_error = result.get('ground_truth_error')
                if gt_error:
                    if 'timeout' in str(gt_error).lower():
                        ground_truth_timeouts += 1
                    else:
                        ground_truth_errors += 1

            # Calculate accuracy excluding questions with ground truth errors
            # Questions with GT errors cannot be evaluated (we don't know if prediction is correct)
            excluded_questions = ground_truth_errors + ground_truth_timeouts
            evaluable_questions = total - excluded_questions
            accuracy = (correct / evaluable_questions * 100.0) if evaluable_questions > 0 else 0.0

            evaluation = {
                'database': db_name,
                'total_questions': total,
                'evaluable_questions': evaluable_questions,
                'excluded_questions': excluded_questions,
                'correct': correct,
                'accuracy': accuracy,
                'prediction_errors': prediction_errors,
                'prediction_timeouts': prediction_timeouts,
                'ground_truth_errors': ground_truth_errors,
                'ground_truth_timeouts': ground_truth_timeouts,
                'results': eval_results.get('results', {}),
                'metadata': {
                    'evaluated_at': datetime.now().isoformat(),
                    'questions_file': str(self.questions_file),
                    'db_root': str(self.db_root)
                }
            }
            
            # Save if requested
            if save_to:
                save_to = Path(save_to)
                save_to.parent.mkdir(parents=True, exist_ok=True)
                with open(save_to, 'w') as f:
                    json.dump(evaluation, f, indent=2)
            
            return evaluation
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"Evaluation failed for {db_name}: {e}\n{error_details}")
            return {
                'database': db_name,
                'total_questions': 0,
                'evaluable_questions': 0,
                'excluded_questions': 0,
                'correct': 0,
                'accuracy': 0.0,
                'prediction_errors': 0,
                'prediction_timeouts': 0,
                'ground_truth_errors': 0,
                'ground_truth_timeouts': 0,
                'error_details': error_details,
                'metadata': {}
            }


class TestOutputGenerator:
    """
    Test dataset output generator for RoboPhD system.

    This class creates test predictions in the same JSON format as the input
    test.json, but adds a "predictedSQL" field to each entry without any
    ground truth comparison or accuracy calculation.
    """

    def __init__(self):
        """
        Initialize test output generator.

        Args:
        """
        pass

    def generate_output(self,
                       predictions: Dict,
                       questions:List) -> List:
        """
        Generate test output with predictions.

        Args:
            predictions: the str(question_id) -> bird-sql dictionary
            questions: the list of original input questions that were predicted

        Returns:
            Dictionary with test output format
        """
        if predictions is None:
            raise Exception("no output generated")

        # Generate output in same format as input test.json
        output = []

        for question in questions:            
            # Create output entry with same structure as input
            qid = question['question_id']
            output_entry = {
                "question_id": qid,
                "db_id": question['db_id'],
                "question": question['question'],
                "evidence": question.get('evidence', ''),
                "SQL": question.get('SQL', ''),  # Original (empty) SQL field
            }

            # Add predicted SQL
            predicted_sql = predictions.get(str(qid))
            if not predicted_sql:
                raise ValueError(f"could not find any prediction for questionid {qid}")
            
            if '\t----- bird -----\t' in predicted_sql:
                # Remove BIRD marker if present
                predicted_sql = predicted_sql.split('\t----- bird -----\t')[0].strip()
            output_entry["predicted_sql"] = predicted_sql
            output.append(output_entry)

        return output


class DatabaseManager:
    """
    Shared database discovery and management logic.

    Handles:
    1. Finding available databases
    2. Blacklist management
    3. Random sampling
    4. Database metadata
    """
    
    @classmethod
    def get_blacklisted_databases(cls, dataset: str = 'train') -> set:
        """
        Get blacklisted databases for a specific dataset.

        Args:
            dataset: Dataset name ('train', 'train-filtered', 'dev', 'test')

        Returns:
            Set of database names to exclude
        """
        blacklisted = set()

        # retail_world only blacklisted for 'train' (fixed in train-filtered)
        if dataset == 'train':
            blacklisted.add('retail_world')

        return blacklisted

    # Legacy support (will be removed)
    BLACKLISTED_DATABASES = {'donor', 'retail_world'}

    @classmethod
    def get_databases(cls,
                      db_root: Path,
                      blacklist: Optional[set] = None,
                      sample: Optional[int] = None,
                      seed: Optional[int] = None) -> list:
        """
        Get list of available databases.
        
        Args:
            db_root: Root directory containing databases
            blacklist: Additional databases to exclude
            sample: Number of databases to randomly sample
            seed: Random seed for sampling
            
        Returns:
            List of database names
        """
        import random
        
        # Combine blacklists
        excluded = cls.BLACKLISTED_DATABASES.copy()
        if blacklist:
            excluded.update(blacklist)
        
        # Find all database directories
        databases = []
        for db_dir in sorted(db_root.iterdir()):
            if db_dir.is_dir() and db_dir.name not in excluded:
                # Check for .sqlite file
                sqlite_files = list(db_dir.glob('*.sqlite'))
                if sqlite_files:
                    databases.append(db_dir.name)
        
        # Sample if requested
        if sample and sample < len(databases):
            if seed is not None:
                random.seed(seed)
            databases = random.sample(databases, sample)
        
        return sorted(databases)
    
    @classmethod
    def get_database_path(cls, db_root: Path, db_name: str) -> Path:
        """Get path to database SQLite file."""
        db_dir = db_root / db_name
        sqlite_files = list(db_dir.glob('*.sqlite'))
        if not sqlite_files:
            raise FileNotFoundError(f"No SQLite file found in {db_dir}")
        return sqlite_files[0]


def resolve_api_key():
    """Resolve API key from file or environment variable."""
    api_key_file = Path('.anthropic_key')

    # Check if API key file exists and has content
    if api_key_file.exists():
        try:
            with open(api_key_file, 'r') as f:
                file_api_key = f.read().strip()
            if file_api_key:
                print(f"🔑 Using API key from {api_key_file}")
                return file_api_key
        except Exception as e:
            print(f"⚠️ Error reading API key from {api_key_file}: {e}")

    # Fall back to environment variable
    env_api_key = os.getenv(API_KEY_ENV_VAR)
    if env_api_key:
        print(f"🔑 Using API key from environment variable {API_KEY_ENV_VAR}")
        return env_api_key

    # No API key found
    print(f"❌ No API key found in {api_key_file} or {API_KEY_ENV_VAR}")
    return None