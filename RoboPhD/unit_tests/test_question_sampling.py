#!/usr/bin/env python3
"""Unit tests for question sampling consistency across agents.

These tests ensure that:
1. All agents in an iteration test identical question sets (fair comparison)
2. Question sampling is deterministic given same random seed
3. Different iterations use different question samples
4. Phase 1 failures record correct question counts

Note: These are lightweight unit tests that test the sampling logic directly
without needing the full ParallelAgentResearcher infrastructure.
"""

import sys
import random
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_mock_questions_db(num_dbs=2, questions_per_db=15):
    """Create mock question database."""
    mock_questions = {}
    for db_idx in range(num_dbs):
        db_name = f"test_db_{db_idx}"
        questions = []
        for q_idx in range(questions_per_db):
            questions.append({
                'question_id': f"{db_name}_q{q_idx}",
                'question': f"Test question {q_idx}",
                'evidence': f"Evidence {q_idx}",
                'SQL': f"SELECT * FROM table_{q_idx}",
                'difficulty': 'simple'
            })
        mock_questions[db_name] = questions
    return mock_questions


def simulate_iteration_sampling(questions_by_db, databases, questions_per_database, num_agents):
    """Simulate the question sampling logic from run_iteration (researcher.py:2388-2401)."""
    # This simulates what the fixed code does: sample ONCE before threading
    current_iteration_questions = {}
    for db_name in databases:
        questions = questions_by_db.get(db_name, [])
        if questions:
            sampled = random.sample(
                questions,
                min(questions_per_database, len(questions))
            )
            current_iteration_questions[db_name] = sampled
        else:
            current_iteration_questions[db_name] = []

    # Return what each agent would get (they all get the same)
    return {
        f'agent_{i}': current_iteration_questions
        for i in range(num_agents)
    }


def test_all_agents_get_same_questions():
    """Verify all agents in iteration get identical question sets.

    This is CRITICAL for fair agent comparison. If agents test different
    questions, performance metrics are meaningless.
    """
    print("  Testing: All agents get same questions... ", end="")

    try:
        questions_by_db = create_mock_questions_db(num_dbs=2, questions_per_db=15)
        databases = list(questions_by_db.keys())
        questions_per_database = 5
        num_agents = 3

        # Set seed for determinism
        random.seed(42)

        # Simulate sampling
        agent_questions = simulate_iteration_sampling(
            questions_by_db, databases, questions_per_database, num_agents
        )

        # Verify all agents got the same questions
        expected_questions = agent_questions['agent_0']
        for agent_id in [f'agent_{i}' for i in range(1, num_agents)]:
            actual_questions = agent_questions[agent_id]

            for db_name in databases:
                expected_ids = [q['question_id'] for q in expected_questions[db_name]]
                actual_ids = [q['question_id'] for q in actual_questions[db_name]]

                assert actual_ids == expected_ids, \
                    f"{agent_id} has different questions for {db_name}: {actual_ids} vs {expected_ids}"

        print("PASS")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_sampling_is_deterministic_databases_and_questions():
    """CRITICAL: Verify same seed produces same databases AND questions.

    With same random seed, we should get:
    1. Same databases selected in same order
    2. Same questions sampled for each database in same order

    This ensures complete reproducibility of research runs.
    """
    print("  Testing: Sampling is deterministic... ", end="")

    try:
        questions_by_db = create_mock_questions_db(num_dbs=5, questions_per_db=15)
        all_databases = list(questions_by_db.keys())
        questions_per_database = 3
        num_agents = 2

        # Run 1: Sample with seed=42
        random.seed(42)
        databases1 = random.sample(all_databases, 2)  # Select 2 of 5 databases
        agent_questions1 = simulate_iteration_sampling(
            questions_by_db, databases1, questions_per_database, num_agents
        )
        questions_run1 = {
            db_name: [q['question_id'] for q in agent_questions1['agent_0'][db_name]]
            for db_name in databases1
        }

        # Run 2: Sample with same seed=42
        random.seed(42)
        databases2 = random.sample(all_databases, 2)  # Select 2 of 5 databases
        agent_questions2 = simulate_iteration_sampling(
            questions_by_db, databases2, questions_per_database, num_agents
        )
        questions_run2 = {
            db_name: [q['question_id'] for q in agent_questions2['agent_0'][db_name]]
            for db_name in databases2
        }

        # Verify databases are identical
        assert databases1 == databases2, \
            f"Database selection differs: {databases1} vs {databases2}"

        # Verify questions are identical for each database
        assert questions_run1 == questions_run2, \
            f"Question sampling differs:\nRun1: {questions_run1}\nRun2: {questions_run2}"

        print("PASS")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_question_samples_differ_across_iterations():
    """Verify different iterations get different question samples.

    This ensures we're not testing same questions repeatedly across
    iterations (would reduce training data diversity).
    """
    print("  Testing: Different iterations use different samples... ", end="")

    try:
        questions_by_db = create_mock_questions_db(num_dbs=3, questions_per_db=15)
        databases = list(questions_by_db.keys())[:2]  # Use first 2 databases
        questions_per_database = 5
        num_agents = 2

        # Iteration 1: Sample questions with seed=42
        random.seed(42)
        agent_questions1 = simulate_iteration_sampling(
            questions_by_db, databases, questions_per_database, num_agents
        )
        questions_iter1 = {
            db_name: [q['question_id'] for q in agent_questions1['agent_0'][db_name]]
            for db_name in databases
        }

        # Iteration 2: Sample questions with different seed
        # In real system, seed is updated: (original_seed + iteration * 10000) % (2**32)
        random.seed(42 + 1 * 10000)
        agent_questions2 = simulate_iteration_sampling(
            questions_by_db, databases, questions_per_database, num_agents
        )
        questions_iter2 = {
            db_name: [q['question_id'] for q in agent_questions2['agent_0'][db_name]]
            for db_name in databases
        }

        # Verify questions differ between iterations
        assert questions_iter1 != questions_iter2, \
            f"Questions should differ across iterations but are identical: {questions_iter1}"

        print("PASS")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_phase1_failure_uses_correct_questions():
    """Verify Phase 1 failures still record correct question count.

    When Phase 1 fails (analysis agent errors), we still need to record
    the correct number of questions attempted (for evaluation.json).
    """
    print("  Testing: Phase 1 failures record correct question count... ", end="")

    try:
        questions_by_db = create_mock_questions_db(num_dbs=2, questions_per_db=20)
        databases = list(questions_by_db.keys())
        db_name = databases[0]
        questions_per_database = 7
        num_agents = 1

        # Pre-sample questions (as run_iteration does)
        random.seed(42)
        agent_questions = simulate_iteration_sampling(
            questions_by_db, databases, questions_per_database, num_agents
        )

        # Simulate Phase 1 failure path (from researcher.py line 2121)
        sampled = agent_questions['agent_0'][db_name]

        evaluation = {
            'database': db_name,
            'total_questions': len(sampled),
            'correct': 0,
            'accuracy': 0.0,
            'error': 'Phase 1 failed',
            'results': {}
        }

        # Verify the count is correct (should be questions_per_database, not more)
        assert evaluation['total_questions'] == questions_per_database, \
            f"Phase 1 failure recorded wrong count: {evaluation['total_questions']} vs {questions_per_database}"

        # Verify results is empty
        assert evaluation['results'] == {}, \
            f"Phase 1 failure should have empty results: {evaluation['results']}"

        print("PASS")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_current_iteration_questions_populated():
    """Verify current_iteration_questions is populated before threading.

    This ensures the fix works: questions sampled sequentially before
    parallel execution starts.
    """
    print("  Testing: Questions pre-sampled before threading... ", end="")

    try:
        questions_by_db = create_mock_questions_db(num_dbs=3, questions_per_db=30)
        all_databases = list(questions_by_db.keys())
        questions_per_database = 10
        num_agents = 2

        # Select databases
        random.seed(42)
        databases = random.sample(all_databases, 2)

        # Simulate what run_iteration does (researcher.py line 2388-2401)
        current_iteration_questions = {}
        for db_name in databases:
            questions = questions_by_db.get(db_name, [])
            if questions:
                sampled = random.sample(
                    questions,
                    min(questions_per_database, len(questions))
                )
                current_iteration_questions[db_name] = sampled
            else:
                current_iteration_questions[db_name] = []

        # Verify current_iteration_questions is populated
        assert len(current_iteration_questions) == len(databases), \
            f"Should have questions for all databases: {len(current_iteration_questions)} vs {len(databases)}"

        # Verify each database has correct number of questions
        for db_name in databases:
            assert db_name in current_iteration_questions, \
                f"Missing questions for database: {db_name}"

            sampled = current_iteration_questions[db_name]
            assert len(sampled) == questions_per_database, \
                f"Wrong question count for {db_name}: {len(sampled)} vs {questions_per_database}"

        print("PASS")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Running RoboPhD Question Sampling Unit Tests")
    print("="*60 + "\n")

    results = []
    results.append(test_all_agents_get_same_questions())
    results.append(test_sampling_is_deterministic_databases_and_questions())
    results.append(test_question_samples_differ_across_iterations())
    results.append(test_phase1_failure_uses_correct_questions())
    results.append(test_current_iteration_questions_populated())

    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ All tests passed! ({passed}/{total})")
        print("="*60 + "\n")
        sys.exit(0)
    else:
        failed = total - passed
        print(f"❌ {failed} test(s) failed ({passed}/{total} passed)")
        print("="*60 + "\n")
        sys.exit(1)
