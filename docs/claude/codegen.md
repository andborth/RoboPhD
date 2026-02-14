# CodeGen Domain Documentation

Domain-specific documentation for the LiveCodeBench code generation task.

## Overview

The CodeGen domain extends RoboPhD's evolutionary framework to code generation, targeting the LiveCodeBench benchmark. Rather than evolving the code generator directly, we evolve a **critic agent** that reviews code and provides feedback to the coder.

**Key insight**: Learning *what feedback helps* may be more tractable than learning to solve problems directly.

## Dataset: LiveCodeBench v6

**Total problems**: 1055 (May 2023 - April 2025)

| Split | Count | Date Range | Purpose |
|-------|-------|------------|---------|
| Evolution | 767 | May 2023 - Oct 2024 | Sample ~100/iteration for critic evolution |
| Test | 288 | Nov 2024 - Apr 2025 | Final evaluation only, never seen during evolution |

**Temporal split at 2024-11-01** ensures:
1. Test problems could not have contaminated any model's training data
2. ~27% test split provides stable metrics

**Temporal filtering** via `contest_date` field in each problem's metadata:
```python
# Load only evolution set problems
problems = [p for p in all_problems if p["contest_date"] < "2024-11-01"]
```

## Cache Directory Structure

CodeGen uses a versioned cache structure:

```
../robophd_runs/codegen_cache/
├── {model}_v6/           # Cached per model version
│   ├── {problem_id}/
│   │   ├── code_v1.py    # Initial solution from Coder Call 1
│   │   ├── approach.txt  # Self-reported approach from Call 1.5
│   │   └── session.json  # Session ID for resumption
│   └── ...
└── ...
```

## Coder/Critic Architecture

The CodeGen domain uses a 5-phase workflow:

```
┌─────────────────────────────────────────────────────────────┐
│                    Evolution AI (Opus)                       │
│  Evolves critic prompts based on binary pass/fail outcomes  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Coder AI (Call 1)                           │
│  Receives problem, generates initial solution (Code v1)     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│             Coder AI (Call 1.5 — same session)               │
│  Query: "Describe the algorithmic approach you used"        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Critic AI (Evolved)                        │
│  Reviews Code v1, produces structured feedback               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Coder AI (Call 2)                           │
│  Receives feedback, has discretion to accept/reject          │
│  Produces revised solution (Code v2)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Evaluation (Ground Truth)                  │
│  Runs Code v2 against hidden test suite                     │
│  Binary outcome: pass (all tests) / fail (any test fails)   │
└─────────────────────────────────────────────────────────────┘
```

### Phase Details

**Phase 1: Initial Generation (Coder AI — Call 1)**
- Receives problem statement with examples
- Writes initial solution (Code v1)
- Can execute code on visible examples
- Observes: "Example 1 ✓, Example 2 ✓, Example 3 ✓"

**Phase 1.5: Approach Query (same session)**
- Query: "Briefly describe the algorithmic approach you used"
- Returns free-form description (e.g., "DP with binary search optimization")

**Phase 2: Critic Review**
- Critic prompt constructed based on parsed approach
- Receives: Problem + Code v1 + approach description
- Produces: Structured feedback with specific suggestions

**Phase 3: Revision (Coder AI — Call 2)**
- Receives: Problem + Code v1 + Critic feedback
- Has discretion to accept all, some, or none
- Produces: Code v2

**Phase 4: Evaluation**
- Runs Code v2 against hidden test suite
- Binary pass/fail (all tests pass = pass)

## Basic Usage

```bash
# Run CodeGen evolution
python RoboPhD/researcher.py --num-iterations 10 --domain codegen

# Or via config
python RoboPhD/researcher.py --config '{"domain": "codegen", "eval_model": "haiku-4.5"}'

# Quick test
python RoboPhD/researcher.py \
  --domain codegen \
  --num-iterations 2 \
  --config '{"contexts_per_iteration": 3, "problems_per_context": 10}'
```

## Test Execution Methodology

**What Coder and Critic CAN Do:**
- Read the problem statement and constraints
- See the example inputs/outputs (typically 2-3)
- Write and execute code on examples
- Observe whether examples produce expected output

**What Coder and Critic CANNOT Do:**
- Run against hidden test cases
- Know if the solution is actually correct
- See edge cases not covered by examples

**Hidden tests include:**
- Edge cases: empty input, single element, maximum constraints
- Corner cases the examples don't illustrate
- Performance limits: will O(n²) TLE on n=10⁵?

## Critic Agent Structure

```
agents/dp_critic/
├── agent.md                    # Lightweight routing logic
├── eval_instructions.md        # Static coding principles
└── tools/
    ├── route_approach.py       # Parse approach → select heuristics
    └── heuristics/             # Substantial evolved content
        ├── dp_patterns.md
        ├── graph_patterns.md
        ├── binary_search.md
        └── ...
```

The `route_approach.py` script parses the coder's approach description and combines relevant heuristics:

```python
def main():
    context = json.load(open("problem_context.json"))
    approach = context["approach"].lower()

    heuristics = []
    if "dp" in approach or "dynamic programming" in approach:
        heuristics.append(Path("tools/heuristics/dp_patterns.md").read_text())
    if "binary search" in approach:
        heuristics.append(Path("tools/heuristics/binary_search.md").read_text())

    print("\n\n".join(heuristics))
```

## Common Algorithmic Patterns

Patterns that critics can specialize on:

1. **Dynamic Programming**: Memoization, state transitions, optimal substructure
2. **Graph Algorithms**: BFS/DFS, shortest path, connectivity
3. **Greedy**: Local optimum choices, sorting-based approaches
4. **Binary Search**: Search space reduction, monotonic predicates
5. **Data Structures**: Heaps, segment trees, hash maps
6. **Math/Number Theory**: Modular arithmetic, combinatorics
7. **String Processing**: Pattern matching, parsing
8. **Simulation**: Direct implementation, state tracking
9. **Two Pointers/Sliding Window**: Array traversal patterns
10. **Divide and Conquer**: Recursive decomposition

## Metrics

- **Pass@1**: Fraction of problems solved on first attempt
- **Pass@1 (with critic)**: Fraction solved after critic review cycle
- **Critic Precision**: Helpful suggestions / Total suggestions
- **Critic Recall**: Helpful suggestions / Problems that needed help
- **Acceptance Rate**: Suggestions accepted / Total suggestions

## Key Differences from Text2SQL

| Component | Text2SQL | CodeGen |
|-----------|----------|---------|
| **Phase 1 Input** | Database file | Bundle: {question, code_v1, approach} |
| **Phase 1 Output** | system_prompt.txt | critic_prompt.txt |
| **Phase 2 Mechanism** | Fresh API call | Resume Claude Code session |
| **Phase 2 Context** | Generated by Phase 1 | Original coder reasoning preserved |

## Troubleshooting

### Test Execution Timeouts
- **Symptom**: Tests hang on specific problems
- **Solution**: Check for infinite loops or TLE-prone algorithms in Code v2

### Session Resumption Failures
- **Symptom**: "Session not found" errors
- **Solution**: Ensure session IDs are properly cached and not expired

### Missing Approach Description
- **Symptom**: Critic routing fails
- **Solution**: Verify Call 1.5 completed and approach.txt was cached

## Further Reading

- [Full design document](../code_generation_critic/robophd_code_generation.md)
- [Domain abstraction design](../code_generation_critic/domain_abstraction_design.md)
- [Critic evaluation results](../code_generation_critic/critic_evaluation_results.md)
