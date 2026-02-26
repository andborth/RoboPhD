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
│   │   ├── problem.md    # Problem statement (from dataset)
│   │   ├── meta.json     # Metadata: question_id, contest_date, difficulty, session_id
│   │   └── solution.py   # Initial solution from Phase 1
│   └── ...
└── ...
```

## Coder/Critic Architecture

The CodeGen domain uses a 6-phase workflow with verdict branching:

```
┌─────────────────────────────────────────────────────────────┐
│                    Evolution AI (Opus)                       │
│  Evolves critic agents based on binary pass/fail outcomes   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 1: Initial Generation (Coder)            │
│  Receives problem, generates initial solution (Code v1)     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│       Phase 2: Critic Review (tool-only + eval LLM)         │
│  Tool analyzes code → eval LLM produces verdict + feedback  │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
             VERDICT: CORRECT    VERDICT: INCORRECT
                    │                   │
                    │                   ▼
                    │    ┌──────────────────────────────────┐
                    │    │  Phase 3: Revision (Coder)       │
                    │    │  Forked session, receives feedback│
                    │    │  Produces Code v2                │
                    │    └──────────────────────────────────┘
                    │                   │
                    │                   ▼
                    │    ┌──────────────────────────────────┐
                    │    │  Phase 3.5: Acceptance Query     │
                    │    │  Categorizes acceptance:         │
                    │    │  ACCEPTED_ALL / SOME / REJECTED  │
                    │    └──────────────────────────────────┘
                    │                   │
          v2 = symlink to v1            │
                    │                   │
                    └─────────┬─────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Evaluation (Ground Truth)                  │
│  Tests both v1 and v2 against hidden test suite             │
│  6-second timeout per test, one retry on timeout            │
│  Binary outcome: pass (all tests) / fail (any test fails)   │
└─────────────────────────────────────────────────────────────┘
```

### Phase Details

**Phase 1: Initial Generation (Coder)**
- Receives problem statement with examples
- Writes initial solution (`solution.py`)
- Can execute code on visible examples
- Observes: "Example 1 ✓, Example 2 ✓, Example 3 ✓"

**Phase 2: Critic Review**
- Evolved critics typically use **tool-only execution mode**: a Python script performs automated static analysis of the solution, producing `tool_output/critic_feedback.txt`
- An eval LLM receives the tool output + `eval_instructions.md` and produces `feedback.md`
- `feedback.md` starts with `VERDICT: CORRECT` or `VERDICT: INCORRECT`, followed by analysis
- If the verdict line is missing, the critic is re-prompted once to fix it

**Phase 3: Revision (Coder)**
- **Only runs if verdict is INCORRECT** (CORRECT → `solution_v2.py` is symlink to `solution.py`, no revision)
- Forks the coder's original session (preserves original for future iterations)
- Receives critic feedback; has discretion to accept all, some, or none
- Writes `solution_v2.py`

**Phase 3.5: Acceptance Query (Coder)**
- Only runs if revision was attempted and completed
- Post-hoc query on the forked revision session
- Coder categorizes: `ACCEPTED_ALL` / `ACCEPTED_SOME` / `REJECTED_ALL`
- Produces `acceptance.md`

**Phase 4: Evaluation**
- Tests both v1 and v2 against hidden test suite
- 6-second timeout per test with one retry on timeout
- Binary pass/fail (all tests pass = pass)
- Measures improvement (v1 fail → v2 pass) and regression (v1 pass → v2 fail)

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
  --config '{"examples_per_iteration": 3, "problems_per_context": 10}'
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

A typical evolved critic is a **monolithic tool-only analyzer** — a single Python script that performs comprehensive static analysis:

```
agents/<agent_name>/
├── agent.md                   # YAML config: tool_only execution mode
├── eval_instructions.md       # Decision framework for verdict + feedback
└── tools/
    └── analyzer.py            # Static analysis script
```

The `agent.md` YAML frontmatter configures tool-only execution:

```yaml
---
name: <agent_name>
description: <one-line summary of critic approach>
execution_mode: tool_only
tool_command: python tools/analyzer.py
tool_output_file: tool_output/critic_feedback.txt
---
```

The analyzer script reads `solution.py` and `problem.md` from its working directory, performs whatever analysis the evolution strategy designed, and writes structured findings to `tool_output/critic_feedback.txt`. The eval LLM then uses this analysis (along with `eval_instructions.md`) to render a verdict. Common analysis techniques include constraint extraction, complexity estimation, test execution against visible examples, and pattern-specific heuristics.

## Per-Problem Output Files

Each problem directory contains the full audit trail:

```
problems/<problem_id>/
├── problem.md            # Symlink to cache (problem statement)
├── solution.py           # Code v1 (copied from cache)
├── tools/                # Critic's analysis scripts (copied from agent)
├── tool_output/          # Static analysis output (used by critic LLM)
│   └── critic_feedback.txt
├── critic_prompt.md      # Full prompt sent to critic LLM
├── feedback.md           # Critic verdict (CORRECT/INCORRECT) + analysis
├── revision_prompt.md    # [If revised] Feedback formatted as revision request
├── solution_v2.py        # Revised code, or symlink to solution.py if CORRECT
├── acceptance_prompt.md  # [If revised] Acceptance query prompt
├── acceptance.md         # [If revised] ACCEPTED_ALL/SOME/REJECTED_ALL + explanation
└── result.json           # Evaluation result with v1/v2 pass, timing, cost data
```

## Metrics

- **V1 Pass@1**: Fraction of problems solved before critic review
- **V2 Pass@1**: Fraction of problems solved after critic review cycle
- **Improved**: v1 fail → v2 pass (critic helped)
- **Regressed**: v1 pass → v2 fail (critic hurt)
- **Verdict Classification**: TP (INCORRECT + v1 wrong), FP (INCORRECT + v1 right), TN (CORRECT + v1 right), FN (CORRECT + v1 wrong)
- **Acceptance Effectiveness**: Per category (accepted_all / accepted_some / rejected_all), tracks improved / no_help / no_harm / regressed

## Key Differences from Text2SQL

| Component | Text2SQL | CodeGen |
|-----------|----------|---------|
| **Phase 1 Input** | Database file | Problem context (problem + code_v1) |
| **Phase 1 Output** | system_prompt.txt | feedback.md (critic verdict + analysis) |
| **Eval Mechanism** | Fresh API call with system prompt | Fresh API call with tool output + eval_instructions |
| **Revision** | Verification retries with progressive temperature | Critic-driven session fork; coder accepts/rejects feedback |

## Troubleshooting

### Test Execution Timeouts
- **Symptom**: Tests hang on specific problems
- **Solution**: Check for infinite loops or TLE-prone algorithms in Code v2

### Session Resumption Failures
- **Symptom**: "Session not found" errors
- **Solution**: Sessions expire; the system auto-regenerates solution.py with a fresh session when this happens
