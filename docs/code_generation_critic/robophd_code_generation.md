# RoboPhD: Evolving Critics for Code Generation

## Abstract

We extend RoboPhD's evolutionary framework from text-to-SQL to code generation, targeting the LiveCodeBench benchmark. Rather than evolving the code generator directly, we propose evolving a **critic agent** that reviews code and provides feedback to the coder. The coder has discretion to accept or reject suggestions, and we track acceptance patterns post-hoc to understand what kinds of feedback help. Selection is based on binary pass/fail outcomes, consistent with the Text-to-SQL approach. This leverages the insight that learning *what feedback helps* may be more tractable than learning to solve problems directly.

The system is implemented on **Claude Code**, enabling session persistence across multi-week experiments and immediate deployment into production workflows. Evolved critics can plug directly into CI/CD pipelines, code review processes, and developer tooling — making this not just a research contribution but a practical industrial capability.

---

## 1. Introduction

### 1.1 Background: RoboPhD for Text-to-SQL

RoboPhD demonstrates that evolutionary algorithms can improve LLM performance on structured tasks by:

- Employing ELO-based competitive ranking for agent selection
- Processing databases individually to enable progressive learning
- Maintaining a portfolio of evolutionary strategies
- Examining errors to inform prompt improvements
- Optionally using academic papers as "genetic material" for prompt evolution

The system achieved 68.2% execution accuracy on the BIRD benchmark, competitive with fine-tuning approaches.

### 1.2 Extension to Code Generation

Code generation shares key properties with text-to-SQL:

- Natural language input → executable code output
- Clear correctness signal from test execution
- Problem diversity that benefits from specialized strategies
- Rich error signals for the Evolution AI to learn from

LiveCodeBench provides an ideal target: 1000+ problems from LeetCode, AtCoder, and Codeforces, with continuous updates to prevent contamination.

### 1.3 The Coder/Critic Paradigm

Rather than evolving the coder directly, we evolve **instructions to a critic agent**. This is motivated by:

1. **Code review is a solved workflow**: The coder/critic pattern is well-established in software engineering
2. **Meta-learning opportunity**: Learning what feedback helps may generalize better than learning to solve
3. **Richer signal**: We can track which suggestions get accepted and which lead to success
4. **Connection to Mixture-of-Experts**: Category-specific critics can develop specialized heuristics

The generator/critic separation has emerged as a best practice across multiple lines of research:

- **Self-Refine** (Madaan et al., NeurIPS 2023) demonstrated that LLMs can generate feedback on their own output and use it to iteratively improve, achieving ~20% absolute improvement across diverse tasks including code optimization.

- **AlphaCodium** (Ridnik et al., 2024) introduced a test-based, multi-stage iterative flow for competitive programming, improving GPT-4's pass@5 on CodeContests from 19% to 44%.

- **Claude Code Best Practices** (Anthropic, 2025) explicitly recommends multi-agent workflows: "A simple but effective approach is to have one Claude write code while another reviews or tests it... This separation often yields better results than having a single Claude handle everything."

**Our contribution**: We *evolve* the critic through ELO-based selection on binary pass/fail outcomes, rather than using fixed critique prompts.

---

## 2. Related Work

### 2.1 Mixture-of-Prompts (MoP)

Wang et al. (2024) demonstrate that partitioning the problem space into categories, each with specialized prompts, outperforms single-prompt approaches. Key findings:

- Optimal performance around 8-12 expert categories
- K-means clustering on embeddings effectively partitions problems
- Region-specific instruction search complements demo assignment
- 81% win rate against prior methods across benchmarks

**Our contribution**: MoP uses *fixed* clustering. We propose *evolving* the classification scheme alongside category-specific critic prompts.

### 2.2 Self-Repair in Code LLMs

LiveCodeBench evaluates four independent capabilities: code generation, self-repair, code execution, and test output prediction. The **self-repair scenario** tests debugging ability with concrete failure information:

- **Input**: Problem + Known-incorrect code + Failing test case + Error message
- **Output**: Fixed code

This is distinct from our approach:

| | Self-Repair Task | Our Evolved Critic |
|---|---|---|
| **When** | After execution fails | Before execution |
| **Input** | Concrete error + failing test | Code that passes examples |
| **Challenge** | Fix a known bug | Anticipate unknown bugs |
| **Information** | Ground truth provided | Must predict failure modes |

**Our contribution**: The critic operates *pre-execution*, reviewing code that passes all visible examples but may fail hidden tests.

### 2.3 LiveCodeBench

Jain et al. (2024) introduce LiveCodeBench with:

- Continuous problem collection from competition platforms
- Temporal tagging to detect/prevent contamination
- Difficulty stratification (Easy/Medium/Hard)

We focus on the code generation scenario, using Pass@1 as our primary metric.

---

## 3. Method

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Evolution AI (Opus)                       │
│  Evolves critic prompts based on:                           │
│    - Binary pass/fail outcomes (ELO selection)              │
│    - Examination of errors and failure patterns             │
│  Optional: Can read academic papers for inspiration         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Coder AI (Call 1)                           │
│  Receives problem, generates initial solution (Code v1)     │
│  Can execute code on visible examples                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│             Coder AI (Call 1.5 — same session, cheap)        │
│  Query: "Describe the algorithmic approach you used"        │
│  Returns: Free-form description (e.g., "DP with binary      │
│           search optimization for the inner loop")          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Critic AI (Evolved)                        │
│  Prompt constructed from parsed approach description        │
│  Combines heuristics for relevant algorithmic patterns      │
│  Reviews Code v1, produces structured feedback              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Coder AI (Call 2)                           │
│  Receives: Problem + Code v1 + Critic feedback              │
│  Has discretion to accept all, some, or none               │
│  Produces revised solution (Code v2)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Coder AI (Call 3 — cheap)                   │
│  Post-hoc query: "Which suggestions did you accept?"        │
│  Returns structured acceptance decisions                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Evolution (Ground Truth)                   │
│  Runs Code v2 against hidden test suite                     │
│  Binary outcome: pass (all tests) / fail (any test fails)   │
│  Updates ELO rankings; examines failure patterns            │
└─────────────────────────────────────────────────────────────┘
```

**Design Rationale: Separating Code Generation from Approach Query**

Call 1 (code generation) and Call 1.5 (approach query) are separated to enable:

1. **Clean baseline measurement**: Raw coding ability measured independently
2. **Experimental flexibility**: Approach parsing can evolve without invalidating coding baselines
3. **Natural hybrid handling**: Free-form descriptions capture "DP + binary search" without artificial buckets
4. **Session continuity**: Both calls in same session — the coder remembers its reasoning

### 3.2 Information Flow and Execution Capabilities

Both the Coder and Critic have access to code execution via the Claude API. They can run code against the visible examples provided in the problem statement.

**What Coder and Critic CAN Do:**
- Read the problem statement and constraints
- See the example inputs/outputs (typically 2-3 examples)
- Write and execute code on examples
- Observe whether examples produce expected output
- See any runtime errors or exceptions

**What Coder and Critic CANNOT Do:**
- Run against hidden test cases
- Know if the solution is actually correct
- See edge cases not covered by examples

**What Only Evolution Knows:**
- Full test suite results (ground truth)
- Which suggestions were accepted (via post-hoc query)
- Final success/failure outcome

**Why This Makes the Critic's Job Hard:**

The visible examples are typically the "easy" cases — they help humans understand the problem but don't cover edge cases. Hidden tests include:
- Edge cases: empty input, single element, maximum constraints
- Corner cases the examples don't illustrate  
- Performance limits: will O(n²) TLE on n=10⁵?

The critic must reason: "This code passes all examples, but will it pass hidden tests?" This requires anticipating:
- "The examples don't include negative numbers, but constraints allow them"
- "The examples are small, but n can be 10⁵ — this approach will TLE"
- "The examples have unique elements, but the problem doesn't guarantee that"

### 3.3 The Coder/Critic Interaction

The Coder AI is invoked twice: first to generate an initial solution, then to revise based on critic feedback. The Critic AI is invoked once, reviewing the initial attempt.

```
Coder AI (Call 1) → Code v1 → Critic AI → Feedback → Coder AI (Call 2) → Code v2
```

**Phase 1: Initial Generation (Coder AI — Call 1)**
```
Coder receives: Problem statement with examples
Coder writes: Initial solution (Code v1)
Coder executes: Runs code on all visible examples
Coder observes: "Example 1 ✓, Example 2 ✓, Example 3 ✓"
Coder concludes: "Looks correct to me"
```

**Phase 1.5: Approach Query (Coder AI — same session, cheap)**

Two experimental conditions:

**Baseline (open-ended, no meta-evolution):**
```
Query: "Briefly describe the algorithmic approach you used to solve this."

Coder responds: "I used dynamic programming with memoization. The state 
is dp[i][j] representing the minimum cost to process elements i through j.
I also used binary search to optimize the inner loop for finding the 
optimal split point, reducing complexity from O(n³) to O(n² log n)."
```

Evolution learns to parse this free-form description and construct appropriate critic prompts — potentially combining heuristics from multiple categories (DP + binary search).

**With meta-evolution (Section 5.2 of ICLR paper):**
```
Query: [Evolved prompt with structured format]
Categories: [Evolved set, potentially hierarchical]
Parsing rules: [Evolved mapping from response to critic selection]
```

The baseline approach avoids artificial category boundaries and captures hybrid approaches naturally. The open-ended question remains stable across experiments; only the parsing/routing logic evolves.

**Phase 2: Critic Review (Critic AI)**
```
Critic receives: Problem + Code v1 + (optionally) approach description
Critic prompt: Constructed based on parsed approach categories
Critic executes: Can also run code on examples (confirms they pass)
Critic produces: Structured feedback with specific suggestions
```

**Phase 3: Revision (Coder AI — Call 2)**
```
Coder receives: Problem + Code v1 + Critic feedback
Instruction: "You may accept all, some, or none of the feedback"
Coder revises: Produces Code v2
Coder executes: Verifies examples still pass
```

**Phase 4: Post-hoc Acceptance Query (Coder AI — Call 3, cheap)**
```
Query to Coder: "You received these suggestions. For each, did you accept it?"

Coder responds:
  Suggestion 1: "No, I believe greedy is correct here because..."
  Suggestion 2: "Yes, I added an empty check at line 3"
  Suggestion 3: "Partially, I optimized the inner loop but kept the approach"
```

**Phase 5: Ground Truth Evaluation (Evolution Only)**
```
Evolution runs: Code v2 against full hidden test suite
Evolution records: Binary pass/fail (all tests pass = pass, any test fails = fail)
Evolution updates: ELO rankings based on head-to-head outcomes
Evolution examines: Failure patterns to inform prompt improvements
```

### 3.4 Fitness Signal and Error Examination

**Selection is binary**: ELO ranking is based on pass/fail outcomes (see Section 4.3). The analysis below helps the Evolution AI understand *why* things pass or fail.

For each suggestion, we can observe a 2×2 outcome:

| | Outcome: Pass | Outcome: Fail |
|---|---|---|
| **Accepted** | Helpful | Harmful |
| **Rejected** | Unnecessary | Missed opportunity? |

This breakdown informs evolution but doesn't replace binary ELO. It helps answer:
- `w_helpful`: Which accepted suggestions correlate with success?
- `w_harmful`: Which accepted suggestions correlate with failure?
- What patterns appear in "missed opportunity" cases?

**Error Examination**: The Evolution AI examines specific failure cases:
- What bugs did the critic miss? (Rejected suggestions where code failed)
- What harmful advice did the critic give? (Accepted suggestions where code failed)
- What patterns appear across multiple failures?

This qualitative analysis informs prompt improvements that pure win/loss counts might miss.

### 3.5 Approach-Based Critic Routing

Rather than pre-defined categories, critics are constructed based on the coder's self-reported approach.

**Common algorithmic patterns** (not exhaustive, not mutually exclusive):

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

**Baseline approach**: The coder describes its approach in free-form text. Evolution learns to parse this and construct critic prompts by combining relevant heuristics. A solution described as "DP with binary search optimization" triggers both DP-specific and binary search-specific review heuristics.

This baseline is appealing because:
- Evolution discovers what aspects of approach descriptions actually matter
- No need to pre-specify category boundaries or hierarchies
- Hybrid approaches handled naturally without "multi-label classification"
- The free-form description is stable — only parsing logic evolves

**Co-evolution**: Critic prompts for each algorithmic pattern evolve based on pass/fail outcomes and error patterns. The parsing logic that maps descriptions to patterns can also evolve.

---

## 4. Evolutionary Strategies

### 4.1 Critic Prompt Evolution

The Evolution AI modifies critic prompts based on two primary signals:
1. **Binary pass/fail outcomes**: ELO ranking based on pass rates (see Section 4.3)
2. **Error examination**: Direct analysis of what went wrong when critics failed

**Refinement**: Improve existing heuristics based on success patterns
```
Before: "Check for edge cases"
After: "Check for: empty input, single element, maximum constraints, negative values"
```

**Error-Pattern Learning**: Examine failed reviews to identify missed patterns
```
"For problems involving 'maximum subarray' or 'contiguous subsequence', 
 verify Kadane's algorithm handles all-negative arrays correctly."
```
This comes from analyzing cases where the critic failed to catch a bug.

**Category Specialization**: Develop domain-specific heuristics
```
Graph Critic: "For shortest path problems, verify the algorithm handles:
 (1) disconnected components, (2) negative edges if using Dijkstra,
 (3) the distinction between directed and undirected graphs."
```

**Research-Driven (optional)**: Inject insights from academic papers
```
"When reviewing DP solutions, verify that the recurrence relation handles 
 base cases correctly. Common errors include off-by-one in array indexing 
 and incorrect initialization of the memoization table."
```
Note: Paper reading is an optional capability not used in the current experiments.

### 4.2 Approach Parsing and Critic Construction

**Baseline (no meta-evolution of classification):**

The coder provides a free-form description of its approach. Evolution learns to:
- Extract key algorithmic concepts from the description
- Map concepts to relevant critic heuristics
- Combine heuristics when multiple approaches are mentioned

```
Input: "I used DP with binary search optimization for the inner loop"
Parsed: [DP, BinarySearch]
Critic prompt: Combines DP heuristics + binary search heuristics
```

This avoids artificial category boundaries and handles hybrid approaches naturally.

**With meta-evolution (optional, per Section 5.2):**

Classification itself can evolve:
- **Query prompt**: How to ask about the approach
- **Category definitions**: What categories exist, their boundaries
- **Parsing rules**: How to map responses to critic prompts
- **Granularity**: Coarse vs fine-grained classification

This enables:
- Ablation studies comparing fixed vs evolved classification
- Experiments with different category granularities (5 vs 10 vs 20)
- Learning that certain problem phrasings map unexpectedly to certain approaches

### 4.3 Selection Mechanism

Selection follows the same approach as Text-to-SQL:

**Binary outcome per problem**: Each problem results in pass/fail based on LiveCodeBench's hidden test suite.

**Sampling for generalization**: Each evolution round samples a subset of training problems (~100). This prevents overfitting to specific problems and pushes the system to develop generally applicable critic heuristics.

**ELO ranking based on pass rate**: Agent packages (critic variants) are scored by their pass rate on the sampled batch. Higher pass rate = win, lower = loss, equal = tie (moderately rare). ELO ratings update based on these win/loss/tie outcomes.

**What gets compared**: The full pipeline outcome. An "agent package" includes the critic prompt and any associated parsing/routing logic. Evolution improves whichever components lead to higher pass rates.

This keeps the selection mechanism simple and grounded in the same metric reported on the leaderboard.

---

## 5. Experimental Design

### 5.1 Dataset

**LiveCodeBench v6** (1055 problems, May 2023 - April 2025)

| Split | Count | Date Range | Purpose |
|-------|-------|------------|---------|
| Evolution | 767 | May 2023 - Oct 2024 | Sample ~100/iteration for critic evolution |
| Test | 288 | Nov 2024 - Apr 2025 | Final evaluation only, never seen during evolution |

**Rationale**: Following AlphaCodium's approach for prompt/flow optimization, we use only evolution and test splits (no separate validation set). The temporal cutoff (2024-11-01) ensures:
1. Test problems could not have contaminated any model's training data
2. ~27% test split provides stable metrics (~250+ problems, consistent with CodeContests and LeetCodeDataset)

LiveCodeBench explicitly supports custom temporal splits via `--start_date` and `--end_date` flags. The LeetCodeDataset paper (2025) uses a similar methodology with a 2024-07-01 cutoff.

**Temporal filtering**: Each preprocessed problem's `meta.json` includes a `contest_date` field (e.g., `"2024-03-15"`) from LiveCodeBench. Filter by this field to enforce the temporal split:
```python
# Example: Load only evolution set problems
problems = [p for p in all_problems if p["contest_date"] < "2024-11-01"]
```

**Evolution sampling**: Each iteration samples ~100 problems from the evolution set. Different samples each round push the system to develop generally applicable heuristics rather than overfitting to specific problems.

### 5.2 Baselines and Ablations

**Code Generation Baselines:**
1. **No Critic**: Direct code generation without review
2. **Generic Critic**: Fixed prompt ("review for correctness and edge cases")
3. **Single Evolved Critic**: One critic prompt for all problems (no approach-based routing)

**Approach-Based Routing:**
4. **Open-ended + Learned Parsing (baseline)**: Free-form approach description, evolution learns parsing
5. **Structured Categories + Meta-evolution**: Classification prompt and categories also evolve (Section 5.2 of ICLR paper)

### 5.3 Metrics

- **Pass@1**: Fraction of problems solved on first attempt
- **Pass@1 (with critic)**: Fraction solved after critic review cycle
- **Critic Precision**: Helpful suggestions / Total suggestions
- **Critic Recall**: Helpful suggestions / Problems that needed help
- **Acceptance Rate**: Suggestions accepted / Total suggestions

### 5.4 Ablations

- Coder model (Sonnet, Opus, Haiku)
- Critic model (Sonnet, Haiku)
- Number of revision rounds (1, 2, 3)
- **Meta-evolution only**: Number of structured categories (5, 10, 15, 20)

---

## 6. Expected Contributions

1. **Novel architecture**: First system to evolve code review feedback rather than code generation directly

2. **Implicit acceptance tracking**: Post-hoc queries provide clean signal without biasing coder behavior

3. **Co-evolutionary framework**: Joint evolution of problem classifier and category-specific critics

4. **Domain-agnostic methodology**: The coder/critic/evolution pattern could extend to other domains (math, reasoning, planning)

5. **Benchmark results**: Competitive performance on LiveCodeBench with analysis of what makes critics effective

---

## 7. Discussion

### 7.1 Why Evolve the Critic?

The critic operates at a higher level of abstraction than the coder. While the coder must produce correct code (hard), the critic must recognize patterns that lead to incorrect code (potentially easier). By evolving the critic, we're learning:

- What failure modes are common
- What feedback actually gets accepted
- What feedback leads to successful fixes

This meta-knowledge may transfer better across problem types than direct solving ability.

### 7.2 The Role of Discretion

Giving the coder discretion to reject feedback is crucial:

1. **Prevents harmful advice from propagating**: Bad suggestions get filtered
2. **Provides signal for evolution**: Rejection patterns indicate unhelpful feedback
3. **Models real code review**: Human developers also selectively accept suggestions
4. **Enables coder-critic co-adaptation**: The coder learns which critics to trust

### 7.3 Connection to RLHF

This approach shares structure with RLHF:
- The coder is the "policy" being improved
- The critic is like a "reward model" providing feedback
- Evolution replaces gradient-based optimization
- Acceptance patterns replace preference data

Key difference: We evolve the critic (reward model) rather than the coder (policy).

### 7.4 Limitations and Future Work

**Current limitations:**
- Requires multiple model calls per problem: Coder ×4 (generate, approach query, revise, acceptance query) + Critic ×1
- Calls 1.5 and 3 are cheap (small prompts, structured output); Call 1 is cacheable across experiments
- Evolution requires many problems to establish signal

**Future directions:**
- Multi-turn critic interaction (iterative refinement)
- Critic confidence scores to guide when to accept
- Cross-domain transfer (evolve on code, apply to math)
- Ensemble critics with voting mechanisms

---

## 8. Implementation: Claude Code Integration

### 8.1 Platform Choice

RoboPhD is implemented on **Claude Code** (Anthropic, 2025), leveraging:

- **Session persistence**: `--resume <session-id>` maintains context across experiments spanning weeks
- **Code execution**: Claude Code can run generated code against examples natively
- **Flat-rate pricing**: Claude MAX plans enable extensive experimentation without per-call cost concerns
- **Familiar tooling**: Builds on infrastructure already used for agentic coding workflows

This positions RoboPhD not as a standalone research prototype, but as an application of Anthropic's agent platform — demonstrating Claude Code's versatility beyond interactive development.

### 8.2 Session Management for Multi-Week Experiments

```bash
# Week 1: Coder Call 1 — generate initial solution (CACHED)
RESPONSE=$(claude --print "$PROBLEM_TEXT" --output-format json)
SESSION_ID=$(echo "$RESPONSE" | jq -r '.session_id')
CODE_V1=$(echo "$RESPONSE" | jq -r '.code')

# Save code for later use
mkdir -p sessions/code
echo "$CODE_V1" > sessions/code/$PROBLEM_ID.py

# Week 1: Coder Call 1.5 — approach query (same session, open-ended)
APPROACH=$(claude --resume "$SESSION_ID" --print \
  "Briefly describe the algorithmic approach you used to solve this." \
  --output-format json | jq -r '.result.text')

# Cache session info for future experiments
mkdir -p sessions/approaches
echo "$SESSION_ID,$PROBLEM_ID" >> sessions/cache.csv
echo "$APPROACH" > sessions/approaches/$PROBLEM_ID.txt

# Week 3: Parse approach and construct critic prompt (approach is primary input)
CRITIC_PROMPT=$(python construct_critic.py --approach "$APPROACH")

# Week 3: Run critic to generate feedback
CODE_V1=$(cat sessions/code/$PROBLEM_ID.py)
CRITIC_FEEDBACK=$(claude --system-prompt "$CRITIC_PROMPT" --print \
    "Problem: $PROBLEM_TEXT

Code to review:
$CODE_V1

Review this code for potential issues." \
    --output-format json | jq -r '.feedback')

# Week 3: Resume session with critic feedback
claude --resume "$SESSION_ID" --print "
A critic reviewed your code and suggests:
$CRITIC_FEEDBACK

Evaluate each suggestion against your original reasoning.
Accept suggestions that address genuine issues you missed.
Reject suggestions that conflict with decisions you made deliberately.
" --output-format json >> results/experiment_N.jsonl
```

**What's cached vs. what's variable:**

| Component | Cached? | Notes |
|-----------|---------|-------|
| Call 1 (code generation) | ✅ Yes | Session + Code v1 preserved |
| Call 1.5 (approach query) | ✅ Yes | Free-form description, stable across experiments |
| Approach → Critic parsing | ❌ No | This is what evolves |
| Critic prompt construction | ❌ No | Depends on evolved parsing logic |
| Call 2 (revision) | ❌ No | Depends on critic feedback |
| Call 3 (acceptance) | ❌ No | Depends on revision |

The Coder's original reasoning remains in context, enabling meaningful evaluation of critic feedback without re-explaining the problem or approach.

### 8.3 Industrial Applications

The Claude Code integration makes evolved critics immediately deployable in production workflows:

| Use Case | Description |
|----------|-------------|
| **CI/CD Review** | Evolved critics review PRs automatically via `claude --print` in pipelines |
| **Team-Specific Critics** | Evolve on your codebase's actual error patterns, not generic benchmarks |
| **Onboarding Acceleration** | Junior developers receive feedback tuned to senior team's standards |
| **Security Review** | Critics evolve to catch vulnerabilities specific to your stack and dependencies |
| **Legacy Modernization** | Evolve critics that recognize legacy patterns and suggest modern equivalents |
| **Performance Optimization** | Critics learn which patterns cause performance issues in your specific workload |

**Deployment involves a learned parsing layer:**
```python
# construct_critic.py
def construct_critic_prompt(approach_description, problem_text=None, code_v1=None):
    """
    Parses the coder's approach description and constructs an appropriate
    critic prompt by combining relevant heuristics.
    
    Primary input: approach_description (from Phase 1.5)
    Optional: problem_text and code_v1 can provide additional signals
    
    This parsing logic is what evolution improves.
    """
    # Learned parsing: extract algorithmic concepts from description
    concepts = parse_approach(approach_description)  # e.g., ["DP", "binary_search"]
    
    # Combine heuristics for detected concepts
    heuristics = []
    for concept in concepts:
        heuristics.extend(EVOLVED_HEURISTICS[concept])
    
    # Optionally incorporate problem-specific signals
    if problem_text and mentions_constraints(problem_text, "10^5"):
        heuristics.append(COMPLEXITY_CHECK)
    
    return build_critic_prompt(heuristics)

# In the workflow:
CRITIC_PROMPT = construct_critic_prompt(approach, problem, code_v1)
# The critic prompt is a system prompt; problem and code are the user message
claude --system-prompt "$CRITIC_PROMPT" \
    "Problem: $PROBLEM\n\nCode:\n$CODE_V1\n\nReview for issues."
```

The evolved components are:
- `parse_approach()`: How to extract concepts from free-form descriptions
- `EVOLVED_HEURISTICS`: Pattern-specific review instructions
- Any additional routing logic

No new infrastructure required — just a Python script that constructs prompts, running on the same Claude Code developers already use.

### 8.4 From Research to Production

The evolution loop can run continuously:
1. Collect real code review outcomes from production (which suggestions were accepted? did they help?)
2. Use this signal to evolve critic prompts
3. Deploy improved critics back into the workflow
4. Repeat

This creates a flywheel where the critic improves from actual team behavior, not synthetic benchmarks.

---

## 9. Conclusion

We propose extending RoboPhD to code generation by evolving critic agents rather than coders directly. The key insight is that learning to provide helpful feedback may be more tractable than learning to solve problems. Selection is based on binary pass/fail outcomes — the same metric reported on the LiveCodeBench leaderboard — with acceptance patterns analyzed to understand what makes feedback effective. By combining this with post-hoc approach descriptions and pattern-based critics, we create a system that can develop specialized reviewing expertise across different algorithmic domains.

Implementing on Claude Code provides both research and practical benefits: session persistence enables multi-week experiments with preserved context, while the same evolved critics can deploy directly into production CI/CD pipelines and developer workflows. This bridges the gap between academic benchmarks and industrial code review, creating a path from research to deployment that requires no additional infrastructure.

---

## References

- Jain, N., et al. (2024). LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code. *arXiv:2403.07974*

- Madaan, A., et al. (2023). Self-Refine: Iterative Refinement with Self-Feedback. *NeurIPS 2023*. arXiv:2303.17651

- Ridnik, T., et al. (2024). Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering. *arXiv:2401.08500*

- Wang, R., et al. (2024). One Prompt is not Enough: Automated Construction of a Mixture-of-Expert Prompts. *ICML 2024*

- Anthropic (2025). Claude Code: Best practices for agentic coding. *https://anthropic.com/engineering/claude-code-best-practices*

- Anthropic (2025). Building agents with the Claude Agent SDK. *https://anthropic.com/engineering/building-agents-with-the-claude-agent-sdk*

- [RoboPhD text-to-SQL work - citation pending]
