# Domain Abstraction Design: Factoring Out Text2SQL for Code Generation Extension

## Key Insight: Isomorphic Architecture

The Text2SQL and CodeGen architectures are **identical** in structure. The only difference is what gets fed into Phase 1:

| Component | Text2SQL | Code Generation |
|-----------|----------|-----------------|
| **Phase 1 Input** | Database file | Bundle: {question, code_v1, approach_description} |
| **agent.md + tools/** | Analyze schema → DB-specific context | Route approach → select/combine heuristics |
| **eval_instructions.md** | Static SQL principles | Static coding principles |
| **Output** | system_prompt.txt | critic_prompt.txt |

The agent/tools do **lightweight routing** over **substantial evolved heuristics** stored in files like `tools/heuristics/dp_patterns.md`.

---

## What Stays the Same (No Changes)

- Three-artifact agent structure: `agent.md`, `eval_instructions.md`, `tools/`
- Evolution strategies in `evolution_strategies/`
- ELO ranking system
- Agent selection logic
- Checkpoint/resume system
- ConfigManager and config validation
- Report generation
- Deep Focus evolution manager
- Tool-only vs agent-driven execution modes

---

## What Changes

### 1. Minimal Domain Interface

Create `RoboPhD/domains/base.py`:

```python
class DomainInterface(ABC):
    """Minimal interface - domains differ only in Phase 1 input and evaluation."""

    @abstractmethod
    def prepare_phase1_input(self, workspace: Path, problem: Dict) -> Path:
        """
        Prepare the input for Phase 1 analysis.

        Text2SQL: Creates symlink to database.sqlite
        CodeGen: Writes problem_context.json with {question, code_v1, approach}

        Returns: Path to the input file/directory
        """
        pass

    @abstractmethod
    def evaluate(self, solution: str, problem: Dict) -> Dict:
        """
        Evaluate a solution against ground truth.

        Text2SQL: Execute SQL, compare result sets
        CodeGen: Run against hidden tests, binary pass/fail

        Returns: {correct: bool, details: ...}
        """
        pass

    @abstractmethod
    def load_problems(self, dataset: str) -> List[Dict]:
        """Load problems from dataset."""
        pass

    @property
    @abstractmethod
    def phase1_input_name(self) -> str:
        """Name of the Phase 1 input (for logging/prompts)."""
        # Text2SQL: "database"
        # CodeGen: "problem context"
        pass
```

### 2. Directory Structure

```
RoboPhD/
├── domains/
│   ├── __init__.py
│   ├── base.py                    # DomainInterface ABC
│   ├── text2sql/
│   │   ├── __init__.py
│   │   ├── domain.py              # Text2SQLDomain (wraps existing code)
│   │   └── ... (existing core.py classes stay in place)
│   └── codegen/
│       ├── __init__.py
│       ├── domain.py              # CodeGenDomain
│       ├── livecode_bench.py      # Dataset loader
│       └── test_runner.py         # Hidden test execution
├── core.py                        # UNCHANGED (SQLGenerator, Evaluator, etc.)
├── researcher.py                  # Minimal changes: accept --domain flag
└── ...
```

### 3. Text2SQL Adapter (Zero Changes to Existing Code)

`RoboPhD/domains/text2sql/domain.py`:

```python
from RoboPhD.core import SQLGenerator, Evaluator, DatabaseManager

class Text2SQLDomain(DomainInterface):
    """Wraps existing Text2SQL code - no modifications to core.py."""

    def __init__(self, config: Dict):
        self.config = config
        # Use existing classes via composition
        self.sql_generator = SQLGenerator(...)
        self.evaluator = Evaluator(...)

    def prepare_phase1_input(self, workspace: Path, problem: Dict) -> Path:
        """Create symlink to database.sqlite (existing behavior)."""
        db_path = DatabaseManager.get_database_path(...)
        dest = workspace / "database.sqlite"
        dest.symlink_to(db_path)
        return dest

    def evaluate(self, solution: str, problem: Dict) -> Dict:
        """Delegate to existing Evaluator."""
        return self.evaluator.evaluate(...)

    @property
    def phase1_input_name(self) -> str:
        return "database"
```

### 4. CodeGen Domain Skeleton

`RoboPhD/domains/codegen/domain.py`:

```python
class CodeGenDomain(DomainInterface):
    """Code generation with evolved critics."""

    def prepare_phase1_input(self, workspace: Path, problem: Dict) -> Path:
        """
        Write problem_context.json with:
        - question: The problem statement
        - code_v1: Initial solution from Coder Call 1
        - approach: Self-reported approach from Call 1.5

        Note: session_id is in problem dict but NOT written here.
        The critic doesn't need it - Phase 2 uses it for revision.
        """
        context = {
            "question": problem["question"],
            "code_v1": problem["code_v1"],
            "approach": problem["approach_description"]
        }
        context_path = workspace / "problem_context.json"
        context_path.write_text(json.dumps(context, indent=2))
        return context_path

    def evaluate(self, solution: str, problem: Dict) -> Dict:
        """Run against hidden tests - binary pass/fail."""
        result = self.test_runner.run(solution, problem["test_cases"])
        return {"correct": result.all_passed, "details": result}

    @property
    def phase1_input_name(self) -> str:
        return "problem context"
```

### 4.1 Key Architectural Difference: Session Resumption

CodeGen's Phase 2 differs fundamentally from Text2SQL:

| | Text2SQL Phase 2 | CodeGen Phase 2 |
|---|---|---|
| **Mechanism** | Fresh API call | Resume Claude Code session |
| **Input** | system_prompt.txt + question | Critic feedback (session has context) |
| **Context** | Generated by Phase 1 agent | Original coder reasoning preserved |

**Cached problem data** (from preprocessing step, run once per problem):

```python
{
    "question": "...",              # From LiveCodeBench
    "code_v1": "...",               # From Coder Call 1
    "approach_description": "...",  # From Coder Call 1.5
    "session_id": "..."             # Claude Code session for Call 2 resume
}
```

The `session_id` enables the coder to evaluate critic feedback against its original reasoning. Without it, the coder would need to re-understand the problem from scratch, losing the context of *why* it made certain design decisions.

### 5. Evolved Critic Agent Structure

```
agents/dp_critic/
├── agent.md                    # Lightweight routing logic
│   # execution_mode: tool_only
│   # tool_command: python tools/route_approach.py
├── eval_instructions.md        # Static coding principles
└── tools/
    ├── route_approach.py       # Parse approach → select heuristics
    └── heuristics/             # Substantial evolved content
        ├── dp_patterns.md      # "For DP: check base cases..."
        ├── graph_patterns.md   # "For graphs: verify cycles..."
        ├── binary_search.md    # "For binary search: check bounds..."
        └── ...
```

The `route_approach.py` script:
```python
def main():
    # Read problem_context.json
    context = json.load(open("problem_context.json"))
    approach = context["approach"].lower()

    heuristics = []
    if "dp" in approach or "dynamic programming" in approach:
        heuristics.append(Path("tools/heuristics/dp_patterns.md").read_text())
    if "binary search" in approach:
        heuristics.append(Path("tools/heuristics/binary_search.md").read_text())
    # ... etc

    # Output combined heuristics
    print("\n\n".join(heuristics))
```

### 6. Changes to researcher.py

Minimal changes (~50-100 lines):

```python
# Add import
from RoboPhD.domains import get_domain

# In __init__:
self.domain = get_domain(config.get('domain', 'text2sql'), config)

# In process_database/process_problem:
# Replace: db_path = self.db_root / db_name / f"{db_name}.sqlite"
# With:    input_path = self.domain.prepare_phase1_input(workspace, problem)

# Replace: self.evaluator.evaluate(...)
# With:    self.domain.evaluate(solution, problem)
```

### 7. CLI Changes

```bash
# Default (text2sql)
python RoboPhD/researcher.py --num-iterations 10

# Code generation
python RoboPhD/researcher.py --num-iterations 10 --domain codegen

# Or via config
python RoboPhD/researcher.py --config '{"domain": "codegen", ...}'
```

---

## Implementation Steps

### Phase 1: Extract Domain Interface (preserve Text2SQL)
1. Create `RoboPhD/domains/base.py` with `DomainInterface` ABC
2. Create `RoboPhD/domains/text2sql/domain.py` wrapping existing code
3. Add `--domain` flag to researcher.py (default: "text2sql")
4. **Verify**: Run existing Text2SQL experiments, confirm identical results

### Phase 2: Refactor researcher.py
1. Replace direct database/SQL references with domain method calls
2. Update agent_orchestrator.py parameter names (minimal)
3. **Verify**: Run Text2SQL again, confirm still identical

### Phase 3: Add CodeGen Domain
1. Create `RoboPhD/domains/codegen/domain.py`
2. Implement LiveCodeBench dataset loader
3. Implement test runner for hidden tests
4. Create initial critic agent structure

### Phase 4: Adapt Evolution Strategies
1. Create CodeGen variants of evolution strategies (terminology changes)
2. Or: Make strategies domain-aware via template variables

---

## Files to Modify

| File | Changes |
|------|---------|
| `RoboPhD/domains/base.py` | **NEW** - DomainInterface ABC |
| `RoboPhD/domains/text2sql/domain.py` | **NEW** - Wraps existing code |
| `RoboPhD/domains/codegen/domain.py` | **NEW** - Code generation domain |
| `RoboPhD/researcher.py` | ~50-100 lines - Add domain parameter, delegate to interface |
| `RoboPhD/agent_orchestrator.py` | ~20 lines - Rename database→context in some places |
| `RoboPhD/core.py` | **UNCHANGED** |
| `RoboPhD/evolution_strategies/*` | Future: Add codegen variants or make domain-aware |

---

## Verification Strategy

1. **Before any changes**: Run a small Text2SQL experiment, save results
2. **After Phase 1**: Run identical experiment through Text2SQLDomain adapter
3. **Compare**: Results must be byte-for-byte identical
4. **After Phase 2**: Run again after researcher.py refactor, still identical
