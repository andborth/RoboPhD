# RoboPhD Project Bibliography
## Code Generation Scaffolding Systems & Benchmarks
**Compiled: January 2026**

---

## Citation Summary Table

| Paper | Venue | Date | Citations | Key Result |
|-------|-------|------|-----------|------------|
| **AgentCoder** | arXiv | Dec 2023 | **227** | HumanEval 96.3%, MBPP 91.8% (GPT-4) |
| **MapCoder** | ACL 2024 | Aug 2024 | ~50+ | HumanEval 93.9%, CodeContests 28.5% |
| **AlphaCodium** | arXiv | Jan 2024 | ~100+ | CodeContests 19%→44% (pass@5) |
| **ThinkCoder** | ACL 2025 Findings | Jul 2025 | <10 | +3.0% over MapCoder, 6.4% compute |
| **Seed-CTS** | arXiv | Dec 2024 | <10 | 35.1% LiveCodeBench-Hard (Qwen-32B) |
| **CodeSIM** | NAACL 2025 Findings | 2025 | <10 | Simulation-driven planning |

---

## 1. Multi-Agent Code Generation Frameworks

### AgentCoder (Huang et al., 2023) — 227 citations ⭐
- **Architecture**: Three-agent framework (programmer, test designer, test executor)
- **Key Innovation**: Separate test generation from code generation
- **Results**: HumanEval 96.3%, MBPP 91.8% with GPT-4
- **arXiv**: [2312.13010](https://arxiv.org/abs/2312.13010)

### MapCoder (Islam et al., ACL 2024) — ~50+ citations
- **Architecture**: Four-agent framework (retrieval, planning, coding, debugging)
- **Key Innovation**: Plan-derived debugging, confidence-based traversal
- **Results**: HumanEval 93.9%, MBPP 83.1%, CodeContests 28.5%
- **URL**: [ACL Anthology](https://aclanthology.org/2024.acl-long.269/)

### ThinkCoder (Zhang et al., ACL 2025 Findings) — <10 citations (new)
- **Architecture**: Exploration agent + refinement with ReST training
- **Key Innovation**: Preference learning from exploration trajectories
- **Results**: +3.0% over MapCoder with only 6.4% computation cost
- **arXiv**: [2502.17442](https://arxiv.org/abs/2502.17442)

### CodeSIM (Islam et al., NAACL 2025 Findings) — <10 citations (new)
- **Architecture**: Simulation-driven planning and debugging
- **Key Innovation**: Validates plans against I/O simulation before coding
- **URL**: [NAACL 2025](https://aclanthology.org/2025.findings-naacl.285/)

### Seed-CTS (Wang et al., 2024) — <10 citations (new)
- **Architecture**: Token-level MCTS + Chain-of-Thought prompting
- **Key Innovation**: Monte Carlo Tree Search over code generation tokens
- **Results**: Pass@1 = 35.1% on LiveCodeBench-Hard (outperforms GPT-4o pass@100)
- **arXiv**: [2412.12544](https://arxiv.org/abs/2412.12544)

### AlphaCodium (Ridnik et al., 2024) — ~100+ citations
- **Architecture**: Flow engineering (pre-processing → code iteration)
- **Key Innovation**: "Flow engineering > prompt engineering"
- **Results**: GPT-4 improved from 19% to 44% on CodeContests (pass@5)
- **arXiv**: [2401.08500](https://arxiv.org/abs/2401.08500)

---

## 2. Code Generation Benchmarks

### LiveCodeBench (Jain et al., 2024)
- **Content**: 1055 problems (v6, May 2023 - Apr 2025)
- **Sources**: LeetCode, AtCoder, CodeForces
- **Scenarios**: Code generation, self-repair, execution, test prediction
- **Key Feature**: Contamination-free with continuous updates
- **arXiv**: [2403.07974](https://arxiv.org/abs/2403.07974)

### LiveCodeBench Pro (Zhang et al., 2025)
- **Focus**: Harder problems subset
- **Key Finding**: Best models achieve **0% pass@1 on hard tier**
- **Insight**: Conceptual/algorithmic errors dominate (not implementation bugs)
- **arXiv**: [2506.11928](https://arxiv.org/abs/2506.11928)

### BIRD Benchmark (Li et al., 2024)
- **Focus**: Text-to-SQL with real-world databases
- **Scale**: 95 databases, 33.4 GB total
- **Metrics**: Execution accuracy + efficiency
- **Venue**: NeurIPS 2024

### Classic Benchmarks
| Benchmark | Problems | Focus |
|-----------|----------|-------|
| HumanEval | 164 | Python function completion |
| MBPP | 974 | Basic Python programming |
| APPS | 10,000 | Intro/Interview/Competition |
| CodeContests | 165 | Competitive programming |

---

## 3. Evolutionary Prompt Optimization

### APE - Automatic Prompt Engineer (Zhou et al., ICLR 2023)
- Uses LLMs to generate and select prompts automatically

### PromptBreeder (Fernando et al., 2023)
- Self-referential prompt evolution with mutation operators

### EvoPrompt (Guo et al., 2024)
- Combines evolutionary algorithms with LLM-based optimization

---

## 4. Reasoning & Self-Improvement Methods

| Method | Key Idea |
|--------|----------|
| **Chain-of-Thought** (Wei et al., 2022) | Step-by-step reasoning |
| **Tree of Thoughts** (Yao et al., 2023) | Tree-structured exploration |
| **ReAct** (Yao et al., 2023) | Interleaved reasoning + action |
| **Reflexion** (Shinn et al., 2023) | Verbal self-reflection loops |
| **Self-Debug** (Chen et al., 2023) | Code explanation for debugging |

---

## 5. RoboPhD Positioning

### Unique Contributions vs. Existing Work

| System | What Evolves | Approach |
|--------|-------------|----------|
| Base Models | Nothing | Direct prompting |
| MapCoder/ThinkCoder | Fixed scaffolding | Hand-designed flows |
| Seed-CTS | MCTS search | Search over code tokens |
| AlphaCodium | Fixed flow | Hand-designed stages |
| **RoboPhD** | **Critic instructions** | **Evolved feedback strategies** |

### Novel Elements
1. **Evolutionary critic development** — Critics learn what feedback helps
2. **Research-driven strategy** — Autonomous literature review → adaptation
3. **ELO-based selection** — Competitive ranking for evolution (not just evaluation)
4. **Database-centric processing** — Progressive learning per database

### Publication Strategy
- **Baseline comparisons**: Direct prompting, fixed critic, AlphaCodium-style
- **Key metrics**: Pass@1 by difficulty, self-repair performance, cost/problem
- **Temporal split**: Pre-Oct 2024 for evolution, post-Oct 2024 for evaluation

---

## Key Insights from Literature

1. **Hard problems remain unsolved**: LiveCodeBench Pro shows 0% pass@1 on hard tier for all models

2. **Conceptual errors dominate**: LLMs fail at algorithmic reasoning, not implementation

3. **Diminishing returns from iteration**: Most gains happen in first 2-3 refinement rounds

4. **Test-time compute helps**: MCTS and search methods significantly boost performance

5. **Flow > Prompt engineering**: Structured workflows outperform prompt tweaking alone

---

*Bibliography compiled for RoboPhD ICLR submission, January 2026*