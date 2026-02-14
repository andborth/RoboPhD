# Related Work: Evolving Critics for Code Generation

This bibliography covers literature relevant to the RoboPhD code generation critic approach, organized by research area.

---

## 1. Self-Refinement and Self-Correction

### Reflexion (Shinn et al., NeurIPS 2023)
**Reflexion: Language Agents with Verbal Reinforcement Learning**

- Paper: https://arxiv.org/abs/2303.11366
- Code: https://github.com/noahshinn/reflexion

The closest conceptual match to our approach. Reflexion uses verbal self-reflection stored in episodic memory to improve agent performance across trials. For code generation:
- Achieved 88% pass@1 on HumanEval (vs 67% for GPT-4 baseline)
- Uses self-generated test suites for grounded evaluation
- Key finding: **self-reflection-guided refinement outperforms refinement-only by 8%**

**Key difference from RoboPhD**: Reflexion uses fixed reflection prompts; we evolve the critic instructions.

### Self-Refine (Madaan et al., NeurIPS 2023)
**Self-Refine: Iterative Refinement with Self-Feedback**

- Paper: https://arxiv.org/abs/2303.17651
- Code: https://github.com/madaan/self-refine
- Website: https://selfrefine.info/

Three-step iterative approach: generate → get feedback → refine. Demonstrated ~20% improvement across diverse tasks including code optimization. Uses the same model for generation and feedback.

**Key difference**: Fixed feedback mechanism; no evolution.

### CRITIC (Gou et al., 2023)
**CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing**

- Paper: https://arxiv.org/abs/2305.11738

LLMs validate and amend outputs using tool interaction (e.g., code execution, calculators). Similar generate→critique→revise loop with external verification.

**Key difference**: Tool-based verification but fixed critique prompts.

### Critical Survey (Kamoi et al., TACL 2024)
**When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs**

- Paper: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00713
- ArXiv: https://arxiv.org/abs/2406.01297
- ACL Anthology: https://aclanthology.org/2024.tacl-1.78/

Important negative results showing LLMs cannot reliably self-correct without external feedback. Intrinsic self-correction (no external signal) often degrades performance.

**Implication for RoboPhD**: Supports the decision to use execution feedback rather than pure self-evaluation.

---

## 2. Verifier and Filtering Approaches

### AlphaCode (DeepMind, Science 2022)
**Competition-level code generation with AlphaCode**

- Paper: https://www.science.org/doi/10.1126/science.abq1158
- Website: https://alphacode.deepmind.com/

Different paradigm from refinement: generate millions of samples, filter by test execution, cluster by behavior, submit top 10. Uses a trained scoring model to rank candidates.

- Achieved top 54.3% ranking in competitive programming
- AlphaCode 2 (2023) improved to 85th percentile

**Key insight**: Filtering/selection can be as important as generation.

**Relevance to RoboPhD**: The critic could be viewed as a soft filter—rather than hard rejection, it provides feedback for revision.

### AlphaCodium (Ridnik et al., 2024)
**Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering**

- Paper: https://arxiv.org/abs/2401.08500

Test-based, multi-stage iterative flow for competitive programming. Improved GPT-4's pass@5 on CodeContests from 19% to 44%.

---

## 3. Prompt Optimization and Evolution

### DSPy (Stanford, 2023-2025)
**DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines**

- Paper: https://arxiv.org/abs/2310.03714
- Website: https://dspy.ai/
- Code: https://github.com/stanfordnlp/dspy

Framework for algorithmic prompt optimization using training data and metrics. Key optimizers:
- **MIPROv2**: Bayesian optimization over instruction space with few-shot bootstrapping
- **COPRO**: Coordinate ascent over instructions
- **SIMBA**: Stochastic mini-batch sampling with self-reflective improvement rules

**Relevance**: DSPy optimizes prompts for the generator; RoboPhD optimizes prompts for the critic.

### GEPA (Agrawal et al., July 2025)
**GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning**

- Paper: https://arxiv.org/abs/2507.19457
- Code: https://github.com/gepa-ai/gepa
- DSPy integration: https://dspy.ai/api/optimizers/GEPA/overview/

Uses natural language reflection on system trajectories to diagnose problems and propose prompt updates. Employs Pareto-based selection to avoid local optima. Outperforms GRPO by 10% average (up to 20%) while using 35x fewer rollouts; outperforms MIPROv2 by over 10%.

**Most similar to RoboPhD**: Evolves prompts based on execution trajectories and error analysis, though for generators not critics.

### OPRO (DeepMind, 2023)
**Large Language Models as Optimizers**

- Paper: https://arxiv.org/abs/2309.03409

Uses LLMs to iteratively optimize prompts by examining past attempts and their scores.

---

## 4. LLM-as-Judge and Learned Critics

### Survey on LLM-as-a-Judge (2024)
**A Survey on LLM-as-a-Judge**

- Paper: https://arxiv.org/abs/2411.15594

Comprehensive survey covering:
- Correlation with human judgment (~0.81 on code translation)
- Bias mitigation strategies
- Multi-agent evaluation (critics + defenders)

### JudgeLM and CritiqueLLM
**Trained critic models** (not evolved):
- JudgeLM: Fine-tuned for evaluation using reference support/drop paradigms
- CritiqueLLM: Multi-path prompting (pointwise-to-pairwise, referenced-to-reference-free)

**Key difference from RoboPhD**: These are trained via gradient descent on human preferences, not prompt evolution.

### Agent-as-Judge (2025)
**When AIs Judge AIs: The Rise of Agent-as-a-Judge Evaluation for LLMs**

- Paper: https://arxiv.org/abs/2508.02994

Emerging work on training agents specifically for evaluation tasks.

---

## 5. Code Generation Agents and Iterative Refinement

### ReVeal (Jin et al., 2025)
**ReVeal: Self-Evolving Code Agents via Reliable Self-Verification**

- Paper: https://arxiv.org/abs/2506.11442

Multi-turn RL framework with explicit self-verification through test case generation and tool interaction.

### AgentRefine (Fu et al., ICLR 2025)
**AgentRefine: Enhancing Agent Generalization through Refinement Tuning**

- Paper: https://arxiv.org/abs/2501.01702
- Code: https://github.com/Fu-Dayuan/AgentRefine
- OpenReview: https://openreview.net/forum?id=FDimWzmcWn

Connects agent generalization with self-refinement based on environment feedback. Trains models to develop step-level refinement abilities, adjusting decisions based on environment feedback.

### OpenCodeInterpreter (Zheng et al., ACL 2024)
**OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement**

- Paper: https://aclanthology.org/2024.findings-acl.762/
- ArXiv: https://arxiv.org/abs/2402.14658
- Code: https://github.com/OpenCodeInterpreter/OpenCodeInterpreter

Open-source code generation system integrating execution and human feedback for dynamic code refinement. Achieves 83.2% on HumanEval/MBPP, approaching GPT-4's 84.2%.

---

## 6. Positioning of RoboPhD Approach

The RoboPhD code generation critic approach occupies a unique position:

| Approach | What's Evolved | Critic Type | Uses Execution Feedback |
|----------|---------------|-------------|------------------------|
| Reflexion | Coder memory | Fixed self-reflection | ✓ |
| Self-Refine | Nothing (iterative) | Fixed self-feedback | |
| AlphaCode | Nothing (massive sampling) | Trained scorer | ✓ |
| DSPy/GEPA | Generator prompts | N/A | ✓ |
| JudgeLM | Critic weights | Trained model | |
| **RoboPhD** | **Critic prompts** | **Evolved agent** | ✓ |

**Novel contribution**: Evolving critic instructions through a meta-learning loop where critic performance (verdict accuracy, improvement rate) drives evolution. This inverts the typical paradigm of improving the generator while using fixed feedback.

---

## 7. Key Insights from Literature

1. **Self-correction without external feedback often fails** (Huang et al.) — supports using execution results

2. **Reflection-guided refinement > refinement-only** (Shinn et al.) — supports the critic providing structured feedback

3. **Filtering/selection matters as much as generation** (AlphaCode) — the critic acts as a soft filter

4. **Prompt evolution can match or exceed RL** (GEPA) — validates the evolutionary approach

5. **LLM judges achieve ~0.8 correlation with experts on code** (Survey) — ceiling for critic accuracy

---

## 8. Curated Reading Lists

### Awesome-LLM-as-a-judge
https://github.com/llm-as-a-judge/Awesome-LLM-as-a-judge

### Awesome-Code-LLM
https://github.com/codefuse-ai/Awesome-Code-LLM

### AwesomeLLM4SE (SCIS 2025)
https://github.com/iSEngLab/AwesomeLLM4SE
