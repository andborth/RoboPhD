# CodeGen Critic: Project Status

**Updated:** February 10, 2026

## Project Overview

This project extends RoboPhD from Text2SQL to code generation, targeting the LiveCodeBench benchmark (1,055 competition programming problems). Rather than evolving the code generator directly, we evolve **critic agents** that review code and provide feedback. The key insight: learning *what feedback helps* is more tractable than learning to solve problems directly.

The coder generates a solution, the critic reviews it pre-execution (without access to hidden tests), and the coder decides whether to accept the critic's suggestions. Selection is based on binary pass/fail test outcomes, and agents compete via ELO ranking — the same mechanism that drove Text2SQL to 73.67% on BIRD.

## What's Been Built

### Domain Abstraction

The Text2SQL and CodeGen architectures are structurally identical — they differ only in what gets fed into Phase 1. A `DomainInterface` abstraction (`RoboPhD/domains/base.py`) lets `researcher.py` drive both domains without domain-specific logic. The three-artifact agent format (`agent.md`, `eval_instructions.md`, `tools/`), evolution strategies, ELO ranking, deep focus evolution, meta-evolution, and checkpoint/resume all carry over unchanged.

### Critic Evaluation Pipeline

The CodeGen pipeline (`RoboPhD/tools/run_critic_evaluation.py`, ~1,900 lines) runs as a subprocess and implements six phases:

1. **Codegen** — Coder generates initial solution (Code v1) with access to visible test examples
2. **Reflection** — Same session, coder describes its algorithmic approach
3. **Critic Review** — Critic analyzes Code v1 and approach, outputs CORRECT/INCORRECT with structured feedback
4. **Revision** — If INCORRECT, critic generates revised code (Code v2)
5. **Acceptance** — Coder evaluates suggestions, accepts/rejects (ACCEPTED_ALL, ACCEPTED_SOME, REJECTED_ALL)
6. **Test Execution** — Run final code against hidden test suite, binary pass/fail

Solution caching with Claude Code session persistence avoids re-generating Code v1 across evaluation runs. Cache is stored per-model under `../robophd_runs/codegen_cache/{model}_v6/{problem_id}/`.

### LiveCodeBench Integration

| Split | Count | Date Range | Purpose |
|-------|-------|------------|---------|
| Evolution | 767 | May 2023 – Oct 2024 | Sample ~60–100 per iteration for critic evolution |
| Test | 288 | Nov 2024 – Apr 2025 | Final evaluation only, never seen during evolution |

Temporal split at 2024-11-01 ensures test problems post-date model training cutoffs. The ~27% test split provides stable evaluation metrics.

## Key Results

### 3x3 Naive Critic Model Grid

Nine runs testing all combinations of haiku-4.5, sonnet-4.5, and opus-4.5 as coder and critic on 288 test problems:

**Delta matrix (V2 – V1 accuracy):**

|  | haiku critic | sonnet critic | opus critic |
|--|--------------|---------------|-------------|
| **haiku coder** | -0.7% | +4.5% | +8.3% |
| **sonnet coder** | +1.1% | +0.0% | +7.7% |
| **opus coder** | +0.0% | +1.5% | +2.2% |

**Findings:**

- **Capability gap hypothesis.** For meaningful improvement, the critic must be stronger than the coder. Opus critic averages +6.1% across all coders; sonnet only helps haiku (+4.5%); haiku helps no one.
- **Same-tier critics are unreliable.** haiku→haiku hurts (-0.7%), sonnet→sonnet breaks even, opus→opus gains modestly (+2.2% off an 85% baseline).
- **Detection ≠ correction.** Even when critics correctly identify wrong code, they often fail to fix it. Fix rate of true positives ranges from 3.5% (haiku→haiku) to 53.8% (opus→opus).
- **Opus code is hard to improve.** With V1 accuracy at ~85%, only ~42 errors exist in 288 problems. Even opus→opus catches just 32% of them.

### Skip-a-Tier: Evolved Haiku Matches Naive Sonnet

The headline result, replicating the finding from the RoboPhD Text2SQL paper:

| Config | Agent | V1 | V2 | Delta | Net Fixes |
|--------|-------|----|----|-------|-----------|
| haiku → haiku | naive | 51.7% | 51.0% | -0.7% | -2 |
| haiku → sonnet | naive | 52.1% | 56.6% | +4.5% | +13 |
| **haiku → haiku** | **0203_i005 (evolved)** | **52.1%** | **56.6%** | **+4.5%** | **+13** |

The evolved agent (`0203_i005_reflection_refined_critic`) comes from a 14-iteration evolution run on the 767-problem evolution set. It uses tool-only execution mode — an 875-line Python analyzer generates structured critic feedback without any LLM call in the analysis phase. The result: haiku with evolved instructions matches the accuracy of a naive sonnet critic.

**How it works:** The evolved agent flags 138 problems (vs 73 for naive haiku, 75 for naive sonnet) with 78% recall (vs 41% naive haiku, 48% naive sonnet). It compensates for model capability with context — a 14,161-byte prompt vs 944 bytes for naive critics — effectively substituting compute for capability.

**Cost:** 1.3x naive sonnet ($0.194 vs $0.149 per problem). The critic phase is actually cheaper ($0.116 vs $0.123) since haiku's 3x lower per-token pricing offsets the larger prompt. The premium comes from revision: more flagged problems means more revision calls. Trimming `eval_instructions.md` from 294 to ~200 lines would achieve cost parity.

**Caveat:** A second evolved agent from a different evolution run achieved only +0.7%. Evolution quality varies across runs — consistent results remain an open challenge.

## Open Weight Model Support

### Goal

Enable open weight models (e.g., Qwen3 Coder 30B) as coder and/or critic in CodeGen. While Text2SQL uses open weight models via direct API calls through LiteLLMProvider, CodeGen routes everything through Claude Code CLI subprocesses. This is the first use of open weight models through Claude Code.

### Implementation

Per-subprocess environment routing lets local models and the Anthropic API coexist in the same run:

- `get_lmstudio_env(model)` in `config.py` returns `None` for Anthropic models, or `{"ANTHROPIC_BASE_URL": base_url, "ANTHROPIC_AUTH_TOKEN": "lmstudio"}` for local models
- `call_claude_cli()` in `utilities/claude_cli.py` accepts `extra_env` to override environment per subprocess
- Cache directory sanitization handles model names with `/` (e.g., `qwen/qwen3-coder-30b` → `qwen--qwen3-coder-30b`)
- `coder_model_tag` config field isolates cache by quantization variant (e.g., `q4_K_M`)

Both `config.py` (`CLAUDE_CLI_MODEL_MAP`) and `run_critic_evaluation.py` (`MODEL_MAP`) maintain their own model maps; non-Anthropic models bypass validation and pass through directly.

### Status

First test run completed with Qwen3 Coder 30B via LM Studio.

## Architecture Quick Reference

| File | Role |
|------|------|
| `RoboPhD/domains/codegen/domain.py` | CodeGen domain implementation, calls `run_critic_evaluation.py` as subprocess |
| `RoboPhD/tools/run_critic_evaluation.py` | Standalone evaluation pipeline (~1,900 lines), manages coder/critic workflow and test execution |
| `RoboPhD/domains/base.py` | `DomainInterface` ABC — the abstraction that lets `researcher.py` drive both Text2SQL and CodeGen |
| `utilities/claude_cli.py` | `call_claude_cli()` — central function for all Claude CLI subprocess calls, supports `extra_env` |
| `RoboPhD/config.py` | Model configs, `CLAUDE_CLI_MODEL_MAP`, `get_lmstudio_env()` |
| `RoboPhD/config_manager.py` | Delta-based config with defaults, schedules, weighted random |
| `RoboPhD/researcher.py` | Main entry point and experiment orchestrator |
| `RoboPhD/evolution.py` | Evolution strategy selector and orchestration |
| `RoboPhD/deep_focus_evolution_manager.py` | Multi-round evolution with Claude CLI sessions |
| `RoboPhD/meta_evolution_manager.py` | Meta-evolution for evolving strategies themselves |

## Current Status & Next Steps

**Working now:**
- Full CodeGen domain with critic evaluation pipeline
- Solution caching with session persistence
- 3x3 naive model grid evaluated on test set
- Skip-a-tier result demonstrated (evolved haiku matches naive sonnet)
- Open weight model support via LM Studio

**Next steps:**
- Trim evolved agent prompts for cost parity with naive sonnet
- Evolve critics for sonnet and opus coders (current results are haiku-only)
- Investigate why detection is easier than correction
- Test whether evolution can close the gap to opus-level critic performance (+8.3%)
- Explore multi-round critique (critic reviews its own revision)
- Scale open weight model testing beyond initial Qwen3 run
