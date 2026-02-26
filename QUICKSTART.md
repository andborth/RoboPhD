# Quick Start Guide

This guide will get you running RoboPhD in 5 minutes.

## Prerequisites

Ensure you've completed [INSTALLATION.md](INSTALLATION.md) first.

## Your First Run

### 1. Quick Test (2 minutes)

Test that everything works with a minimal run:

```bash
# Text2SQL (requires BIRD dataset)
python RoboPhD/researcher.py \
  --num-iterations 1 \
  --config '{"examples_per_iteration": 1, "problems_per_context": 5}'
```

This runs 1 iteration on 1 database with 5 questions - fast but verifies your setup.

```bash
# Or test CodeGen domain (no dataset needed)
python RoboPhD/researcher.py \
  --num-iterations 1 \
  --config configs/codegen_small_test.json
```

### 2. Dev Set Evaluation (10 minutes)

Evaluate the pre-trained `haiku_best` agent on the dev set:

```bash
python RoboPhD/researcher.py \
  --dev-eval \
  --config '{"initial_agents": ["haiku_best"], "eval_model": "haiku-4.5"}'
```

Expected accuracy: ~66%.

### 3. Run Evolution (1+ hours)

Start an evolution run to improve agents:

```bash
# Text2SQL evolution
python RoboPhD/researcher.py \
  --num-iterations 10 \
  --config configs/primary_production.json

# CodeGen evolution
python RoboPhD/researcher.py \
  --num-iterations 10 \
  --config configs/codegen_with_simple_meta_evolution.json
```

This will:
- Test agents on random task subsets
- Evolve new agents using cross-pollination
- Track performance with ELO rankings
- Generate reports after each iteration

## Understanding the Output

### Directory Structure

Each run creates a timestamped directory:

```
output/robophd_YYYYMMDD_HHMMSS/
├── checkpoint.json          # Resume state
├── final_report.md          # Summary with ELO rankings
├── iteration_001/           # First iteration results
│   ├── agent_XXX/           # Per-agent results
│   └── error_analysis_report.md
├── iteration_002/           # Second iteration
└── evolution_output/        # Evolution artifacts
    └── iteration_002/
        ├── agent.md         # New evolved agent
        └── reasoning.md     # Evolution reasoning
```

### Key Files

- `final_report.md` - Overall results and ELO rankings
- `checkpoint.json` - State for resuming runs
- `iteration_XXX/interim_report.md` - Per-iteration results

## Common Commands

### Resume a Run

```bash
python RoboPhD/researcher.py --resume output/robophd_20251031_043607
```

### Extend a Completed Run

```bash
python RoboPhD/researcher.py \
  --resume output/robophd_20251031_043607 \
  --extend 5
```

### Change Configuration Mid-Run

```bash
python RoboPhD/researcher.py \
  --resume output/robophd_20251031_043607 \
  --from-iteration 5 \
  --modify-config '{"evolution_strategy": "refinement_tool_only"}'
```

## Configuration Options

### Models

```jsonc
// Text2SQL
{
  "eval_model": "haiku-4.5",      // SQL generation (cheapest)
  "analysis_model": "haiku-4.5",  // Database analysis
  "evolution_model": "opus-4.6"   // Agent evolution (best quality)
}
// CodeGen
{
  "coder_model": "haiku-4.5",     // Code generation
  "critic_model": "haiku-4.5",    // Code review critic
  "evolution_model": "opus-4.6"   // Agent evolution (best quality)
}
```

### Scale

```json
{
  "examples_per_iteration": 8,     // More = better signal, slower (databases for Text2SQL)
  "problems_per_context": 40,      // More = better signal, more API cost (questions per DB)
  "agents_per_iteration": 3        // Agents tested per iteration
}
```

### Evolution

```json
{
  "evolution_strategy": "cross_pollination_tool_only",
  "new_agent_test_rounds": 2       // Deep focus refinement rounds
}
```

## Tips for Best Results

1. **Start with `primary_production.json`** (Text2SQL) or `codegen_with_simple_meta_evolution.json` (CodeGen)
2. **Use `opus_best` as initial agent** (Text2SQL) - Our best performer. Use `naive` to start fresh. For CodeGen, start with `naive_critic`.
3. **Run 20+ iterations** - Evolution needs time to find improvements
4. **Pre-compute ground truth** (Text2SQL) - Prevents database timeout errors

## Next Steps

- Read [CLAUDE.md](CLAUDE.md) for comprehensive documentation
- Explore evolution strategies in `RoboPhD/evolution_strategies/` (Text2SQL) or `RoboPhD/evolution_strategies_codegen/` (CodeGen)
- Check out pre-trained agents in `RoboPhD/agents/` (Text2SQL) or `RoboPhD/codegen_agents/` (CodeGen)

## Getting Help

If you encounter issues:
1. Check [INSTALLATION.md](INSTALLATION.md) troubleshooting section
2. Review error messages in iteration reports
3. Open an issue on GitHub
