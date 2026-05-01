"""
Base class for meta-evolution strategies.

A meta-evolution strategy bundles every design choice that distinguishes one
meta-evolution approach from another into a single Python class:

- The strategy body (`instructions_for_llm`) — was the .md file content.
- The CLAUDE.md content (`claude_md_section`) the meta-evolution AI sees as
  context for every firing.
- The initial-firing prompt (`initial_firing_prompt`) and follow-up firing
  prompt (`followup_firing_prompt`).
- The reflection prompt (`reflection_prompt`) requested at the end of each
  firing.

The base class provides defaults that reproduce the behavior the manager
implemented inline before this refactor; subclasses override only what
differs. All hook methods take explicit keyword arguments rather than a
context dict so the input contract is visible from the signature alone —
an LLM editing a subclass does not need to read the manager to know what
data is available.
"""

from abc import ABC, abstractmethod
from typing import ClassVar


META_EVOLUTION_ENVIRONMENT_GUIDE = """\
# Meta-Evolution Environment

## Cadence

You are called every {cadence} iterations: first firing at iter {first_iteration}, then iter {first_plus_cadence}, {first_plus_2cadence}, … Plan your `meta_config_schedule.json` decisions with this {cadence}-iteration horizon in mind — any change you propose will run for ~{cadence} iterations before you see its effect and can revise.

The Claude Code session persists across all firings within a run; subsequent firings deliver brief status updates against this same session.

## Reading Elo Signals

Elo measures *relative* win-rate among the agents that have actually been compared head-to-head — not absolute quality. A leader at high Elo means "wins more often against the agents we've tried"; it does not mean "no better agent is possible."

In principle a run could produce an agent that is genuinely unbeatable, but this is rare. Far more often, a long-unchanged leader signals that evolution has stopped finding new directions to explore — not that it has hit a ceiling. When you see a leader that has been stable for several iterations, default to the second interpretation: the search has narrowed, not the space of possible improvements.

Remember that your job is to help evolution produce something better than the incumbent. An incumbent that is difficult to beat means that you did your job well in previous iterations, but you are always searching for new ways to improve it. If you fail, then this evolutionary run will yield the current incumbent, but you want to try to do even better.

## Working Directory

Your working directory is the run's `meta_evolution_output/` directory, which is stable across all firings within a run (so the persistent Claude Code session can be resumed each iteration). Iteration-specific subdirectories live as children:
- `iteration_NNN/` — per-firing output (reasoning.md, meta_config_schedule.json, new_strategies/, etc.)
- `../iteration_NNN/` — per-iteration outputs from the main run: interim_report.md, cost_report.md, error_analysis_report.md, plus `agent_<name>/` subdirectories holding each participating agent's per-problem evaluation outputs (NOT the agent code).
- `../agents/<name>/` — installed agent packages (the actual agent code).
- `../evolution_strategies/` — installed evolution strategies (yours land here after validation)

## Per-Iteration Reports

These reports are generated after each iteration at `../iteration_NNN/` (relative to your working dir):
- `error_analysis_report.md` — cross-agent score comparison & failure summary
- `error_index.json` — raw per-problem score data (source for the report)
- `cost_report.md` — per-agent LLM cost breakdown (tokens, cache hits, USD)

## CLI Tools

{cli_tools}

## Required Outputs

Each firing must produce, at minimum:
- `iteration_NNN/reasoning.md` — your analysis. Each meta-evolution strategy is expected to specify what reasoning.md should contain; follow your strategy's instructions. (If your strategy doesn't specify, document your decisions and rationale at your own discretion.)
- `iteration_NNN/meta_config_schedule.json` — config changes for upcoming iterations. Can be empty (`{{}}`) if you propose no schedule changes; the file itself must exist.

Optional, only if your strategy authorizes:
- `iteration_NNN/config_delta.json` — immediate parameter change starting next iteration (persists until overwritten).
- `iteration_NNN/new_strategies/<name>/strategy.md` — a new evolution strategy.

Missing required outputs trigger a correction prompt within the same session; persistent failure terminates the run.

## Strategy Packages

If you create a new evolution strategy, it lives at `iteration_NNN/new_strategies/<name>/` as a package containing:
- `strategy.md` (required) — YAML frontmatter with `name` and `description` fields, followed by instructions for the evolution AI on how to create agents.
- `strategy_tools/` (optional) — Python helper scripts the evolution AI can run (custom error analysis, state tracking, specialized reports). Details below.

Review existing strategies under `../evolution_strategies/` for patterns and structure to follow. The format and content of `reasoning.md` is whatever your meta-evolution strategy specifies.

### Strategy tools details

When you include `strategy_tools/` in a package, those tools are **symlinked into the evolution working directory** as `strategy_tools/`. Reference them as `python strategy_tools/<script>.py` in your strategy.md instructions.

- Tools should use only stdlib and libraries already installed in the environment
- Include `--help` support so Claude can discover usage
- Reference them with imperative language in strategy.md (e.g., "Run `python strategy_tools/analyze_failures.py ...`" not "If the tool is available...")
- The symlink will exist — do NOT include fallback instructions suggesting the tool might be missing

## Strategy Naming

Pick a hyphenated, lowercase name for your strategy (e.g. `cost_mechanism_aware`) — it ends up in installed paths and the schedule, so legibility matters.

When you create a new evolution strategy at `iteration_NNN/new_strategies/<name>/`, the system installs it as `evolution_strategies/iter{{N}}_<name>/` — your name is automatically prefixed with `iter{{N}}_` to keep each firing's contribution unique (mirroring how evolved agents get an iter prefix). **Reference the prefixed form in `meta_config_schedule.json`.**

For example, if at iteration 7 you create `new_strategies/cost_mechanism_aware/`, it installs as `evolution_strategies/iter7_cost_mechanism_aware/`. Your schedule should reference `"evolution_strategy": "iter7_cost_mechanism_aware"`, not `"cost_mechanism_aware"`.

If you reference a name that doesn't resolve, you'll get a correction prompt with the full list of installed strategies and the prefixed form of any strategy you just created.

## Configuration Persistence

Configurations persist across iterations once set. A `meta_config_schedule.json` entry like `{{"4": {{"evolution_strategy": "X"}}}}` does NOT mean "use X at iteration 4 only" — it means "starting at iteration 4, use X until another entry overrides it." To restrict X to a single iteration, schedule both the change AND the revert: `{{"4": {{"evolution_strategy": "X"}}, "5": {{"evolution_strategy": "Y"}}}}`.

## Schedule Format

`meta_config_schedule.json` is a top-level mapping from iteration-number strings to delta dicts. Example:

```json
{{
  "11": {{"evolution_strategy": "iter11_my_strategy"}},
  "13": {{"evolution_strategy": "iter4_my_other_strategy"}}
}}
```

Iteration 11 starts using `iter11_my_strategy` (a strategy you just created); iteration 12 inherits it (no override scheduled); iteration 13 switches to `iter4_my_other_strategy` (an older strategy you created in a prior firing). See Configuration Persistence above for the inheritance rule.

## Horizon

The run may be extended beyond the current iteration count. Don't treat any iteration as "final" or optimize for a specific end point — make decisions based on strategy performance trends, not on how many iterations remain.

## Framework Behavior (post-firing)

After your firing completes, the framework will:
- Discover strategies by scanning `iteration_NNN/new_strategies/` for subdirectories
- Validate each strategy package (frontmatter, syntax) and prompt you to correct any errors
- Install valid strategies to `evolution_strategies/iter{{N}}_<name>/`
- Validate that every `evolution_strategy` reference in your schedule resolves to an installed strategy; prompt for correction if not
- Integrate `meta_config_schedule` via ConfigManager (your changes take effect at their scheduled iterations)"""


class MetaEvolutionStrategy(ABC):
    """
    Base class for a meta-evolution strategy.

    A subclass must set `name` and `description` class vars and override
    `instructions_for_llm`. All other hooks have defaults that reproduce the
    pre-class behavior of the manager.
    """

    name: ClassVar[str]
    description: ClassVar[str]

    @abstractmethod
    def instructions_for_llm(self) -> str:
        """
        The strategy body — instructions the meta-evolution AI sees in its
        initial firing. Was the .md file content prior to the refactor.

        Return a plain string starting at the first heading (e.g.
        "# Minimal Guidance"). The default `initial_firing_prompt` wraps this
        body with the framework's Current State / Your Task scaffold.
        """

    def claude_md_section(
        self,
        *,
        domain_background: str,
        domain_objective: str,
        cadence: int,
        first_iteration: int,
        cli_tools: str,
    ) -> str:
        """
        The full content of the run's `meta_evolution_output/CLAUDE.md` —
        what the meta-evolution AI sees as persistent environmental context.

        Default: optional Domain Background + Domain Objective sections (when
        the corresponding strings are non-empty) followed by
        META_EVOLUTION_ENVIRONMENT_GUIDE formatted with cadence, first_iteration,
        cli_tools. Subclasses may rearrange or omit any section.
        """
        sections = []
        if domain_background:
            sections.append(f"# Domain Background\n\n{domain_background}")
        if domain_objective:
            sections.append(f"# Domain Objective\n\n{domain_objective}")
        sections.append(
            META_EVOLUTION_ENVIRONMENT_GUIDE.format(
                cadence=cadence,
                first_iteration=first_iteration,
                first_plus_cadence=first_iteration + cadence,
                first_plus_2cadence=first_iteration + 2 * cadence,
                cli_tools=cli_tools,
            )
        )
        return "\n\n".join(sections)

    def initial_firing_prompt(
        self,
        *,
        iteration: int,
        interim_reports: str,
        budget_status: str,
        domain_background: str,
        domain_objective: str,
    ) -> str:
        """
        Full text of the prompt sent to Claude on the first firing of a run.

        Default: `instructions_for_llm()` + Current State + Your Task scaffold.
        Default ignores `budget_status`, `domain_background`, `domain_objective`
        — today's initial firing sources domain context from CLAUDE.md and
        shows no budget. The arguments are passed in regardless so subclasses
        that want to weave them in directly can do so without any manager
        change.
        """
        return f"""
{self.instructions_for_llm()}

## Current State (Iteration {iteration})

### Recent Performance
{interim_reports}

## Your Task

Produce the artifacts for this firing in `iteration_{iteration:03d}/`. Per your strategy:
- `reasoning.md` (REQUIRED) — your analysis, formatted per your strategy's instructions
- `meta_config_schedule.json` (REQUIRED) — can be `{{}}` if no changes
- `new_strategies/<name>/strategy.md` and/or `config_delta.json` — only if your strategy authorizes them

See `CLAUDE.md` (already in your context) for: cadence, strategy-package structure, naming convention, schedule format, schedule semantics, and the framework's post-firing actions.
"""

    def followup_firing_prompt(
        self,
        *,
        iteration: int,
        cadence: int,
        budget_status: str,
        domain_background: str,
        domain_objective: str,
    ) -> str:
        """
        Full text of the prompt sent to Claude on follow-up firings (within
        the persistent session, so the strategy body and CLAUDE.md from the
        initial firing are still in context).

        Default: header, latest reports list, budget_status, next-firing line,
        Your Task, completion sentinel. Default ignores `domain_background` /
        `domain_objective` — the persistent session retains them via CLAUDE.md.
        """
        return f"""## Meta-Evolution Firing — Iteration {iteration}

Iteration {iteration} has just completed. Updated reports for this iteration (paths relative to your meta_evolution_output/ working dir):
- Interim report: `../iteration_{iteration:03d}/interim_report.md`
- Cost report: `../iteration_{iteration:03d}/cost_report.md`
- Error analysis: `../iteration_{iteration:03d}/error_analysis_report.md`

{budget_status}

Next firing: iteration {iteration + cadence} (or run end if budget exhausts first).

Produce the artifacts for this firing in `iteration_{iteration:03d}/`. Per your strategy:
- `reasoning.md` (REQUIRED) — your analysis, formatted per your strategy's instructions
- `meta_config_schedule.json` (REQUIRED) — can be `{{}}` if no changes
- `new_strategies/<name>/strategy.md` and/or `config_delta.json` — only if your strategy authorizes them

After completing, respond with: "META-EVOLUTION ITERATION {iteration} COMPLETE"
"""

    def reflection_prompt(self, *, iteration: int) -> str:
        """
        Prompt asking Claude to reflect on the firing's work.

        Default: the existing reflection prompt the manager used inline
        prior to the refactor.
        """
        return f"""Take a moment to reflect on this firing's work. You're in a persistent session — your next firing (a few iterations from now) will have this iteration's context already in memory, so this reflection serves two purposes: an audit-trail checkpoint for the human reviewing the run, and a chance to consolidate the insights you most want to carry forward.

Please consider:
- What patterns or insights from this iteration's data are worth emphasizing for next time?
- What was challenging or time-consuming about the analysis or implementation?
- Were the provided tools and reports helpful? Anything you wished you had?
- What would you do differently in the next firing?
- Any prompt or tooling changes worth flagging for the human maintainer of this system?

**Keep your reflection concise - 300 lines or less.**

Save your reflection to `iteration_{iteration:03d}/meta_evolution_reflection.md`.

After saving the reflection, respond with: "REFLECTION COMPLETE"
"""
