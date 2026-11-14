# Evolution Reflection — DS-1000 agent (from iteration 12)

## The single most important insight

**This task converged to a stable answer around iteration 6, and iterations 7–12 were
largely re-confirming it plus adding tiny format fixes.** The settled architecture:

> One `GPT_5_4.generate()` call, default reasoning ("none"), iter6's TINY preamble,
> `max_tokens=1024`, mini fallback on error, + a few deterministic *format-only*
> extraction fixes (html.unescape, function-body reindent, target-variable
> assignment-wrap). Exactly ONE candidate — no verify, no self-consistency, no
> reasoning escalation.

If you are a future instance, **start from the current leader and make one surgical,
provably-safe change** rather than re-exploring. The exploration was already done and
is expensive to repeat.

## What worked

- **Persistent memory was the highest-leverage tool by far.** The `MEMORY.md` +
  `ds1000-agent-insights.md` file carried the full falsification history across
  sessions. Without it, every session would re-derive (and re-pay for) the same dead
  ends. Future instances: READ IT FIRST, TRUST IT, and APPEND your batch result +
  next design to it before finishing.
- **"Fix-or-no-op" format fixes are the only reliably-positive lever.** unescape,
  reindent, assign-wrap each remove a *confirmed* loss class and provably cannot break
  a working answer (they touch the form of the code, never the model's chosen answer,
  and there's still one candidate). This is the safe way to make monotone progress on
  a task dominated by n=20 noise.
- **Diagnosing failures by diffing `extracted_code.md` across agents** was decisive for
  attribution — it let me prove iter11's dip was sampling noise (different model
  answer) rather than a regression in my post-processing. Always check whether a
  failure is a *content* difference or a *format* artifact before reacting.

## What was challenging / time-consuming

- **n=20 batch noise (~1σ per problem) is the dominant confound.** Score swings of
  1–3 problems between agents are almost always sampling noise, not signal. Many
  iterations "learned" a lesson from a gap that was noise. Do NOT over-fit a single
  batch; weight the cross-batch record and the mechanism (why did this specific
  problem fail?) over the aggregate number.
- **GPT_5_4 is stochastic on the same prompt** — it emits a function body indented or
  not, assigns the target var or not, from run to run. Several "regressions" were the
  same agent sampling differently. This is exactly why the deterministic format fixes
  (which normalize those stochastic slips) are valuable.
- **Distinguishing real reasoning wins from unfixable-hard problems** took care. Some
  consensus failures (dtype-coercion traps, opaque hidden-data loaders, exact
  melt/merge column ordering) are near-unguessable and chasing them regresses easy
  problems. The record now flags these — don't re-chase them.

## Tools

- `python_session` / `sandbox()` are **unmetered** — only `get_model()` counts toward
  cost. This is a big deal that isn't obvious; execution-based verification is "free"
  in cost terms. (It still didn't help, because wrong answers execute cleanly, but the
  cost model is worth knowing.)
- No `strategy_tools/` directory was provided this session; analysis was manual
  (reading `extracted_code.md` / `test_result.md`, diffing agents). A helper that
  auto-diffs the current-leader's per-problem extracted code vs the new agent's, and
  flags content-vs-format differences, would save time.
- A local test harness (`test_extract.py`) for the pure-Python extraction logic was
  easy to stub (mock the inspect/model imports) and gave fast, deterministic
  confidence in the fix-or-no-op fixes without a full eval.

## What I'd do differently / suggestions for the process

1. **The evolution is near its ceiling; diminishing returns are real.** Iters 7–12 each
   moved by one small format fix. Future value is limited to (a) catching a *new*
   confirmed format loss class, or (b) a genuinely better model/backbone if one appears.
   Consider telling future instances this explicitly so they don't burn effort
   re-exploring falsified machinery.
2. **Cost was never binding.** Every leader ran at ~$0.0013–0.0016 vs the $0.003 free
   zone — 2× headroom, single call. The cost table in the prompt is well-explained but
   in practice a single strong-model call is always safe; future instances shouldn't
   agonize over it.
3. **Prompt suggestion:** surface the current leader agent's name + recipe + its
   cross-batch score history directly in the iteration prompt (not just "3 agents were
   tested"). I had to reconstruct "who is the leader and why" from memory each time.
4. **Prompt suggestion:** the per-batch score deltas invite over-fitting. A one-line
   caveat in the prompt — "n≈20; a 1–3 problem gap is within noise, prefer the
   cross-batch trend and mechanism" — would steer future instances away from the most
   common mistake in this project's own history.
5. **Bigger evaluation batches (n≥50) would change the game** — they'd let real signal
   separate from noise and allow confident A/B of ideas that currently wash out. If the
   harness can afford it, larger batches are the single biggest process improvement.
