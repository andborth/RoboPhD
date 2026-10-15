Name: ensemble-exec-vote-escalate

# Analysis of iterations 1–5

| Agent | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Completed-problem record | Mean cost |
|---|---|---|---|---|---|---|
| baseline (one-shot) | 13/20 | 16/20 | – | – | 29/40 | $0.011 |
| iter2_exec_verified_sonnet | 15/20 | 18/20 | 18/20 | 20/20 | 71/80 | $0.013 |
| iter3/4/5 ensemble-exec-vote family | – | 17/20 (3 harness kills) | 18/20 | 20/20 | 55/57 | $0.042–0.048 |

The ensemble family's only two losses on problems it finished are 165 and 284 (iteration 4), both
"reference dtype is a side effect of a sloppy accepted answer". Iteration 5 added a dtype-split judge
for the 165 pattern (correct candidate present, outvoted) and declared 284 (no candidate had the float
column) irreducible. Iteration 5's batch was 20/20 for all three agents, so the score table gives no
ranking signal; everything below comes from reading `agent_stdout`.

## What the iteration-5 logs show

Decision-rule distribution on the 20 problems: 11 unanimous, 2 majority-of-2, 5 three-way splits,
2 static (skeleton needs `load_data`). Of the five three-way splits, the Opus judge decided three
(517, 961, 962) and **two were decided by `fallback pick -> B`** (662 Matplotlib, 918 Sklearn), i.e.
the judge never ran. Both logs end with `est_cost=$0.126` / `$0.104` against `SOFT_CAP = 0.10`, while
the harness billed $0.077 and $0.067. The agent's cost estimator adds `reasoning_tokens` on top of
`output_tokens` for every provider; checking against `eval_cost` shows OpenAI and Anthropic already
include reasoning in `output_tokens` (only Gemini bills it on top). A GPT-5.4 medium call with ~3k
reasoning tokens therefore inflated the estimate by ~$0.05 and tripped the cap, and the single most
valuable call in the pipeline, the judge on a three-way split, was skipped. The fallback (highest
priority loop-free candidate) happened to be right both times, which is luck, not design. With the
corrected formula the two samples estimate to $0.0774 and $0.0667, matching `eval_cost` to four digits.

Other observations:
- The judge is 8/8 over iterations 4–5 whenever it ran (398, 999, 838, 887, 977, 517, 961, 962).
- Three-way splits are ~25% of problems (5/20 in iteration 5, 2/20 plus three static judgements in
  iteration 4). This is where the remaining risk concentrates; unanimous picks have been right 100%.
- Gemini 3.1 Pro stalled past its 240 s timeout again (662); its successful calls take 5–62 s.
- Mean cost $0.048 against a $0.127 free zone: more than half the free budget is unused, and the
  free zone is a batch average, so money spent only on the disagreeing ~25% is cheap in the mean.

# Approach

Keep iteration 5's pipeline (parser, runner, three-provider generation, execution, repair, majority
vote, dtype-split judge, generality check, loop guard, bounded waits, best-so-far emission) and change
the decision path on disagreement, which is the only place where uncertainty remains.

1. **Fix the cost estimator** (no reasoning double-count for OpenAI/Anthropic) and give the judges
   their own cap (`JUDGE_CAP = 0.30`) separate from the optional-step cap (`SOFT_CAP = 0.16`). The
   disagreement judge, the dtype judge and the generality judge can no longer be skipped by an
   estimate; a skipped judge is now logged explicitly.
2. **Escalation on pairwise disagreement.** When no two runnable candidates agree on the example
   output, two more candidates are generated in parallel (Claude Sonnet 4.6 medium, GPT-5.5) and
   executed. With five voices, a group of ≥3 wins outright; otherwise the judge sees one
   representative per distinct output, annotated with how many models produced it, and a note that
   agreement is evidence rather than proof. This turns "three different answers, pick one" into
   "here is how five independent models split". Bounded: 240 s, skipped past `OPTIONAL_AFTER` or when
   the sandbox is dead or the estimate is already over `SOFT_CAP`.
3. **Judge with a little reasoning** (Opus 4.8, `reasoning_effort="low"`). The judge was 8/8 without
   it, so this is insurance on the hardest cases rather than a fix; "low" is the cheapest opt-in.
4. **Gemini timeout 150 s** (was 240 s); the Sonnet top-up still restores three voices.
5. **Replayable logs**: each runnable candidate logs the hash of its strict and value-normalised
   output and its dtype lines, so the next session can count splits and dtype-splits from
   `agent_stdout` without reconstructing dumps (the iteration-5 reflection's main request).

## Cost

Iteration 5 measured $0.048. Escalation adds two generations (~$0.02–0.06) on ~25% of problems
(+$0.01 mean); judge reasoning adds ~$0.01–0.03 on the same subset (+$0.005 mean); the corrected
estimator lets a few previously-skipped judges run (+$0.003). Expected mean ≈ $0.065, half the free
zone; a pathological batch where every problem escalates and is judged stays near $0.12.

# Why this should score higher

- The one concrete defect found in the logs (judge silently skipped on two of five three-way splits)
  is fixed; the floor is iteration 5's behaviour because every new step falls back to the old pick.
- Unanimous and majority paths, which have been 100% correct, are untouched (the `need` threshold is
  still 2 of 3 when no escalation happens).
- The three-way split is the only path with real uncertainty, and it now gets strictly more
  evidence (five executed candidates with counts) and a judge that always runs and may think.
- No prompt or rule is tuned to a specific problem; the generator prompt is unchanged, so candidate
  quality on the 11/20 unanimous problems is identical.

# Validation

`test_dryrun.py` (stubbed models, local `python_session`) extended with a `GPT_5_5` stub and
per-candidate code overrides (`A=... E=...`). Scenarios run: three distinct outputs with two agreeing
escalation candidates → `[2,1,1,1]`, judged with counts; `[2,2,1]` → judged, `CHOOSE: E` resolves to
an escalation candidate not in the deduplicated pool; five distinct → judged; Sonnet stalling during
escalation → GPT-5.5 alone joins and the judge runs; value-agree/dtype-split → dtype judge; loop-guard
(269), static judging with a dead sandbox (883), Matplotlib (611) regressions. The cost estimator was
unit-checked against the billed `eval_cost` of iteration-5 samples 918 and 662 (exact match).
