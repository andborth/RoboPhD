Name: iter13_style_aware_lean

# Analysis

## Iteration 12 results (3-way comparison on 20 problems)

| Agent | Mean Score | Cost | Avg cost/problem |
|---|---|---|---|
| iter8_robust_judge | 95.000 | $0.20 | $0.018 |
| iter9_mpl_aware_judge | 95.000 | $0.19 | $0.021 |
| iter12_version_aware_judge | 89.990 | $0.88 | $0.044 |

iter12 lost to iter8/iter9 by ~5 points despite carrying every iter11 strength plus
new "version-aware" features. The 20-problem batch shows it cleanly:

- **iter12 won 1 problem (269)**: a `df.stack/unstack` reshape problem with a
  hidden `assert "for" not in tokens` style assertion. iter8 and iter9 emitted
  a list-comprehension-based answer (`[f"{c}_{i+1}" for ... ]`) — execution
  passed, the style assertion failed. iter12's style-hint detection +
  loop-free-aware judge pushed it to a `df.stack().to_frame().T` solution that
  satisfied both the value test and the style assertion.

- **iter12 lost 2 problems (906, 860)**: both Sklearn problems with
  `load_data()` setups. All four model candidates produced nearly correct code
  but smoke failed because the smoke fixture stubs `load_data() -> None`.
  iter12's "alt-approach regeneration" feature kicked in (because all smokes
  failed and the chosen code matched a failing candidate), generated a NEW
  solution with sonnet@0.5, and adopted it via the rule "if alt smoke also
  fails the same NoneType-fixture way and the alt error is different, adopt
  it anyway." This rule throws away validation — it adopts an unverified
  alternative on top of a fixture-noise signal. iter9 in this slot fell
  through to its first candidate (which was correct) and won.

- **iter12 was tagged a small cost penalty on 11 problems** (0.008–0.030
  each) because of the 5th candidate (haiku) plus extra LLM calls (style
  retry, alt-approach regen). These are tiebreakers — small, not decisive.

The key takeaway: the alt-approach feature is broken (adopts unverified
alternatives based on fixture noise). The style-hint loop-free preference is
clearly correct (won 269 cleanly). The 5th candidate, pandas-tuple hint, and
TF Keras 3 hint were unproven on this batch (no problem in the batch
exercised them).

## Strategy

Build on **iter9_mpl_aware_judge** (the cheaper of the two top scorers) and
add ONLY the iter12 changes that demonstrably converted a problem:

1. **Style-hint detection + loop-free-aware judge selection**: detect
   "idiomatic", "efficient", "vectorized", "without a for loop", "cleanest",
   "elegant", "one-liner" etc. in the prompt; tokenize each candidate and
   tag `loop_free`; in the judge prompt and selection logic, prefer
   loop-free candidates over loop-using ones when style hint is present.
   This is the iter12 change that won problem 269.

2. **Style retry**: if the chosen candidate uses `for`/`while` and the
   prompt has a style hint, do ONE rewrite with sonnet@0 asking for a
   loop-free version. Adopt only when:
   - the rewrite has no `for`/`while` tokens, AND
   - its smoke output matches the original chosen candidate's smoke REPR
     (so we don't regress correctness).

3. **Code-consensus short-circuit when smoke fails with NoneType**: if all
   candidates produced equivalent code AND all smokes failed because
   `load_data()` returned None (fixture noise), pick the leading candidate
   directly. This is the small iter11 feature that turns fixture noise into
   a benign signal. It is the safe inverse of the broken alt-approach
   feature: instead of replacing potentially-correct code with a random
   alternative when smoke is uninformative, we just trust the unanimous code.

4. **DROP the alt-approach regeneration feature entirely**. This is the
   iter12 feature that lost 906 and 860. Generating an alternative when
   smoke is uninformative and adopting it without verification is strictly
   worse than trusting unanimous candidates.

5. **DROP the 5th candidate (haiku)**. iter9's 4 candidates (sonnet, gpt54,
   mini, gemini) were enough for 95% on this batch. The 5th adds cost and
   no demonstrated correctness.

6. **DROP the unproven TF Keras 3 and pandas-tuple system-prompt
   additions**. They were targeting iter11-batch problems (706, 165) that
   didn't appear in iter12's batch. They might help on unseen problems but
   they grow the system prompt (more input tokens × 4 candidates × every
   problem). Keep the system prompt focused on iter9's checked-in rules.

## Why I expect this to outperform iter8/iter9

iter9 missed problem 269 (style assertion). iter12's style-aware addition
fixes that pattern. Without iter12's broken alt-approach feature, the iter8
and iter9 strengths on 906/860 are preserved. So I expect:

- Match iter8/iter9 on every problem they solved correctly (no
  alt-approach to mess things up).
- Pick up problems where the prompt has a style hint and the natural
  candidate uses a list-comp / for-loop — like 269 in iter12, but the
  pattern is general (any "idiomatic", "efficient", "vectorized" prompt
  with a hidden style assertion).
- Cost stays in the iter9 range (~$0.02/problem), well under the $0.04
  free-zone threshold per problem so cost penalty stays at 0.

The bet: the loop-free style preference is a generally-applicable
correctness gain, while alt-approach is fragile. We keep the gain, drop
the loss.
