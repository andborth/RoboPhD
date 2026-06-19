Name: iter10_literal_consensus

# Analysis & Plan — Iteration 10

## What the data shows

Three agents were evaluated in iteration 9 and **all three tied at 80% (16/20)**:

| Agent | Score | Mean cost | First shot | Extra levers |
|---|---|---|---|---|
| iter3_fmt_strong_cascade | 80.0 | $0.0042 | GPT_5_4 low | self-check + repair |
| iter7_agree_escalate | 80.0 | $0.0104 | GPT_5_4 low | cross-model value agreement + escalate |
| iter9_reason_agree | 80.0 | $0.0221 | GPT_5_4 **medium** | both levers, high-reasoning tiebreak |

Two facts drive my plan:

1. **The per-agent differences are noise.** The three "split" problems (269, 706,
   812) were each solved by exactly one agent, and the agents differ only in
   reasoning effort and when they escalate — i.e. these splits are model
   stochasticity, not a reproducible capability gap. Spending 5× more (iter9 vs
   iter3) bought *zero* accuracy on this batch. So pouring more money/voters into
   the architecture is not where the next point comes from.

2. **There is exactly one systematic, addressable failure pattern.** The two
   *consensus* misses (445, 883) — wrong for **all three** agents — share a single
   root cause: every agent produced the *cleverer / more statistically faithful*
   answer, while the DS-1000 reference uses the **direct literal transform**.
   - **445** ("highest-to-lowest, the reverse of rankdata"): agents wrote
     `rankdata(-a, method='min')`; reference is `len(a) - rankdata(a).astype(int)`.
     The two agree everywhere except the unique-max element, where the clever
     negation is off by one → wrong.
   - **883** ("data is a 2-D distance matrix, cluster it"): agents condensed the
     matrix (`squareform`) and used `average` linkage — statistically the *correct*
     reading. The reference naively feeds the raw matrix straight into
     `linkage(..., 'ward')` + `cut_tree`. The faithful answer fails the test.

   Crucially, **voting and self-checking cannot fix this class**: there is no
   oracle in the sandbox, and both independent strong models make the *same*
   clever substitution, so they agree on the wrong answer. Only a prompt-level
   prior toward the literal reading can move these.

## Cost head-room

iter9's $0.022 mean is 2.3× under the $0.05 free zone (86% of its spend is the
medium-reasoning GPT_5_4 first shot). Cost is therefore not the binding
constraint — accuracy is the entire objective — but there is no reason to inflate
it either, so I keep iter9's proven spend profile unchanged.

## Approach: keep the proven machine, add a literal-reference prior

I start from **iter9_reason_agree** because it is strictly the most complete prior
agent — it has the fullest base instructions (dtype/shape matching, construct-
undefined-objects, function-arity rules), format-aware module/function handling, a
numpy/pandas/torch-aware value serializer, free in-sandbox self-check, cross-model
repair on crashes, and value-agreement with a high-reasoning escalation tiebreak.
Re-deriving any of that risks regressions for no gain.

My change is **a single targeted lever aimed only at the consensus-miss class**,
applied in two places so it can't disturb the 80% that already passes:

1. **A new base-instruction rule** teaching the model that DS-1000 references favor
   the *most direct, literal* reading and the simplest call that reproduces the
   shown example — with the two concrete traps spelled out generically:
   - "reverse / opposite / descending of F" → use a direct arithmetic transform of
     F's normal output (`len(a) - rankdata(a)`, `max(x) - x`), **not** negated
     inputs into F (`rankdata(-a)`), which diverges at ties/boundaries;
   - feed data in the *same literal form* the problem presents (pass the given
     matrix straight in), and don't add preprocessing the problem never mentioned
     (squareform/condensing/reshaping/extra normalization).
   - when a plain one-liner and a smarter multi-step version both match the
     example, choose the one-liner.

2. **The disagreement tiebreaker prompt** gets the same lesson at the exact moment
   two strong candidates split — so when one attempt is the literal one-liner and
   the other is the clever equivalent, the high-reasoning judge is told to break
   the tie toward the literal answer.

Nothing else changes: same models, same reasoning efforts, same escalation graph,
same serializer, same guards (the whole agreement/repair layer is wrapped so it can
only convert a 0 into a possible 1, never the reverse).

## Why this should generalize (and not regress)

- The literal-reference rule is **not** in tension with the existing "prefer the
  canonical library idiom" rule — it refines it: *among* valid library approaches,
  pick the one that most directly mirrors the plain-language description and the
  shown example. It discourages over-engineering, which is a recurring DS-1000
  reference style (445, 883, and likely the 269/812-style "simplest expression"
  references too), not a one-batch quirk.
- It is additive and conservative: it only changes behavior on problems where the
  model was about to substitute a "smarter" equivalent — exactly the cases that
  silently score 0 today. Problems with a single obvious answer are unaffected.
- I deliberately did **not** add more voters or a third model family. iter5/iter6
  showed that folding in extra (weak) voters *lost* accuracy, and on this batch the
  extra spend in iter7→iter9 bought nothing. The marginal, low-regression-risk move
  is the prompt prior, not more ensembling.

Expected effect: retain the 80% baseline, recover a meaningful share of the
literal-vs-clever consensus-miss class on unseen batches, at iter9's already-free
cost.
