Name: iter11_ds1000_tridtype_judge

# Analysis & Strategy — Iteration 11

## What the data shows

All three agents in iteration 10 scored 100% (20/20), so the iter10 batch gives no
fresh failure signal. The useful signal is the longer history across iters 4–10,
which separates the agents that *generalize* from the ones that got a lucky batch:

| Agent | i5 | i6 | i7 | i8 | i9 | i10 |
|---|---|---|---|---|---|---|
| iter3_ensemble_judge | 100 | 90 | 90 | 85 | 80 | 100 |
| iter8_strongjudge | — | — | — | 100 | 95 | 100 |
| iter9_execverify_judge | — | — | — | — | 95 | — |
| iter10_dtypeverify | — | — | — | — | — | 100 |

Key facts extracted from the error reports:

- **iter3** (the 2-candidate, medium-reasoning judge, value-only diagnostics) is the
  weakest generalizer — it drifts to 80–90% on harder batches. The **stronger judge
  (reasoning="high"), the always-judge-on-disagreement discipline, dtype-rich
  diagnostics, and the verify+repair loop** are what lifted iter8→iter10 above it.
- The one **all-agent consensus failure** in iter9 was **problem 165**: a
  dtype-coercion case where the reference builds the frame with `np.column_stack`,
  silently coercing an int column to string. Value-only diagnostics made the two
  candidate frames print identically, so a value-only consensus short-circuit emitted
  a wrong answer. `iter10_dtypeverify` fixed exactly this by printing per-column
  `dtypes` + `repr` and comparing those in the consensus check — and it took the
  iter10 batch to 100%.
- The **iter4 lesson** (recorded in iter7's header): emitting a runtime-value
  *majority* without a judge does **not** beat always-judging. So consensus must be a
  conservative shortcut, never a substitute for the judge.
- **iter7_triverified_judge** validated a separate, orthogonal lever: a **third
  candidate from a different model family (GEMINI_3_1_PRO_PREVIEW)**. Cross-family
  diversity raises the odds that at least one candidate is already correct — the exact
  material the judge needs to pick a winner rather than repair from scratch.

## Approach: synthesize the two strongest, separately-validated levers

`iter10_dtypeverify` is the best current design, but it has only ONE candidate-family
pair (GPT_5_4 + CLAUDE_SONNET_4_6). `iter7` showed a third family helps. These levers
are independent and additive, and cost leaves ample room: dtypeverify's mean spend was
**$0.0283** against the **$0.08** free-zone threshold, so one extra low-reasoning
candidate generation (Gemini's cheapest path) keeps me comfortably free-zone — no
score penalty, pure accuracy play.

**iter11_ds1000_tridtype_judge** keeps the entire proven iter10 core verbatim and adds
the third family:

1. **THREE diverse candidates** — GPT_5_4 (low), CLAUDE_SONNET_4_6 (low),
   GEMINI_3_1_PRO_PREVIEW (default `low`, override omitted for the cheapest path).
2. **Dtype-rich execution** of every candidate whenever the setup runs (matplotlib
   included): per-column `DataFrame.dtypes`, `Series` dtype+name, ndarray `.dtype`,
   and `repr(result)` — unchanged from iter10.
3. **Unanimity-only shortcut** (the iter4 discipline, generalized to 3 candidates):
   short-circuit *only* when **all** cleanly-run candidates (≥2 of them) agree on the
   dtype-rich summary. Any disagreement — including dtype-only disagreement — falls
   through to the judge. This is strictly safer than a bare majority vote.
4. **Strong output-grounded judge** (GPT_5_4, reasoning="high") now sees **three**
   execution-grounded candidates plus their per-column dtypes — a strictly larger
   information set than iter10's two, with the same dtype-coercion guidance.
5. **Verify + up to two traceback-informed repairs**, then graceful fallback to any
   clean-running candidate — unchanged from iter10.

## Why this should generalize better

- It is a **superset** of the iter10 leader: every path iter10 took is preserved, and
  the only change is *more diverse candidate material*. The third family can only add
  correct options for the judge to recognize; the unanimity gate prevents it from
  loosening the consensus shortcut (it actually makes the shortcut stricter, since one
  more candidate must agree).
- Cross-family diversity directly attacks the dominant failure mode of these agents:
  "runs cleanly but is subtly wrong." When GPT and Claude share a blind spot (same
  wrong idiom), Gemini frequently doesn't — giving the judge a correct reference to
  prefer instead of having to invent one.
- It retains the two things that empirically separated the strong generalizers from
  iter3: the **high-reasoning judge** and **dtype-rich diagnostics**.
- Cost stays deep in the free zone, so the diversity is "free" in score terms.

## Risks & mitigations

- *Cost creep* from a third candidate + occasional extra judge fall-through: bounded —
  Gemini's cheapest path plus iter10's already-modest $0.028 mean keeps the projected
  mean well under $0.08.
- *A weak/empty Gemini candidate*: handled by the existing per-candidate `_gen`
  try/except (returns ""), the run guard (only non-empty candidates execute), and the
  fallback chain. A missing third candidate degrades gracefully to the iter10 behavior.
