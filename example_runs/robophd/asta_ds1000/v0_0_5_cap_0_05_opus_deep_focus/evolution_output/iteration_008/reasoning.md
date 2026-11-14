Name: iter8_perspective_consensus_ds1000

# Analysis

## What the data says

Across iterations the ranking is stable and the lesson is consistent:

- **iter3_safe_repair (95%, $0.002)** is the champion: simplicity-first prompt, normalize
  func-body indentation, execute the candidate, and **repair ONLY on a real traceback —
  never second-guess a clean run.**
- **iter6_grounded_repair (85%)** and **iter7_grounded_reconcile (90%)** both added
  LLM-driven *reconciliation* of clean runs against the prompt's expected output. Both
  scored **lower** than iter3 and cost 2–3× more. Inspecting their split-decision losses:
  - **667** (tensorflow): the right answer is `x.assign(114514); result = x.numpy().item()`.
    iter7 emitted `result = 114514`, dropping the side-effect on `x`. A reconcile/arbitration
    step rewrote a correct answer into a wrong one.
  - **723** (scipy sparse): reference is the literal `result = sA.multiply(sB)`. iter7
    over-engineered into `sA.multiply(sB.toarray().ravel()).tocsr()` and failed.
  iter3 got both right by staying literal and never touching a clean run.
- iter3's **only** miss was **129**: a quirky pandas problem whose reference promotes the
  whole frame to float via `df.update`. iter7 happened to fix it by casting to float first.
  This is a *clean-but-wrong* failure (runs fine, wrong dtype) — exactly the bucket iter3's
  error-only repair cannot see.

**Meta-lesson:** free LLM arbitration over a clean answer is net-negative here — it breaks
more correct answers than it fixes. The proven-safe core is iter3. With only 20 problems
per batch, differences of 1–2 problems are noise, so I must avoid overfitting and instead
add only changes with positive *expected* value and **no regression path** below iter3.

## Approach: multi-perspective consensus on an objective signal

I keep iter3's entire proven scaffold byte-for-byte (simplicity guide, indentation
normalization, func/var probe, error-only repair, fall-back-to-best-clean) and replace the
single generation with a small **prompt-perspective ensemble decided by objective
execution agreement** — no model arbitration anywhere.

1. Generate **3 candidates** from the same cheap model (GPT_5_4_MINI, low reasoning) under
   3 framings: two simplicity-biased (idiom / general-robust) and one
   exact-output-form-biased (match the dtype/shape/column-order/quirks shown in the
   prompt). Using distinct prompts guarantees diversity even if temperature is ignored,
   and the output-form variant directly targets the 129-class dtype miss.
2. Execute all three in the free sandbox and group them by their **exact probe output**
   (an objective signal, not an LLM opinion). If ≥2 candidates agree on an output, ship a
   candidate producing it — preferring the simplicity variants and the shortest code, so
   ties resolve toward the literal answer the reference favors. A single over-engineered
   outlier is automatically out-voted.
3. **No majority** (all three clean outputs differ, or <2 ran clean) ⇒ the problem is
   genuinely uncertain, so escalate **once** to the stronger GPT_5_4 (low reasoning). Ship
   its answer only if it runs clean; otherwise fall back to the first clean candidate.
4. **All three error** ⇒ iter3's error-only repair on the first candidate.
5. Matplotlib / no probeable target / no sandbox ⇒ ship the first candidate (iter3 behavior).

### Why this can't regress below iter3 and has real upside
- Every shipped answer is a clean run (or iter3's repaired/fallback answer) — we never ship
  an errored candidate.
- We never let an LLM rewrite a clean consensus answer (the exact thing that sank
  iter6/iter7). The strong model only fires when there is **no consensus to protect**.
- Consensus across independent framings reduces variance on the uncertain middle of the
  distribution, and the output-form perspective recovers some clean-but-wrong dtype/shape
  misses that iter3 structurally cannot.
- Cost: ~3 cheap calls on easy problems (~$0.006–0.008) plus a rare strong call; mean stays
  far inside the $0.05 free zone (iter3 spent $0.002, budget headroom is ~25×).

This is a conservative, generalization-oriented bet: identical floor to the proven
champion, with an objective consensus layer that only adds correctness.
