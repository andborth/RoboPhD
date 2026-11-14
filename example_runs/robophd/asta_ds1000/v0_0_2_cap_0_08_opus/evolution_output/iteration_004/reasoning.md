Name: iter4_ds1000_idiom_probe

# Analysis & Strategy — Iteration 4

## What the data shows

Iteration 3 scores: iter3_ds1000_format_aware **80%** (best) > iter2_verify_repair 75% > seed 65%.
iter3 is the clear base: CLAUDE_SONNET_4_6 reasoning=high + deterministic
re-indent + sandbox verify-repair for runnable column-0 problems. Its only
liability over the 75% agent is none — it strictly dominates. I build on it.

### Consensus failures (all 3 agents): 706, 165, 269, 420

I read every diagnostic. The failures fall into distinct, **generalizable**
buckets — not random model misses:

| # | Lib | Root cause | Caught by sandbox? |
|---|-----|-----------|--------------------|
| 269 | Pandas | **Style test**: `assert "for" not in tokens`. Answer used a list comprehension (`for` token). Code *ran fine* (`sandbox OK on attempt 0`) but the hidden `test_string` rejects the token. | No — sandbox only catches tracebacks |
| 420 | Numpy | **Invent-signature function**: skeleton sets `x, x_min, x_max`; problem says "define function named `smoothclamp`". Model wrote `smoothclamp(x, mi, mx)`; hidden test calls `smoothclamp(x)` → `TypeError: missing 2 args`. Reference uses skeleton vars as globals. Function defined-but-never-called, so `sandbox OK on attempt 0`. | No — function never invoked in the solution |
| 706 | TF | Deprecated `save_format='tf'` in Keras 3 + protobuf env noise; skeleton comment literally says `#Save the model in "export/1"` (reference: `tf.saved_model.save(model,"export/1")`). Repairs ran but didn't converge. | Partially — needs a stronger model / hint use |
| 165 | Pandas | Subtle dtype trap: `np.column_stack` upcasts int counts to string; expected output is object dtype. Naive `pd.DataFrame({...})` keeps ints. | No — wrong-value, not a traceback |

The recurring theme: **iter3's sandbox verify only catches tracebacks**, so it
is blind to (a) style-token failures and (b) defined-but-uncalled functions —
two of the four consensus misses. These are systematic DS-1000 failure modes
(style/idiom tests and "define a function" problems are common), so fixing
them generalizes well to unseen problems.

## Approach: iter3 + three targeted, generalizable upgrades

I keep iter3's proven pipeline verbatim (model, re-indent, traceback
verify-repair) and add three guarded layers. Cost headroom is huge (iter3 mean
$0.0096 vs $0.08 free-zone), so extra verification rounds are free.

1. **Idiom / style-constraint enforcement (fixes 269-class).**
   Detect idiom language ("idiomatic", "without a loop", "vectorized",
   "efficient/clean way", "not one by one", "one-liner", "no loop"). When
   present, (a) strengthen the prompt to forbid `for`/`while` *anywhere,
   including comprehensions/generators*, and (b) after generation, AST-scan the
   candidate for `For`/`While`/comprehension nodes; if found, regenerate (≤2×)
   with an explicit "the grader tokenizes and rejects any `for`/`while` token —
   use pure vectorized pandas/numpy (`.map` with a format string, broadcasting,
   `np.where`) with NO comprehensions" instruction. A loop-free vectorized
   answer is correct whether or not a string test exists, so this never hurts.

2. **Invent-signature function probe (fixes 420-class).**
   Detect "define a function named X" / "def X" *completion* problems where the
   skeleton does NOT supply the `def` line. Add prompt guidance: the hidden
   test calls the function with only the principal example input; treat other
   skeleton-set values (bounds, n, config) as module globals — keep the
   signature minimal. Then **actively probe**: in the sandbox, after defining,
   call `X(<first non-config example variable>)`; an arity `TypeError` is fed
   into the repair loop (sandbox now exercises the function the way the grader
   will, instead of leaving it uncalled).

3. **Smarter repair triage + strong-model escalation (helps 706-class).**
   Only treat output as pure env noise when there is *no* traceback frame in
   the executed program (`<string>`); an actionable in-program error (e.g. the
   Keras `save_format` deprecation) is fed to repair even amid protobuf noise.
   Bump repairs to 3. If a runnable problem still has an actionable error after
   repairs exhaust, make one final attempt with **CLAUDE_OPUS_4_7** (stronger
   trap/hint reasoning) — a cheap, targeted use of the strongest handle on the
   genuinely-hard runnable cases only.

I also add concise gotcha lines (idiomatic-construction dtype, e.g.
`np.column_stack` upcasting; following literal skeleton comment hints like a
save path) to nudge 165/706-class cases without overfitting.

## Why this beats iter3

Every change is additive and guarded behind a detector, so the 16/20 iter3
already solves are untouched (no regression risk). The new layers convert two
of the four *consensus* failures from structural blind spots into handled
cases (style-token scan → 269-class; function call-probe → 420-class) and give
706-class hard runnable problems a stronger second model. These are systematic
DS-1000 categories, so the expected lift carries to the unseen held-out set,
not just this batch.
