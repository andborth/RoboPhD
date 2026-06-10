# Evolution Reflection — Iteration 14 (iter14_filemock_adjudicate)

## What worked well

1. **Forensic depth on a single failure beats broad re-analysis.** By iteration
   13 the lineage was at 100%, so the only signal was iter12's one failure
   (919). Reading the actual artifacts — `extracted_code.md`, `test_result.md`
   (the grader's traceback), `agent_stdout` (the pipeline's own log lines) —
   revealed not a one-off mistake but a *structural* gap: contexts that read
   data files could never execute in the sandbox, so the entire verification
   machinery was blind for that whole problem class. The per-problem
   `agent_stdout` was the decisive artifact: `ok_a=False ok_b=False` showed the
   pipeline knew its evidence was inconclusive. Future instances: always read
   the stdout of the *failing* agent, not just the diff between submissions.

2. **The lineage's "three-layer fix" pattern is the right template.** Every
   durable improvement in this run (loop-token guard, cut_tree convention,
   fnsig defense, now file mocking) attacked one failure mode at three layers
   simultaneously: a generation rule, a critique/adjudication checklist item,
   and a *deterministic* harness/evidence upgrade. The deterministic layer is
   the most valuable — rules can be ignored by models; synthesized execution
   evidence (e.g. appending `result = f(x)`, creating the missing CSV) makes
   the grader's failure mode reproduce in the sandbox before submission.

3. **Conservative layering with explicit blast-radius testing.** Copying the
   best agent verbatim and gating new behavior behind a narrow trigger
   (regex on the *extracted context*, not the whole prompt) made regression
   risk auditable. I grepped every historical problem for the trigger pattern
   and ran the actual routing function on each hit — 919 routes to the new
   path, 284 (read_csv only in prose) doesn't, 861/910 keep their old
   `load_data` path. Ten minutes of negative testing buys real confidence.

4. **Local simulation of the harness.** The pure-Python parts of agent.py
   (context extraction, target detection, trigger regexes, exec semantics) can
   be unit-tested locally by `ast`-extracting the functions from agent.py
   without importing inspect_ai. I reproduced the grader's exact NameError on
   iter12's real submitted code with a mocked CSV — the strongest possible
   pre-deployment evidence that the fix works.

## What was challenging / time-consuming

- **Local environment drift.** The local pandas is stricter than the eval
  sandbox's pinned version (categorical `replace` raises locally, only warns
  there), and sklearn isn't installed locally. I had to stub sklearn and
  neutralize the dtype quirk to validate the mechanism. Knowing the sandbox's
  pinned versions (or having a requirements list in CLAUDE.md) would have
  saved a detour.
- **Path confusion at the start.** The "experiment directory structure" uses
  paths relative to the working dir (`../../agents/`), but my first instinct
  was `evolution_output/agents/`. One failed `ls` fixed it; an absolute-path
  example in the prompt would prevent it.
- **A 1500-line base agent.** Reading iter13 in full is necessary before
  editing (the mock flags thread through four stages), but it exceeds one Read
  call. The thorough docstring at the top of the lineage agents — summarizing
  the pipeline stage by stage and what each iteration added — was enormously
  helpful. Future instances: maintain that docstring discipline; it is the
  cheapest knowledge transfer mechanism in this whole process.

## On the tools and data

- `error_index.json` + per-problem dirs were sufficient and well-structured.
  `jq` over `by_agent.failed_ids` across all iterations gives the full failure
  history in one command — do this first; it instantly shows which failure
  modes are already fixed (by which iteration's rule) and which were never
  addressed.
- The cost report mattered less this round (lineage is comfortably in the free
  zone at ~$0.05 vs $0.06), but it confirms where spend concentrates (GPT_5_5
  adjudication ≈ 55–60%) — that's the lever if cost ever becomes binding.
- No `strategy_tools/` existed this round; not missed.

## What I'd do differently / advice to future instances

1. **When the visible batch is saturated (100%), mine *history*, not the
   current batch.** The current batch gives zero gradient. The gradient lives
   in: (a) the one cross-agent split decision, (b) historical failures whose
   root cause was patched *symptomatically* (a rule) rather than
   *structurally* (evidence), and (c) classes where the pipeline's own logs
   show it operating blind. Category (c) is gold: search agent_stdout files
   for `ok_a=False ok_b=False`, `verdict=no`, `(not executed)` — every such
   line is a problem the pipeline solved by luck.
2. **Prefer evidence upgrades over rule additions.** The RULES block is at 25
   items; each new rule dilutes the others and risks prompt-bloat
   misgeneralization. Deterministic harness improvements (file mocks, call
   synthesis, token checks) don't dilute anything and can't be ignored.
3. **Don't rewrite; graft.** Every winning iteration here was the previous
   winner + one narrowly-gated addition. The two times the batch contained an
   older agent (iter8) it also hit 100% — but it lacks five generations of
   defenses that the held-out set will eventually probe. Score parity on one
   20-problem batch is weak evidence of equivalence.
4. **Verify the failure reproduces before fixing it.** I ran iter12's literal
   submitted code under the new harness path and watched the grader's exact
   error appear. If you can't reproduce the failure mechanism locally or in
   the sandbox, your fix is speculative.

## Suggestions for the process/prompts

- **Include sandbox package versions** (or a pip-freeze excerpt) in CLAUDE.md;
  version drift between local python and the eval container cost time and
  could mislead validation.
- **Surface pipeline-blindness metrics in the error analysis report**: for
  each agent×problem, whether sandbox execution succeeded. A column
  "exec-evidence available: yes/no" would have exposed the 919 class many
  iterations earlier (284 was probably solved blind for the same reason).
- **Keep the clone-detection note** — it correctly flagged that iter13 and
  iter8 were indistinguishable *on this batch*, which is exactly the right
  caution about saturated batches.
- 20-problem batches saturate quickly with agents this strong. If feasible,
  larger or harder-skewed batches (oversample libraries/classes with
  historical failures: sklearn corrected-code, torch/TF, matplotlib styling)
  would restore gradient.
