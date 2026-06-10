Name: iter14_filemock_adjudicate

# Iteration 14: data-file mocking + prose-vs-context rule on top of iter13

## What the iteration-13 data shows

| Agent | Score | Failures | Mean cost |
|---|---|---|---|
| iter13_fnsig_adjudicate | 100.0 (20/20) | — | $0.0480 |
| iter8_refquirk_adjudicate | 100.0 (20/20) | — | $0.0378 |
| iter12_thirdvote_adjudicate | 95.0 (19/20) | 919 | $0.0523 |

iter13 (the newest lineage member, carrying every accumulated defense through the
function-signature fix) went 20/20 with comfortable cost headroom. The only
failure in the batch — iter12 on **919** — is the most informative artifact, and
digging into it revealed a *systemic evidence gap* that has been silently
degrading the pipeline on a whole problem class:

### Anatomy of the 919 failure

919 is a "corrected, runnable code" sklearn problem. The prose shows the asker's
broken script (which contains `logReg = LogisticRegression()`); the runnable
`<code>` context keeps only the read/clean part:

```python
dataframe = pd.read_csv(filename, dtype='category')   # animalData.csv
...
dataframe.replace(cleanup, inplace=True)
```

iter12 submitted code that called `logReg.fit(...)` **without defining
`logReg`** — assuming the object from the prose snippet exists. The hidden test
program is only context + solution, so it died with
`NameError: name 'logReg' is not defined`.

Why did three layers of defense miss it?

1. **Evidence gap**: the sandbox cross-check executes context + candidate, but
   `animalData.csv` doesn't exist in the sandbox — the context itself raised
   FileNotFoundError for *every* candidate. iter12's stdout confirms it:
   `ok_a=False ok_b=False`, third opinion also errored, the final-exec check
   also errored. All execution evidence was inconclusive noise, so the
   adjudicator had to judge on code semantics alone and guessed wrong. The
   existing mock stage only triggers on the `load_data()` pattern, not on file
   reads.
2. **Knowledge gap**: no rule told generators/critique/adjudicator that the
   hidden test program contains ONLY the runnable `<code>` context — objects
   shown solely in the asker's prose snippet must be created by the solution.
   (iter13 passed 919 with a lucky `try: logReg / except NameError:` defensive
   pattern from its generator; iter8 passed by simply defining `logReg` fresh.
   Neither was systematic.)

The other historical failures I audited (440 → direct-formula rule 16; 10 →
sloppy-display handling; 420 → iter13's signature defense; 269/883 → loop-token
and cut_tree rules) are all already covered by accumulated defenses.

## Approach: iter13 verbatim + a three-layer file-read/prose-object defense

Mirroring how the loop-token, cut_tree, and fnsig fixes were engineered, I keep
iter13 bit-for-bit and add the missing defense at all three layers:

1. **File-mock stage (the key fix, unmetered except one cheap GPT_5_4 call)**:
   when the extracted context contains a data-file read
   (`read_csv/read_table/read_excel/read_json/read_pickle/read_parquet/read_fwf/
   loadtxt/genfromtxt` — matched on the call name, since the filename is often a
   variable, as in 919), a cheap call synthesizes Python that *creates the
   file(s)* before the harness runs. DS-1000 prompts usually display the file
   contents verbatim (919 shows the entire CSV); the mock prompt instructs exact
   reproduction and flags it with a first-line `# VERBATIM` marker. With the
   file present, the cross-check/third-vote/final-exec machinery runs for real:
   a candidate that references `logReg` raises the *grader's exact NameError*
   inside the harness, breaking false consensus and arming repair.
   - Verbatim mocks: executed values are treated as real evidence (the
     expectation check and value-mismatch repair stay enabled).
   - Non-verbatim mocks: value-level checks are skipped (like `load_data`
     mocks), but error evidence still flows, with an explicit caveat to the
     adjudicator/repairer that mock-data artifacts are possible while a
     NameError on a never-defined object is real.
2. **Generation rule 25** (seen by both generators, the adjudicator, and
   repair): the hidden test program is ONLY the runnable `<code>` context plus
   your code; objects appearing only in the asker's prose script (`logReg`,
   fit calls, splits) must be created by the solution.
3. **Critique checklist item 17 + adjudication note**: flags any reference to a
   name the runnable context never defines, so even dual-model consensus on the
   wrong assumption can't sail through the fast path, and a NameError in
   execution evidence is explicitly tied to rule 25 for the adjudicator.

### Verification (local, deterministic pieces)

- `agent.py` parses; on the real 919 prompt the context extractor + trigger
  route it to the file-mock path; with a verbatim CSV mock prepended, executing
  iter12's actual submitted code reproduces the grader's exact
  `NameError: name 'logReg' is not defined` while the reference-style candidate
  runs clean and assigns `predict` (see `test_919_filemock.py`).
- Negative tests on every historical problem containing a file-read string:
  **284** (read_csv only in prose; context builds the DataFrame inline) →
  no trigger, pipeline unchanged; **861/910** (`load_data()` prompts) → routed
  to the existing data-mock path by the `elif` ordering, unchanged. So the new
  path activates *only* on the 919 class.

## Why this should score higher on held-out problems

- iter13 is the strongest base (100%, and the only lineage member carrying the
  fnsig defense that 420-class problems require); everything it does is
  preserved exactly, so there is no regression risk on classes it already
  handles — the new trigger is inert unless the runnable context reads a file.
- "Corrected, runnable code" problems with CSV reads recur in DS-1000's sklearn
  and pandas sections, and the 919 failure mode was structural: *no* execution
  evidence on this whole class, for any agent, in any prior iteration. Turning
  the class from evidence-blind adjudication into normal verified execution
  upgrades every downstream defense at once (cross-check, third vote,
  expectation check, final repair).
- Cost: one extra GPT_5_4 mock call (~$0.001) on the rare file-read problem;
  everything else is unmetered sandbox work. iter13 ran at $0.0480 against the
  $0.06 free-zone threshold, so headroom comfortably absorbs it. Expected mean
  cost stays ≈ $0.05.
