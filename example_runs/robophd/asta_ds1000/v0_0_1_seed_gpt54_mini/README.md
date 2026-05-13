# asta_ds1000 / v0_0_1_seed_gpt54_mini

One-shot GPT-5.4-mini baseline, submitted alongside `v0_0_1_soft_cap_0_16` at the same SHA. The cheapest of our submitted points.

## Recorded score

| | Value |
|---|---|
| Score (RoboPhD-internal eval) | *(filled in after `astabench eval`)* |
| Solver model | `openai/gpt-5.4-mini` |
| Agent size | 46 lines |

## Submission metadata

| Field | Value |
|---|---|
| Agent name | GPT-5.4-mini one-shot DS-1000 baseline (v0.1) |
| Openness | Open source, closed weights |
| Tools tier | Standard |
| Leaderboard URL | *(filled in after submission)* |

## Approach

46-line one-shot solver. Sends the problem prompt to GPT-5.4-mini, wraps the response in `<code>...</code>` tags, and returns. No self-check, no `python_session` calls, no library-specific scaffolding, no per-problem detectors. See `agent.py`.

The agent.py here is **byte-identical** to `examples/asta_ds1000/seeds/baseline/agent.py` at the SHA recorded with the submission — the canonical seed at HEAD.

## Resilience wrapper

The submitted `agent.py` (in the leaderboard tarball) is **not** the literal `agent.py` from this directory. It's a small auto-generated wrapper that imports `make_solver` from `agent_inner.py` (a renamed copy of this file) and from a bundled `seed_agent.py`, and provides a **two-tier** safety net:

```python
try:
    return await inner(state, generate)        # the primary (this seed)
except Exception as primary:
    try:
        return await seed(state, generate)     # retry with bundled seed
    except Exception as fallback:
        state.output.completion = ""           # scorer marks "I" → 0
        return state
```

For this seed-as-primary submission the "fallback tier" is a same-agent retry — the bundled `seed_agent.py` is a copy of the same file. It still provides a safety net against transient provider failures (e.g., a one-off 5xx from OpenAI). The wrapper recipe lives in [`scripts/asta_ds1000_submit.py`](../../../../scripts/asta_ds1000_submit.py) (`WRAPPER_TEMPLATE`); template + bundling were introduced in commit `0370bad`.

## Reproducing the submission

```
cd /path/to/repo
python scripts/asta_ds1000_submit.py
```

The script copies this `agent.py` into a working dir, wraps it in the resilience template, runs `astabench eval --solver agent.py --model openai/gpt-5.4-mini --split test --task DS_1000_test`, scores, and tarballs for upload.
