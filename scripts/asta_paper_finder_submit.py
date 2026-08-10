#!/usr/bin/env python3
"""
PaperFindingBench → AstaBench leaderboard submission runner.

Run from anywhere — paths are resolved relative to the repo root:

    python scripts/asta_paper_finder_submit.py [--only NAME ...] [--limit N]

Per submission, the script:
  1. Verifies the working tree is clean (otherwise eval_spec.revision.dirty
     would record `true`, weakening the reproducibility claim).
  2. Verifies provider keys, ASTA_TOOL_KEY / HF_ACCESS_TOKEN, and that the
     installed litellm's BUNDLED price map covers every model the staged
     agent calls (the DS-1000 v0_0_3 lesson: an unpriced model makes
     `astabench score` emit cost=null and the entry ships costless).
  3. Stages a working dir at submissions/asta_paper_finder/<name>/
     (gitignored) with agent.py (resilience wrapper), agent_inner.py (the
     evolved agent from example_runs/), seed_agent.py (fallback tier), and
     model_registry.py.
  4. Runs `astabench eval` against paper_finder_test (no Docker — this
     task has no sandbox tier).
  5. Runs `astabench score` to produce scores.json + summary_stats.json,
     refusing to continue on cost=null.
  6. Tarballs logs/full_test/ for HuggingFace upload (full runs only —
     a --limit smoke run logs to logs/smoke_limit_<N>/ and is never
     tarred, so a partial run can't masquerade as a submission or trip
     the full run's idempotency skip).

The script does NOT submit. After a full run, one .tar.gz per selected
submission is ready for manual upload via the HF Spaces leaderboard form.

Cost / time (full test split, 267 queries). Judge spend dominates — it
was 98.7% of the v0_0_8 bill and 93.4% of v0_0_9's — and it scales with
how much the agent ships, not with the agent's own price. Three measured
points:

    v0_0_7_soft_cap_0_06_fable    $192 judge + $15 agent    ~19 hr
        976 chars/paper evidence, 250 papers/semantic query,
        ~12 min/semantic query at --max-samples 4.   $0.0040/paper
    v0_0_8_soft_cap_0_033_opus    $117 judge + $1.59 agent  1h32m
        750 chars/paper, 203.5 papers/query, --max-samples 6.
                                                     $0.0030/paper
    v0_0_9_cap_0_063_opus5        $197 judge + $14 agent    7h18m
        747 chars/paper, 250 papers/query, --max-samples 6.
                                                     $0.00407/paper
    v0_0_9_cap_0_355_opus5        $206 judge + $67 agent    6h58m
        765 chars/paper, 250 papers/query, --max-samples 6.
                                                     $0.00426/paper

So a cheap agent is not merely cheap to run — shorter submission lists
cut the judge bill too, which is the term that actually matters. Wall
clock is unscored officially, so duration buys evidence quality, not
points.

But note what v0_0_9 falsified: it shipped 747 chars/paper, within 3
chars of v0_0_8's 750, and still billed at v0_0_7's $0.0040 rate rather
than v0_0_8's $0.0030. Evidence LENGTH alone does not predict the
per-paper rate — list length does (250 vs 203.5 papers/query is the one
thing v0_0_9 shares with v0_0_7 and not v0_0_8), and something in the
evidence beyond its character count moves the judge's own output tokens.
Project from the $0.0040 ceiling below, not from a length comparison; a
pre-run estimate of ~$145 built that way came in $52 under.

Official judging is UNCAPPED (every submitted paper, no top-K cap),
billed to OPENAI_API_KEY during the eval — internal capped+cached
judging is much cheaper and is not a guide. Stock astabench keeps a
persistent verdict cache (detailed_reference.json inside the installed
package) that accumulates across every run on this machine — and the
scoring loop inserts cached verdicts at submission position while
appending fresh ones after, so the nDCG rank term of a warm-cache run
scores a permuted ranking (see ../robophd_runs/docs/astabench_judge_ordering_issue.md).
A fresh FULL run therefore moves that cache aside first
(backup_stock_judge_cache) and judges cold: deterministic, reproducible
ordering at full judge spend. Resumed full runs and smoke runs keep the
cache as-is (a resume's completed samples already baked their ordering;
a smoke's warmth is cleared later by the full run's backup anyway).

A `--limit 3` smoke run costs ~$3 and validates the whole path first.

Prerequisites:
  - OPENAI_API_KEY               (agent models gpt-5.4-mini/gpt-5.4 AND the
                                  benchmark's GPT-4o relevance judge)
  - ANTHROPIC_API_KEY            (or ANTHROPIC_API_KEY_FOR_ROBOPHD;
                                  model_registry instantiates all handles
                                  at import, Anthropic validates eagerly)
  - GOOGLE_API_KEY               (registry import needs it present)
  - ASTA_TOOL_KEY                (Asta MCP corpus tools — the task's only
                                  retrieval surface)
  - HF_ACCESS_TOKEN              (test-split dataset download)
  - pip install litellm==1.88.1  (submission-scoring price map)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

# Repo root (this script lives at <repo>/scripts/).
REPO = Path(__file__).resolve().parent.parent

EXAMPLES_DIR = REPO / "examples" / "asta_paper_finder"
SOURCE_BASE = REPO / "example_runs" / "robophd" / "asta_paper_finder"
WORKING_BASE = REPO / "submissions" / "asta_paper_finder"
FULL_LOG_SUBDIR = "logs/full_test"
# The astabench CLI --task filter matches the config's task NAME
# (config/v1.0.0.yml: name PaperFindingBench_test, path
# astabench/paper_finder_test) — the path does not match.
TASK_NAME = "PaperFindingBench_test"

# Every model the staged agents call, as litellm bundled-map keys (litellm
# strips the provider prefix). Checked against the installed litellm before
# any eval spend — an unpriced model would surface only after the full run
# as `astabench score` cost=null.
#
# The union across SUBMISSIONS, not a per-submission list: the preflight is
# a cheap all-or-nothing gate, and an agent that never calls a listed model
# costs nothing to over-verify. v0_0_9 added the Anthropic pair — it is the
# first submission here whose agent calls anything but OpenAI, and a model
# absent from this list is simply never checked.
AGENT_MODELS = [
    "gpt-5.4-mini",
    "gpt-5.4-2026-03-05",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
]

# Official-judging cost projection, printed before eval. Calibrated on
# three completed official runs rather than an internal capped estimate:
#   v0_0_7  $192 judge / 194 x 250 = 48.5K verdicts  ≈ $0.0040/paper (976 chars/paper)
#   v0_0_8  $117 judge / 194 x 203.5 = 39.5K verdicts ≈ $0.0030/paper (750 chars/paper)
#   v0_0_9c063 $197 judge / 194 x 250 = 48.5K verdicts ≈ $0.00407/paper (747 chars)
#   v0_0_9c355 $206 judge / 194 x 250 = 48.5K verdicts ≈ $0.00426/paper (765 chars)
# v0_0_9c063 rules out the obvious model: its evidence is 3 chars/paper
# shorter than v0_0_8's and it still billed at v0_0_7's rate, so per-paper
# cost does NOT track evidence length — 976, 747 and 765 chars/paper all
# land in a $0.0040-0.0043 band.
#
# The printed figure is NOT a ceiling, and as of 2026-08-07 it is not a floor
# either. print_cost_projection computes SEMANTIC_TEST_QUERIES x
# PAPERS_PER_SEMANTIC_QUERY x JUDGE_COST_PER_PAPER_USD = 194 x 250 x $0.0040 =
# $194 flat. Three full-250 runs came in ABOVE it — $197.32 for c063, $206.38
# for c355_opus5 (a 6.0% under-estimate; 12.38 short of 206.38, taken over the
# ACTUAL, which is the base for how far an estimate missed) and $203.61 for
# c355_fable — but c063_fable came in BELOW at $187.38 ($0.00386/paper), the
# first full-250 run under the flat figure.
#
# That low point is NOT a pricing change: the agent did less work. Its official
# run ran 14h33m with a median semantic sample of 1529s against 1318s
# internally, and its time-boxed expansion rounds return partial results rather
# than fail, so it graded fewer candidates and shipped thinner evidence — judge
# INPUT tokens 42.3M against c355_fable's 45.5M over the identical 48,500
# papers. Wall-clock pressure moves this number, so the band is wider than the
# per-paper rate alone suggests.
#
# It over-predicts hard when the agent ships SHORT LISTS (~$219 printed against
# $118.68 actual for v0_0_8 at 203.5 papers/query). Budget a full-250 agent from
# the measured $0.0039-0.0043 band, not from the printed number.
JUDGE_COST_PER_PAPER_USD = 0.0040
SEMANTIC_TEST_QUERIES = 194
PAPERS_PER_SEMANTIC_QUERY = 250


class Submission(NamedTuple):
    # Dir name under both example_runs/robophd/asta_paper_finder/ and
    # submissions/asta_paper_finder/. Becomes the .tar.gz basename too.
    name: str
    # agent.py path relative to the example_runs source dir for this
    # submission. Snapshots are run-shaped (ds1000 precedent): the
    # winning agent nests under agents/<agent_name>/.
    agent_rel_path: str
    # --model arg for `astabench eval`. `none` for multi-model agents so
    # the recorded eval.model doesn't claim a single primary; per-call
    # usage is captured in stats.model_usage.
    model_arg: str


SUBMISSIONS = [
    Submission(
        name="v0_0_7_soft_cap_0_06_fable",
        agent_rel_path="agents/iter12_body_conjunction/agent.py",
        model_arg="none",
        # Run robophd-asta_paper_finder-003 (fable-5-evolved), winner
        # iter12_body_conjunction (Elo 1581, 8 test rounds). Internal test
        # 0.3724 mean F1 @ $0.0556/query (free zone $0.06). Models:
        # gpt-5.4-mini + gpt-5.4-2026-03-05 — both priced in the litellm
        # 1.88.1 bundled map (verified; see AGENT_MODELS preflight).
        # Patch 0_0_7 continues the cross-benchmark sequence after
        # DS-1000's v0_0_6.
    ),
    Submission(
        name="v0_0_8_soft_cap_0_033_opus",
        agent_rel_path="agents/iter9_rerank_rich_v1/agent.py",
        model_arg="none",
        # Run robophd-asta_paper_finder-006 (opus-4.8-evolved; luna
        # no-prose training judge), winner iter9_rerank_rich_v1 (Elo 1589,
        # 7 test rounds) — the platform's own Elo pick, submitted as such
        # even though iter14 finished 1.07 Elo behind on a higher train
        # mean (see the snapshot README). Internal test 0.2754 mean F1 @
        # $0.006/query on a full stock GPT-4o re-eval (free zone $0.033).
        # Sole model gpt-5.4-mini, already covered by AGENT_MODELS.
        # The cheap counterpart to v0_0_7, not a replacement: the two are
        # distinct Pareto points (0.3749 @ $0.0533 vs 0.2754 @ $0.006).
        # model_arg stays "none" despite the single model — the recorded
        # eval.model would otherwise claim a primary, and per-call usage
        # is already captured in stats.model_usage.
    ),
    Submission(
        name="v0_0_9_cap_0_063_opus5",
        agent_rel_path="agents/iter15_verdict_repair/agent.py",
        model_arg="none",
        # Run robophd-asta_paper_finder-010 (opus-5-evolved; luna no-prose
        # training judge), winner iter15_verdict_repair (Elo 1571, 5 test
        # rounds). Internal test 0.3839 mean F1 @ $0.0533/query on the
        # stock GPT-4o basis with canonical ordering. Prices are quoted to
        # three decimals throughout — the board's precision, beyond which
        # two entries are a cost TIE decided on score alone.
        # The $0.063 free zone is a competitor's price, not a round number:
        # Asta Paper Finder's cheaper frontier point is $0.063. Setting the
        # gate AT that figure buys the cost half of a Pareto-dominance
        # claim by construction and leaves only the score half to win (the
        # v0_0_8 $0.033 gate's logic, aimed this time at the frontier point
        # directly above ours). At $0.053 this ties v0_0_7 on cost, so its
        # +0.009 score displaces our own entry rather than joining it.
        # FIRST leaderboard submission of any kind evolved by Opus 5, which
        # spends its allowance far more readily than Opus 4.8 (cap
        # utilization 75% over 3 runs vs 52% over 5) — see the snapshot
        # README; unspent free zone is forgone score.
        # Unlike v0_0_8 this is not a new Pareto point beside v0_0_7 — at
        # the same $0.0533 it carries +0.009 more score, so if it transfers
        # it displaces our own entry rather than joining it.
        # FIRST submission whose agent calls non-OpenAI models: four
        # handles over two providers (gpt-5.4-mini, gpt-5.4-2026-03-05,
        # claude-haiku-4-5-20251001, claude-sonnet-4-6). All four are in
        # the litellm 1.88.1 bundled map and all four are now listed in
        # AGENT_MODELS — the preflight only checks what it is told about.
        # The name drops the soft_/sharp_ prefix: cost_per_error now
        # defaults to 10% of the threshold, so the slope that prefix
        # encoded no longer varies between submissions.
    ),
    Submission(
        name="v0_0_9_cap_0_355_opus5",
        agent_rel_path="agents/iter21_gold_rubric_and_hard_predicates/agent.py",
        model_arg="none",
        # Run robophd-asta_paper_finder-011 (opus-5-evolved; luna no-prose
        # training judge), winner iter21_gold_rubric_and_hard_predicates
        # (Elo 1602, 3 test rounds). Internal test 0.4222 mean F1 @
        # $0.246/query — our highest on this task and the first above Asta
        # Paper Finder's 0.397. It opens a new region of the curve between
        # Ai2's two entries rather than displacing anything of ours.
        # SAME patch number as cap_0_063 on purpose: the version tracks the
        # RoboPhD code base, which did not meaningfully change between the
        # two runs (one engine-side elo_reachability fix, no solver
        # change). Same code, different cap.
        # $0.355 is Asta Paper Finder's TOP entry to three decimals — the
        # same competitor's-price construction as the $0.033 and $0.063
        # gates. This one left headroom: it landed at 69% of the gate,
        # where cap_0_063 engineered to its threshold and landed on it.
        # Models: gpt-5.4-2026-03-05, gpt-5.4-mini, claude-sonnet-4-6 —
        # three handles over two providers, all already in AGENT_MODELS.
        # NAME: the winner's name is the evolution model's own and is kept
        # unedited, but it OVERSTATES the code. The agent reconstructs a
        # rubric from the raw query; no gold reaches it (AST-verified: no
        # id-shaped constants, no sample_id branching). Three worked
        # examples in the planner prompt come from TRAINING gold, with zero
        # topical overlap in the held-out 267. The snapshot README carries
        # the full disclosure — read it before answering any reviewer
        # question about the name.
    ),
    Submission(
        name="v0_0_9_cap_0_355_fable",
        agent_rel_path="agents/iter18_cocite_largegold_v1/agent.py",
        model_arg="none",
        # Run robophd-asta_paper_finder-012 (FABLE-5-evolved; luna no-prose
        # training judge), winner iter18_cocite_largegold_v1 (Elo 1594, 5
        # test rounds). Internal test 0.4383 mean F1 @ $0.278/query on the
        # stock GPT-4o basis with canonical ordering.
        # FIRST entry that would DOMINATE the board leader rather than tie
        # it: Asta Paper Finder's top point is 0.433 @ $0.355, and this is
        # 0.438 @ $0.278 — higher AND 22% cheaper, so it takes rank 1 and
        # knocks that entry off the curve. cap_0_355_opus5 landed 0.00093
        # BELOW the leader and sat beside it; this clears it. Projected
        # frontier: five slots, four ours.
        # Same $0.355 competitor's-price gate as cap_0_355_opus5, and
        # identical in every other deliberate respect — same guard, judges,
        # budget, examples/iteration, harness. ONLY the evolution model
        # differs, which makes the pair the cleanest evolution-model A/B in
        # the campaign. The whole +0.016 is metadata (0.215 -> 0.342, zeros
        # 16 -> 5) at 59% LOWER spend on that category; semantic and
        # specific are a wash.
        # Models: gpt-5.4-2026-03-05 + gpt-5.4-mini — two handles over ONE
        # provider, both already in AGENT_MODELS (no preflight change). Note
        # against cap_0_355_opus5's three handles over two providers: the
        # higher score did not come from more model diversity.
        # Version stays v0_0_9: the patch number tracks the code base, and
        # only 41933a6e (a luna pricing-table update, not a solver change)
        # landed between the two runs.
    ),
    Submission(
        name="v0_0_9_cap_0_063_fable",
        agent_rel_path="agents/iter14_title_channel/agent.py",
        model_arg="none",
        # Run robophd-asta_paper_finder-013 (FABLE-5-evolved; luna no-prose
        # training judge), winner iter14_title_channel (Elo 1607 on 8 test
        # rounds — the best-measured winner in the campaign). Internal test
        # 0.38738 mean F1 @ $0.05828/query, stock GPT-4o basis, canonical
        # ordering, ZERO test-set timeouts.
        # COMPLETES THE 2x2 of {opus-5, fable-5} x {$0.063, $0.355}. This is
        # the missing $0.063 x fable-5 cell; all four now sit on one frozen
        # stack (zero commits to examples/asta_paper_finder/ between -012
        # and this run). fable-5 leads opus-5 by +0.0034 here against -010's
        # 0.38394 — same direction as the $0.355 cell's +0.0085, at ~40% the
        # size. n=1 per cell, and note the $0.355 comparison is
        # official-basis while this one is internal-basis.
        # The CATEGORY SHAPE replicates: vs -010 on matched IDs, metadata
        # +0.0262, specific +0.0351, semantic -0.0101 — the same three signs
        # as the $0.355 cell (+0.1202 / +0.0430 / -0.0184). fable-5 trades
        # semantic for the other two at BOTH gates, so the headline is a
        # trade rather than a uniform gain.
        # Same $0.063 competitor's-price gate as cap_0_063_opus5 — Asta
        # Paper Finder's 0.397 @ $0.063, the one non-RoboPhD entry still
        # holding a frontier slot. Second attempt at that target, with the
        # evolution model as the only deliberate change.
        # Cap-hugging replicates too: $0.0583 is 93% of the $0.063 cap
        # against opus-5's 85% at the same gate (78% vs 71% at $0.355).
        # Models: gpt-5.4-2026-03-05 + gpt-5.4-mini — two handles over ONE
        # provider, both already in AGENT_MODELS (no preflight change).
        # Version stays v0_0_9: this run executed 2026-08-04 on a stack
        # byte-identical to -012's. The 2026-08-06 commits (rejudge_test
        # --uncapped / --from-eval-log, evaluator persist_full_evidence) are
        # test/offline tooling and touch neither solver nor training.
    ),
]


def verify_clean_tree() -> None:
    """Refuse to run if git status --porcelain returns anything.

    A dirty tree would cause Inspect to record eval_spec.revision.dirty=true
    in the .eval log, weakening the reproducibility claim for the leaderboard
    submission.
    """
    out = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if out:
        sys.stderr.write(
            "Working tree is dirty — refusing to run.\n"
            "The recorded eval_spec.revision.dirty would be true, weakening\n"
            "the leaderboard submission's reproducibility claim.\n"
            "Either commit or stash the following before re-running:\n\n"
            f"{out}\n"
        )
        sys.exit(1)


def resolve_anthropic_key() -> str:
    """Return ANTHROPIC_API_KEY value to inject into subprocesses.

    Convention: keep ANTHROPIC_API_KEY unset in the interactive shell so
    the user's Claude Code CLI subscription credentials aren't clobbered,
    and use ANTHROPIC_API_KEY_FOR_ROBOPHD for evaluation (model_registry
    reads either). Injected into subprocess env only.
    """
    return (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY_FOR_ROBOPHD")
        or ""
    )


def verify_credentials() -> None:
    """Hard-fail at startup if any required env var is missing.

    All three provider keys are required even though the agent only calls
    OpenAI models — model_registry.py creates handles for every family at
    import time, and the Anthropic provider validates its key eagerly.
    ASTA_TOOL_KEY is the task's only retrieval surface; HF_ACCESS_TOKEN
    gates the test-split download. Failing here beats failing minutes into
    a paid run.
    """
    missing = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not resolve_anthropic_key():
        missing.append("ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY_FOR_ROBOPHD)")
    if not os.environ.get("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY")
    if not os.environ.get("ASTA_TOOL_KEY"):
        missing.append("ASTA_TOOL_KEY")
    if not os.environ.get("HF_ACCESS_TOKEN"):
        missing.append("HF_ACCESS_TOKEN")
    if missing:
        sys.stderr.write(
            f"Missing required env vars: {', '.join(missing)}\n"
            "See examples/asta_paper_finder/README.md for setup instructions.\n"
        )
        sys.exit(1)


def verify_model_pricing() -> None:
    """Assert the installed litellm's bundled map prices every AGENT_MODELS
    entry — before any eval spend.

    `astabench score` runs with LITELLM_LOCAL_MODEL_COST_MAP=True (it
    refuses otherwise), which forces the BUNDLED map: any model the agent
    called that the map lacks prices the whole run to cost=null, and the
    leaderboard entry would ship without a cost figure (DS-1000 v0_0_3).
    """
    try:
        import litellm
    except ImportError:
        sys.stderr.write(
            "litellm is not importable — `pip install litellm==1.88.1` "
            "(the submission-scoring price map) and re-run.\n"
        )
        sys.exit(1)
    unpriced = [
        m for m in AGENT_MODELS
        if not (litellm.model_cost.get(m) or {}).get("input_cost_per_token")
    ]
    if unpriced:
        sys.stderr.write(
            f"litellm's bundled price map does not price: {unpriced}\n"
            "`astabench score` would emit cost=null for the whole run.\n"
            "Upgrade litellm (1.88.1 is known-good for these models) and "
            "re-run.\n"
        )
        sys.exit(1)
    print(f"[pricing] litellm bundled map prices all agent models: {AGENT_MODELS}")


def print_cost_projection(limit: int | None) -> None:
    judge = SEMANTIC_TEST_QUERIES * PAPERS_PER_SEMANTIC_QUERY * JUDGE_COST_PER_PAPER_USD
    print(
        f"\n[cost] Official judging is fresh + uncapped: "
        f"~{SEMANTIC_TEST_QUERIES} semantic queries × "
        f"~{PAPERS_PER_SEMANTIC_QUERY} papers × ${JUDGE_COST_PER_PAPER_USD}/paper "
        f"≈ ${judge:.0f} judge + ~$15 agent ≈ ${judge + 15:.0f} total (full run)."
    )
    if limit is not None:
        print(f"[cost] --limit {limit} smoke run: a few dollars; logs are "
              f"kept separate and never tarred.")


def run(cmd: list[str], *, cwd: Path, extra_env: dict | None = None,
        tee: Path | None = None) -> int:
    """Stream subprocess output live. Returns exit code.

    `tee` also writes the child's combined output to a file. The agent's own
    stage prints ("[t+NNNs] plan: ...", "DEADLINE: solver timed out", trim
    messages) go to stdout and are NOT captured anywhere in the .eval log —
    inspect records model events, not prints. Without a tee they exist only in
    whatever terminal scrollback happens to survive, which is how the
    2026-08-06 post-mortem lost its most direct evidence.

    The file is APPENDED, never truncated: eval_submission reuses one log_dir
    across resumes — that is how `inspect eval-set`'s retry path reuses
    completed samples — so opening "w" would delete the interrupted attempt's
    stdout, which is precisely what a post-mortem of that interruption needs.
    Each attempt gets a banner so the boundaries are greppable.
    """
    env = {**os.environ, **(extra_env or {})}
    print(f"\n$ cd {cwd}")
    print(f"$ {' '.join(cmd)}\n")
    if tee is None:
        return subprocess.run(cmd, cwd=cwd, env=env).returncode
    tee.parent.mkdir(parents=True, exist_ok=True)
    # The child block-buffers stdout when it is a pipe rather than a tty, so
    # without this the agent's prints reach the file in ~8 KB gulps and a run
    # that dies loses whatever is still in the buffer — the tail being exactly
    # the part worth reading.
    env["PYTHONUNBUFFERED"] = "1"
    with subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, bufsize=1,
                          text=True) as p, tee.open("a") as fh:
        fh.write("\n===== attempt started %s =====\n"
                 % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        fh.flush()
        for line in p.stdout:          # keeps the live-stream contract
            sys.stdout.write(line)
            fh.write(line)
        return p.wait()


# Auto-generated resilience wrapper inserted as `agent.py` in the staged
# working dir. The original agent code is renamed to `agent_inner.py` and
# imported from here.
#
# Why this exists: RoboPhD's evolution evaluator runs each sample in a
# subprocess, so a per-sample solver crash returns raw_score=0 and the run
# continues. AstaBench's CLI runs all samples in one process and aborts on
# any uncaught solver exception. The wrapper bridges those two contracts so
# the AstaBench-CLI score reflects the same crash-tolerance the recorded
# RoboPhD-internal score was produced under.
#
# Caught: any `Exception` subclass (which on Python 3.11+ includes
# asyncio.TimeoutError). NOT caught: KeyboardInterrupt, SystemExit,
# asyncio.CancelledError — BaseException-only signals stay unhandled so
# user/runtime cancellation still works correctly.
WRAPPER_TEMPLATE = '''"""Auto-generated resilience wrapper for AstaBench submission.

Two-tier fallback: primary agent (agent_inner.py) -> seed agent
(seed_agent.py) -> empty-but-valid submission. Each layer catches
Exception so a transient provider failure or evolved-agent bug on one
query doesn't abort the eval, and the seed gets a shot at recovering the
score before we give up.

The last-resort output is NOT an empty string: PaperFindingBench's scorer
falls back to an LLM re-parse on unparseable completions, so we emit the
schema-valid empty submission ({"output": {"query_id": ..., "results": []}}),
which parses cleanly and scores 0 with no extra machinery.

Per-sample wall-clock timeout: both tiers are wrapped in
`asyncio.wait_for(..., timeout=1500)` so a hung primary (rate-limit
retry storm, provider connection deadlock, snippet_search stall) can't
wedge the eval indefinitely. Internal training capped queries at 1800s;
1500s leaves room for the seed tier within the same ballpark.
asyncio.TimeoutError is a subclass of Exception on Python 3.11+, so the
`except Exception` blocks catch the timeout and fall through naturally.

Tool-call launch pacing: `astabench eval` runs all samples as asyncio
tasks in ONE process (--max-samples concurrent), all sharing the Asta MCP
key and its ~10 req/s per-endpoint budget. Each sample's tools are wrapped
with tool_pacer.pace_tools (in-process mode, ~8 launches/s per endpoint
globally) so concurrent samples cannot aggregate into the HTTP 500
contention windows observed in training. Standard-tier legal: stdlib
pacing around the task-provided tools, same category as this wrapper's
retry/fallback plumbing.

Per-sample module isolation: agent_inner and seed_agent are loaded as FRESH
module instances per sample (see _isolated below), so anything an evolved
agent keeps at module scope is per-sample here exactly as it is under
RoboPhD's one-subprocess-per-sample training harness. Without this, six
concurrent samples share one module namespace and per-sample state silently
becomes global — which killed v0_0_9_cap_0_063_fable's deadline pacing.
model_registry and tool_pacer deliberately stay shared.

Wrapper recipe lives in scripts/asta_paper_finder_submit.py:WRAPPER_TEMPLATE.
"""
import asyncio
import importlib.util
import itertools
import json
import sys
import traceback
import uuid
from pathlib import Path

from inspect_ai.solver import Generate, TaskState, solver

import tool_pacer
# LOAD-BEARING — do not delete as an "unused import". This runs while
# inspect's chdir_python() still has the submission dir on sys.path, which
# pins model_registry into sys.modules. agent_inner does `from model_registry
# import ...` at ITS module scope, and the per-sample loads below happen long
# after chdir_python.__exit__ restored sys.path (inspect_ai/_util/path.py:60).
# Without this, every sample dies with ModuleNotFoundError.
import model_registry  # noqa: F401


_HERE = Path(__file__).resolve().parent
_TAG = uuid.uuid4().hex[:8]   # load_module() re-execs this file on every
_SEQ = itertools.count()      # solver_from_spec, so a bare counter collides
_WARNED = set()


def _warn_once(key: str, detail: str) -> None:
    """One line per distinct problem, not one per sample.

    These conditions are not fatal to a sample but they ARE load-bearing for
    the run, so swallowing them is wrong: the eval would finish and score
    plausibly with no trace of what changed.
    """
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[WRAPPER] WARNING: {detail}", flush=True)


# Private inspect API, resolved once at load rather than per sample. If it
# moves, the wrapper still runs, but agent_inner's @solver keeps the bare
# "make_solver" registry entry it clobbers on every load — and eval-set's
# RESUME path (evalset.py resolve_solver, inside task_identifier) looks the
# solver up by that name, so a resumed run would execute agent_inner with NO
# wrapper: no wait_for ceiling, no seed tier, no tool pacing, and the shared
# module state this whole wrapper exists to prevent. Loud here, once.
try:
    from inspect_ai._util.registry import registry_add as _registry_add
    from inspect_ai._util.registry import registry_info as _registry_info
except Exception as _reg_err:  # pragma: no cover - private API moved
    _registry_add = _registry_info = None
    print(f"[WRAPPER] WARNING: inspect_ai._util.registry unavailable "
          f"({type(_reg_err).__name__}: {_reg_err}); the solver registry entry "
          f"cannot be restored after per-sample loads. Do NOT resume this "
          f"eval-set — a resume may run agent_inner with no wrapper.",
          flush=True)


def _isolated(stem: str):
    """Load <stem>.py as a FRESH module: its module scope becomes per-sample.

    `astabench eval` runs --max-samples samples as asyncio tasks in ONE
    process; RoboPhD training and internal test run one sample per subprocess.
    Agents are therefore evolved where module scope means "per sample" and
    graded where it means "shared across concurrent samples". v0_0_9's
    cap_0_063_fable kept its deadline clock (_START/_DEADLINE) and tool
    semaphore (_TOOL_SEM) at module scope: under concurrency each starting
    sample re-stamped the shared clock, _remaining() never counted down, every
    trim gate the agent had evolved was dead, and 30% of samples ran to their
    own wait_for. This makes that class of bug structurally impossible.

    Isolation boundary: modules loaded from THIS directory under a synthetic
    name. model_registry and tool_pacer stay SHARED by design (canonical names
    in sys.modules — one connection pool, one global launch budget). An agent
    that mutates a shared module's globals is not isolated by this.
    """
    # chdir_python() had this dir on sys.path only while agent.py itself was
    # loading, then __exit__ restored the snapshot. Third-line defence only:
    # the load below uses an absolute path, and model_registry is already
    # pinned in sys.modules by the module-scope import above. It is here
    # because the failure it guards (ModuleNotFoundError on every sample)
    # zeroes an entire run at full judge cost.
    # The entry persists for the process lifetime. That is safe because
    # `astabench eval` runs exactly ONE submission per process — the submit
    # script spawns a subprocess per submission — so no second submission dir
    # can ever be shadowed by it.
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    name = f"{stem}__iso{_TAG}_{next(_SEQ)}"
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{stem}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build a module spec for {stem}.py")
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: the @solver decorator calls find_spec() on this
    # name, which is a full sys.path scan whenever the name is absent.
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        # Drop the name so hundreds of module dicts don't accumulate. Never
        # mod.__dict__.clear() — a cancelled asyncio task may still be
        # unwinding through these globals.
        sys.modules.pop(name, None)
    # agent_inner's own @solver registered under the same bare name
    # ("make_solver") and clobbered ours. inspect's eval-set resume path
    # resolves the solver by that name, so put it back. Absent during the
    # warm-up below, which runs before the wrapper is defined.
    _wrapper = globals().get("make_solver")
    if _wrapper is not None and _registry_add is not None:
        try:
            _registry_add(_wrapper, _registry_info(_wrapper))
        except Exception as e:
            _warn_once(
                "registry_restore",
                f"could not restore the solver registry entry after loading "
                f"{stem} ({type(e).__name__}: {e}). Samples still run, but do "
                f"NOT resume this eval-set — a resume may run agent_inner "
                f"with no wrapper.")
    return mod


# Warm-up load, at module scope so it runs INSIDE inspect's chdir_python
# window. Compiles the pyc once so no sample pays compile cost, and surfaces a
# broken agent_inner here rather than on sample 1. Must stay ABOVE the @solver
# below so the wrapper is the last thing registered.
_isolated("agent_inner")


PRIMARY_TIMEOUT_S = 3000  # 50 min — a hang bound, not a work budget.
# It has to clear whatever pacing the staged agent has, and agents differ:
# v0_0_7's iter12 self-paced (SOFT_DEADLINE=1300 / TAIL_DEADLINE=1550,
# evolved against training's 1800s external cap), while v0_0_8's iter9 has
# no deadline constants at all — its runtime comes from a fixed work plan,
# and with no per-call timeouts a hung tool call runs until this ceiling
# fires. For a self-pacing agent a tight ceiling guillotines it INSIDE its
# planned budget: at 1500s the first official attempt seed-fell-back on 5
# of 9 samples (0.57->0.06 on semantic_5); at 2100 the second watched
# semantic_25 finish 48s short as MCP latency stretched solves to ~30 min.
# For a non-pacing agent the ceiling should only ever fire on a true hang.
# Either way firing means a seed-tier fallback for that sample. Wall clock
# is unscored officially, so generosity costs nothing — size this off the
# slowest agent in SUBMISSIONS, never the newest.
SEED_TIMEOUT_S = 1500     # fallback tier, applied independently


def _empty_submission(state: TaskState) -> str:
    return json.dumps({"output": {"query_id": str(state.sample_id), "results": []}})


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.metadata is None:
            state.metadata = {}
        # Global in-process launch pacing across concurrent samples (see
        # module docstring). Wrapping here covers the seed tier too, since
        # it sees the same state.tools.
        state.tools = tool_pacer.pace_tools(state.tools)
        try:
            # Fresh module per sample. INSIDE the try on purpose: a load
            # failure (transient OSError, MemoryError, a half-written source
            # file) must degrade to the seed tier, not abort the whole eval.
            inner = _isolated("agent_inner").make_solver()
            return await asyncio.wait_for(inner(state, generate), timeout=PRIMARY_TIMEOUT_S)
        except Exception as primary:
            print(f"[{state.sample_id}] WRAPPER primary caught {type(primary).__name__}: {primary}")
            print(f"[{state.sample_id}] WRAPPER primary traceback (truncated):")
            print(traceback.format_exc()[:1500])
            state.metadata["__wrapper_primary_caught"] = repr(primary)[:500]
            state.metadata["__wrapper_primary_traceback"] = traceback.format_exc()[:2000]
            try:
                # Lazy: the seed is loaded only on the failure path, so the
                # happy path pays zero seed loads.
                seed = _isolated("seed_agent").make_solver()
                return await asyncio.wait_for(seed(state, generate), timeout=SEED_TIMEOUT_S)
            except Exception as fallback:
                print(f"[{state.sample_id}] WRAPPER seed fallback ALSO caught {type(fallback).__name__}: {fallback}")
                print(f"[{state.sample_id}] WRAPPER seed fallback traceback (truncated):")
                print(traceback.format_exc()[:1500])
                state.output.completion = _empty_submission(state)
                state.metadata["__wrapper_fallback_caught"] = repr(fallback)[:500]
                state.metadata["__wrapper_fallback_traceback"] = traceback.format_exc()[:2000]
                return state

    return solve
'''


def work_dir(s: Submission, suffix: str = "") -> Path:
    """Working dir for a submission.

    A non-empty suffix gives an A/B arm its own dir, so the real submission
    dir — frozen once its full run succeeds — is never touched.
    """
    return WORKING_BASE / f"{s.name}{suffix}"


def stage(s: Submission, suffix: str = "", restage: bool = False) -> Path:
    """Stage a working dir with the two-tier resilience wrapper.

    Layout in dst_dir:
      agent.py          — auto-generated wrapper (the file --solver references)
      agent_inner.py    — the primary evolved agent source (renamed)
      seed_agent.py     — the baseline seed agent (the fallback tier)
      model_registry.py — copied from examples/asta_paper_finder/
      tool_pacer.py     — copied from examples/asta_paper_finder/ (global
                          launch pacing across concurrent samples)

    FROZEN once the submission has a successful full log. Staging is
    unconditional and runs BEFORE eval_submission's skip-on-success check, so
    without this guard any re-run — including a bare invocation with no --only,
    which iterates every entry — silently replaces the on-disk record of what
    produced a posted leaderboard entry. That record has already drifted once:
    v0_0_7_soft_cap_0_06_fable still carries its original 3702-byte wrapper and
    no tool_pacer.py, both predating the current template. The eval itself is
    skipped either way, so the loss would be invisible.

    --restage overwrites deliberately; A/B arms should pass a suffix instead.

    Returns the working dir path.
    """
    dst_dir = work_dir(s, suffix)
    status, completed = _log_status(dst_dir / FULL_LOG_SUBDIR)
    if status == "success" and not restage:
        print(f"[skip stage] {s.name}: frozen (successful full log, "
              f"{completed} samples). Pass --restage to overwrite, or "
              f"--work-suffix to stage an A/B arm elsewhere.")
        return dst_dir
    src_agent = SOURCE_BASE / s.name / s.agent_rel_path
    src_seed = EXAMPLES_DIR / "seeds" / "baseline" / "agent.py"
    src_registry = EXAMPLES_DIR / "model_registry.py"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_agent, dst_dir / "agent_inner.py")
    shutil.copy(src_seed, dst_dir / "seed_agent.py")
    (dst_dir / "agent.py").write_text(WRAPPER_TEMPLATE)
    shutil.copy(src_registry, dst_dir / "model_registry.py")
    shutil.copy(EXAMPLES_DIR / "tool_pacer.py", dst_dir / "tool_pacer.py")
    return dst_dir


def log_subdir(limit: int | None) -> str:
    """Full runs log to logs/full_test (the tarred, idempotency-checked
    dir). --limit smoke runs log to their own dir so a partial run can
    neither be tarred as a submission nor trip the full run's skip."""
    return FULL_LOG_SUBDIR if limit is None else f"logs/smoke_limit_{limit}"


def _log_status(log_dir: Path) -> tuple[str | None, int]:
    """(status, completed_samples) of the newest .eval in log_dir, read
    header-only. (None, 0) when no log exists.

    Status matters: an interrupted run leaves a log with status
    "cancelled"/"error" that contains only the samples that finished.
    Treating any *.eval as done would skip the eval on re-run and then
    score+tar a PARTIAL log as a submission. Instead, only "success"
    skips; anything else re-runs `astabench eval` with the same log dir,
    which flows into `inspect eval-set`'s retry path — completed samples
    are reused (dataset unshuffled + stable query ids), so a resume
    re-runs only what didn't finish.
    """
    logs = sorted(log_dir.glob("*.eval")) if log_dir.exists() else []
    if not logs:
        return None, 0
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(logs[-1]), header_only=True)
    completed = getattr(getattr(log, "results", None), "completed_samples", 0) or 0
    return log.status, completed


def backup_stock_judge_cache() -> None:
    """Move astabench's accumulated judge-verdict cache aside (timestamped
    sibling, never deleted) so a fresh full run judges cold.

    Cold judging is the only reproducible official-scoring mode: with a warm
    cache, cached verdicts land at submission position while fresh ones are
    appended after the scoring loop, so the nDCG rank term depends on how
    much this submission's papers overlap with whatever was scored on this
    machine before (../robophd_runs/docs/astabench_judge_ordering_issue.md). Costs the full
    judge spend — the budgeted figure — instead of a discounted but
    cache-order-contaminated number.

    The path comes from astabench itself, never hardcoded. Missing file
    (fresh install) is a silent no-op.
    """
    from astabench.evals.paper_finder.paper_finder_utils import (
        detailed_reference_path,
    )

    path = Path(detailed_reference_path)
    if not path.exists():
        print("[judge-cache] no stock detailed_reference.json — already cold")
        return
    bak = path.with_name(
        f"{path.name}.bak-{time.strftime('%Y%m%d_%H%M%S')}"
    )
    size_mb = path.stat().st_size / 1e6
    path.rename(bak)
    print(f"[judge-cache] moved stock cache aside ({size_mb:.1f} MB) -> "
          f"{bak} — full run will judge cold (reproducible rank ordering)")


def eval_submission(s: Submission, working_dir: Path, limit: int | None) -> bool:
    status, completed = _log_status(working_dir / log_subdir(limit))
    if status == "success":
        print(f"[skip eval] {s.name}: successful .eval "
              f"({completed} samples) in {working_dir / log_subdir(limit)}")
        return True
    if status is not None:
        print(f"[resume] {s.name}: previous eval status={status!r} with "
              f"{completed} completed sample(s) — re-running; inspect "
              f"eval-set reuses completed samples and runs the rest")
    elif limit is None:
        # Fresh full run: judge cold. A resume keeps the cache — its
        # completed samples' verdict ordering is already baked into the log.
        backup_stock_judge_cache()
    log_dir = working_dir / log_subdir(limit)
    log_dir.mkdir(parents=True, exist_ok=True)
    # LITELLM_LOCAL_MODEL_COST_MAP is required by `astabench score`;
    # passing during eval too keeps env consistent. ANTHROPIC_API_KEY is
    # injected here (not in the parent process) — see resolve_anthropic_key().
    extra_env = {
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "ANTHROPIC_API_KEY": resolve_anthropic_key(),
    }
    cmd = [
        "astabench", "eval",
        "--solver", "agent.py",
        "--model", s.model_arg,
        "--split", "test",
        "--task", TASK_NAME,
        "--log-dir", str(log_dir),
        "--display", "plain",
        # Concurrency: no Docker tier for this task, so no --max-sandboxes.
        # --max-samples 6 keeps aggregate Asta MCP tool traffic under the
        # 10 req/s per-endpoint server-side rate limit (the agent fans out
        # snippet searches within a query; it evolved and was internally
        # tested at 8-wide against this endpoint, so 6 is inside its
        # native habitat). 40 connections: agents' grading
        # calls and the scorer's per-paper GPT-4o judge fan-out share ONE
        # pool — at 20, agents queued behind judge bursts and samples
        # stretched from ~12 min (smoke, no scoring overlap) to ~25 min,
        # into the wrapper ceiling.
        "--max-samples", "6",
        "--max-connections", "40",
    ]
    if limit is not None:
        # Forwarded to `inspect eval-set` via agenteval's
        # ignore_unknown_options Click context.
        cmd += ["--limit", str(limit)]
    rc = run(cmd, cwd=working_dir, extra_env=extra_env,
             tee=log_dir / "submit_stdout.log")
    if rc != 0:
        return False
    # Belt-and-suspenders: a zero exit with a non-success log (cancelled
    # mid-write, partial eval-set) must not flow into score+tar.
    status, completed = _log_status(log_dir)
    if status != "success":
        print(f"!! eval exited 0 but log status={status!r} "
              f"({completed} samples) — not treating as success")
        return False
    return True


def score_submission(working_dir: Path, limit: int | None) -> bool:
    # LITELLM_LOCAL_MODEL_COST_MAP=True is REQUIRED by `astabench score`
    # (it refuses to run without it). It forces litellm's *bundled* price
    # map — verify_model_pricing() pre-checked coverage, and the null-cost
    # gate below catches anything that still slips through.
    rc = run(
        ["astabench", "score", str(working_dir / log_subdir(limit))],
        cwd=working_dir,
        extra_env={"LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )
    if rc != 0:
        return False
    stats_path = working_dir / log_subdir(limit) / "summary_stats.json"
    try:
        stats = json.loads(stats_path.read_text())["stats"]
        # One task per run; resolve its key instead of hardcoding, so an
        # astabench task-name change fails loudly here rather than KeyError.
        # Match both naming styles ("PaperFindingBench_test" config name,
        # "paper_finder_test" task path).
        task_keys = [
            k for k in stats
            if "paperfinding" in k.lower().replace("_", "")
            or "paperfinder" in k.lower().replace("_", "")
        ] or list(stats)
        cost = stats[task_keys[0]]["cost"]
    except Exception as e:
        print(f"!! could not read cost from {stats_path}: {e}")
        return False
    if cost is None:
        print(
            "!! astabench score produced cost=null — a model in the eval "
            "log isn't in the installed litellm's bundled price map. "
            "Upgrade litellm (bundled map must cover every model the "
            "agent called) and re-run. Refusing to tar a costless "
            "submission."
        )
        return False
    print(f"scored: cost/problem = ${cost:.4f}")
    return True


def tar_submission(s: Submission, working_dir: Path) -> bool:
    out = WORKING_BASE / f"{s.name}.tar.gz"
    rc = run(
        # -C cd's into the working dir, then we tar the leaf dir name
        # so the archive contents are `full_test/...eval` rather than
        # absolute or deeply-nested paths.
        ["tar", "czfv", str(out), "-C", str(working_dir / "logs"), "full_test"],
        cwd=working_dir,
    )
    return rc == 0


def summarize(limit: int | None) -> None:
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for s in SUBMISSIONS:
        scores_path = WORKING_BASE / s.name / log_subdir(limit) / "scores.json"
        if not scores_path.exists():
            print(f"  {s.name:30s}  (no scores.json)")
            continue
        try:
            data = json.loads(scores_path.read_text())
            result = data["results"][0]
            metrics = {m["name"]: m["value"] for m in result["metrics"]}
            # PaperFindingBench's headline metric is adjusted_f1_micro_avg
            # (grouped mean over all samples); fall back to printing
            # everything if the name shifts upstream.
            headline = next(
                (v for k, v in metrics.items() if "adjusted_f1" in k or k.endswith("_micro_avg")),
                None,
            )
            if headline is not None:
                print(f"  {s.name:30s}  adjusted_f1 = {headline:.4f}")
            else:
                print(f"  {s.name:30s}  metrics = {metrics}")
        except Exception as e:
            print(f"  {s.name:30s}  (parse error: {e})")
    print("=" * 70)
    if limit is not None:
        print(f"\n--limit {limit} smoke run: logs kept in "
              f"{log_subdir(limit)}/ — NOT tarred, not a submission.")
        return
    print("\nReady-to-upload tarballs:")
    for s in SUBMISSIONS:
        out = WORKING_BASE / f"{s.name}.tar.gz"
        marker = "✓" if out.exists() else "✗"
        print(f"  {marker} {out}")
    print("\nUpload at:")
    print("  https://huggingface.co/spaces/allenai/asta-bench-leaderboard")
    print("\nMetadata for the form (suggested):")
    print("  Openness:    Open source, closed weights")
    print("  Tools tier:  Standard")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="restrict to a subset of SUBMISSIONS")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="smoke run: evaluate only the first N samples; "
                             "logs to a separate dir, never tarred")
    parser.add_argument("--restage", action="store_true",
                        help="overwrite a frozen working dir (one whose full "
                             "log already succeeded). Destroys the on-disk "
                             "record of what produced a posted entry — prefer "
                             "--work-suffix for experiments")
    parser.add_argument("--work-suffix", default="", metavar="SUFFIX",
                        help="stage into <name><SUFFIX>/ instead of <name>/, "
                             "e.g. __ab_new. Leaves the real submission dir "
                             "untouched and is never tarred")
    args = parser.parse_args()

    selected = SUBMISSIONS
    if args.only:
        names = set(args.only)
        unknown = names - {s.name for s in SUBMISSIONS}
        if unknown:
            print(f"unknown submission(s): {sorted(unknown)}")
            print(f"known: {[s.name for s in SUBMISSIONS]}")
            return 2
        selected = [s for s in SUBMISSIONS if s.name in names]

    verify_clean_tree()
    verify_credentials()
    verify_model_pricing()
    print_cost_projection(args.limit)

    failures = []
    for s in selected:
        print(f"\n{'#' * 70}\n# {s.name}\n{'#' * 70}")
        working_dir = stage(s, args.work_suffix, args.restage)
        if not eval_submission(s, working_dir, args.limit):
            print(f"!! eval failed for {s.name}; continuing to next submission")
            failures.append((s.name, "eval"))
            continue
        if not score_submission(working_dir, args.limit):
            print(f"!! score failed for {s.name}")
            failures.append((s.name, "score"))
            continue
        # A/B arms are never tarred: tar_submission writes <name>.tar.gz off
        # s.name, which would collide with the real submission's archive.
        if args.limit is None and not args.work_suffix \
                and not tar_submission(s, working_dir):
            print(f"!! tar failed for {s.name}")
            failures.append((s.name, "tar"))

    summarize(args.limit)
    if failures:
        print(f"\nFailures: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
