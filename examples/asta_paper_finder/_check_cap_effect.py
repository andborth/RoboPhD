#!/usr/bin/env python3
"""Measure the capped-vs-uncapped judging effect from an OFFICIAL run's own
verdicts. No LLM calls, no cost.

Why this exists
---------------
The campaign repeatedly attributed internal->official score movements to the
capped->uncapped judging switch (internal judges only the top-K estimate;
official judges all 250 submitted papers). That attribution was never measured,
because measuring it looked like it needed a re-judge. It does not.

    adjusted_f1 = harmonic(rank, estimated_recall_at_estimate)

verified exactly on every semantic sample of every official run. The recall
term is ALREADY K-windowed, so judging depth cannot move it -- capping can only
change the nDCG `rank` term. An official run judged all 250 papers, so its own
per-paper verdicts are sufficient to recompute what the CAPPED score would have
been: truncate the ordered grade list to K and re-score.

What it found (2026-08-07, all four v0_0_9 runs)
------------------------------------------------
    run                    gate     cap effect   published internal->official
    cap_0_063_opus5       $0.063      -0.0008          -0.0117
    cap_0_063_fable       $0.063      -0.0023          -0.0271
    cap_0_355_opus5       $0.355      -0.0071          +0.0130
    cap_0_355_fable       $0.355      -0.0054          +0.0033

Capping always LOWERS the score, and lowers it 3-4x more for the deep-grading
$0.355 agents than the cheap ones -- so the mechanism is real and
gate-dependent, as supposed. But it is far too small to carry the observed
transfers, and at the $0.063 gate it has the WRONG SIGN: those runs moved
negative while the cap pushes positive. The residual is agent run-to-run
variance, which dominates at the cheap gate.

Method note (why the sample count is < 194)
-------------------------------------------
Grades are parsed from the judge's own model events, in call order, and the
result is VALIDATED by recomputing `rank` and comparing against the official
value. It reproduces exactly on ~151/194 samples; those mismatches are the
samples where the scorer skipped a document (see the "Skipping" warnings),
which shifts the ordered list. Only verified samples are reported -- an
unverified ordering would silently produce a wrong nDCG.

A SECOND, unrelated cause also drops samples: a query with no k_estimate in
the --k-from run. That says nothing about judging -- it means the run dir does
not cover this submission's queries -- so the two are counted and reported
separately in the "dropped" column. Reading a --k-from shortfall as evidence
about capped judging would be a mistake; the run prints a warning when any
occur.

Usage
-----
    python _check_cap_effect.py <submission_name> [<submission_name> ...]
    python _check_cap_effect.py --k-from <run_dir> v0_0_9_cap_0_063_fable

`--k-from` supplies k_estimate per query from a completed run's
test_problems/*/score_meta.json. K is a property of the QUERY (the benchmark's
recall denominator), not of the agent, so any run of the same split serves;
defaults to the newest run under ../../../robophd_runs/robophd/.
"""

import argparse
import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from inspect_ai.log import read_eval_log  # noqa: E402
from astabench.evals.paper_finder import relevance as rel  # noqa: E402
from astabench.evals.paper_finder.relevance import rj_4l_codes  # noqa: E402
from astabench.evals.paper_finder.eval import (  # noqa: E402
    lower_bound_corrected_ndcg,
)

JUDGE = "openai/gpt-4o-2024-11-20"
SUBMISSIONS = HERE.parent.parent / "submissions" / "asta_paper_finder"


def _harmonic(a: float, b: float) -> float:
    return 2 * a * b / (a + b) if (a + b) else 0.0


def _grades(sample) -> list[int]:
    """Per-paper 0-3 grades in judge-call order, replicating the scorer's own
    pipeline (rj_4l_codes -> calculate_relevance_criteria_score ->
    calculate_0_to_3_relevance) rather than approximating it."""
    tgt = sample.target if isinstance(sample.target, str) else (sample.target or [""])[0]
    try:
        criteria_raw = json.loads(tgt).get("relevance_criteria") or []
    except (json.JSONDecodeError, TypeError):
        return []
    crits = [
        rel.RelevanceCriterion(
            name=c["name"], description=c["description"], weight=c["weight"]
        )
        for c in criteria_raw
    ]
    required = {c.name for c in crits}
    out: list[int] = []
    for e in sample.events or []:
        if getattr(e, "event", "") != "model" or e.model != JUDGE:
            continue
        try:
            parsed = rel.extract_json_from_response(e.output.completion or "")
        except Exception:
            continue
        if not parsed or "criteria" not in parsed:
            continue
        rcj = []
        for cname, j in parsed["criteria"].items():
            if not isinstance(j, dict) or j.get("relevance") not in rj_4l_codes:
                rcj = None
                break
            # The scorer strips underscores from criterion names (a common
            # LLM naming error); mirror that or the weight lookup misses.
            rcj.append(
                {"name": cname.replace("_", " "), "relevance": rj_4l_codes[j["relevance"]]}
            )
        if rcj is None:
            continue
        if any(n not in {r["name"] for r in rcj} for n in required):
            continue  # the scorer skips these too
        try:
            score = rel.calculate_relevance_criteria_score(crits, rcj)
        except Exception:
            continue
        out.append(rel.calculate_0_to_3_relevance(score))
    return out


def _k_estimates(run_dir: Path) -> dict[str, int]:
    ks: dict[str, int] = {}
    tp = run_dir / "test_problems"
    if not tp.is_dir():
        raise SystemExit(f"{tp} not found — pass --k-from a completed run dir")
    for sid in os.listdir(tp):
        meta = tp / sid / "score_meta.json"
        if meta.exists():
            try:
                k = json.loads(meta.read_text()).get("k_estimate")
            except json.JSONDecodeError:
                continue
            if k:
                ks[sid] = k
    return ks


def analyse(name: str, ks: dict[str, int]) -> dict | None:
    logs = glob.glob(str(SUBMISSIONS / name / "logs" / "full_test" / "*.eval"))
    if not logs:
        print(f"{name}: no official log found", file=sys.stderr)
        return None
    log = read_eval_log(logs[0])
    verified, total, unc, cap = 0, 0, [], []
    # Two DIFFERENT causes shrink verified/total below, and conflating them
    # misreads the result: a rank mismatch is a property of the LOG (the
    # scorer skipped a document, shifting the ordered list), while a missing
    # k_estimate is a property of the --k-from RUN not covering this query.
    # The second says nothing about judging and should not be read as if it
    # did — it just means you pointed at the wrong run dir.
    rank_mismatch, no_k = 0, 0
    for s in log.samples or []:
        if not str(s.id).startswith("semantic"):
            continue
        meta = list(s.scores.values())[0].metadata or {}
        rank_off = meta.get("rank")
        recall = meta.get("estimated_recall_at_estimate")
        if rank_off is None or recall is None:
            continue
        grades = _grades(s)
        if not grades:
            continue
        total += 1
        # VALIDATION: our parsed+ordered grades must reproduce official rank.
        if abs(lower_bound_corrected_ndcg(grades)["rank"] - rank_off) >= 1e-6:
            rank_mismatch += 1
            continue
        k = ks.get(str(s.id))
        if not k:
            no_k += 1
            continue
        verified += 1
        unc.append(_harmonic(rank_off, recall))
        cap.append(_harmonic(lower_bound_corrected_ndcg(grades[:k])["rank"], recall))
    if not verified:
        print(f"{name}: no samples passed rank validation", file=sys.stderr)
        return None
    return dict(
        name=name, verified=verified, total=total,
        rank_mismatch=rank_mismatch, no_k=no_k,
        uncapped=st.mean(unc), capped=st.mean(cap),
        effect=st.mean(cap) - st.mean(unc),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="+", help="submission dir names")
    ap.add_argument("--k-from", type=Path, default=None,
                    help="run dir supplying k_estimate (default: newest paper_finder run)")
    args = ap.parse_args()

    run_dir = args.k_from
    if run_dir is None:
        runs = sorted(glob.glob(str(HERE.parent.parent.parent /
                                    "robophd_runs" / "robophd" / "asta_paper_finder_*")))
        if not runs:
            raise SystemExit("no run dir found for k_estimate — pass --k-from")
        run_dir = Path(runs[-1])
    ks = _k_estimates(run_dir)
    print(f"k_estimate from {run_dir.name} ({len(ks)} queries)\n")
    print(f"{'submission':28s} {'n':>5s} {'uncapped':>9s} {'capped':>9s} "
          f"{'cap effect':>11s}  {'dropped':>16s}")
    missing_k = False
    for name in args.names:
        r = analyse(name, ks)
        if not r:
            continue
        missing_k = missing_k or r["no_k"] > 0
        print(f"{r['name']:28s} {r['verified']:>3d}/{r['total']:<3d} "
              f"{r['uncapped']:9.5f} {r['capped']:9.5f} {r['effect']:+11.4f}"
              f"  {r['rank_mismatch']:>5d} rank {r['no_k']:>4d} no-k")
    if missing_k:
        print(f"\n!! some queries have no k_estimate in {run_dir.name}: those "
              f"samples are dropped for a reason unrelated to judging.\n"
              f"   Point --k-from at the run that produced these submissions.")
    print("\ncap effect = capped - uncapped, semantic mean over rank-validated samples.")
    print("dropped: 'rank' = our parsed ordering did not reproduce the official rank")
    print("(the scorer skipped a document); 'no-k' = the --k-from run lacks that query.")
    print("Negative means uncapped judging scores HIGHER, i.e. the switch to official")
    print("judging adds that much on its own — before any agent-resampling difference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
