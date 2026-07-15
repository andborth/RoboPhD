#!/usr/bin/env python3
"""Judge calibration study: GPT-5.4-nano vs GPT-4o as the relevance judge.

Gates the training-judge switch (plan Workstream C). Training may switch to a
cheaper judge to cut cost, but the held-out test set is scored with GPT-4o. If
the two judges disagree materially, optimizing against the cheaper judge stops
predicting test performance — so measure agreement BEFORE switching.

Method: reconstruct real judge inputs from a completed run's diagnostics — each
semantic query's `relevance_criteria` (gold_criteria.md) paired with the
`markdown_evidence` each agent actually submitted (agent_output). Judge every
(criteria, evidence) doc with BOTH models and compare. This is the honest
comparison: identical inputs, two graders. `load_relevance_judgement` is pure
(it does not touch the verdict cache), so this script pollutes nothing.

Usage:
    python _check_judge_calibration.py --run-dir <run> [--n 150] [--seed 0]
        [--baseline openai/gpt-4o-2024-11-20] [--candidate openai/gpt-5.4-nano]

Requires OPENAI_API_KEY (both models are OpenAI). Makes 2*N real judge calls.
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BASELINE_DEFAULT = "openai/gpt-4o-2024-11-20"
CANDIDATE_DEFAULT = "openai/gpt-5.4-nano"

# Repair-path bookkeeping, so a run reports how much of a candidate's output
# needed rescuing rather than burying it in per-doc warnings.
_REPAIR = {"strict_ok": 0, "recovered": 0, "unrecoverable": 0}


def _lenient_extract_json(response: str):
    """Drop-in for astabench's extract_json_from_response that, when strict
    decoding fails, repairs the near-JSON some models emit — trailing commas
    and a dangling `,"` before a closing brace — then retries. Strict output
    is untouched. This isolates *format* non-compliance (which a production
    grader could eliminate with JSON mode) from genuine judgement differences,
    so the agreement number reflects the model's verdicts, not its punctuation.

    NB measurement-only: the live scorer uses the strict parser, so recovered
    docs would be DROPPED in production unless the grader call enforces JSON."""
    s = response[response.find("{"): response.rfind("}") + 1] if "{" in response else ""
    if not s:
        _REPAIR["unrecoverable"] += 1
        return None
    try:
        out = json.loads(s)
        _REPAIR["strict_ok"] += 1
        return out
    except json.JSONDecodeError:
        pass
    repaired = re.sub(r",(\s*[}\]])", r"\1", s)          # trailing commas
    repaired = re.sub(r',\s*"\s*([}\]])', r"\1", repaired)  # dangling ,"  before } ]
    try:
        out = json.loads(repaired)
        _REPAIR["recovered"] += 1
        return out
    except json.JSONDecodeError:
        _REPAIR["unrecoverable"] += 1
        return None


def _install_json_repair() -> None:
    """Patch the extractor the judge actually calls (relevance.py binds it by
    from-import, so patch that binding)."""
    from astabench.evals.paper_finder import relevance as rel
    rel.extract_json_from_response = _lenient_extract_json


def _extract_results_lenient(text: str):
    """Extract complete result objects from an agent_output that is truncated
    to 1000 chars (evaluator persists only the completion's head). Decodes
    objects from inside the `results` array until the truncation cuts one off,
    so we recover the top-ranked papers' real (paper_id, markdown_evidence)."""
    i = text.find('"results"')
    if i < 0:
        return []
    i = text.find("[", i)
    if i < 0:
        return []
    dec = json.JSONDecoder()
    out = []
    j = i + 1
    while j < len(text):
        while j < len(text) and text[j] in " ,\n\r\t":
            j += 1
        if j >= len(text) or text[j] != "{":
            break
        try:
            obj, end = dec.raw_decode(text, j)
        except ValueError:
            break  # truncated mid-object: stop at the last complete one
        if isinstance(obj, dict):
            out.append(obj)
        j = end
    return out


def _load_docs(run_dir: Path):
    """Yield (query_id, criteria_list, [(paper_id, evidence), ...]) per semantic
    query problem dir found under the run. Deduplicates (query, paper, evidence)
    triples so identical resubmissions are judged once."""
    seen: set[tuple] = set()
    per_query: dict[str, dict] = {}
    # followlinks=True: cached problem dirs are symlinks, which pathlib.rglob
    # would skip.
    gold_files = []
    for root, _dirs, files in os.walk(run_dir, followlinks=True):
        if "gold_criteria.md" in files:
            gold_files.append(Path(root) / "gold_criteria.md")
    for gc in gold_files:
        pdir = gc.parent
        if not pdir.name.startswith("semantic"):
            continue
        try:
            gold = json.loads(gc.read_text())
            criteria = gold.get("relevance_criteria") or []
            if not criteria:
                continue
            # agent_output is truncated to 1000 chars, so parse leniently.
            results = _extract_results_lenient((pdir / "agent_output").read_text())
        except OSError:
            continue
        qid = pdir.name
        entry = per_query.setdefault(qid, {"criteria": criteria, "docs": []})
        for r in results:
            pid = str(r.get("paper_id", "")).strip()
            ev = (r.get("markdown_evidence") or "").strip()
            if not pid or not ev:
                continue
            key = (qid, pid, ev)
            if key in seen:
                continue
            seen.add(key)
            entry["docs"].append((pid, ev))
    return per_query


async def _judge(model: str, criteria_raw: list, docs: list):
    """Judge docs (list of (pid, evidence)) with `model`; return {pid: label}."""
    from astabench.evals.paper_finder import relevance as rel
    from astabench.evals.paper_finder.relevance import Document, RelevanceCriterion

    criteria = [
        RelevanceCriterion(name=c["name"], description=c["description"], weight=c["weight"])
        for c in criteria_raw
    ]
    entities = [Document(corpus_id=pid, markdown=ev) for pid, ev in docs]
    prev = rel.GRADER_MODEL_NAME
    rel.GRADER_MODEL_NAME = model  # read at call time in _llm_judge_relevance_with_criteria
    try:
        return await rel.load_relevance_judgement(entities, criteria)
    finally:
        rel.GRADER_MODEL_NAME = prev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=150, help="max docs to judge")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", default=BASELINE_DEFAULT)
    ap.add_argument("--candidate", default=CANDIDATE_DEFAULT)
    ap.add_argument("--no-repair-json", action="store_true",
                    help="Disable the near-JSON repair fallback (see the strict "
                         "parser drop everything it can't decode, as production "
                         "would).")
    args = ap.parse_args()

    if not args.no_repair_json:
        _install_json_repair()

    from astabench.evals.paper_finder.relevance import bench_rj_2_pf_rj

    per_query = _load_docs(args.run_dir)
    flat = [(qid, e["criteria"], pid, ev)
            for qid, e in per_query.items() for pid, ev in e["docs"]]
    if not flat:
        print("No (criteria, evidence) docs found under", args.run_dir)
        return 1
    random.Random(args.seed).shuffle(flat)
    flat = flat[: args.n]
    # regroup the sample by query so each judge call uses the right criteria
    by_q: dict[str, dict] = {}
    for qid, crit, pid, ev in flat:
        g = by_q.setdefault(qid, {"criteria": crit, "docs": []})
        g["docs"].append((pid, ev))

    print(f"Judging {len(flat)} docs across {len(by_q)} queries")
    print(f"  baseline : {args.baseline}")
    print(f"  candidate: {args.candidate}")
    print(f"  json repair: {'off' if args.no_repair_json else 'on'}\n")

    pairs: list[tuple[str, str]] = []  # (baseline_label, candidate_label)
    # docs submitted vs verdicts returned per model — a verdict can go missing
    # from an unrecoverable JSON parse OR a criteria-name mismatch (the model
    # returned the wrong/renamed criteria). Both mean the paper is dropped.
    submitted = {args.baseline: 0, args.candidate: 0}
    returned = {args.baseline: 0, args.candidate: 0}
    for qid, g in by_q.items():
        b = asyncio.run(_judge(args.baseline, g["criteria"], g["docs"]))
        c = asyncio.run(_judge(args.candidate, g["criteria"], g["docs"]))
        for model, res in ((args.baseline, b), (args.candidate, c)):
            submitted[model] += len(g["docs"])
            returned[model] += sum(1 for pid, _ in g["docs"] if pid in res)
        for pid, _ in g["docs"]:
            if pid in b and pid in c:
                pairs.append((b[pid], c[pid]))

    if not pairs:
        print("No paired verdicts produced (judge parse failures?).")
        return 1

    n = len(pairs)
    exact = sum(1 for x, y in pairs if x == y) / n
    # Scoring-relevant binarization: only PERFECT counts toward F1.
    PERF = "perfectly_relevant_papers"
    b_perf = [x == PERF for x, _ in pairs]
    c_perf = [y == PERF for _, y in pairs]
    perf_agree = sum(1 for a, b in zip(b_perf, c_perf) if a == b) / n
    both_perf = sum(1 for a, b in zip(b_perf, c_perf) if a and b)
    # Cohen's kappa on the PERFECT/not binarization (chance-corrected).
    pb, pc = sum(b_perf) / n, sum(c_perf) / n
    p_e = pb * pc + (1 - pb) * (1 - pc)
    kappa = (perf_agree - p_e) / (1 - p_e) if (1 - p_e) else 1.0
    # Ordinal correlation on the 0-3 scale (Spearman without scipy).
    bx = [bench_rj_2_pf_rj.get(x, 0) for x, _ in pairs]
    cy = [bench_rj_2_pf_rj.get(y, 0) for _, y in pairs]
    spearman = _spearman(bx, cy)

    print(f"n paired verdicts        : {n}")
    print(f"exact label agreement    : {exact:.3f}")
    print(f"PERFECT/not agreement    : {perf_agree:.3f}")
    print(f"PERFECT/not Cohen kappa  : {kappa:.3f}")
    print(f"ordinal Spearman (0-3)   : {spearman:.3f}")
    print(f"baseline PERFECT rate    : {pb:.3f}  candidate PERFECT rate: {pc:.3f}")
    print(f"both-PERFECT count       : {both_perf}")
    print("\nconfusion (baseline \\ candidate):")
    _print_confusion(pairs)

    print("\njudge output reliability (docs that produced a usable verdict):")
    for model in (args.baseline, args.candidate):
        sub, ret = submitted[model], returned[model]
        drop = (sub - ret) / sub if sub else 0.0
        print(f"  {model:32} {ret}/{sub} verdicts  ({drop:.1%} dropped)")
    if not args.no_repair_json:
        r = _REPAIR
        print(f"  json repair: {r['strict_ok']} strict, {r['recovered']} recovered, "
              f"{r['unrecoverable']} unrecoverable")
        print("  NB drops that remain are criteria-name mismatches, not JSON — "
              "repair can't fix those. In production the strict parser would ALSO "
              "drop the 'recovered' docs unless the grader call enforces JSON mode.")

    print(
        "\nDecision: switch training to the candidate only if PERFECT/not kappa "
        "is high (≈0.7+) AND the PERFECT rates are close AND its verdict-drop "
        "rate is low. Recall/precision both hinge on the PERFECT label, so "
        "agreement THERE is what protects the train→test link; a high drop rate "
        "separately injects score noise (dropped papers count as not-relevant)."
    )
    return 0


def _spearman(a: list, b: list) -> float:
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    vb = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def _print_confusion(pairs):
    labels = ["perfectly_relevant_papers", "highly_relevant_papers",
              "somewhat_relevant_papers", "not_relevant_papers"]
    short = {l: l.split("_")[0][:5] for l in labels}
    cnt = Counter((x, y) for x, y in pairs)
    hdr = "        " + "".join(f"{short[l]:>7}" for l in labels)
    print(hdr)
    for x in labels:
        row = "".join(f"{cnt.get((x, y), 0):>7}" for y in labels)
        print(f"{short[x]:>7} {row}")


if __name__ == "__main__":
    sys.exit(main())
