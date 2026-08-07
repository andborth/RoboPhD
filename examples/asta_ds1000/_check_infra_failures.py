#!/usr/bin/env python3
"""Post-hoc detector for evaluations that scored 0 because the HARNESS broke.

Docker/OrbStack wedges, subprocess timeouts, OOM kills and provider quota
errors zero out problems the agent never got to attempt. In the score those
zeros are indistinguishable from wrong answers, so they commit into Elo and
distort selection. Observed twice at full strength: asta_ds1000_20260610_203253
(10/20 zeroed, a 90%-accurate agent recorded at 45.0) and
asta_ds1000_20260720_163409 (461/900 test problems zeroed, 0.7722 recorded as
0.3733). Both were caught by eye; a partial wedge of three or four timeouts
would commit unnoticed.

The run's own error accounting cannot see any of this. THREE signals are dead:
  * `result.json["error"]` is always false — core computes
    `has_error = "error.md" in diagnostics` while this evaluator emits the
    key "error", so the flag never fires.
  * `iteration_*/error_index.json` `total_errors` is 0 in every entry of every
    run, for the same reason.
  * `checkpoint.json`'s zero_accuracy_cases / exception_failures /
    five_hour_limit_incidents are empty in every DS-1000 run.
The surviving evidence is the `error` FILE beside each result.json, plus a
zero-`eval_cost` fingerprint for the failures that never wrote one (an agent
that never ran spends nothing; every genuine wrong answer in the catalogue
cost money).

WHAT THIS DOES NOT ASSERT. It does not decide that a flagged round is
worthless, and it does not re-score anything: the `adjusted` mean it prints is
"the mean over the evals that actually ran", which is an estimate of what the
round would have recorded, not a measurement. It cannot see a wedge that broke
the AGENT's own sandbox calls rather than the evaluator's subprocess — that
surfaces as a genuine low score with no error file and real spend, and is
indistinguishable from the agent being wrong.

READ-ONLY, and the report goes to stdout — never into a run directory. That is
a design constraint, not an accident: evolution sessions are handed the
experiment directory and read what they find there, so a report file written
inside a run could teach evolution about Docker state and invite it to evolve
retry logic against the harness instead of the task. Do not add a
--write-report flag; redirect stdout instead. For the same reason none of this
belongs in `aggregate_explanation`, which flows into evolution_prompt.md.

Unlike its sibling _check_* scripts this file has NO sys.path block and imports
nothing from the task or from RoboPhD — it reads run artifacts off disk and
nothing else, so it cannot trigger an evaluator or a model call as a side
effect. Everything below the TASK CONFIG block is generic over the core domain
layout; copying this to docfinqa/arc_agi_1/sudoku is an edit of that block.

Usage:
    python _check_infra_failures.py --run-dir <run> [<run> ...]
    python _check_infra_failures.py --runs-glob '/path/to/robophd_runs/*/asta_ds1000_*'
    python _check_infra_failures.py --run-dir <run> --include-archived
"""

import argparse
import glob as globmod
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ===========================================================================
# TASK CONFIG — the only DS-1000-specific values. Edit when copying.
# ===========================================================================

# Positive allow-list, matched by SUBSTRING. Substring and not startswith
# because test-path errors are head+tail truncated by main.py::_trunc
# (head=100, tail=400, middle elided), so anchored matching fails on the long
# docker tracebacks. An allow-list and not a deny-list because genuine agent
# bugs also write the error file — NameError("name 'win_cands' is not
# defined") and IndexError('Replacement index 0 out of range...') both appear
# in the catalogue and must not be swallowed as infrastructure.
INFRA_TOKENS = (
    "subprocess timed out after",            # evaluator.py:940, subprocess wall clock
    "subprocess failed (exit -",             # negative exit == signal; -9 == SIGKILL == OOM
    "subprocess produced no valid output",   # evaluator.py:955
    "Docker compose command",                # compose up --wait timed out
    "Failed to create container directory",  # docker.sock unreachable mid-eval
    "No services started.",                  # compose up produced nothing
    "PrerequisiteError",                     # inspect's pre-flight docker gate
    "docker.sock",                           # any orbstack/docker daemon socket failure
    "timeout after ",                        # older domain-level format
)

# Read from agent_stdout for rows ALREADY flagged infra, to name the cause when
# no error file exists. This is what identifies the OpenAI 429 quota wedge in
# asta_ds1000_20260707_033647, which wrote no error file at all.
PROVIDER_TOKENS = (
    "RateLimitError", "insufficient_quota", "AuthenticationError",
    "APIConnectionError", "PermissionDeniedError", "InternalServerError",
    "overloaded_error",
)

TEST_GLOB = "test_results_*"


def score_ok(score) -> bool:
    """True when a per-problem score counts as a pass.

    NOT `score == 1`: six early runs record scores on a 0-100 scale, and four
    of those apply a per-problem cost penalty that pushes raw-zero problems
    strictly NEGATIVE (35 such rows exist). NOT `score == 0` on the failure
    side either, for the same reason. There is no partial credit anywhere in
    the catalogue, so `> 0` is exact on both scales.
    """
    return score > 0


# ===========================================================================
# Layout constants — generic over the core external-domain shape
# ===========================================================================

PRUNE_DIRS = {".git", ".claude", "__pycache__", "node_modules", "debug_logs"}
ZERO_COST_EPS = 1e-9
TRUNC = 200          # inline truncation for printed error text
CAUSE_WIDTH = 108    # width of a collapsed cause line


# ---------------------------------------------------------------------------
# Tiny helpers — none of these raise
# ---------------------------------------------------------------------------

def _read_json(path: Path):
    """Parsed JSON, or None on any read/parse failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _read_text(path: Path, limit: int = 200_000) -> str:
    """File text (truncated), or "" on any read failure."""
    try:
        with open(path, errors="replace") as f:
            return f.read(limit)
    except OSError:
        return ""


def _infra_token(text: str):
    """First INFRA_TOKENS member contained in `text`, else None."""
    if not text:
        return None
    for tok in INFRA_TOKENS:
        if tok in text:
            return tok
    return None


def _provider_cause(problem_dir: Path) -> str:
    """First agent_stdout line naming a provider error, else "".

    Called only for rows already classified infra, so the extra read is
    bounded by the number of failures, not by catalogue size.
    """
    text = _read_text(problem_dir / "agent_stdout", limit=60_000)
    if not text:
        return ""
    for line in text.splitlines():
        if any(tok in line for tok in PROVIDER_TOKENS):
            return line.strip()[:CAUSE_WIDTH]
    return ""


def _collapse(texts, width: int = CAUSE_WIDTH):
    """[(count, text[:width])] sorted by count desc.

    Turns 461 identical docker tracebacks into one line.
    """
    counts = Counter(t.replace("\n", " ")[:width] for t in texts if t)
    return [(n, t) for t, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def _mean(values):
    return (sum(values) / len(values)) if values else None


def _fmt_mean(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


# ---------------------------------------------------------------------------
# Training path — RoboPhD engine only
# ---------------------------------------------------------------------------

def find_problem_cells(run_dir: Path, include_archived: bool):
    """Every `problems/` directory under `run_dir`, one per (iteration, agent).

    Keys on the directory NAME rather than a path template, so all four
    observed nestings are covered by one rule: iteration_N/agent_X/problems,
    evolution_output/iteration_N/iteration_M_test/agent_X/problems, and the
    archived_* mirrors of both.

    os.walk defaults to followlinks=False but still LISTS symlinked dirs in
    `dirnames`; pruning them is what prevents ever visiting a cached tree.
    Descent stops at `problems/` itself, which is what keeps a full-catalogue
    scan to a few seconds instead of stat-ing ~340k files.
    """
    cells = []
    for dirpath, dirnames, _files in os.walk(run_dir):
        here = Path(dirpath)
        dirnames[:] = [
            n for n in dirnames
            if n not in PRUNE_DIRS
            and not (here / n).is_symlink()
            and (include_archived or not n.startswith("archived_"))
        ]
        if here.name == "problems":
            cells.append(here)
            dirnames[:] = []
    return sorted(cells)


def scan_cell(problems_dir: Path) -> dict:
    """Read one (iteration, agent) cell into rows, skipping cached symlinks.

    `is_symlink()` is checked STRICTLY BEFORE `is_dir()`: os.DirEntry.is_dir()
    follows links by default, so the other order would walk into the cache and
    count one infra failure again in every later iteration that reused it.
    Checking is_symlink() first also handles dangling symlinks for free (it is
    True regardless of whether the target still exists), which is why this does
    not de-duplicate by resolve() — that raises on exactly those.
    """
    rows, cached_qids = [], []
    try:
        entries = sorted(os.scandir(problems_dir), key=lambda e: e.name)
    except OSError:
        return {"dir": problems_dir, "rows": [], "cached_qids": []}

    for entry in entries:
        if entry.is_symlink():
            # Cached: never classified (it was produced in an earlier, healthy
            # round), but its SCORE still belongs to this round's mean — the
            # recorded average_score covers fresh and cached alike, and a
            # fresh-only mean would read 0.00 for a cell whose passes were all
            # cache hits. Scores are resolved in classify_cell from
            # evaluation.json, falling back to the symlink target.
            cached_qids.append(entry.name)
            continue
        if not entry.is_dir(follow_symlinks=False):
            continue
        pdir = Path(entry.path)
        result = _read_json(pdir / "result.json")
        score = result.get("score") if isinstance(result, dict) else None
        cost = result.get("eval_cost") if isinstance(result, dict) else None
        rows.append({
            "qid": entry.name,
            "score": score if isinstance(score, (int, float)) else None,
            "cost": cost if isinstance(cost, (int, float)) else None,
            "err": _read_text(pdir / "error", limit=20_000).strip(),
            "dir": pdir,
        })
    return {"dir": problems_dir, "rows": rows, "cached_qids": cached_qids}


def classify_cell(cell: dict, run_dir: Path, min_cluster: int) -> dict:
    """Bucket every row, then label the cell.

    The chain is strict if/elif, so the buckets are mutually exclusive by
    construction and a row can never be double-counted. In particular a row
    with BOTH an error file and zero cost (the common wedge shape) lands in
    `error-file`; the zero-cost branch only ever sees rows the error-file
    branch declined.
    """
    infra, unclassified, genuine, incomplete, ok = [], [], [], [], []
    causes = []

    for row in cell["rows"]:
        if row["score"] is None:
            incomplete.append(row)
            continue
        if score_ok(row["score"]):
            ok.append(row)
            continue
        if row["err"]:
            if _infra_token(row["err"]):
                infra.append(row)
                causes.append(row["err"].splitlines()[0] if row["err"] else "")
            else:
                unclassified.append(row)
            continue
        if row["cost"] is not None and row["cost"] <= ZERO_COST_EPS:
            infra.append(row)
            causes.append(_provider_cause(row["dir"]) or "zero eval_cost, no agent spend, no error file")
            continue
        genuine.append(row)

    # evaluation.json is a CROSS-CHECK and a score source for cached rows, but
    # never the walk's source: four agent cells in the catalogue have none at
    # all because they were killed mid-eval, and those are exactly the cells
    # most worth reporting.
    recorded, cache_note = None, None
    ev_results = {}
    ev = _read_json(cell["dir"].parent / "evaluation.json")
    n_cached = len(cell["cached_qids"])
    if isinstance(ev, dict):
        summary = ev.get("summary") or {}
        recorded = summary.get("average_score")
        results = ev.get("results") or {}
        if isinstance(results, dict):
            ev_results = results
            marked = sum(1 for r in results.values()
                         if isinstance(r, dict) and r.get("cached"))
            if marked != n_cached:
                cache_note = (f"evaluation.json marks {marked} cached, "
                              f"{n_cached} symlinks on disk")

    # Cached scores: one evaluation.json read covers the whole cell; only fall
    # back to following the symlink when that file is missing.
    cached_scores = []
    for qid in cell["cached_qids"]:
        entry = ev_results.get(qid)
        score = entry.get("score") if isinstance(entry, dict) else None
        if not isinstance(score, (int, float)):
            target = _read_json(cell["dir"] / qid / "result.json")
            score = target.get("score") if isinstance(target, dict) else None
        if isinstance(score, (int, float)):
            cached_scores.append(score)

    fresh_scored = [r["score"] for r in cell["rows"] if r["score"] is not None]
    fresh_survivors = [r["score"] for r in (ok + genuine + unclassified)
                       if r["score"] is not None]
    scored = fresh_scored + cached_scores
    survivors = fresh_survivors + cached_scores

    try:
        rel = str(cell["dir"].parent.relative_to(run_dir))
    except ValueError:
        rel = str(cell["dir"].parent)

    return {
        "rel": rel,
        "archived": any(p.startswith("archived_") for p in Path(rel).parts),
        "n_fresh": len(cell["rows"]),
        "n_cached": n_cached,
        "n_survivors": len(survivors),
        "n_infra": len(infra),
        "n_unclassified": len(unclassified),
        "n_genuine": len(genuine),
        "n_incomplete": len(incomplete),
        "n_ok": len(ok),
        "raw_mean": _mean(scored),
        "adjusted_mean": _mean(survivors),
        "recorded": recorded,
        "cache_note": cache_note,
        "causes": _collapse(causes),
        "infra_qids": [r["qid"] for r in infra],
        "unclassified": [(r["qid"], r["err"].replace("\n", " ")[:TRUNC]) for r in unclassified],
        "wedge": len(infra) >= min_cluster,
    }


def scan_training(run_dir: Path, include_archived: bool, min_cluster: int) -> dict:
    """Whole-run training-path scan.

    Returns zero cells for GEPA and Autoresearch runs — those engines write no
    iteration_*/agent_*/problems/* artifacts at all. That is reported, not
    failed.
    """
    cells = [classify_cell(scan_cell(p), run_dir, min_cluster)
             for p in find_problem_cells(run_dir, include_archived)]
    totals = defaultdict(int)
    for c in cells:
        for key in ("n_fresh", "n_cached", "n_infra", "n_unclassified",
                    "n_genuine", "n_incomplete", "n_ok"):
            totals[key] += c[key]
    return {
        "cells": cells,
        "wedges": [c for c in cells if c["wedge"]],
        "singletons": [c for c in cells if c["n_infra"] and not c["wedge"]],
        # A cell with unreadable result.json files and no error file was killed
        # mid-eval — itself an infra tell, and invisible to the buckets above.
        "incomplete_cells": [c for c in cells if c["n_incomplete"] and not c["n_infra"]],
        "n_cells": len(cells),
        **totals,
    }


# ---------------------------------------------------------------------------
# Test path — all three engines
# ---------------------------------------------------------------------------

def find_test_artifacts(run_dir: Path):
    """Discover and pair test_results_* files.

    Discrimination is by PARSED JSON SHAPE, never by filename: four historical
    naming variants exist (`.per_problem.json`,
    `.per_problem.budget500_iter13.json`, `.json.suspect-orbstack-failure`, and
    `.per_problem.summary.json` — which is a DICT despite the name). A
    list-of-dicts is per-problem records; a dict is a summary. Iterating a
    summary as if it were records yields str keys and an AttributeError.
    """
    per_problem, summaries, unreadable = {}, {}, []
    for path in sorted(run_dir.glob(TEST_GLOB)):
        if not path.is_file():
            continue
        doc = _read_json(path)
        if isinstance(doc, list) and doc and isinstance(doc[0], dict):
            per_problem[path.name] = (path, doc)
        elif isinstance(doc, dict):
            summaries[path.name] = (path, doc)
        else:
            unreadable.append(path.name)

    artifacts, paired_summaries = [], set()
    for name, (path, records) in per_problem.items():
        summary_name = None
        for sname, (_sp, sdoc) in summaries.items():
            if sdoc.get("per_problem_file") == name:
                summary_name = sname
                break
        if summary_name is None:
            candidate = name.replace(".per_problem", "", 1)
            if candidate in summaries:
                summary_name = candidate
        summary = summaries[summary_name][1] if summary_name else None
        if summary_name:
            paired_summaries.add(summary_name)
        artifacts.append({
            "name": summary_name or name,
            "pp_name": name,
            "records": records,
            "summary": summary,
        })

    for sname, (_sp, sdoc) in summaries.items():
        if sname not in paired_summaries:
            artifacts.append({"name": sname, "pp_name": None,
                              "records": None, "summary": sdoc})

    return sorted(artifacts, key=lambda a: a["name"]), unreadable


def scan_test_artifact(art: dict, test_wedge_rate: float) -> dict:
    """Classify one test artifact's records and assign a verdict.

    Every headline is RECOMPUTED from the records rather than read from the
    summary: `n_fallback_used` is None on all legacy summaries, including one
    whose records carry 25 real errors.

    The (error, primary_error) pair names the failure mode. `primary_error`
    set with `error` unset is the dangerous one — the primary agent died and
    the SEED's score was substituted, so the artifact reports a healthy number
    that is partly not the candidate's work.
    """
    out = {
        "name": art["name"], "pp_name": art["pp_name"],
        "n": 0, "n_infra_hard": 0, "n_fallback_sub": 0, "n_unclassified": 0,
        "n_null_sample": 0, "raw_mean": None, "adjusted_mean": None,
        "summary_mean": None, "cost": None, "reported_fallback": None,
        "causes": [], "legacy": False, "verdict": "PASS", "records_seen": False,
    }
    summary = art["summary"] or {}
    out["summary_mean"] = summary.get("mean_test_score")
    out["cost"] = summary.get("test_eval_cost_usd")
    out["reported_fallback"] = summary.get("n_fallback_used")

    records = art["records"]
    if records is None:
        out["verdict"] = "PASS"
        return out
    out["records_seen"] = True

    causes, scored, survivors = [], [], []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        out["n"] += 1
        if "fallback_used" not in rec and "primary_error" not in rec:
            out["legacy"] = True
        err = rec.get("error") or ""
        perr = rec.get("primary_error") or ""
        score = rec.get("score")
        if isinstance(score, (int, float)):
            scored.append(score)
        if rec.get("sample_id") is None:
            out["n_null_sample"] += 1

        bucket = "ok"
        if err:
            # Primary died; if a fallback also errored or none was configured,
            # the score was forced to 0 either way.
            if _infra_token(err) or _infra_token(perr):
                bucket = "infra_hard"
            else:
                bucket = "unclassified"
            causes.append(err)
        elif perr:
            # Primary died, fallback answered: the reported score is the
            # SEED's, not the candidate's.
            if _infra_token(perr):
                bucket = "fallback_sub"
            else:
                bucket = "unclassified"
            causes.append(perr)

        if bucket == "infra_hard":
            out["n_infra_hard"] += 1
        elif bucket == "fallback_sub":
            out["n_fallback_sub"] += 1
        elif bucket == "unclassified":
            out["n_unclassified"] += 1
        elif isinstance(score, (int, float)):
            survivors.append(score)

    out["raw_mean"] = _mean(scored)
    out["adjusted_mean"] = _mean(survivors)
    out["causes"] = _collapse(causes)

    n = out["n"] or 1
    infra_rate = out["n_infra_hard"] / n
    fallback_rate = out["n_fallback_sub"] / n
    out["infra_rate"], out["fallback_rate"] = infra_rate, fallback_rate
    worst = max(infra_rate, fallback_rate)
    # max() of BOTH rates, not infra alone: an artifact can report a perfectly
    # healthy mean while a tenth of its scores belong to the seed.
    out["verdict"] = ("FAIL" if worst >= test_wedge_rate
                      else "WARN" if worst > 0 else "PASS")
    return out


def scan_test(run_dir: Path, test_wedge_rate: float) -> dict:
    artifacts, unreadable = find_test_artifacts(run_dir)
    scanned = [scan_test_artifact(a, test_wedge_rate) for a in artifacts]
    # Cost baseline for a flagged artifact: the healthiest sibling in the same
    # run. Absolute cost thresholds are meaningless here — legitimate values
    # span $0.045 to $88.22 across the catalogue — but a sibling comparison
    # inside one run is decisive.
    clean = [a for a in scanned if a["verdict"] == "PASS" and a["cost"]]
    for art in scanned:
        art["sibling"] = None
        if art["verdict"] in ("FAIL", "WARN"):
            for other in clean:
                # Same problem count or the comparison is meaningless — a
                # 90-problem experiment eval is not a cost baseline for a
                # 900-problem final one.
                if other["name"] != art["name"] and other["n"] == art["n"]:
                    art["sibling"] = other
                    break
    return {
        "artifacts": scanned,
        "unreadable": unreadable,
        "n_fail": sum(1 for a in scanned if a["verdict"] == "FAIL"),
        "n_warn": sum(1 for a in scanned if a["verdict"] == "WARN"),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def resolve_run_dirs(args):
    """Union of --run-dir and expanded --runs-glob, de-duped, existing only."""
    paths = []
    for group in (args.run_dir or []):
        paths.extend(group)
    for pattern in (args.runs_glob or []):
        paths.extend(globmod.glob(pattern))
    seen, out = set(), []
    for p in paths:
        path = Path(p)
        key = str(path.resolve())
        if key in seen or not path.is_dir():
            continue
        seen.add(key)
        out.append(path)
    return sorted(out)


def scan_run(run_dir: Path, args) -> dict:
    training = ({"cells": [], "wedges": [], "singletons": [], "n_cells": 0}
                if args.skip_training
                else scan_training(run_dir, args.include_archived, args.min_cluster))
    test = ({"artifacts": [], "unreadable": [], "n_fail": 0, "n_warn": 0}
            if args.skip_test else scan_test(run_dir, args.test_wedge_rate))
    scannable = bool(training.get("n_cells") or test["artifacts"])
    return {
        "run_dir": str(run_dir),
        "name": run_dir.name,
        "engine": run_dir.parent.name,
        "training": training,
        "test": test,
        "scannable": scannable,
        "failed": bool(training.get("wedges") or test["n_fail"]),
    }


def _print_cell(cell: dict, args, indent: str = "  "):
    label = "WEDGE" if cell["wedge"] else "infra"
    print(f"{indent}{label}  {cell['rel']}")
    cached = f"  ({cell['n_cached']} more were cached; round size "\
             f"{cell['n_fresh'] + cell['n_cached']})" if cell["n_cached"] else ""
    print(f"{indent}       {cell['n_infra']} of {cell['n_fresh']} fresh evals "
          f"infra-zeroed{cached}")
    print(f"{indent}       raw_mean {_fmt_mean(cell['raw_mean'])}  ->  adjusted "
          f"{_fmt_mean(cell['adjusted_mean'])} over the "
          f"{cell['n_survivors']} surviving evals")
    if cell["recorded"] is not None:
        print(f"{indent}       evaluation.json recorded average_score {cell['recorded']}")
    for count, text in cell["causes"][: args.max_detail]:
        print(f"{indent}       {count}x  {text}")
    if cell["infra_qids"]:
        shown = " ".join(cell["infra_qids"][: args.max_detail])
        more = "" if len(cell["infra_qids"]) <= args.max_detail else " ..."
        print(f"{indent}       problems: {shown}{more}")
    if cell["n_incomplete"]:
        print(f"{indent}       {cell['n_incomplete']} problem dirs have no readable "
              f"result.json (cell killed mid-eval)")
    if cell["archived"]:
        print(f"{indent}       NOTE inside archived_* — a rolled-back iteration that "
              f"was re-run; compare the live cell of the same agent")
    if cell["cache_note"]:
        print(f"{indent}       note: {cell['cache_note']}")


def _print_test_artifact(art: dict, args, indent: str = "  "):
    if not art["records_seen"]:
        print(f"{indent}{art['name']:<44} PASS  summary only, no per-problem sidecar"
              f" (mean {_fmt_mean(art['summary_mean'])})")
        return
    print(f"{indent}{art['name']:<44} {art['verdict']:<4}  {art['n']} evals, "
          f"{art['n_infra_hard']} infra, {art['n_fallback_sub']} fallback, "
          f"mean {_fmt_mean(art['raw_mean'])}"
          f"{'  [legacy schema]' if art['legacy'] else ''}")
    if art["verdict"] == "PASS" and not art["n_unclassified"]:
        return
    pad = indent + "    "
    if art["n_infra_hard"] or art["n_fallback_sub"]:
        print(f"{pad}mean {_fmt_mean(art['raw_mean'])}  ->  adjusted "
              f"{_fmt_mean(art['adjusted_mean'])} over the "
              f"{art['n'] - art['n_infra_hard'] - art['n_fallback_sub']} surviving evals")
    if art["n_infra_hard"]:
        print(f"{pad}{art['n_infra_hard']} hard failures (primary and fallback both "
              f"errored -> score forced to 0)")
    if art["n_fallback_sub"]:
        print(f"{pad}{art['n_fallback_sub']} fallback substitutions (reported score is "
              f"the SEED's, not the candidate's)")
    if art["n_unclassified"]:
        print(f"{pad}{art['n_unclassified']} error records match no infra token "
              f"— classify by hand")
    cost_str = "n/a" if art["cost"] is None else f"${art['cost']:.4f}"
    print(f"{pad}n_fallback_used {art['reported_fallback']}   null sample_id "
          f"{art['n_null_sample']}/{art['n']}   test_eval_cost_usd {cost_str}")
    for count, text in art["causes"][: args.max_detail]:
        print(f"{pad}{count}x  {text}")
    sib = art.get("sibling")
    if sib and sib["cost"] and art["cost"]:
        pct = 100.0 * art["cost"] / sib["cost"]
        print(f"{pad}sibling in this run: {sib['name']}  mean "
              f"{_fmt_mean(sib['raw_mean'])}  ${sib['cost']:.4f}  "
              f"-> cost collapsed to {pct:.0f}% of the sibling")


def print_run_report(rep: dict, args):
    print()
    print(f"== {rep['name']} ({rep['engine']}) ==")
    if not rep["scannable"]:
        print("nothing to scan — no problem artifacts and no test_results_* files")
        return

    tr = rep["training"]
    if not args.skip_training:
        print()
        print("-- training path --")
        if not tr["n_cells"]:
            print("0 cells (engine writes no training-path problem artifacts)")
        else:
            print(f"{tr['n_cells']} iteration/agent cells, {tr['n_fresh']} fresh problem "
                  f"evals ({tr['n_cached']} cached symlinks skipped)")
            verdict = "FAIL" if tr["wedges"] else "PASS"
            wedge_note = (f"{tr['n_infra']} in {len([c for c in tr['cells'] if c['n_infra']])} "
                          f"cell(s), {len(tr['wedges'])} above the min-cluster "
                          f"{args.min_cluster} threshold") if tr["n_infra"] else "0"
            print(f"  infra-zeroed evals       {verdict}  {wedge_note}")
            print(f"  unclassified errors      {'WARN' if tr['n_unclassified'] else 'PASS'}"
                  f"  {tr['n_unclassified']}")
            print(f"  incomplete problem dirs  {'WARN' if tr['n_incomplete'] else 'PASS'}"
                  f"  {tr['n_incomplete']}")
            print(f"  genuine zeros            --    {tr['n_genuine']}  "
                  f"(score<=0, agent spent money, no error file)")
            print()
            for cell in tr["wedges"] + tr["singletons"]:
                _print_cell(cell, args)
            for cell in tr.get("incomplete_cells", []):
                print(f"  killed  {cell['rel']}")
                print(f"          {cell['n_incomplete']} of {cell['n_fresh']} problem dirs "
                      f"have no readable result.json"
                      f"{' and the cell has no evaluation.json' if cell['recorded'] is None else ''}")
            for cell in tr["cells"]:
                for qid, text in cell["unclassified"]:
                    print(f"  unclassified  {cell['rel']}  q{qid}")
                    print(f"                {text}")

    if not args.skip_test:
        print()
        print("-- test path --")
        if not rep["test"]["artifacts"]:
            print("no test_results_* artifacts in this run")
        for art in rep["test"]["artifacts"]:
            _print_test_artifact(art, args)
        for name in rep["test"]["unreadable"]:
            print(f"  {name:<44} SKIP  unreadable JSON")


def print_batch_report(reports, args):
    flagged = [r for r in reports if r["failed"]]
    engines = Counter(r["engine"] for r in reports)
    print("== DS-1000 infrastructure-failure audit ==")
    print(f"scope       {len(reports)} run dirs   " +
          "   ".join(f"{k} {v}" for k, v in sorted(engines.items())))
    print(f"archives    {'included' if args.include_archived else 'skipped'}"
          f"{'' if args.include_archived else ' (pass --include-archived for rolled-back iterations)'}")
    print(f"thresholds  cell WEDGE at >={args.min_cluster} infra evals   "
          f"test WEDGE at >={args.test_wedge_rate:.1%} infra-or-fallback rate")

    print()
    print("-- flagged runs " + "-" * 62)
    if not flagged:
        print("none")
    else:
        print(f"{'run':<46}{'cells':>6}{'infra':>7}{'wedge':>7}{'arts':>6}"
              f"{'infra':>7}{'fb':>5}{'wedge':>7}")
        for r in sorted(flagged, key=lambda x: x["name"]):
            tr, te = r["training"], r["test"]
            arts = te["artifacts"]
            print(f"{r['engine'] + '/' + r['name']:<46}"
                  f"{tr.get('n_cells', 0):>6}{tr.get('n_infra', 0):>7}"
                  f"{len(tr.get('wedges', [])):>7}{len(arts):>6}"
                  f"{sum(a['n_infra_hard'] for a in arts):>7}"
                  f"{sum(a['n_fallback_sub'] for a in arts):>5}{te['n_fail']:>7}")
    clean = len(reports) - len(flagged)
    if clean:
        print(f"({clean} further run{'s' if clean != 1 else ''} clean)")

    print()
    print(f"-- training wedges (>={args.min_cluster} infra evals in one cell) " + "-" * 30)
    any_wedge = False
    for r in reports:
        for cell in r["training"].get("wedges", []):
            any_wedge = True
            print(f"FAIL  {r['name']}")
            _print_cell(cell, args, indent="      ")
    if not any_wedge:
        print("none")

    singles = [(r, c) for r in reports for c in r["training"].get("singletons", [])]
    if singles:
        print()
        print(f"-- singleton infra evals (below min-cluster {args.min_cluster}; "
              f"not exit-code failing) " + "-" * 10)
        for r, cell in singles:
            print(f"{r['name']}  {cell['rel']}  "
                  f"q{' q'.join(cell['infra_qids'][: args.max_detail])}")

    recurring = Counter()
    for r in reports:
        for cell in r["training"].get("cells", []):
            for qid in cell["infra_qids"]:
                recurring[qid] += 1
    repeats = [(q, n) for q, n in recurring.items() if n >= 3]
    if repeats:
        print()
        print("-- recurring problem ids (same question infra-failed in >=3 cells) " + "-" * 15)
        for qid, n in sorted(repeats, key=lambda kv: -kv[1]):
            print(f"question {qid}   {n} cells   -> pathological problem, not a round wedge")

    print()
    print(f"-- test wedges (>={args.test_wedge_rate:.1%} infra-or-fallback rate) " + "-" * 30)
    any_test = False
    for r in reports:
        for art in r["test"]["artifacts"]:
            if art["verdict"] == "FAIL":
                any_test = True
                print(f"FAIL  {r['name']}")
                _print_test_artifact(art, args, indent="      ")
    if not any_test:
        print("none")

    unclassified = [(r, c, q, t) for r in reports
                    for c in r["training"].get("cells", [])
                    for q, t in c["unclassified"]]
    if unclassified:
        print()
        print("-- unclassified error files (error present, no infra token) " + "-" * 20)
        for r, cell, qid, text in unclassified:
            print(f"{r['name']}  {cell['rel']}  q{qid}")
            print(f"    {text}")
        print("-> classify by hand. If one is infra, add its token to INFRA_TOKENS "
              "and re-run —")
        print("   an unclassified error is how a NEW infra family first appears.")

    tot_fresh = sum(r["training"].get("n_fresh", 0) for r in reports)
    tot_infra = sum(r["training"].get("n_infra", 0) for r in reports)
    tot_wedge = sum(len(r["training"].get("wedges", [])) for r in reports)
    tot_records = sum(a["n"] for r in reports for a in r["test"]["artifacts"])
    tot_test_bad = sum(a["n_infra_hard"] + a["n_fallback_sub"]
                       for r in reports for a in r["test"]["artifacts"])
    tot_test_fail = sum(r["test"]["n_fail"] for r in reports)

    print()
    print("== Summary ==")
    print(f"{len(reports)} runs scanned.  {len(flagged)} carry an infra fingerprint; "
          f"{clean} clean.")
    pct = (100.0 * tot_infra / tot_fresh) if tot_fresh else 0.0
    print(f"training  {tot_infra} of {tot_fresh} fresh evals ({pct:.2f}%) infra-zeroed; "
          f"{tot_wedge} cell(s) exceed the min-cluster {args.min_cluster} threshold.")
    print(f"test      {tot_test_bad} of {tot_records} records affected; "
          f"{tot_test_fail} artifact(s) exceed the {args.test_wedge_rate:.1%} rate.")
    if tot_wedge or tot_test_fail:
        print(f"FAIL: {tot_wedge} training wedge(s) + {tot_test_fail} test wedge(s). "
              f"Discount or re-run those rounds before trusting their Elo.")
    else:
        print("PASS: no round exceeds either threshold.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", nargs="+", action="append", metavar="PATH",
                    help="run directory to scan; repeatable, accepts shell-expanded globs")
    ap.add_argument("--runs-glob", action="append", metavar="PATTERN",
                    help="glob expanded by the script (quote it); repeatable")
    ap.add_argument("--include-archived", action="store_true",
                    help="also scan archived_*/ (rolled-back iterations that were "
                         "re-run; off by default so counts reflect what reached Elo)")
    ap.add_argument("--min-cluster", type=int, default=3, metavar="N",
                    help="infra evals in one iteration/agent cell before it is a WEDGE")
    ap.add_argument("--test-wedge-rate", type=float, default=0.02, metavar="R",
                    help="infra-or-fallback fraction of a test eval before it is WEDGED")
    ap.add_argument("--max-detail", type=int, default=12, metavar="N",
                    help="cap per-cell problem-id and cause lines")
    ap.add_argument("--strict", action="store_true",
                    help="also exit 1 on unclassified errors and singleton infra evals")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-run blocks; print the batch summary only")
    ap.add_argument("--json", action="store_true",
                    help="emit one machine-readable JSON object on stdout")
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

    run_dirs = resolve_run_dirs(args)
    if not run_dirs:
        patterns = (args.runs_glob or []) + [p for g in (args.run_dir or []) for p in g]
        print("No run directories resolved from:", ", ".join(patterns) or "(nothing given)")
        print("Pass --run-dir <path> or --runs-glob '<pattern>'.")
        return 1

    reports = [scan_run(d, args) for d in run_dirs]

    n_wedges = sum(len(r["training"].get("wedges", [])) for r in reports)
    n_test_fail = sum(r["test"]["n_fail"] for r in reports)
    n_unclassified = sum(r["training"].get("n_unclassified", 0) for r in reports)
    n_singleton = sum(len(r["training"].get("singletons", [])) for r in reports)
    # Singletons and unclassified errors are listed but never fail by default:
    # single timeouts happen, and a permanently-red audit is a dead signal.
    code = 1 if (n_wedges or n_test_fail
                 or (args.strict and (n_unclassified or n_singleton))) else 0

    if args.json:
        print(json.dumps({
            "runs": [{
                "name": r["name"], "engine": r["engine"], "run_dir": r["run_dir"],
                "failed": r["failed"],
                "training": {k: v for k, v in r["training"].items()
                             if k not in ("cells", "wedges", "singletons")},
                "wedges": [{k: v for k, v in c.items() if k != "unclassified"}
                           for c in r["training"].get("wedges", [])],
                "test": [{k: v for k, v in a.items() if k != "sibling"}
                         for a in r["test"]["artifacts"]],
            } for r in reports],
            "n_training_wedges": n_wedges,
            "n_test_wedges": n_test_fail,
            "exit_code": code,
        }, indent=2, default=str))
        return code

    if len(reports) > 1 or args.quiet:
        if not args.quiet:
            for rep in reports:
                print_run_report(rep, args)
            print()
        print_batch_report(reports, args)
    else:
        print_run_report(reports[0], args)
        print()
        tr, te = reports[0]["training"], reports[0]["test"]
        print("== Summary ==")
        print(f"{tr.get('n_infra', 0)}/{tr.get('n_fresh', 0)} training evals and "
              f"{sum(a['n_infra_hard'] + a['n_fallback_sub'] for a in te['artifacts'])}/"
              f"{sum(a['n'] for a in te['artifacts'])} test evals carry an infra "
              f"fingerprint.")
        if code:
            print(f"FAIL: {len(tr.get('wedges', []))} training wedge(s), "
                  f"{te['n_fail']} test wedge(s).")
        else:
            print("PASS: no round exceeds either threshold.")
        if not args.include_archived:
            print("Archives skipped (pass --include-archived for rolled-back iterations).")

    return code


if __name__ == "__main__":
    sys.exit(main())
