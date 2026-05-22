#!/usr/bin/env python3
"""
Catalog sandbox_denials.jsonl records across RoboPhD runs.

Classifies every denial into one of a handful of well-known patterns
(true positive, false positive, parser limitation, etc.) so a sweep
across many runs answers the question "are we still seeing FP class X
in production?" by counting rather than reading.

This script exists because **individual records under-surface
patterns**. The Claude-CLI tool-results readback FP sat open for a
week — visible in single records but easy to dismiss as a one-off —
and only became impossible to park once a cross-run sweep showed it
as "FP-OPEN: 2 across 2 runs." Re-run after a batch of runs (or on
a cron) to keep open false-positive classes accountable instead of
relying on individual records to surface trends.

Usage:
    # Catalog all asta_ds1000 runs under ../robophd_runs + ../alt_robophd_runs
    python scripts/catalog_sandbox_denials.py

    # Specific domain or engine
    python scripts/catalog_sandbox_denials.py --domain text2sql
    python scripts/catalog_sandbox_denials.py --engine gepa

    # Custom roots (repeatable)
    python scripts/catalog_sandbox_denials.py \
        --runs-dir ../robophd_runs --runs-dir ../alt_robophd_runs

    # Machine-readable output
    python scripts/catalog_sandbox_denials.py --json

    # Show unclassified records for triage (always shown by default if any)
    python scripts/catalog_sandbox_denials.py --show-unclassified

Categories:
    TP          true positive — sandbox correctly denied
    FP-FIXED    false positive that would NOT recur at HEAD
    FP-OPEN     false positive that still recurs at HEAD
    LIMITATION  fail-closed by design (e.g. shlex parse-fail)
    HISTORICAL  produced by retired hook branches; no longer reachable
    ERROR       hook internal error
    UNKNOWN     unclassified — add a pattern below
"""

import argparse
import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------
# Classifier
#
# Each pattern is (label, category, predicate). FIRST MATCH WINS, so
# order specifically before generally. Predicates take the parsed JSON
# record dict.
#
# When a new FP class is closed in code, MOVE its pattern from FP-OPEN
# to FP-FIXED — historical jsonl records keep classifying correctly,
# and the open->fixed transition is visible over time. We deliberately
# do NOT tag the closing commit hash in the label (SHAs rot under
# rebase, and labels are user-visible in reports); use ``git log -S
# "<label phrase>"`` from the closing PR/commit to find when a pattern
# was migrated.
#
# Predicates that match on blocked_path patterns alone (e.g., the
# sed-script regex) are tightened with a second cmd-awareness check
# so a real out-of-scope path that *syntactically* matches the FP
# regex isn't silently mislabeled FP-FIXED.
# ---------------------------------------------------------------------


def _b(rec):
    return rec.get("blocked_path", "") or ""


def _cmd(rec):
    return rec.get("command", "") or ""


def _scope(rec):
    return rec.get("scope") or ""


_INTERP_PATH_RE = re.compile(
    r"^/(opt/anaconda3|usr|usr/local|opt/homebrew)/.*?/(python|node|ruby)\d*"
)
_SED_SCRIPT_RE = re.compile(r"^/\^?[^/]*[ ,].*[,$]p?$")


PATTERNS = [
    # ---- FP-FIXED: no longer recurs at HEAD ----
    ("FP-FIXED: interpreter binary path (multi-line cd + abs python)",
     "FP-FIXED",
     # No $ anchor: real paths look like /opt/.../bin/python3.11 — version
     # suffix has a non-digit (.) so end-anchoring with \d*$ never matches.
     # Tighten with cmd-awareness via DIRNAME (not the full blocked path):
     # blocked is realpath'd (`python` -> `python3.11`), cmd keeps the
     # symlink name; the shared signal is the interpreter directory.
     lambda r: _scope(r) == "read"
               and bool(_INTERP_PATH_RE.match(_b(r)))
               and os.path.dirname(_b(r)) in _cmd(r)),

    ("FP-FIXED: sed script as path (/regex/,$p)",
     "FP-FIXED",
     # Tighten: real out-of-scope paths CAN syntactically match the sed
     # script regex (e.g., /foo,bar/p$baz). Require `sed` in the
     # command so we don't mislabel a genuine recon read as a fixed FP.
     lambda r: _scope(r) == "read"
               and bool(_SED_SCRIPT_RE.match(_b(r)))
               and bool(re.search(r"\bsed\b", _cmd(r)))),

    ("FP-FIXED: auto-memory write under this-run slug",
     "FP-FIXED",
     lambda r: _scope(r) == "write"
               and ".claude/projects/" in _b(r) and "/memory/" in _b(r)),

    ("FP-FIXED: Claude CLI tool-results spill readback",
     "FP-FIXED",
     # Read-only carve-out — a write attempt to the same path is a TP
     # (the read-only intent), so require scope=read.
     lambda r: _scope(r) == "read"
               and ".claude" in _b(r) and "/tool-results/" in _b(r)),

    # ---- FP-OPEN: still recurs at HEAD ----
    ("FP-OPEN: tr -d / operand (parked — needs substitution-parser rework)",
     "FP-OPEN",
     lambda r: _scope(r) == "read" and _b(r) == "/"
               and bool(re.search(r"\btr\s+-d\s+/", _cmd(r)))),

    # ---- TPs: sandbox correctly catching out-of-policy actions ----
    ("TP: find / full-filesystem scan",
     "TP",
     lambda r: _scope(r) == "read" and _b(r) == "/"
               and " find " in " " + _cmd(r) + " "),

    ("TP: cross-run recon (engine-level or higher)",
     "TP",
     # Under a runs-root, but NOT pointing into a specific <task>_<ts>
     # run dir — e.g. `ls <engine>/`, `ls <engine>/agents/`,
     # `find <runs-root> -name model_registry*`, `cd <engine> && ls`.
     # The negation must match an actual <task>_<timestamp>/ segment.
     # Task names contain internal underscores (`asta_ds1000`), so
     # [a-z_]+ greedy-then-_ doesn't reach the digits; need a stricter
     # body class that ends with _<8digits>_<6digits>.
     lambda r: _scope(r) == "read"
               and bool(re.search(r"/(alt_)?robophd_runs(/|$)", _b(r)))
               and not bool(re.search(
                   r"/(alt_)?robophd_runs/[^/]+/[a-z][a-z0-9_]*_\d{8}_\d{6}/",
                   _b(r),
               ))),

    ("TP: sibling-run or source-repo read",
     "TP",
     lambda r: _scope(r) == "read"
               and ("/cc/RoboPhD/" in _b(r) or bool(re.search(
                   r"/asta_ds1000_\d{8}_\d{6}", _b(r),
               )))),

    ("TP: write outside cwd",
     "TP",
     lambda r: _scope(r) == "write"
               and not _b(r).startswith(os.path.expanduser("~/.claude"))),

    ("TP: /tmp scratch attempt outside scope",
     "TP",
     lambda r: _b(r).startswith(("/tmp", "/private/tmp"))),

    # ---- LIMITATION: fail-closed by design ----
    ("LIMITATION: shlex parse fail (fail-closed)",
     "LIMITATION",
     lambda r: _scope(r) == "parse" or r.get("reason") == "shlex.split failed"),

    # ---- ERROR: hook internal failure ----
    ("ERROR: hook internal error",
     "ERROR",
     lambda r: bool(r.get("error"))),

    # ---- HISTORICAL: retired hook branches ----
    ("HISTORICAL: retired subprocess_bypass / classifier scope",
     "HISTORICAL",
     lambda r: _scope(r) in ("subprocess_bypass", "classifier")),
]


def classify(rec: dict) -> tuple:
    """Return (label, category) for a denial record."""
    for label, cat, pred in PATTERNS:
        try:
            if pred(rec):
                return label, cat
        except Exception:
            continue
    return "UNKNOWN", "UNKNOWN"


# ---------------------------------------------------------------------
# Filesystem walk
# ---------------------------------------------------------------------


def find_denial_files(roots: list, domain: str = None,
                      engine: str = None) -> list:
    """Locate sandbox_denials.jsonl files under the given roots.

    Layout assumed (matches `../robophd_runs/`): each root contains
    one sub-dir per engine (``robophd``, ``gepa``, ``autoresearch``),
    each of those contains run dirs (``<domain>_<timestamp>``), each
    run dir may contain a ``sandbox_denials.jsonl``.
    """
    files: list = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        engines = [engine] if engine else None
        for engine_dir in sorted(root.iterdir()) if not engines else [root / e for e in engines]:
            if not engine_dir.is_dir():
                continue
            for run_dir in sorted(engine_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                if domain and not run_dir.name.startswith(domain + "_"):
                    continue
                f = run_dir / "sandbox_denials.jsonl"
                if f.exists():
                    files.append(f)
    return files


def load_records(path: Path) -> list:
    recs: list = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return recs


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------


def render_text(files, all_runs, totals_by_cat, totals_by_label,
                unclassified, show_unclassified):
    print(f"Scanned {len(files)} sandbox_denials.jsonl file(s) across "
          f"{sum(1 for _, n, _ in all_runs if n)} run(s) with denials "
          f"({sum(1 for _, n, _ in all_runs if not n)} clean).\n")

    print("=== PER-RUN SUMMARY ===")
    print(f"{'run':70} {'n':>3}  by-label")
    for path_label, n, run_counts in all_runs:
        if not n:
            continue
        parts = ", ".join(
            f"{label.split(':', 1)[0]}={count}"
            for label, count in run_counts.most_common()
        )
        print(f"{path_label:70} {n:>3}  {parts}")

    print("\n=== CATEGORY TOTALS ===")
    for cat in ("TP", "FP-FIXED", "FP-OPEN", "LIMITATION",
                "HISTORICAL", "ERROR", "UNKNOWN"):
        n = totals_by_cat.get(cat, 0)
        if n:
            print(f"  {cat:12} {n}")

    print("\n=== LABEL TOTALS ===")
    for label, n in totals_by_label.most_common():
        print(f"  {n:>4}  {label}")

    if unclassified and show_unclassified:
        print(f"\n=== UNCLASSIFIED ({len(unclassified)}) — add a pattern ===")
        for run_label, rec in unclassified[:50]:
            print(f"  [{run_label}] tool={rec.get('tool')} "
                  f"scope={rec.get('scope')} blocked={rec.get('blocked_path')!r}")
            cmd = (rec.get("command", "") or "")[:150]
            print(f"      cmd={cmd!r}")
    elif unclassified:
        print(f"\n({len(unclassified)} unclassified — re-run with "
              f"--show-unclassified to inspect)")

    # Headline accountability line.
    open_n = totals_by_cat.get("FP-OPEN", 0)
    print()
    if open_n:
        print(f"⚠  {open_n} FP-OPEN denial(s) — false positives still "
              f"recurring at HEAD. See LABEL TOTALS above.")
    else:
        print("✓ No FP-OPEN denials in this sweep.")


def render_json(files, all_runs, totals_by_cat, totals_by_label, unclassified):
    out = {
        "scanned_files": len(files),
        "runs": [
            {"run": label, "n_denials": n,
             "by_label": dict(c.most_common())}
            for label, n, c in all_runs
        ],
        "totals_by_category": dict(totals_by_cat),
        "totals_by_label": dict(totals_by_label.most_common()),
        "unclassified": [
            {"run": run_label, "record": rec}
            for run_label, rec in unclassified
        ],
    }
    json.dump(out, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs-dir", action="append", type=Path, default=None,
        help="Root directory containing engine subdirs (repeatable). "
             "Default: ../robophd_runs and ../alt_robophd_runs (if present) "
             "relative to repo root.",
    )
    parser.add_argument(
        "--domain", default=None,
        help="Filter to runs whose dir name starts with '<domain>_' "
             "(e.g. asta_ds1000, text2sql). Default: all domains.",
    )
    parser.add_argument(
        "--engine", default=None, choices=("robophd", "gepa", "autoresearch"),
        help="Filter to one engine subdir. Default: all engines.",
    )
    parser.add_argument(
        "--show-unclassified", action="store_true",
        help="Print full UNCLASSIFIED records for triage.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of the text report.",
    )
    args = parser.parse_args(argv)

    if args.runs_dir is None:
        roots = [_PROJECT_ROOT.parent / "robophd_runs",
                 _PROJECT_ROOT.parent / "alt_robophd_runs"]
    else:
        roots = args.runs_dir
    roots = [r.resolve() for r in roots if r.exists()]
    if not roots:
        print("No runs roots found.", file=sys.stderr)
        return 1

    files = find_denial_files(roots, domain=args.domain, engine=args.engine)

    totals_by_cat: collections.Counter = collections.Counter()
    totals_by_label: collections.Counter = collections.Counter()
    unclassified: list = []
    all_runs: list = []

    for f in files:
        recs = load_records(f)
        run_counts: collections.Counter = collections.Counter()
        # Per-run label: <runs-root-name>/<engine>/<run-name>
        run_label = str(f.parent.relative_to(f.parents[3])) \
            if len(f.parents) >= 4 else str(f.parent)
        for rec in recs:
            label, cat = classify(rec)
            run_counts[label] += 1
            totals_by_label[label] += 1
            totals_by_cat[cat] += 1
            if cat == "UNKNOWN":
                unclassified.append((run_label, rec))
        all_runs.append((run_label, len(recs), run_counts))

    if args.json:
        render_json(files, all_runs, totals_by_cat, totals_by_label, unclassified)
    else:
        render_text(files, all_runs, totals_by_cat, totals_by_label,
                    unclassified, args.show_unclassified)

    # Exit non-zero only on UNKNOWN — drives "add a pattern" pressure.
    # FP-OPEN is loud but doesn't fail the run (it's expected to be
    # nonzero whenever a parked limitation is in play).
    return 2 if totals_by_cat.get("UNKNOWN", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
