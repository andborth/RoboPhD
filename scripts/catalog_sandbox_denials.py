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

Categories (FP-FIXED/FP-OPEN/TP/TP-NOW-ALLOWED are decided by REPLAYING
each record through the live hook policy — see the Classifier section —
not asserted by a pattern, so they can't drift from the code):
    TP              true positive — correct deny, still denied at HEAD
    TP-NOW-ALLOWED  judged a true positive, but the live hook no longer
                    denies it — policy loosened; possible scope regression
    FP-FIXED        false positive that no longer recurs at HEAD
    FP-OPEN         false positive that still recurs at HEAD
    LIMITATION      fail-closed by design (e.g. shlex parse-fail)
    HISTORICAL      produced by retired hook branches; no longer reachable
    ERROR           hook internal error
    UNKNOWN         unclassified — add a pattern below

Pass --no-replay to skip the live-hook replay (FP intents then report as
FP-OPEN conservatively); useful for a fast, FS-independent pass.

REPLAY BLIND SPOT — own-iteration write records.
A denial record stores only {ts, tool, scope, blocked_path, command, cwd}.
It does NOT store ROBOPHD_EVOLUTION_ITERATION_DIR, the per-iteration env
var the harness sets and the hook uses to anchor the WRITE scope on the
iteration root (sandbox_hook commit 8ef3d45, 2026-05-24) rather than the
literal cwd. So when replay re-runs a write record, that env var is unset
and the hook falls back to write_root=cwd. A record where the agent wrote
to its OWN current iteration dir from a deeper subdir (e.g. cwd
.../iteration_015/<test>/<agent>, blocked .../iteration_015/agent.py) was
a real FP BEFORE the fix, but the live hook ALLOWS it now — yet replay
still denies it, because it can't reconstruct the env var. Such records
therefore surface as "TP: write outside cwd" / still-denied even though
HEAD no longer denies them. Treat the still-denied verdict on a write
that targets the run's own iteration dir as UNRELIABLE, not as proof the
scope is still tight. (Writes to a PRIOR iteration are correctly denied
at HEAD regardless — that's the immutable-history boundary, not this
blind spot.) Fixing it properly would require logging the iteration dir
into the record; until then the caveat lives here and on the pattern.
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

# The live hook is the single source of truth for "does HEAD still deny
# this?". Import lazily-tolerant: if utilities/ isn't importable for some
# reason, classify() still works in its --no-replay fallback mode.
sys.path.insert(0, str(_PROJECT_ROOT / "utilities"))
try:
    import sandbox_hook as _hook
except Exception:  # pragma: no cover - defensive
    _hook = None


def _make_replay(run_dir):
    """Build a replay callable bound to a run dir (the experiment_dir for
    every record in that run's sandbox_denials.jsonl).

    Returns ``None`` if the live hook isn't importable, so callers fall
    back to static classification.
    """
    if _hook is None:
        return None

    def _replay(rec):
        return _hook.replay_denial_record(rec, str(run_dir))

    return _replay


# ---------------------------------------------------------------------
# Classifier
#
# Each pattern is (label, intent, predicate). FIRST MATCH WINS, so
# order specifically before generally. Predicates take the parsed JSON
# record dict.
#
# A pattern's `intent` is the human judgement about the denial — what it
# WOULD be if the policy stayed as-is:
#
#     "FP"  this denial is a false positive (legit work wrongly blocked)
#     "TP"  this denial is correct (out-of-policy access, rightly blocked)
#
# The FINAL category (FP-FIXED / FP-OPEN / TP / TP-NOW-ALLOWED) is NOT
# written in the pattern — it is decided by REPLAYING the record through
# the live hook policy (sandbox_hook.replay_denial_record) and combining
# that verdict with the intent:
#
#     intent FP + replay still-denies  -> FP-OPEN     (needs fixing)
#     intent FP + replay now-allows    -> FP-FIXED    (closed in code)
#     intent TP + replay still-denies  -> TP          (correct, verified)
#     intent TP + replay now-allows    -> TP-NOW-ALLOWED (policy loosened!)
#
# This is why FP-FIXED can no longer drift from reality: a pattern can
# only CLAIM something is a false positive; whether it is actually fixed
# is observed from the code, not asserted by a regex. When you fix an FP
# in the hook, you do NOT edit this file — the next sweep simply sees the
# replay flip from still-denies to now-allows and the record moves from
# FP-OPEN to FP-FIXED on its own.
#
# Records that can't be faithfully replayed (parse/error/historical, or
# a shape replay_denial_record raises on) keep a static category from
# STATIC_PATTERNS and are never replay-adjusted.
#
# Predicates that match on blocked_path alone (e.g. the sed-script regex)
# still carry a cmd-awareness check so a real out-of-scope path that
# syntactically matches an FP regex isn't mis-INTENTED as FP — though
# even a mislabel is now caught by replay (a real recon read replays as
# still-denies, so an FP-intented recon would surface as FP-OPEN for
# triage rather than silently passing as FP-FIXED).
# ---------------------------------------------------------------------


def _b(rec):
    return rec.get("blocked_path", "") or ""


def _cmd(rec):
    return rec.get("command", "") or ""


def _scope(rec):
    return rec.get("scope") or ""


def _tool(rec):
    return rec.get("tool") or ""


# Tools that carry the auto-memory WRITE carve-out in the live hook
# (sandbox_hook.WRITE_TOOLS). A write to the run's memory dir is only a
# false positive when it arrives through one of these — a Bash write to
# the same dir is denied BY DESIGN (injection-hardening), so it is a TP,
# not an open FP. Kept in sync with the hook by mirroring the set here.
_WRITE_TOOLS = {"Edit", "Write", "NotebookEdit", "MultiEdit"}


def _is_memory_write(rec):
    return (_scope(rec) == "write"
            and ".claude/projects/" in _b(rec) and "/memory/" in _b(rec))


_INTERP_PATH_RE = re.compile(
    r"^/(opt/anaconda3|usr|usr/local|opt/homebrew)/.*?/(python|node|ruby)\d*"
)
# An interpreter BINARY in a bin dir — mirrors sandbox_hook.INTERPRETER_BIN_RE
# (the read-scope exemption). Used for the interpreter-FP intent so it
# matches only the exempt binary, NOT a library/site-packages path that
# merely contains `python` (e.g. `.../lib/python3.11/site-packages/...`,
# which is a real out-of-scope read, not an interpreter run).
_INTERP_BIN_RE = re.compile(
    # `usr` covers `/usr/local/...` too via the `(.*/)?` middle, so no
    # separate `usr/local` alternative is needed.
    r"^/(opt/anaconda3|opt/miniconda3|opt/homebrew|usr)/"
    r"(.*/)?bin/(python|node|ruby|perl|pypy)[0-9.]*$"
)
_SED_SCRIPT_RE = re.compile(r"^/\^?[^/]*[ ,].*[,$]p?$")
# An awk program token blocked-as-path: a `/regex/,/regex/` range or a
# `/regex/{action}` guard. Both start with `/` and contain a second
# unescaped `/` (range comma-join) or a `{`/space (action/multi-expr).
_AWK_PROGRAM_RE = re.compile(r"^/.*(/,|/\{|\}|\{|, )")


# Replayable patterns: (label, intent, predicate) with intent in
# {"FP", "TP"}. The final FP-FIXED/FP-OPEN/TP/TP-NOW-ALLOWED category is
# decided by replaying the record (see classify()).
INTENT_PATTERNS = [
    # ---- intent FP: legit work that was (or is) wrongly blocked ----
    # Each of these names a shell shape that should NOT be denied. Whether
    # it's actually fixed is decided by replay, not asserted here.
    ("FP: sed script as path (/regex/,$p)",
     "FP",
     # Real out-of-scope paths CAN syntactically match the sed script
     # regex (e.g., /foo,bar/p$baz); require `sed` in the command so a
     # genuine recon read isn't mis-INTENTED as FP. (Replay is the final
     # backstop either way.)
     lambda r: _scope(r) == "read"
               and bool(_SED_SCRIPT_RE.match(_b(r)))
               and bool(re.search(r"\bsed\b", _cmd(r)))),

    ("FP: awk program as path (/regex/ range or action)",
     "FP",
     lambda r: _scope(r) == "read"
               and bool(_AWK_PROGRAM_RE.match(_b(r)))
               and bool(re.search(r"\bawk\b", _cmd(r)))),

    ("FP: grep pattern as path (slash/regex operand)",
     "FP",
     # A `grep` regex operand starting with `/` is misread as a path:
     #   grep -v "/agents/"            -> blocked '/agents'
     #   grep -E "/ ?1000|million|/1e" -> blocked the whole alternation
     # Require `grep` in the command and a blocked path that's either a
     # slash-delimited word (`/agents`) or carries regex metachars
     # (`|`, `?`, `*`, `\`) — neither is a real filesystem read.
     lambda r: _scope(r) == "read"
               and bool(re.search(r"\bgrep\b", _cmd(r)))
               and (bool(re.search(r"[|?*\\]", _b(r)))
                    or bool(re.match(r"^/[A-Za-z0-9_]+$", _b(r))))),

    ("FP: tr operand (character set, not a path)",
     "FP",
     # `tr -d /` etc. — the lone `/` operand is a character set. Require
     # `tr` and exclude a real `find /` (also blocked='/').
     lambda r: _scope(r) == "read" and _b(r) == "/"
               and bool(re.search(r"\btr\b", _cmd(r)))
               and not bool(re.search(r"\bfind\s+/", _cmd(r)))),

    ("FP: $((arithmetic)) split a path token",
     "FP",
     # `iter${v}_v$((v-2))/agent.py` split on `((`/`))`, leaving a stray
     # `/agent.py`. Require `$((` in the command.
     lambda r: _scope(r) == "read"
               and "$((" in _cmd(r)
               and bool(re.match(r"^/[A-Za-z0-9._-]+$", _b(r)))),

    ("FP: command wrapper prefix (timeout/time/nice <interp>)",
     "FP",
     # `timeout 60 /opt/.../python foo.py`, `time /opt/.../python ...`:
     # the wrapper pushes the interpreter into argument position so its
     # abs path read-denies. Legit run of the interpreter. (Currently
     # OPEN — replay will confirm — until the wrapper-strip hook fix
     # lands; then it flips to FIXED with no edit here.)
     lambda r: _scope(r) == "read"
               and bool(_INTERP_PATH_RE.match(_b(r)))
               and bool(re.search(r"\b(timeout|time|nice|ionice|stdbuf|nohup)\b",
                                  _cmd(r)))),

    ("FP: auto-memory write under this-run slug (Write tool)",
     "FP",
     # Only the WRITE_TOOLS branch carves out memory writes. A write here
     # via Write/Edit is the legit relocated auto-memory -> FP. A Bash
     # write to the same dir is a deliberate deny (see the TP pattern
     # below), so gate on the tool to mirror the hook's own split.
     lambda r: _is_memory_write(r) and _tool(r) in _WRITE_TOOLS),

    ("FP: Claude CLI tool-results spill readback",
     "FP",
     lambda r: _scope(r) == "read"
               and ".claude" in _b(r) and "/tool-results/" in _b(r)),

    # ---- intent TP: sandbox correctly catching out-of-policy actions ----
    ("TP: Bash write to auto-memory dir (injection-hardening deny)",
     "TP",
     # A write to the run's own memory dir is carved out ONLY for the
     # Write tool; a Bash `cat >> ~/.claude/.../memory/x` is denied by
     # design (a shell write to ~/.claude is an injection vector, and the
     # sanctioned path is the Write tool). Correct deny, not an open FP.
     # Ordered before the memory-write FP can't fire (that's now Write-
     # tool-gated) so any non-Write memory write lands here.
     lambda r: _is_memory_write(r) and _tool(r) not in _WRITE_TOOLS),

    # `find /` is also blocked='/' like the tr FP above, but ordering and
    # the tr predicate's `find /` exclusion keep them separate.
    ("TP: find / full-filesystem scan",
     "TP",
     # `find /` (root scan). Match the `find` token at any word boundary
     # so a newline-preceded `find` after a multi-line `cd` still counts
     # (the old `" find "` substring missed `\nfind /`).
     lambda r: _scope(r) == "read" and _b(r) == "/"
               and bool(re.search(r"\bfind\s+/", _cmd(r)))),

    # Interpreter-path denials are ALL false positives now. The hook
    # exempts system interpreter binaries from read scope
    # (sandbox_hook.INTERPRETER_BIN_RE), because running an interpreter is
    # benign and a bare leading `/opt/.../python foo.py` was always
    # allowed. So every shape that read-denied an interpreter path — the
    # `PY=/opt/.../python; $PY` assignment, `find -exec python`, loop
    # lists, probes like `ls /opt/.../python`, leading-token runs, heredoc
    # bodies — is a historical FP that replay now reports FIXED. (Earlier
    # this split into a probe-TP vs run-FP; that distinction is gone — the
    # hook no longer treats ANY interpreter-path read as a real deny.)
    ("FP: interpreter path read (now exempt — assign/exec/probe/run)",
     "FP",
     # Match only an interpreter BINARY (the exempt shape), not any path
     # containing `python` — a `.../lib/python3.11/site-packages` read or
     # a `find` into the conda lib tree is a real out-of-scope access (TP),
     # caught by the find/system-dir patterns below, not exempted here.
     lambda r: _scope(r) == "read"
               and bool(_INTERP_BIN_RE.match(_b(r)))),

    ("TP: read of hallucinated path (malformed run-name segment)",
     "TP",
     # Path is shaped like a real experiment-dir read (under
     # <runs>/<engine>/<X>/<deeper>...) but the run-name slot <X>
     # doesn't match the canonical <task>_<8digits>_<6digits> form —
     # the agent built an absolute path with a wrong run-name. Distinct
     # from engine-level recon (a deliberate `ls <engine>/`). Tracks
     # agent path-hallucination recurrence.
     lambda r: _scope(r) == "read"
               and bool(re.search(
                   r"/(alt_)?robophd_runs/[^/]+/[^/]+/[^/]+", _b(r),
               ))
               and not bool(re.search(
                   r"/(alt_)?robophd_runs/[^/]+/[a-z][a-z0-9_]*_\d{8}_\d{6}(/|$)",
                   _b(r),
               ))),

    ("TP: cross-run recon (engine-level or higher)",
     "TP",
     # Under a runs-root, but NOT pointing into a specific <task>_<ts>
     # run dir — e.g. `ls <engine>/`, `find <runs-root> -name ...`.
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
     # CAVEAT: when the blocked path is the run's OWN current iteration
     # dir (cwd .../iteration_N/<sub>, blocked .../iteration_N/agent.py),
     # the still-denied replay verdict is UNRELIABLE — replay can't set
     # ROBOPHD_EVOLUTION_ITERATION_DIR, so it re-denies an own-iteration
     # write the live hook now allows. See "REPLAY BLIND SPOT" in the
     # module docstring. A write to a PRIOR iteration is a correct deny.
     lambda r: _scope(r) == "write"
               and not _b(r).startswith(os.path.expanduser("~/.claude"))),

    ("TP: /tmp scratch attempt outside scope",
     "TP",
     lambda r: _b(r).startswith(("/tmp", "/private/tmp"))),

    ("TP: find on out-of-scope system path (e.g. conda env, /usr, /opt)",
     "TP",
     # The narrower shape where the agent picks a specific out-of-scope
     # find root — `find /opt/anaconda3/envs/<env> -name model_*`.
     lambda r: _scope(r) == "read"
               and bool(_b(r)) and _b(r) != "/"
               and bool(re.search(rf"\bfind\s+{re.escape(_b(r))}\b",
                                  _cmd(r)))),

    ("TP: probe of out-of-scope system directory (ls/cat/stat ...)",
     "TP",
     # An out-of-scope SYSTEM dir read that isn't an interpreter binary
     # (those are now exempt) and isn't a find root (caught above) —
     # e.g. `ls /opt/anaconda3/envs/`, `cat /etc/hosts`, `ls /usr/lib`.
     # The blocked path is under a real system root and the command names
     # it as an operand. Correct deny: out-of-policy read of a system
     # location. (INTERPRETER_BIN_RE-shaped paths never reach here — the
     # interpreter FP matches them first.)
     lambda r: _scope(r) == "read"
               and bool(re.match(r"^/(opt|usr|etc|var|private/var|Library|System)/",
                                 _b(r)))
               and _b(r) in _cmd(r)),
]


# Static patterns: non-replayable records (no live decision to observe).
# (label, category, predicate) — category is final, never replay-adjusted.
STATIC_PATTERNS = [
    ("LIMITATION: shlex parse fail (fail-closed)",
     "LIMITATION",
     lambda r: _scope(r) == "parse" or r.get("reason") == "shlex.split failed"),

    ("ERROR: hook internal error",
     "ERROR",
     lambda r: bool(r.get("error"))),

    ("HISTORICAL: retired subprocess_bypass / classifier scope",
     "HISTORICAL",
     lambda r: _scope(r) in ("subprocess_bypass", "classifier")),
]


def _match_intent(rec: dict):
    """Return (label, intent) for the first matching INTENT_PATTERN, else
    (None, None)."""
    for label, intent, pred in INTENT_PATTERNS:
        try:
            if pred(rec):
                return label, intent
        except Exception:
            continue
    return None, None


def _match_static(rec: dict):
    """Return (label, category) for the first matching STATIC_PATTERN, else
    (None, None)."""
    for label, cat, pred in STATIC_PATTERNS:
        try:
            if pred(rec):
                return label, cat
        except Exception:
            continue
    return None, None


def classify(rec: dict, replay=None) -> tuple:
    """Return (label, category) for a denial record.

    ``replay`` is an optional callable ``(rec) -> (still_denies: bool,
    blocked)`` — normally ``_make_replay(run_dir)`` — that runs the
    record through the live hook policy. When provided, the FP/TP intent
    from INTENT_PATTERNS is combined with the replay verdict to yield the
    final category (see module docstring). When ``replay`` is None (or
    raises), the intent maps to a conservative static category so the
    classifier still works without filesystem access (used by unit tests
    and the --no-replay path).

    Static (non-replayable) records — parse/error/historical — bypass
    replay entirely.
    """
    # Static categories first: these have no live decision to observe.
    label, cat = _match_static(rec)
    if label is not None:
        return label, cat

    label, intent = _match_intent(rec)
    if label is None:
        return "UNKNOWN", "UNKNOWN"

    if replay is None:
        # No live policy available — fall back to a self-consistent
        # static mapping: an FP intent is reported as FP-OPEN (assume
        # unfixed; the safe, attention-drawing direction), a TP as TP.
        return label, ("FP-OPEN" if intent == "FP" else "TP")

    try:
        still_denies, blocked_now = replay(rec)
    except Exception:
        # Unreplayable shape — keep the intent's conservative mapping.
        return label, ("FP-OPEN" if intent == "FP" else "TP")

    if intent == "FP":
        if not still_denies:
            return label, "FP-FIXED"
        # Replay still denies. If it denies on the SAME path the record
        # logged, the named FP genuinely still recurs -> FP-OPEN. If it
        # denies on a DIFFERENT path, the record's denial is no longer
        # explained by this FP — the command has a separate, still-valid
        # block (e.g. an interpreter path inside a `cat > /tmp/x` whose
        # real block is the /tmp write; or an interpreter-regex match
        # that's actually a `find` into the conda env). Surface it as
        # OTHER-STILL-DENIED so a real TP isn't mislabeled as an open FP.
        if _same_path(blocked_now, _b(rec)):
            return label, "FP-OPEN"
        return f"OTHER-STILL-DENIED (matched {label})", "TP"
    # intent == "TP"
    return label, ("TP" if still_denies else "TP-NOW-ALLOWED")


def _same_path(a, b) -> bool:
    """Whether two blocked paths refer to the same target. Robust to the
    /tmp <-> /private/tmp symlink (macOS) and trailing slashes."""
    if not a or not b:
        return a == b

    def norm(p):
        p = p.rstrip("/")
        if p.startswith("/private/tmp"):
            p = p[len("/private"):]
        return p
    return norm(a) == norm(b)


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


# "Benign-intent friction" labels: denials that are policy-CORRECT (TP)
# but where the agent's intent was legitimate and it simply had no
# sanctioned path — so the deny is friction, not a caught attack. These
# don't trip the ⚠ accountability lines (they're true positives, not
# FP-OPEN / TP-NOW-ALLOWED), so without a dedicated headline a rising
# trend would sit buried in LABEL TOTALS. The TP/FP axis answers "is the
# deny correct?"; this set answers the orthogonal "did we block benign
# work?" — a signal that, if it grows, argues for a sanctioned affordance
# (e.g. a memory-append path, a scratch dir) rather than a policy change.
# Labels must match the pattern labels in INTENT_PATTERNS exactly.
FRICTION_LABELS = (
    "TP: Bash write to auto-memory dir (injection-hardening deny)",
    "TP: /tmp scratch attempt outside scope",
)


def _friction_breakdown(totals_by_label):
    """Return (total, [(label, n), ...]) for friction labels that occurred."""
    items = [(lab, totals_by_label.get(lab, 0)) for lab in FRICTION_LABELS]
    items = [(lab, n) for lab, n in items if n]
    return sum(n for _, n in items), items


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
    for cat in ("TP", "TP-NOW-ALLOWED", "FP-FIXED", "FP-OPEN", "LIMITATION",
                "HISTORICAL", "ERROR", "UNKNOWN"):
        n = totals_by_cat.get(cat, 0)
        if n:
            print(f"  {cat:14} {n}")

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

    # Headline accountability lines.
    open_n = totals_by_cat.get("FP-OPEN", 0)
    loosened_n = totals_by_cat.get("TP-NOW-ALLOWED", 0)
    print()
    if loosened_n:
        print(f"⚠  {loosened_n} TP-NOW-ALLOWED denial(s) — the live hook NO "
              f"LONGER denies a record we judged a true positive. The policy "
              f"loosened; confirm this is intended (possible scope regression).")
    if open_n:
        print(f"⚠  {open_n} FP-OPEN denial(s) — false positives still "
              f"recurring at HEAD. See LABEL TOTALS above.")
    elif not loosened_n:
        print("✓ No FP-OPEN or TP-NOW-ALLOWED denials in this sweep.")

    # Friction line: correct denies of benign-intent work. Informational
    # (does NOT affect exit code) — a rising count argues for a sanctioned
    # affordance, not a policy fix.
    friction_n, friction_items = _friction_breakdown(totals_by_label)
    if friction_n:
        parts = ", ".join(f"{lab.split(':', 1)[1].strip()}={n}"
                          for lab, n in friction_items)
        print(f"ℹ  {friction_n} benign-intent friction denial(s) — correct "
              f"denies of legitimate work with no sanctioned path ({parts}). "
              f"If rising, add an affordance rather than loosening policy.")


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
        "benign_intent_friction": dict(_friction_breakdown(totals_by_label)[1]),
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
    parser.add_argument(
        "--no-replay", action="store_true",
        help="Skip replaying records through the live hook. FP intents "
             "then report as FP-OPEN (conservative) rather than the "
             "observed FP-FIXED/FP-OPEN split. Faster; no FS dependency.",
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
        # Replay binds to this run's dir as the experiment_dir for every
        # record in its log. --no-replay disables it (static fallback).
        replay = None if args.no_replay else _make_replay(f.parent)
        for rec in recs:
            label, cat = classify(rec, replay=replay)
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

    # Exit non-zero on UNKNOWN (drives "add a pattern" pressure) or on
    # TP-NOW-ALLOWED (a true-positive the live policy no longer denies —
    # a possible scope regression that must be looked at). FP-OPEN is
    # loud but doesn't fail the run (a parked FP can legitimately be
    # nonzero until its hook fix lands).
    return 2 if (totals_by_cat.get("UNKNOWN", 0)
                 or totals_by_cat.get("TP-NOW-ALLOWED", 0)) else 0


if __name__ == "__main__":
    sys.exit(main())
