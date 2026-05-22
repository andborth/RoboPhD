"""Unit tests for ``scripts/catalog_sandbox_denials.py`` classifier.

The catalog is a meta-tool: a classification error produces a wrong
LABEL in a report, not a wrong sandbox decision. So the bar is lower
than the hook itself — but the script has 12 regex/predicate patterns
with first-match-wins ordering, and the bug class it's exposed to is
exactly the kind unit tests catch (an over-anchored regex that fails
to match real records, or an under-tightened predicate that
silently mislabels a real out-of-scope path as a fixed FP).

Each test pins ONE pattern with a matches-case and at least one
deliberately-close-but-different non-matches case so the predicate's
*intent* (not its literal regex) is what's fenced.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import catalog_sandbox_denials as catalog  # noqa: E402


def _rec(**kw):
    return kw


# ---------------------------------------------------------------------
# True positives
# ---------------------------------------------------------------------


def test_find_root_classifies_tp():
    label, cat = catalog.classify(_rec(
        scope="read", blocked_path="/", command="find / -name foo",
    ))
    assert cat == "TP"
    assert "find /" in label


def test_cross_run_recon_engine_dir_classifies_tp():
    _, cat = catalog.classify(_rec(
        scope="read",
        blocked_path="/Users/x/Desktop/cc/robophd_runs/robophd",
        command="cd /Users/x/Desktop/cc/robophd_runs/robophd && ls",
    ))
    assert cat == "TP"


def test_cross_run_recon_engine_subdir_classifies_tp():
    """Engine-subdir reads like `ls <engine>/agents/` (no specific
    run-timestamp segment) are also engine-level recon."""
    _, cat = catalog.classify(_rec(
        scope="read",
        blocked_path="/Users/x/Desktop/cc/alt_robophd_runs/robophd/agents",
        command="ls /Users/x/Desktop/cc/alt_robophd_runs/robophd/agents/",
    ))
    assert cat == "TP"


def test_path_inside_specific_run_does_not_match_engine_recon():
    """A path that lives INSIDE a specific timestamped run dir is NOT
    engine-level recon — it's either in-scope or a sibling-run read."""
    label, _ = catalog.classify(_rec(
        scope="read",
        blocked_path=("/Users/x/Desktop/cc/robophd_runs/robophd/"
                      "asta_ds1000_20260101_000000/agents/iter1/agent.py"),
        command="cat ...",
    ))
    assert "engine-level" not in label


def test_tmp_scratch_classifies_tp():
    _, cat = catalog.classify(_rec(
        scope="read", blocked_path="/private/tmp",
        command="cd /tmp && mkdir x",
    ))
    assert cat == "TP"


# ---------------------------------------------------------------------
# FP-FIXED (must classify correctly on real records but NOT
# silently absorb genuine TPs with similar shapes)
# ---------------------------------------------------------------------


def test_interpreter_binary_with_invocation_in_cmd_is_fp_fixed():
    """Real interpreter-FP record: blocked is the resolved-symlink
    path (``python3.11``) while cmd uses the symlink (``python``).
    The cmd-awareness check must use the DIRNAME (shared between
    resolved and unresolved), not the full blocked path."""
    _, cat = catalog.classify(_rec(
        scope="read",
        blocked_path="/opt/anaconda3/envs/robophd_demo/bin/python3.11",
        command=("cd /Users/x/iter_010\n"
                 "/opt/anaconda3/envs/robophd_demo/bin/python -c '...'"),
    ))
    assert cat == "FP-FIXED"


def test_interpreter_path_without_invocation_in_cmd_is_not_fp_fixed():
    """Cmd-awareness defense: a path syntactically matching the
    interpreter regex but whose dirname is absent from cmd must NOT
    be silently mislabeled FP-FIXED. (e.g., a genuine recon read of
    a file whose path happens to start with /opt/anaconda3/.)"""
    label, _ = catalog.classify(_rec(
        scope="read",
        blocked_path="/opt/anaconda3/envs/some_env/bin/python3.11",
        command="cat /Users/me/secret",
    ))
    assert "interpreter binary" not in label


def test_sed_script_with_sed_in_cmd_is_fp_fixed():
    _, cat = catalog.classify(_rec(
        scope="read",
        blocked_path="/^FORMAT_VAR = /,$p",
        command=("diff <(sed -n '/^FORMAT_VAR = /,$p' a.py) "
                 "<(sed -n '/^FORMAT_VAR = /,$p' b.py)"),
    ))
    assert cat == "FP-FIXED"


def test_sed_lookalike_path_without_sed_in_cmd_is_not_fp_fixed():
    """The sed-script regex is loose enough that a real malicious
    path like /foo,bar/p$baz can syntactically match. Cmd-aware
    tightening: no ``sed`` in cmd -> NOT FP-FIXED, so a genuine
    recon read isn't silently dismissed as a fixed FP."""
    label, _ = catalog.classify(_rec(
        scope="read",
        blocked_path="/foo,bar/p$baz",
        command="cat /foo,bar/p$baz",
    ))
    assert "sed script" not in label


def test_auto_memory_write_classifies_fp_fixed():
    _, cat = catalog.classify(_rec(
        scope="write",
        blocked_path="/Users/x/.claude/projects/-slug/memory/insight.md",
        command="",
    ))
    assert cat == "FP-FIXED"


def test_tool_results_read_classifies_fp_fixed():
    _, cat = catalog.classify(_rec(
        scope="read",
        blocked_path=("/Users/x/.claude/projects/-slug/sess-uuid/"
                      "tool-results/spill.txt"),
        command="",
    ))
    assert cat == "FP-FIXED"


def test_tool_results_write_is_not_fp_fixed():
    """Carve-out is READ-only — a WRITE to the same path stays a TP
    (the read-only intent is enforced and the catalog must reflect
    that, not absorb the write into the fixed label)."""
    label, _ = catalog.classify(_rec(
        scope="write",
        blocked_path=("/Users/x/.claude/projects/-slug/sess/"
                      "tool-results/evil.txt"),
        command="echo pwned > ...",
    ))
    assert "tool-results" not in label
    # And explicitly: writes outside cwd that aren't memory writes are
    # a TP — but the predicate excludes ~/.claude/* from write-outside-cwd
    # so this falls through to UNKNOWN, surfacing as triage pressure.
    # That's acceptable; the load-bearing assertion is "NOT FP-FIXED".


# ---------------------------------------------------------------------
# FP-OPEN, LIMITATION, ERROR, HISTORICAL
# ---------------------------------------------------------------------


def test_tr_dash_d_slash_classifies_fp_open():
    _, cat = catalog.classify(_rec(
        scope="read", blocked_path="/", command="ls | tr -d /",
    ))
    assert cat == "FP-OPEN"


def test_find_root_not_swallowed_by_tr_pattern():
    """Both `find /` and `tr -d /` have blocked='/'; first-match-wins
    ordering puts the cmd-specific tr predicate before the generic
    find one, but a find-root command without `tr -d /` must still
    classify as TP (not silently absorbed by an over-loose tr regex)."""
    _, cat = catalog.classify(_rec(
        scope="read", blocked_path="/", command="find / -name foo",
    ))
    assert cat == "TP"


def test_parse_fail_classifies_limitation():
    _, cat = catalog.classify(_rec(
        scope="parse", reason="shlex.split failed", command="...",
    ))
    assert cat == "LIMITATION"


def test_internal_error_classifies_error():
    _, cat = catalog.classify(_rec(error="boom", tool="Bash"))
    assert cat == "ERROR"


def test_retired_scope_classifies_historical():
    _, cat = catalog.classify(_rec(
        scope="subprocess_bypass", blocked_path="", command="python <<EOF",
    ))
    assert cat == "HISTORICAL"


# ---------------------------------------------------------------------
# Sentinel: unrecognized records produce UNKNOWN so the script's
# exit-2-on-UNKNOWN triage signal stays alive.
# ---------------------------------------------------------------------


def test_unrecognized_record_classifies_unknown():
    """A novel pattern we haven't seen must fall through to UNKNOWN,
    not silently absorb into any existing label. UNKNOWN is what
    drives `add a pattern` pressure via exit code 2."""
    _, cat = catalog.classify(_rec(
        scope="read",
        blocked_path="/var/log/wifi.log",
        command="cat /var/log/wifi.log",
    ))
    assert cat == "UNKNOWN"
