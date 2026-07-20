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


# Replay stubs: the catalog now decides FP-FIXED/FP-OPEN/TP-NOW-ALLOWED
# from the LIVE hook verdict, not a static pattern. A test that asserts
# one of those outcomes supplies a stub standing in for
# sandbox_hook.replay_denial_record.
def _replay_allows(rec):
    """Live hook NO LONGER denies (the FP is fixed / policy loosened)."""
    return (False, None)


def _replay_denies_same(rec):
    """Live hook still denies, on the SAME path the record logged."""
    return (True, rec.get("blocked_path", ""))


def _replay_denies_other(rec):
    """Live hook still denies, but on a DIFFERENT path than logged."""
    return (True, "/some/other/still-blocked/path")


# ---------------------------------------------------------------------
# True positives
# ---------------------------------------------------------------------


def test_find_root_classifies_tp():
    label, cat = catalog.classify(_rec(
        scope="read", blocked_path="/", command="find / -name foo",
    ))
    assert cat == "TP"
    assert "find /" in label


def test_hallucinated_path_malformed_run_name_classifies_tp():
    """Real observed case: agent constructed `/.../asta_ds1000_20264_
    214900/iteration_002/agent_seed_X/problems/420/test_result.md`
    when the actual run is `asta_ds1000_20260524_214900` and iterations
    live under `evolution_output/`. Two-axis hallucination (timestamp
    digits AND missing dir segment), goes deep enough to look real."""
    label, cat = catalog.classify(_rec(
        scope="read",
        blocked_path=("/Users/andrew/Desktop/cc/robophd_runs/robophd/"
                      "asta_ds1000_20264_214900/iteration_002/"
                      "agent_seed_rbihfvr4/problems/420/test_result.md"),
        command="",
    ))
    assert cat == "TP"
    assert "hallucinated" in label


def test_real_sibling_run_path_not_classified_as_hallucinated():
    """A path with a VALID <task>_<8d>_<6d> run-name segment is a
    real sibling-run access, not a hallucination — must fall through
    to the sibling-run TP, not be absorbed as a malformed-name TP."""
    label, _ = catalog.classify(_rec(
        scope="read",
        blocked_path=("/Users/andrew/Desktop/cc/robophd_runs/robophd/"
                      "asta_ds1000_20260101_000000/evolution_output/"
                      "iteration_002/agent.py"),
        command="cat ...",
    ))
    assert "hallucinated" not in label


def test_engine_level_recon_not_classified_as_hallucinated():
    """`ls <engine>/agents/` only goes one level under engine, not
    enough for the hallucination pattern (which requires 2+ deeper
    segments). Falls through to engine-level recon."""
    label, _ = catalog.classify(_rec(
        scope="read",
        blocked_path="/Users/andrew/Desktop/cc/alt_robophd_runs/robophd/agents",
        command="ls /Users/andrew/Desktop/cc/alt_robophd_runs/robophd/agents/",
    ))
    assert "hallucinated" not in label


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


def test_find_on_conda_env_classifies_tp():
    """Agent searches `/opt/anaconda3/envs/<env>` for a module — a
    real out-of-scope find seen in the catalog. Distinct shape from
    `find /` (covered separately)."""
    _, cat = catalog.classify(_rec(
        scope="read",
        blocked_path="/opt/anaconda3/envs/robophd_demo",
        command='find /opt/anaconda3/envs/robophd_demo -name "model_registry*"',
    ))
    assert cat == "TP"


def test_find_on_system_dir_requires_path_in_cmd():
    """The find-on-system-path predicate is cmd-aware: a deny on the
    same path via cat/grep doesn't get silently absorbed as a find."""
    label, cat = catalog.classify(_rec(
        scope="read",
        blocked_path="/opt/anaconda3/envs/robophd_demo",
        command="cat /opt/anaconda3/envs/robophd_demo/something",
    ))
    assert "find on out-of-scope" not in label


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


def test_interpreter_binary_read_is_fp_fixed_when_replay_allows():
    """Any read of an interpreter BINARY — assignment, find -exec, loop
    list, probe, leading-token run — is INTENTed FP now that the hook
    exempts interpreter binaries from read scope. With a replay that
    allows it, FP-FIXED."""
    label, cat = catalog.classify(_rec(
        scope="read",
        blocked_path="/opt/anaconda3/envs/robophd_demo/bin/python3.11",
        command=("PY=/opt/anaconda3/envs/robophd_demo/bin/python\n"
                 "$PY -c '...'"),
    ), replay=_replay_allows)
    assert cat == "FP-FIXED"
    assert "interpreter path read" in label


def test_interpreter_binary_intent_fp_open_when_replay_denies():
    """Same intent, but a replay that STILL denies (same path) -> FP-OPEN.
    Proves the catalog reads the verdict from the live hook, not a static
    'fixed' assertion."""
    _, cat = catalog.classify(_rec(
        scope="read",
        blocked_path="/opt/anaconda3/envs/robophd_demo/bin/python3.11",
        command="PY=/opt/anaconda3/envs/robophd_demo/bin/python; $PY foo.py",
    ), replay=_replay_denies_same)
    assert cat == "FP-OPEN"


def test_interpreter_library_path_is_not_interpreter_fp():
    """A `.../lib/pythonX/site-packages/...` read (or a find into the
    conda lib tree) merely CONTAINS `python`; it's a real out-of-scope
    access, NOT the exempt interpreter-binary shape. Must NOT match the
    interpreter FP — it falls to a find/system-dir TP instead."""
    label, cat = catalog.classify(_rec(
        scope="read",
        blocked_path="/opt/anaconda3/envs/x/lib/python3.11/site-packages/foo",
        command="find /opt/anaconda3/envs/x/lib/python3.11/site-packages/foo",
    ), replay=_replay_denies_same)
    assert "interpreter path read" not in label
    assert cat == "TP"


def test_grep_slash_pattern_is_fp():
    """`grep -v "/agents/"` — the slash-delimited regex operand is misread
    as a path. With `grep` in the command it's INTENTed FP; a replay that
    still denies (the tokenizer FP is unfixed) -> FP-OPEN."""
    label, cat = catalog.classify(_rec(
        scope="read", blocked_path="/agents",
        command='find . -name "*.py" | grep -v "/agents/" | head',
    ), replay=_replay_denies_same)
    assert "grep pattern" in label
    assert cat == "FP-OPEN"


def test_grep_regex_alternation_is_fp():
    """A `grep -E "/ ?1000|million|/1e"` alternation string starting with
    `/` and full of regex metachars is misread as a path -> grep FP."""
    label, _ = catalog.classify(_rec(
        scope="read", blocked_path="/ ?1000|/ ?1_000|million|/1e",
        command='grep -iE "/ ?1000|/ ?1_000|million|/1e" prog | head',
    ), replay=_replay_denies_same)
    assert "grep pattern" in label


def test_grep_real_path_arg_not_swallowed_by_grep_fp():
    """Guard: a genuine out-of-scope FILE arg to grep (a normal path, no
    regex metachars, not a bare /word) must not be mislabeled the grep
    pattern FP — it's a real read. `grep foo /sibling/run/x` denies."""
    label, cat = catalog.classify(_rec(
        scope="read",
        blocked_path="/Users/x/robophd_runs/robophd/other_20260101_000000/agents/a/agent.py",
        command="grep foo /Users/x/robophd_runs/robophd/other_20260101_000000/agents/a/agent.py",
    ), replay=_replay_denies_same)
    assert "grep pattern" not in label


def test_system_dir_probe_is_tp():
    """`ls /opt/anaconda3/envs/` — listing a system directory (not an
    interpreter binary, not a find root) is a correct out-of-scope deny."""
    label, cat = catalog.classify(_rec(
        scope="read", blocked_path="/opt/anaconda3/envs",
        command="/opt/x/bin/python -c 1; ls /opt/anaconda3/envs/ 2>/dev/null",
    ), replay=_replay_denies_same)
    assert cat == "TP"
    assert "system directory" in label


def test_find_root_after_newline_cd_is_tp():
    """`cd <run>\\nfind / ...` — the `find /` token is newline-preceded
    after a multi-line cd. The TP must still catch it (the old `" find "`
    substring missed it)."""
    _, cat = catalog.classify(_rec(
        scope="read", blocked_path="/",
        command="cd /Users/x/run\nfind / -name doc 2>/dev/null | head",
    ), replay=_replay_denies_same)
    assert cat == "TP"


def test_sed_script_with_sed_in_cmd_is_fp_fixed():
    _, cat = catalog.classify(_rec(
        scope="read",
        blocked_path="/^FORMAT_VAR = /,$p",
        command=("diff <(sed -n '/^FORMAT_VAR = /,$p' a.py) "
                 "<(sed -n '/^FORMAT_VAR = /,$p' b.py)"),
    ), replay=_replay_allows)
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
    # The auto-memory-write FP pattern is Write-tool-gated (a Bash write
    # to the same dir is a deliberate injection-hardening deny, not an
    # FP), so the record must carry a WRITE_TOOL to exercise it.
    _, cat = catalog.classify(_rec(
        tool="Write",
        scope="write",
        blocked_path="/Users/x/.claude/projects/-slug/memory/insight.md",
        command="",
    ), replay=_replay_allows)
    assert cat == "FP-FIXED"


def test_tool_results_read_classifies_fp_fixed():
    _, cat = catalog.classify(_rec(
        scope="read",
        blocked_path=("/Users/x/.claude/projects/-slug/sess-uuid/"
                      "tool-results/spill.txt"),
        command="",
    ), replay=_replay_allows)
    assert cat == "FP-FIXED"


def test_tmp_task_output_spill_read_classifies_fp_fixed():
    """The CLI's own background-task output spill under
    <tmp>/claude-<uid>/<slug>/... is a legit readback (auto_scratch_dirs
    carve-out), so a read the live hook now allows classifies FP-FIXED —
    the direct analog of the ~/.claude tool-results spill. Both /tmp and
    the macOS /private/tmp realpath form are covered."""
    for path in (
        "/tmp/claude-501/-slug-iteration-020/uuid/tasks/abc.output",
        "/private/tmp/claude-501/-slug-iteration-020/uuid/tasks/abc.output",
    ):
        _, cat = catalog.classify(_rec(
            scope="read", blocked_path=path, command=f"cat {path}",
        ), replay=_replay_allows)
        assert cat == "FP-FIXED", path


def test_tmp_task_output_spill_still_denied_is_fp_open():
    """A /tmp session-spill read the live hook STILL denies (e.g. a
    cross-session slug) stays visible as FP-OPEN rather than silently
    passing — the replay backstop, same as the tool-results pattern."""
    _, cat = catalog.classify(_rec(
        scope="read",
        blocked_path="/tmp/claude-501/-other-slug/uuid/tasks/x.output",
        command="cat /tmp/claude-501/-other-slug/uuid/tasks/x.output",
    ), replay=_replay_denies_same)
    assert cat == "FP-OPEN"


def test_tmp_write_is_not_task_output_fp_fixed():
    """The spill carve-out is READ-only — a WRITE to /tmp stays a TP
    (scratch attempt), not absorbed into the fixed label."""
    label, _ = catalog.classify(_rec(
        scope="write",
        blocked_path="/tmp/claude-501/-slug/uuid/tasks/evil.output",
        command="echo x > /tmp/claude-501/-slug/uuid/tasks/evil.output",
    ), replay=_replay_allows)
    assert "task-output spill" not in label


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


def test_tr_dash_d_slash_classifies_fp_fixed():
    """`tr -d /` is fixed in the hook (BASH_NO_PATH_OPERANDS); with a
    replay that allows it, it's FP-FIXED."""
    _, cat = catalog.classify(_rec(
        scope="read", blocked_path="/", command="ls | tr -d /",
    ), replay=_replay_allows)
    assert cat == "FP-FIXED"


def test_find_root_not_swallowed_by_tr_pattern():
    """Both `find /` and `tr -d /` have blocked='/'; first-match-wins
    ordering puts the cmd-specific tr predicate before the generic
    find one, but a find-root command without `tr -d /` must still
    classify as TP (not silently absorbed by an over-loose tr regex)."""
    _, cat = catalog.classify(_rec(
        scope="read", blocked_path="/", command="find / -name foo",
    ), replay=_replay_denies_same)
    assert cat == "TP"


def test_find_root_with_tr_elsewhere_still_tp():
    """Guard the tr predicate's `find /` exclusion: a command that pipes
    a real `find /` scan into tr must classify TP (the find-root deny),
    not be absorbed as a tr FP."""
    _, cat = catalog.classify(_rec(
        scope="read", blocked_path="/",
        command="find / -name x 2>/dev/null | tr -d ' '",
    ), replay=_replay_denies_same)
    assert cat == "TP"


def test_awk_range_program_classifies_fp_fixed():
    """awk `/regex/,/regex/` range blocked as a path is now fixed."""
    _, cat = catalog.classify(_rec(
        scope="read", blocked_path="/^## Step 2/,/^## Step 3",
        command="awk '/^## Step 2/,/^## Step 3/' trace.md",
    ), replay=_replay_allows)
    assert cat == "FP-FIXED"


def test_awk_action_program_classifies_fp_fixed():
    """awk `/regex/{action}` guard blocked as a path is now fixed."""
    _, cat = catalog.classify(_rec(
        scope="read", blocked_path="/[multiblock]/{found=0} /n=2/{c++}",
        command="cat x | awk '/[multiblock]/{found=0} /n=2/{c++}'",
    ), replay=_replay_allows)
    assert cat == "FP-FIXED"


def test_fp_intent_open_when_replay_still_denies_same_path():
    """An FP-intented record whose replay STILL denies on the SAME path is
    FP-OPEN — the named false positive genuinely still recurs at HEAD
    (e.g. the wrapper-prefix interpreter FP before its hook fix lands)."""
    _, cat = catalog.classify(_rec(
        scope="read", blocked_path="/^## Step 2/,/^## Step 3",
        command="awk '/^## Step 2/,/^## Step 3/' trace.md",
    ), replay=_replay_denies_same)
    assert cat == "FP-OPEN"


def test_fp_intent_other_still_denied_when_replay_denies_different_path():
    """An FP-intented record whose replay denies on a DIFFERENT path is
    NOT the named FP — the command has a separate still-valid block (e.g.
    an interpreter path inside `cat > /tmp/x` whose real deny is the /tmp
    write). Surfaced as a TP so a real positive isn't mislabeled FP."""
    label, cat = catalog.classify(_rec(
        scope="read", blocked_path="/opt/anaconda3/envs/x/bin/python3.11",
        command="cat > /tmp/t.py <<'PY'\n/opt/anaconda3/envs/x/bin/python /tmp/t.py\nPY",
    ), replay=_replay_denies_other)
    assert cat == "TP"
    assert "OTHER-STILL-DENIED" in label


def test_arithmetic_split_path_classifies_fp_fixed():
    """A `/agent.py`-style stray component from a `$((...))` split is
    now fixed (collapse_arithmetic_expansions)."""
    _, cat = catalog.classify(_rec(
        scope="read", blocked_path="/agent.py",
        command="f=/run/iter${v}_v$((v-2))/agent.py; head $f",
    ), replay=_replay_allows)
    assert cat == "FP-FIXED"


def test_arithmetic_pattern_requires_dollar_paren():
    """Guard: the arithmetic FP pattern must require `$((` in the cmd so
    a genuine out-of-scope `/agent.py`-shaped read (no arithmetic) isn't
    INTENTed FP. With no `$((`, no FP pattern matches -> a replay that
    denies yields a non-FP category."""
    label, cat = catalog.classify(_rec(
        scope="read", blocked_path="/agent.py",
        command="cat /agent.py",
    ), replay=_replay_denies_same)
    assert cat not in ("FP-FIXED", "FP-OPEN"), (label, cat)


def test_parse_fail_no_replay_classifies_limitation():
    """Without a live policy to replay against, a parse-fail stays
    LIMITATION (conservative — we can't observe whether it's since been
    fixed)."""
    _, cat = catalog.classify(_rec(
        scope="parse", reason="shlex.split failed", command="...",
    ))
    assert cat == "LIMITATION"


def test_parse_fail_now_parses_classifies_parse_fixed():
    """A parse-fail the live hook now parses AND allows is a fixed FP —
    PARSE-FIXED, observed by replay (not frozen as LIMITATION). This is
    the heredoc-elision case: the fail-closed deny is gone at HEAD."""
    _, cat = catalog.classify(_rec(
        scope="parse", blocked_path="", command="python - <<'EOF'\nx\nEOF",
    ), replay=_replay_allows)
    assert cat == "PARSE-FIXED"


def test_parse_fail_still_unparseable_stays_limitation():
    """A parse-fail the live hook STILL can't parse (replay denies with an
    empty blocked path — a parse deny) stays LIMITATION."""
    # _replay_denies_same echoes the record's blocked_path, which is ""
    # for a parse record -> (True, "") -> still a parse deny.
    _, cat = catalog.classify(_rec(
        scope="parse", blocked_path="", command="cat 'unterminated",
    ), replay=_replay_denies_same)
    assert cat == "LIMITATION"


def test_parse_fail_now_denies_on_real_scope_classifies_tp():
    """A parse-fail that now PARSES but denies on a real scope (replay
    returns a non-empty blocked path) was never a false positive — it's a
    correct deny (TP) previously mis-attributed to a parse failure."""
    _, cat = catalog.classify(_rec(
        scope="parse", blocked_path="",
        command="cat <<'EOF' > /etc/x 2>&1 | tail\ny\nEOF",
    ), replay=_replay_denies_other)
    assert cat == "TP"


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
    drives `add a pattern` pressure via exit code 2. Use a path under no
    system root and no runs root (a bare home-dir read) so no TP/FP
    pattern claims it."""
    _, cat = catalog.classify(_rec(
        scope="read",
        blocked_path="/Users/someone/Documents/notes.txt",
        command="cat /Users/someone/Documents/notes.txt",
    ))
    assert cat == "UNKNOWN"


# ---------------------------------------------------------------------
# Replay-combination contract (the load-bearing redesign): FP-FIXED /
# FP-OPEN / TP-NOW-ALLOWED are derived from the live hook verdict, never
# asserted statically.
# ---------------------------------------------------------------------


def test_no_replay_fallback_reports_fp_intent_as_open():
    """Without a replay (replay=None / --no-replay), an FP-intented record
    can't be confirmed fixed, so it reports the conservative FP-OPEN
    rather than a stale FP-FIXED. The safe, attention-drawing direction."""
    _, cat = catalog.classify(_rec(
        scope="read", blocked_path="/", command="ls | tr -d /",
    ), replay=None)
    assert cat == "FP-OPEN"


def test_no_replay_fallback_reports_tp_intent_as_tp():
    _, cat = catalog.classify(_rec(
        scope="read", blocked_path="/", command="find / -name foo",
    ), replay=None)
    assert cat == "TP"


def test_tp_now_allowed_when_replay_allows_a_tp():
    """A record we judge a true positive but the live hook NO LONGER
    denies -> TP-NOW-ALLOWED (a possible scope regression the sweep must
    surface, not bury)."""
    _, cat = catalog.classify(_rec(
        scope="read", blocked_path="/", command="find / -name foo",
    ), replay=_replay_allows)
    assert cat == "TP-NOW-ALLOWED"


def test_replay_exception_falls_back_to_conservative_mapping():
    """If replay raises (unreconstructable record), classify must not
    crash — it falls back to the intent's conservative static mapping."""
    def _boom(rec):
        raise RuntimeError("cannot reconstruct")
    _, cat = catalog.classify(_rec(
        scope="read", blocked_path="/", command="ls | tr -d /",
    ), replay=_boom)
    assert cat == "FP-OPEN"


def test_static_records_bypass_replay():
    """ERROR / HISTORICAL records have no live decision to observe — they
    keep their static category even when a (bogus) replay is given. (A
    parse LIMITATION is NOT static — it IS replayed now; see the
    PARSE-FIXED tests above.)"""
    def _wrong(rec):
        return (False, None)  # would say 'allowed' if consulted
    _, cat = catalog.classify(_rec(error="boom", tool="Bash"), replay=_wrong)
    assert cat == "ERROR"
    _, cat = catalog.classify(_rec(
        scope="subprocess_bypass", blocked_path="", command="x",
    ), replay=_wrong)
    assert cat == "HISTORICAL"
