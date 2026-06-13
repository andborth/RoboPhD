#!/usr/bin/env python3
"""PreToolUse hook: enforce experiment-dir read scope and iteration-dir
write scope.

Invoked once per tool call by Claude CLI when the per-experiment
.claude/settings.local.json wires it in. Reads JSON from stdin
({"tool_name", "tool_input", "cwd", ...}), writes JSON decision to
stdout, and on denial appends a structured record to
$ROBOPHD_EXPERIMENT_DIR/sandbox_denials.jsonl. researcher.py tails
that file in a daemon thread and re-emits each denial via its
standard logger, so denials surface to the run's normal log output.

Policy:
  - Read scope: anywhere under $ROBOPHD_EXPERIMENT_DIR.
  - Write scope: anywhere under $ROBOPHD_EVOLUTION_ITERATION_DIR (the harness-
    declared iteration root). If unset (legacy / non-evolution
    callers), falls back to the tool's literal cwd — the historical
    behavior. The iteration-rooted policy means the agent can edit
    `<iter>/agent.py` regardless of whether it has `cd`'d into a
    nested test subdir for probing. The security boundary is
    unchanged: a sibling iteration's dir, a sibling run, the source
    repo, and ~/.claude/* all remain outside write scope (they are
    NOT under $ROBOPHD_EVOLUTION_ITERATION_DIR).

Fail-closed: if the hook can't classify a command or hits an internal
error, the tool call is denied (exit 2) so a misclassified Bash
command never silently leaks reads.
"""

import json
import os
import os.path
import re
import shlex
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Bash variable-assignment prefix (`x=...`, `MY_VAR=...`). Used to
# identify leading-token assignments like `x=/sibling/a; cat $x` or
# the env-prefix form `x=/sibling/a cat $x`, neither of which has the
# path token in `args` where the classifier scans.
ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Heredoc start: `<<DELIM`, `<< DELIM`, `<<-DELIM`, `<<'DELIM'`, `<<"DELIM"`.
# DELIM is a bash identifier (letters/digits/underscore) when unquoted.
# `<<-` strips leading tabs from the body but doesn't change DELIM matching
# semantics for our purposes (we only need the body span).
HEREDOC_START_RE = re.compile(
    r"<<-?\s*"
    r"(?:"
    r"'([^'\n]+)'"               # 'DELIM'
    r"|"
    r'"([^"\n]+)"'              # "DELIM"
    r"|"
    r"([A-Za-z_][A-Za-z0-9_]*)"  # bare DELIM
    r")"
)

# Tools whose tool_input describes a single read.
READ_TOOLS = {"Read", "Glob", "Grep"}

# Tools whose tool_input describes a single write.
WRITE_TOOLS = {"Edit", "Write", "NotebookEdit", "MultiEdit"}

# Bash command names that read files. Path tokens map to the read scope.
# `awk` is NOT here — its first bare positional is a program (a
# `/regex/` range looks like a path), so it needs the dedicated branch
# in classify_bash_segment (mirrors `sed`).
BASH_READ_COMMANDS = {
    "cat", "head", "tail", "less", "more", "bat",
    "find", "fd",
    "jq", "ls", "du", "wc",
    "file", "stat", "diff", "comm", "cmp",
    "xxd", "od", "strings", "tree",
}

# grep-family: `grep [OPTS] PATTERN [FILE...]`. Like awk/sed, the first
# bare positional is the inline PATTERN (a regex), NOT a path — a
# `/regex/`-style operand (`grep -v "/agents/"`, `grep -E "/re|.."`)
# would otherwise read-deny as an absolute path. Only the trailing FILE
# operands are paths. Handled by the dedicated grep branch in
# classify_bash_segment. (`grep` was previously in BASH_READ_COMMANDS,
# which mis-classified the pattern as a read.)
BASH_GREP_COMMANDS = {"grep", "egrep", "fgrep", "rg", "ag"}

# Bash command names whose positional operands are NEVER filesystem
# paths. `tr SET1 SET2` operands are character sets — `tr` only ever
# reads stdin and writes stdout, so an operand is never opened as a
# file. Redirect targets (`> file`) are extracted before command
# dispatch and `$(...)` substitutions split into their own scope-checked
# segment, so neither tr's stdin source nor its output file escapes the
# check — only the bare SET operand (e.g. the lone `/` in `tr -d '/'`)
# is exempt.
#
# NOTE: `echo`/`printf` are deliberately NOT here even though their
# operands are also "just strings". Their *output* can become a path
# via command substitution — `cat $(echo /etc/passwd)` — and that read
# is caught precisely because the inner `echo /etc/passwd` segment
# scope-checks its `/etc/passwd` operand. Exempting echo would reopen
# that hole. tr is safe because its stdout is never re-interpreted as a
# path by the surrounding command (the substitution segment would be
# `tr ...`, whose own output we don't and needn't resolve). The cost is
# a rare spurious deny on `echo "$a / $b"` (arithmetic-looking literal),
# which is the strictly safer direction.
BASH_NO_PATH_OPERANDS = {"tr"}

# Bash command names that write to their final positional arg (others
# are read-from sources). E.g., `cp src1 src2 dst`, `mv src dst`.
BASH_WRITE_LAST_POSITIONAL = {"cp", "mv"}

# Bash command names where every positional path is a write target.
# `rm <path>...`, `mkdir <path>`, `touch <path>`, etc.
# Note: `dd` is NOT here — it uses `if=src` (read) and `of=dst` (write)
# named args, handled separately in classify_bash_segment.
BASH_WRITE_ALL_POSITIONAL = {
    "rm", "rmdir", "mkdir", "touch", "tee",
}

# Bash command names that have no path tokens worth scoping (we still
# scan tokens; if any path-shaped token appears it's checked, but we
# don't fail-close just because we don't know the command).
BASH_PASSTHROUGH = {
    "pwd", "whoami", "date", "hostname", "id", "uname",
    "echo", "printf", "true", "false", "test",
    "which", "type", "command",
    "git", "uv", "pip", "pip3", "python", "python3", "node", "npm",
    "env", "export", "alias", "unalias",
    "sleep", "yes", "seq",
}

# Command wrappers that run another command: `timeout 60 CMD`, `time
# CMD`, `nice -n 10 CMD`, `nohup CMD`, `stdbuf -oL CMD`, `ionice CMD`.
# Stripped down to the wrapped CMD so an interpreter relocated into
# argument position (`timeout 600 /opt/.../python foo.py`) is treated
# like the bare leading invocation it stands in for (command token not
# scope-checked). See the wrapper-strip block in classify_bash_segment.
BASH_COMMAND_WRAPPERS = {
    "timeout", "time", "nice", "ionice", "stdbuf", "nohup",
}

# Of those, the wrappers that take a bare POSITIONAL operand before the
# command: only `timeout DURATION CMD`. `nice`/`ionice` do NOT — their
# niceness is always flag-borne (`nice -n 10 CMD`, `ionice -c 2 CMD`);
# the bare form `nice CMD` runs CMD directly. Listing them here would
# make the strip consume the real command as a phantom operand and let
# its args escape (`nice cat /sibling/x` -> `/sibling/x` becomes the
# unchecked command token). `time` likewise takes no leading positional.
BASH_WRAPPER_LEADING_OPERAND = {"timeout"}

# A `timeout` DURATION operand: digits with an optional decimal and an
# optional unit suffix (s/m/h/d), e.g. `600`, `1.5s`, `2m`. Anchored so
# it never matches a path or a flag — only a genuine duration is consumed.
DURATION_RE = re.compile(r"^\d+(\.\d+)?[smhd]?$")

# Bash control-flow keywords. shlex collapses newlines to whitespace,
# so a multi-line `for ...; do echo a; cat /path; done` lands as
# segments split on `;` whose body segment is `do echo a` (or `do cat
# /path` etc.) — leading token is `do`, which isn't a command. We
# strip leading control keywords from each segment so the classifier
# reaches the real body command underneath.
BASH_CONTROL_KEYWORDS = {
    "for", "while", "until", "if", "case", "select",
    "do", "then", "else", "elif",
    "done", "fi", "esac",
}

# Special path strings that are never scope-checked.
PATH_EXEMPT = {"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty", "-"}

# A language interpreter living in a system bin dir, e.g.
# `/opt/anaconda3/envs/x/bin/python3.11`, `/usr/bin/python3`,
# `/opt/homebrew/bin/node`, `/usr/local/bin/ruby2.7`. Read-scope exempt:
# running the interpreter is benign, and a bare leading `/opt/.../python
# foo.py` is ALREADY allowed (the command token is never scope-checked).
# Denying it only when it lands in a non-command position — `PY=/opt/.../
# python; $PY ...`, `find ... -exec /opt/.../python ...`, `for p in
# /opt/.../python ...` — is a pure false positive: the same benign action,
# blocked by spelling. Exempting it removes that friction.
#
# Safety: matched against the realpath'd token (see check_read_paths), so
# a symlink `ln -s /etc/passwd ./py` resolves to /etc/passwd and FAILS
# this pattern — no exfil bypass. The path must point at an interpreter
# binary in a bin dir; a secret data file can't masquerade as one, and
# reading the public interpreter binary itself exfiltrates nothing.
INTERPRETER_BIN_RE = re.compile(
    r"^/(opt/anaconda3|opt/miniconda3|opt/homebrew|usr|usr/local)/"
    r"(.*/)?bin/(python|node|ruby|perl|pypy)[0-9.]*$"
)


def looks_like_path(token: str) -> bool:
    """Heuristic: does this token reference a filesystem path?"""
    if not token:
        return False
    if token.startswith("-") and not token.startswith("--"):
        return False  # short flag like -v
    if token.startswith("--") and "=" not in token:
        return False  # long flag like --recursive
    if token in (".", ".."):
        return True
    if token.startswith("~"):
        return True
    if token.startswith("/"):
        return True
    if "/" in token:
        return True
    return False


def normalize(path_str: str, cwd: str) -> str:
    """Resolve `path_str` against cwd and canonicalize via realpath.

    realpath collapses symlinks so an agent can't `ln -s /etc/passwd alias`
    and then read `alias` to bypass the scope check.
    """
    expanded = os.path.expanduser(path_str)
    if not os.path.isabs(expanded):
        expanded = os.path.join(cwd, expanded)
    return os.path.realpath(expanded)


def is_under(path: str, root: str) -> bool:
    """True iff `path` is `root` itself or a descendant of `root`."""
    root_real = os.path.realpath(root)
    path_real = os.path.realpath(path)
    if path_real == root_real:
        return True
    return path_real.startswith(root_real + os.sep)


def classify_bash_segment(tokens: list) -> tuple:
    """Classify a single Bash command segment.

    Returns ``(read_paths, write_paths, fail_reason)`` where
    ``fail_reason`` is always ``None`` in the current implementation —
    the third return slot is preserved for compatibility with callers
    and possible future re-introduction of fail-closed categories.

    Policy: visible path tokens are classified as reads or writes by
    command name (see ``BASH_READ_COMMANDS``, ``BASH_WRITE_*``,
    ``sed -i``, ``dd of=``). Constructs that run nested commands
    (``find -exec``, ``xargs``, ``$(...)``, subshells, process
    substitution) used to fail closed under a ``subprocess_bypass``
    branch; that branch was retired (2026-05-06) because (a) the
    out-of-scope cases it caught are also caught by the read-scope
    check on the visible path tokens those constructs accept, and
    (b) the in-scope cases were false positives that interfered with
    legitimate evolution work (find -exec on the iteration's own
    files, in-scope $() substitutions, etc.). The user explicitly
    accepted the loss of recall on inner-command invisibility: the
    write violations within the experiment dir that this branch
    uniquely caught are hypothetical, not observed.

    Unknown commands with path-shaped tokens default to read scope
    (path tokens flow into ``read_paths``); see the policy comment
    inside the function body.

    Handles redirects (``>``, ``>>``, ``&>``, ``2>``, ``<``) by
    extracting the redirect target out of ``tokens`` before command
    classification. Strips leading ``env VAR=val`` and shell control
    keywords (``for``, ``do``, ``then``, ...) before lookup.
    """
    read_paths: list = []
    write_paths: list = []

    # Pull redirects out first. We process tokens left-to-right; when we
    # see a redirect operator the next token is the redirect target.
    remaining: list = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in (">", ">>", "&>", "2>"):
            if i + 1 < len(tokens):
                write_paths.append(tokens[i + 1])
                i += 2
                continue
        if t == "<":
            if i + 1 < len(tokens):
                read_paths.append(tokens[i + 1])
                i += 2
                continue
        # Embedded redirect like `>file.txt` (no space).
        if t.startswith(">>") and len(t) > 2:
            write_paths.append(t[2:])
            i += 1
            continue
        if t.startswith(">") and len(t) > 1 and not t.startswith(">>"):
            write_paths.append(t[1:])
            i += 1
            continue
        remaining.append(t)
        i += 1

    # Strip leading `env VAR=val ...` prefix.
    while remaining and remaining[0] == "env":
        remaining = remaining[1:]
        while remaining and "=" in remaining[0] and not remaining[0].startswith("="):
            # Looks like KEY=value
            if "/" in remaining[0].split("=", 1)[0]:
                break
            remaining = remaining[1:]

    # Strip leading shell control keywords (`for`, `while`, `do`, `then`,
    # ...). shlex eats newlines, so a multi-line for-loop body whose
    # statements are newline-separated collapses to a single `;`-segment
    # whose first token is `do`. Without this strip the classifier would
    # try to look up `do` as a command and fail closed despite the actual
    # body command (cat / echo / etc.) being right behind it.
    while remaining and remaining[0] in BASH_CONTROL_KEYWORDS:
        remaining = remaining[1:]

    # Strip leading command-wrapper prefixes (`timeout 600 CMD ...`,
    # `time CMD ...`, `nice -n 10 CMD ...`, `nohup CMD ...`, ...). The
    # wrapper relocates the real CMD into argument position; since CMD is
    # an interpreter path (`/opt/.../python`) it then read-denies even
    # though a bare leading `/opt/.../python foo.py` is allowed (the
    # command token is never scope-checked — see test_*_interpreter_*).
    # Stripping the wrapper down to the wrapped CMD restores that policy.
    #
    # We deliberately handle ONLY the unambiguous, observed shapes —
    # `timeout DURATION CMD` and `<wrapper> CMD` — and bail on anything
    # with options. Per-flag arity is unknowable (does `-n` take the next
    # token? does a bundled `-o/path` carry a value?), and every wrong
    # guess is either a silent path escape or a phantom command token. So
    # the moment we see a flag (or a path where a bare operand should be)
    # we STOP stripping and leave the segment intact: the real command
    # stays in argument position and flows through normal scope-checking.
    # That can spuriously deny an exotic `nice -n 10 <interp>` (none seen
    # in practice) — the safe direction (a blocked legit run, never an
    # escape). Bare `nice CMD` / `time CMD` / `timeout N CMD` still strip.
    while remaining and remaining[0] in BASH_COMMAND_WRAPPERS:
        wrapper = remaining[0]
        rest = remaining[1:]
        if not rest:
            break
        # `timeout` takes a bare numeric DURATION before the command
        # (`timeout 600 CMD`, `timeout 1.5s CMD`). Consume it only if it
        # is a pure duration token (digits + optional unit suffix), never
        # a path or flag.
        if wrapper in BASH_WRAPPER_LEADING_OPERAND and DURATION_RE.match(rest[0]):
            rest = rest[1:]
            if not rest:
                break
        # Next token is the wrapped command itself — it MAY be a path
        # (`timeout 600 /opt/.../python`); that's the case we're here to
        # normalize, and a leading command token is never scope-checked,
        # so it's safe to make it the new head. We only bail on a FLAG,
        # which would mean unparsed wrapper options we won't guess at.
        #
        # Bailing leaves the segment for the normal unknown-command
        # branch, which defaults path args to READ scope. That is
        # deliberately PERMISSIVE, consistent with that branch's own
        # policy (see its comment): rather than fail-close into false
        # positives on legit in-scope work, accept a bounded residual —
        # a flagged-wrapper command that is actually a WRITE could write
        # within experiment_dir (e.g. a sibling iter dir, like
        # `timeout -k 5 rm /exp/agents/iterNNN/x`). That's a run-to-run
        # hygiene leak, NOT a sandbox escape: the path is still confined
        # to experiment_dir (read scope), so nothing reaches /etc, a
        # sibling RUN, the repo, or ~/.claude. The owner's stated
        # preference is false negatives over false positives, and this
        # shape has 0 occurrences in real runs. A path FUSED into an
        # option (`-o/sib/x`) likewise isn't caught — the same
        # pre-existing looks_like_path limitation; see
        # test_known_limitation_bundled_path_option_not_caught.
        if rest[0].startswith("-"):
            break
        remaining = rest

    # Strip leading variable-assignment prefixes (`x=/path/a`,
    # `A=1 B=/path cmd args`). These appear in two shapes:
    #   `x=/sibling/a; cat $x`     — assignment followed by `;` then use
    #   `x=/sibling/a cat $x`      — env-prefix on a command (no `env` kw)
    # In both shapes the path is *inside* the assignment token (left of
    # the rest of remaining), not in `args`, so the read-default scan
    # would miss it. Capture the value as a read-target if it's
    # path-shaped, then strip the assignment so the actual command
    # (if any) becomes the leading token.
    while remaining and ASSIGNMENT_RE.match(remaining[0]):
        _, _, value = remaining[0].partition("=")
        if value and looks_like_path(value):
            read_paths.append(value)
        remaining = remaining[1:]

    if not remaining:
        return read_paths, write_paths, None

    cmd = remaining[0]
    args = remaining[1:]

    # `sed [opts] [SCRIPT] FILE...`. `-i` makes the FILEs writes;
    # otherwise they're reads. Subtleties this block must get right:
    #   * the inline SCRIPT is NOT a path — a sed address range like
    #     `/^FOO/,$p` starts with `/` and would otherwise read as an
    #     absolute path (spurious deny; observed in production on a
    #     legitimate in-scope `diff <(sed -n '/re/,$p' a) <(... b)`).
    #   * in-place can be a BUNDLED short cluster (`-ni`, `-i.bak`) or
    #     `--in-place`; missing those classifies an in-place write as a
    #     read and would *allow* a write outside write scope (the only
    #     unsafe-direction error in the file — every other limitation
    #     fails toward deny). Any single-dash short cluster containing
    #     `i` counts; ambiguity over-classifies as write (safe: at
    #     worst a spurious deny, never an escape).
    #   * a `-f SCRIPTFILE` program file is always a READ, even under
    #     `-i` (only the edited target is the write).
    if cmd == "sed":
        def _is_inplace(a: str) -> bool:
            if a == "--in-place" or a.startswith("--in-place"):
                return True
            if a.startswith("-") and not a.startswith("--"):
                return "i" in a[1:]
            return False

        is_write = any(_is_inplace(a) for a in args)
        sink = write_paths if is_write else read_paths
        have_program = False      # set once -e/-f/--file or the inline
        expect_script_file = False  # next token is the `-f` program file
        j = 0
        while j < len(args):
            a = args[j]
            if expect_script_file:
                if looks_like_path(a):
                    read_paths.append(a)  # script file: always a read
                have_program = True
                expect_script_file = False
            elif a in ("-f", "--file"):
                expect_script_file = True
            elif a.startswith("--file="):
                t = a[len("--file="):]
                if looks_like_path(t):
                    read_paths.append(t)
                have_program = True
            elif a in ("-e", "--expression"):
                have_program = True
                j += 1  # skip the following inline program (not a path)
            elif a.startswith(("-e", "--expression=")):
                have_program = True
            elif a.startswith("-"):
                pass  # other option (incl. bundled -i forms)
            elif not have_program:
                have_program = True  # first bare positional = inline program
            elif looks_like_path(a):
                sink.append(a)
            j += 1
        return read_paths, write_paths, None

    # `awk [opts] 'PROGRAM' FILE...` — like sed, the inline PROGRAM is
    # not a path. An awk program is full of `/regex/` constructs (a
    # range like `/^## Step 2/,/^## Step 3/`, an action guard like
    # `/foo/{print}`) that all start with `/` and would otherwise
    # read-deny as absolute paths (observed in production on in-scope
    # `awk '/range/' trace.md` reads). Only the trailing FILE operands
    # are paths. Subtleties:
    #   * `-f PROGFILE` supplies the program from a file (a READ); the
    #     next token is that file, and no inline program is then expected.
    #   * `-v VAR=val` is an assignment option, never a path.
    #   * the FIRST bare positional (when no `-f`) is the inline program
    #     and is skipped; everything after it is a file operand.
    if cmd == "awk":
        have_program = False
        expect_progfile = False
        j = 0
        while j < len(args):
            a = args[j]
            if expect_progfile:
                if looks_like_path(a):
                    read_paths.append(a)  # -f program file: a read
                have_program = True
                expect_progfile = False
            elif a in ("-f", "--file"):
                expect_progfile = True
            elif a.startswith("--file="):
                t = a[len("--file="):]
                if looks_like_path(t):
                    read_paths.append(t)
                have_program = True
            elif a in ("-v", "--assign"):
                j += 1  # skip the following VAR=val (not a path)
            elif a.startswith(("-v", "--assign=")):
                pass  # bundled `-vVAR=val` / `--assign=VAR=val`
            elif a.startswith("-"):
                pass  # other option
            elif not have_program:
                have_program = True  # first bare positional = inline program
            elif looks_like_path(a):
                read_paths.append(a)  # file operand
            j += 1
        return read_paths, write_paths, None

    # `grep [OPTS] PATTERN [FILE...]` — like awk, the first bare
    # positional is the inline PATTERN (a regex), not a path. A
    # `/regex/`-style pattern (`grep -v "/agents/"`, `grep -iE "/re|.."`)
    # would otherwise read-deny as an absolute path. Only the trailing
    # FILE operands are paths. Subtleties:
    #   * `-e PATTERN` / `-f FILE` supply the pattern via flag, so there
    #     is NO inline-pattern positional — every bare positional is then
    #     a FILE. We MUST detect these (set have_pattern) or a real
    #     out-of-scope FILE would be skipped as the "pattern" (an escape).
    #     `-f FILE`'s own argument is itself a read (a pattern file).
    #   * value-taking option flags (`-m N`, `-A N`, `--include=GLOB`,
    #     ...) must not have their value misread as a pattern/path. We
    #     skip a known set of separated-value short flags; `--long=val`
    #     forms carry their value inline so need no lookahead.
    if cmd in BASH_GREP_COMMANDS:
        have_pattern = False
        expect_pattern_file = False       # token after -f is a pattern FILE (read)
        expect_skip_value = False         # token after a value-flag: skip
        # Separated-value flags whose NEXT token is a non-path value.
        grep_value_flags = {
            "-e", "--regexp", "-m", "--max-count", "-A", "--after-context",
            "-B", "--before-context", "-C", "--context", "-d", "--directories",
            "--color", "--colour", "--label", "--group-separator",
        }
        j = 0
        while j < len(args):
            a = args[j]
            if expect_pattern_file:
                if looks_like_path(a):
                    read_paths.append(a)   # -f pattern file: a read
                have_pattern = True
                expect_pattern_file = False
            elif expect_skip_value:
                expect_skip_value = False  # consume the flag's value
            elif a in ("-f", "--file"):
                expect_pattern_file = True
            elif a.startswith("--file="):
                t = a[len("--file="):]
                if looks_like_path(t):
                    read_paths.append(t)
                have_pattern = True
            elif a in ("-e", "--regexp"):
                have_pattern = True        # pattern supplied by flag
                expect_skip_value = True   # ...the next token IS that pattern
            elif a.startswith("--regexp="):
                have_pattern = True
            elif a in grep_value_flags:
                expect_skip_value = True
            elif a.startswith("--") and "=" in a:
                pass  # long option with inline value (no path operand)
            elif a.startswith("-"):
                pass  # other flag / bundled short cluster
            elif not have_pattern:
                have_pattern = True        # first bare positional = the PATTERN
            elif looks_like_path(a):
                read_paths.append(a)       # file operand
            j += 1
        return read_paths, write_paths, None

    # Commands whose positional operands are never paths (`tr` character
    # sets — see BASH_NO_PATH_OPERANDS). Redirect targets were already
    # pulled out above, so `tr a b > /out/file` still write-checks
    # `/out/file`; only the bare operands (e.g. the lone `/` in
    # `tr -d '/'`) are exempted here.
    if cmd in BASH_NO_PATH_OPERANDS:
        return read_paths, write_paths, None

    # `dd if=SRC of=DST bs=N count=N` — `if=` is a read source, `of=` a
    # write target. Other key=val args (bs, count, conv, status...) are
    # not paths. Falling through to BASH_WRITE_ALL_POSITIONAL would have
    # mis-classified `if=` as a write — caught in code review.
    if cmd == "dd":
        for a in args:
            if a.startswith("if="):
                target = a[len("if="):]
                if looks_like_path(target):
                    read_paths.append(target)
            elif a.startswith("of="):
                target = a[len("of="):]
                if looks_like_path(target):
                    write_paths.append(target)
            # Other dd named args (bs=, count=, conv=, status=, etc.)
            # aren't paths; ignore.
        return read_paths, write_paths, None

    if cmd in BASH_WRITE_LAST_POSITIONAL:
        positional = [a for a in args if not a.startswith("-")]
        if positional:
            write_paths.append(positional[-1])
            for a in positional[:-1]:
                read_paths.append(a)
        return read_paths, write_paths, None

    if cmd in BASH_WRITE_ALL_POSITIONAL:
        for a in args:
            if not a.startswith("-") and looks_like_path(a):
                write_paths.append(a)
        return read_paths, write_paths, None

    if cmd in BASH_READ_COMMANDS:
        for a in args:
            if not a.startswith("-") and looks_like_path(a):
                read_paths.append(a)
        return read_paths, write_paths, None

    if cmd in BASH_PASSTHROUGH:
        for a in args:
            if looks_like_path(a):
                read_paths.append(a)
        return read_paths, write_paths, None

    # Unknown command with path-shaped tokens. We can't tell read-vs-write,
    # but rather than fail-closing into an arms race against creative
    # agents inventing novel command shapes, default to treating path
    # tokens as reads. The read-scope check still denies anything outside
    # experiment_dir; the residual risk is that an unknown command which
    # is actually a write would write somewhere in experiment_dir
    # (possibly a sibling iter dir) — bounded, undesirable for run-to-run
    # hygiene, but not a sandbox-breaking exfil. The fail-closed
    # alternative was producing real false positives in evolution runs
    # (e.g., `frobnicate <path-in-scope>` denied even when intent was
    # clearly a read).
    for a in args:
        if not a.startswith("-") and looks_like_path(a):
            read_paths.append(a)
    return read_paths, write_paths, None


def strip_heredoc_bodies(command: str) -> str:
    """Remove heredoc bodies from a Bash command string before tokenizing.

    A heredoc is a way to send multi-line stdin to a command:

        cat <<EOF > scratch.py
        # Run with `python scratch.py`
        print(1)
        EOF

    The body is opaque content, not bash code we can usefully classify.
    Without stripping, our tokenizer sees the body's `(`, `)`, and
    backticks as standalone shell constructs and false-positives on
    legitimate file writes — a Python script with `print(1)` becomes a
    "subshell," a comment containing markdown backticks becomes
    "command substitution," etc.

    Stripping keeps the redirect target visible (the `> scratch.py`
    part still gets write-scope-checked) while leaving the body
    invisible. The tradeoff: an agent could exfil sibling files via
    heredoc-body backtick substitution (`<<EOF` is not literal — bash
    *will* substitute), but that exfil shape has not been observed in
    practice and the user's policy is "defend the patterns we've
    actually seen, don't preemptively block legitimate work."

    Handles:
      * `<<DELIM`, `<<-DELIM`, `<< DELIM`, `<<'DELIM'`, `<<"DELIM"`
      * Multiple heredocs in one command (rare; processed iteratively)
      * Unterminated heredoc (no closing DELIM line) — strips to end
    """
    out: list = []
    pos = 0
    while pos < len(command):
        m = HEREDOC_START_RE.search(command, pos)
        if not m:
            out.append(command[pos:])
            break
        delim = m.group(1) or m.group(2) or m.group(3)
        # Body begins after the next newline. The same-line tokens
        # *between* the heredoc start and that newline (e.g.,
        # `> outfile` redirects, pipes, additional args) MUST stay
        # visible to the classifier — appending only up to m.end()
        # would drop the redirect target and silently allow writes.
        nl = command.find("\n", m.end())
        if nl < 0:
            # No newline after heredoc start — there's no body. Append
            # the rest verbatim and we're done.
            out.append(command[pos:])
            break
        # Append everything up to and including the newline (covers
        # heredoc-start + same-line redirects/pipes/args).
        out.append(command[pos:nl + 1])
        # Find the closing delimiter on its own line. Bash matches a
        # line consisting only of DELIM (with optional leading whitespace
        # for the `<<-` form; we accept any leading whitespace since the
        # body is being discarded either way).
        end_pattern = re.compile(
            rf"^[ \t]*{re.escape(delim)}\s*$",
            re.MULTILINE,
        )
        em = end_pattern.search(command, nl + 1)
        if not em:
            # Unterminated heredoc — strip to end of command.
            break
        # Skip past the closing delim line.
        line_end = command.find("\n", em.end())
        pos = em.end() if line_end < 0 else line_end + 1
    return "".join(out)


def collapse_arithmetic_expansions(command: str) -> str:
    """Replace ``$((expr))`` arithmetic expansions with a literal ``0``.

    ``split_compound`` uses ``punctuation_chars=True``, which treats
    ``(`` and ``)`` as token separators (needed for subshell / process-
    substitution splitting). That mangles an arithmetic expansion glued
    inside a path token: ``iter${v}_v$((v-2))/agent.py`` tokenizes to
    ``iter${v}_v$``, ``((``, ``v-2``, ``))``, ``/agent.py`` — and the
    trailing ``/agent.py`` then read-denies as an absolute path (observed
    in production on an in-scope read of the run's own agents dir).

    Bash substitutes an arithmetic expansion with its integer result, so
    collapsing ``$((...))`` to ``0`` is faithful in shape: the enclosing
    token stays a single token whose path-ness (and scope) is unchanged
    (``iter${v}_v0/agent.py``). Handles nesting via paren depth-counting;
    an unterminated ``$((`` is left as-is (it will fail to parse and
    fail closed downstream, which is correct).

    Note: a literal ``$((`` inside single quotes is NOT an arithmetic
    expansion to bash, but collapsing it there is still harmless — the
    only effect is on punctuation-driven token splitting, and quoted
    content is opaque data we don't scope-check by value anyway.
    """
    out: list = []
    i, n = 0, len(command)
    while i < n:
        if command.startswith("$((", i):
            # Count paren depth starting from the first '(' (the two
            # opening parens of `$((` take depth to 2). The expansion
            # ends when depth returns to 0 — that index is the second
            # ')' of the closing '))'. Nested `$((...))` are handled
            # naturally since inner parens balance out.
            depth = 0
            j = i + 1
            while j < n:
                if command[j] == "(":
                    depth += 1
                elif command[j] == ")":
                    depth -= 1
                    if depth == 0:
                        out.append("0")
                        i = j + 1
                        break
                j += 1
            else:
                # Ran off the end without closing — unterminated. Leave
                # the literal char; downstream parse fails closed.
                out.append(command[i])
                i += 1
            continue
        out.append(command[i])
        i += 1
    return "".join(out)


def split_statement_newlines(command: str) -> str:
    """Replace unquoted, top-level newlines with ';'.

    shlex (used by ``split_compound``) treats a newline as plain
    whitespace, so ``a\\nb\\nc`` collapses into ONE segment and the
    leading command's classifier wrongly consumes the *later*
    statements' tokens. Observed in production: a ``cd <in-scope-dir>``
    on line 1 swallowed line 2's absolute ``/opt/.../python`` as one of
    *its* arguments, producing a spurious read-scope denial of the
    interpreter path. Newlines ARE statement separators in bash; making
    that explicit lets each statement be classified on its own command.

    Newlines inside single/double quotes are preserved (a quoted
    multi-line string is data, not a statement boundary — e.g.
    ``python -c "import os\\nos.do()"``). Heredoc bodies are already
    removed by ``strip_heredoc_bodies`` before this runs. A
    backslash-newline line-continuation is removed entirely (bash
    JOINS the two physical lines with no separator: ``a\\<nl>b`` -> the
    single token ``ab``). Emitting a space instead would be wrong and
    actively unsafe: ``/exp\\<nl>iltrate/secret`` would split into the
    in-scope token ``/exp`` plus a harmless-looking relative token,
    while bash reads the out-of-scope ``/expiltrate/secret``. Joining
    keeps it one token that the scope check correctly denies.

    Strictly increases splitting precision: more, smaller segments can
    only classify a path under a *narrower* command (or none, which
    still hits the read-scope default) — it never hides a path or
    widens scope.
    """
    out: list = []
    in_single = in_double = False
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if (c == "\\" and i + 1 < n and command[i + 1] == "\n"
                and not in_single and not in_double):
            # bash line-continuation: remove BOTH chars, join with
            # nothing (a\<nl>b -> ab). See docstring for why a space
            # here would be an exploitable mis-split.
            i += 2
            continue
        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
        if c == "\n" and not in_single and not in_double:
            out.append(";")
        else:
            out.append(c)
        i += 1
    return "".join(out)


def split_compound(command: str) -> list:
    """Split a Bash command on top-level pipes/semicolons/&&/|| boundaries.

    Returns a list of token-lists, one per segment. If shlex fails
    (unbalanced quotes, heredocs we don't model), returns None to
    signal "unparseable — fail closed."

    Uses ``punctuation_chars=True`` so separators glued to other tokens
    (``false;`` → ``false ;``) split correctly. Plain ``shlex.split``
    keeps ``;`` glued to the preceding token, which silently broke
    every for/while loop body classification.

    Also breaks on subshell / process- / command-substitution
    boundaries (``(``, ``)``, ``<(``, ``>(``). Without this an inner
    command keeps its outer command's classifier rules — e.g.
    ``diff <(sed -n '/re/,$p' f) ...`` was classified as ``diff`` so
    the ``sed`` script ``/re/,$p`` was scanned as a path and spuriously
    denied. Splitting here only makes classification *more* precise
    (each command judged by its own rules); every path token still
    flows through the same read/write scope check, so nothing is hidden
    and scope is never widened.
    """
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        all_tokens = list(lexer)
    except ValueError:
        return None

    separators = {"|", "||", "&&", ";", "&", "(", ")", "<(", ">("}
    segments: list = []
    current: list = []
    for tok in all_tokens:
        if tok in separators:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        segments.append(current)
    return segments


def append_denial_record(record: dict) -> None:
    """Append a JSON line to $ROBOPHD_EXPERIMENT_DIR/sandbox_denials.jsonl.

    Best-effort: if the env var or the directory is missing, write a
    fallback record to /tmp/robophd_sandbox_denials.jsonl so the failure
    isn't completely silent.
    """
    exp_dir = os.environ.get("ROBOPHD_EXPERIMENT_DIR")
    if exp_dir and os.path.isdir(exp_dir):
        out_path = os.path.join(exp_dir, "sandbox_denials.jsonl")
    else:
        out_path = "/tmp/robophd_sandbox_denials.jsonl"
    try:
        with open(out_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        # Last resort: stderr (the hook's stderr is captured by Claude
        # CLI's stderr buffer; researcher.py won't see it live, but
        # something is better than nothing).
        sys.stderr.write(f"[sandbox] could not append denial record: {record!r}\n")


def emit_decision(decision: str, reason: str = "") -> None:
    """Write the JSON decision envelope to stdout (Claude CLI reads it)."""
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
        }
    }
    if reason:
        payload["hookSpecificOutput"]["permissionDecisionReason"] = reason
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()


def parse_extra_read_paths(argv: list) -> list:
    """Parse --extra-read=PATH args from argv (any order, repeatable).

    Each PATH is canonicalized via realpath. The hook installer is
    responsible for passing absolute paths; relative paths are
    resolved against the hook's own cwd at parse time, which is
    rarely meaningful — kept as a defensive fallback.
    """
    extras: list = []
    for arg in argv[1:]:
        if arg.startswith("--extra-read="):
            extras.append(os.path.realpath(arg[len("--extra-read="):]))
    return extras


def project_slug(abs_path: str) -> str:
    """Reproduce Claude Code's path -> project-slug transform.

    Observed ground truth (pinned in test_sandbox_hook):
      /Users/a/Desktop/cc/robophd_runs/robophd/asta_ds1000_20260514_231614
        -> -Users-a-Desktop-cc-robophd-runs-robophd-asta-ds1000-20260514-231614
      /Users/a/Desktop/cc/RoboPhD
        -> -Users-a-Desktop-cc-RoboPhD

    i.e. each of '/', '_', '.' becomes '-'; case is preserved; a
    leading '/' yields a leading '-'.

    NON-INJECTIVE BY DESIGN: collapsing '/', '_', '.' to a single '-'
    means distinct absolute paths can map to the same slug (e.g.
    ``/a/b_c`` and ``/a/b/c`` both -> ``-a-b-c``). This lossiness is
    inherited from Claude Code's own transform, not introduced here —
    Claude itself would merge the memory of two slug-colliding project
    roots. We must mirror it exactly (an injective slug would simply
    fail to match Claude's real memory path), so the carve-out is
    precisely as collision-safe as Claude's native memory isolation,
    no weaker. test_sandbox_hook pins both the transform (ground truth)
    and the collision boundary explicitly.

    This transform — and the upstream rule that the auto-memory project
    root anchors at the experiment dir — is UNDOCUMENTED and
    version-unstable in Claude Code. We depend on it anyway only because
    the failure is safe and visible (see auto_memory_dir).
    """
    return re.sub(r"[/_.]", "-", abs_path)


def auto_memory_dir(experiment_dir: str):
    """Run-scoped Claude auto-memory carve-out dir, or None.

    Claude Code's built-in memory feature reads/writes under
    ``~/.claude/projects/<project-slug>/memory/``. Empirically the
    project slug for a sandboxed evolution session anchors at the
    EXPERIMENT ROOT (the ``.claude/`` dir ``_install_evolution_sandbox``
    drops there is what makes Claude treat it as the project root), so
    the slug is computed from ``ROBOPHD_EXPERIMENT_DIR``. Whitelisting
    exactly that one ``<slug>/memory/`` dir lets evolution use its own
    within-run memory while keeping a sibling run or the source repo out
    of reach.

    The containment is NOT structural — ``project_slug`` is lossy — it
    holds because: (1) ``EXPERIMENT_DIR`` is created by RoboPhD as
    ``<runs>/robophd/<task>_<timestamp>``, so its name is
    timestamp-unique and a prompt injection cannot induce a colliding
    EXP (it controls only the write target, not EXP); (2) every
    *meaningful* collision target (another run, the dev repo) differs in
    the timestamp component or in length — distinct timestamps do not
    fold together. The lossy-slug collision case is a known boundary,
    inherited from Claude's own transform and pinned explicitly in
    test_sandbox_hook, not an unbounded escape.

    Fragile but FAIL-SAFE: if Claude's anchor or slug transform drifts,
    the memory write targets a slug that is *not* this dir -> ordinary
    deny -> sandbox_denials.jsonl -> surfaced as a WARNING by
    researcher.py's tail thread. We notice; nothing escapes. This is an
    interim measure; the durable fix is a PreToolUse ``updatedInput``
    redirect that does not depend on the undocumented anchor at all.
    """
    home = os.path.expanduser("~")
    if not home or home == "~":
        return None
    slug = project_slug(experiment_dir)
    return os.path.realpath(
        os.path.join(home, ".claude", "projects", slug, "memory")
    )


def auto_session_dirs(cwd: str) -> list:
    """READ-ONLY carve-outs for Claude CLI's own session-project state.

    Claude Code stashes per-session state under
    ``<config-dir>/projects/<slug(cwd)>/`` — the session transcript
    ``<session-id>.jsonl`` and large-tool-output *spills* at
    ``<session-id>/tool-results/<id>.txt``. When a tool produces output
    too large to inline, the CLI writes the full content there and the
    model must ``Read`` it back to see the full result. Without this
    carve-out that Read is denied (outside ``EXPERIMENT_DIR``),
    silently degrading every iteration that produces a large tool
    output.

    Differences vs ``auto_memory_dir``:
      * keyed on the runtime ``cwd`` (the iteration dir for evolution
        sessions), NOT ``EXPERIMENT_DIR`` — Claude uses different
        project roots for transcripts/tool-results (cwd-derived) vs
        auto-memory (experiment-root). Observed empirically; both
        slugs appear under ``~/.claude/projects/`` for the same run.
      * READ-ONLY: the CLI writes these files itself, the model only
        Reads them back; the write-tool branch deliberately does not
        honor these dirs, keeping any ``~/.claude/...`` write denied.

    Tries multiple config-dir locations: ``$CLAUDE_CONFIG_DIR`` (the
    official override), ``~/.claude`` (default), ``~/.claude-secondary``
    (observed in alt installations). Dual raw/realpath cwd-slug for
    symlinked-root robustness, same as the memory carve-out. Same
    fragile-but-fail-safe profile: a slug drift just yields ordinary
    deny + logged record, never an escape.
    """
    home = os.path.expanduser("~")
    if not home or home == "~":
        return []
    # Prefer $CLAUDE_CONFIG_DIR when set (the official override); only
    # then is whitelisting the conventional fallbacks unnecessary. Else
    # we don't know which of ~/.claude vs ~/.claude-secondary Claude
    # will actually use (both observed in practice), so include both.
    env_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if env_dir:
        config_dirs = [env_dir]
    else:
        config_dirs = [
            os.path.join(home, ".claude"),
            os.path.join(home, ".claude-secondary"),
        ]

    out: list = []
    seen_slugs: set = set()
    for cand in (cwd, os.path.realpath(cwd)):
        slug = project_slug(cand)
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        for cd in config_dirs:
            p = os.path.realpath(os.path.join(cd, "projects", slug))
            if p not in out:
                out.append(p)
    return out


def deny_message(
    experiment_dir: str,
    write_root: str,
    blocked_path: str,
    scope: str,
    extra_read_roots: list = None,
    write_root_is_iteration: bool = False,
) -> str:
    """Format the human-readable denial reason shown to the agent.

    ``write_root`` is the policy-defined write root — the
    ``$ROBOPHD_EVOLUTION_ITERATION_DIR`` when set, else the tool's
    cwd. ``write_root_is_iteration`` indicates which mode is active;
    when True we add a clarifying note so the agent doesn't think it
    must `cd` to ``write_root`` (the whole point of the iteration-
    rooted policy is that it doesn't).
    """
    extra_lines = ""
    if extra_read_roots:
        bullets = "\n".join(f"      {p}" for p in extra_read_roots)
        extra_lines = f"    plus task-specific resource roots:\n{bullets}\n"
    if write_root_is_iteration:
        write_note = (
            f"  • Write (Edit/Write/Bash redirects, cp/mv/rm/mkdir/touch/sed -i targets): "
            f"anywhere under your evolution iteration directory:\n"
            f"    {write_root}\n"
        )
    else:
        write_note = (
            f"  • Write (Edit/Write/Bash redirects, cp/mv/rm/mkdir/touch/sed -i targets): "
            f"only under\n"
            f"    {write_root}\n"
        )
    return (
        "Sandbox denied. This Claude CLI session has two scopes:\n"
        f"  • Read (Read/Glob/Grep/Bash file inputs): anywhere under\n"
        f"    {experiment_dir}\n"
        f"{extra_lines}"
        f"{write_note}"
        "\n"
        f"Blocked: {blocked_path} — outside {scope} scope. Sibling experiment runs and the source repo are out of scope by policy."
    )


def evaluate_bash(
    command: str,
    cwd_real: str,
    experiment_dir: str,
    write_root: str,
    extra_read_roots: list,
) -> tuple:
    """Pure Bash scope decision shared by main() and replay.

    Runs the same transform + tokenize + per-segment classify + scope-
    check pipeline main() uses, with NO IO (no emit, no append, no env
    reads). Returns ``(decision, scope, blocked)`` where:

      * ``("deny", "parse", "")``   — unparseable (heredoc / bad quotes)
      * ``("deny", "read", path)``  — a read token outside read scope
      * ``("deny", "write", path)`` — a write token outside write scope
      * ``("allow", None, None)``   — every segment cleared

    Extracting this lets ``catalog_sandbox_denials.py`` replay a logged
    denial through the LIVE policy ("does HEAD still deny this?") using
    the exact code main() runs, so the catalog's FP-FIXED verdict can
    never drift from the hook's real behavior.
    """
    if not command:
        return "allow", None, None

    command = strip_heredoc_bodies(command)
    command = collapse_arithmetic_expansions(command)
    command = split_statement_newlines(command)

    segments = split_compound(command)
    if segments is None:
        # shlex couldn't parse — fail closed (see main()'s parse branch).
        return "deny", "parse", ""

    for tokens in segments:
        if not tokens:
            continue
        read_paths, write_paths, _fail_reason = classify_bash_segment(tokens)
        ok, blocked = check_read_paths(read_paths, experiment_dir,
                                       extra_read_roots, cwd_real)
        if not ok:
            return "deny", "read", blocked
        ok, blocked = check_paths(write_paths, write_root, cwd_real, "write")
        if not ok:
            return "deny", "write", blocked
    return "allow", None, None


def replay_denial_record(
    record: dict,
    experiment_dir: str,
    write_root: str = None,
    extra_read_roots: list = None,
) -> tuple:
    """Replay a logged denial through the LIVE policy.

    Answers "would HEAD still deny this record?" by reconstructing the
    decision inputs from the record and running the same code main()
    runs. Returns ``(still_denies: bool, blocked: str | None)``.

    ``experiment_dir`` is the run dir (the parent of the
    ``sandbox_denials.jsonl`` the record came from). ``write_root``
    defaults to the record's ``cwd`` — the legacy (narrowest) write
    root; an iteration-rooted run had an equal-or-broader write scope,
    so a cwd-rooted replay can only ever be MORE likely to say "still
    denies", never less. That's the safe direction for an FP-FIXED claim
    (we never falsely declare a write fixed).

    Replay is best-effort: a record we can't faithfully reconstruct
    (missing fields, unreadable shape) raises, and the caller falls back
    to pattern classification.
    """
    tool = record.get("tool", "")
    cwd = record.get("cwd") or experiment_dir
    if extra_read_roots is None:
        extra_read_roots = []
    if write_root is None:
        write_root = cwd

    exp = os.path.realpath(experiment_dir)
    cwd_real = os.path.realpath(cwd)
    write_root_real = os.path.realpath(write_root)

    # Reconstruct the same auto carve-outs main() applies, so a denial
    # that HEAD would now allow via a carve-out replays as "fixed". The
    # auto-memory dir is derived from experiment_dir; the auto-session
    # read dirs from cwd. Both are recomputed from the record's own
    # inputs, so the replay matches what the live hook would grant.
    memory_carveouts = []
    md = auto_memory_dir(exp)
    if md:
        memory_carveouts.append(md)
    read_roots = list(extra_read_roots) + memory_carveouts
    for d in auto_session_dirs(cwd_real):
        if d not in read_roots:
            read_roots.append(d)

    if tool == "Bash":
        decision, _scope, blocked = evaluate_bash(
            record.get("command", ""), cwd_real, exp,
            write_root_real, read_roots,
        )
        return decision == "deny", blocked

    # Read/Write tools log the normalized blocked_path (already an
    # absolute realpath) rather than the raw file_path. Re-check that
    # path under the current scope rules — normalize is idempotent on an
    # absolute realpath, so this faithfully reproduces the tool branch.
    blocked_path = record.get("blocked_path", "")
    if not blocked_path:
        # parse/error records, or a cwd-outside-experiment denial with no
        # usable path — not replayable here.
        raise ValueError("record has no replayable blocked_path")

    if tool in READ_TOOLS:
        ok, blocked = check_read_paths([blocked_path], exp,
                                       read_roots, cwd_real)
        return (not ok), blocked
    if tool in WRITE_TOOLS:
        ok, blocked = check_paths([blocked_path], write_root_real,
                                  cwd_real, "write")
        # Same Write-tool auto-memory carve-out main() applies.
        if not ok and any(is_under(blocked, mc) for mc in memory_carveouts):
            ok, blocked = True, None
        return (not ok), blocked

    raise ValueError(f"unreplayable tool: {tool!r}")


def check_read_paths(
    paths: list,
    experiment_dir: str,
    extra_read_roots: list,
    cwd: str,
) -> tuple:
    """Return (ok, blocked_path) for a list of read-target path tokens.

    A path is OK iff its realpath is under experiment_dir OR under any
    extra read root.
    """
    for p in paths:
        if p in PATH_EXEMPT:
            continue
        norm = normalize(p, cwd)
        if is_under(norm, experiment_dir):
            continue
        if any(is_under(norm, root) for root in extra_read_roots):
            continue
        # A system interpreter binary (matched on the REALPATH, so a
        # symlink to a secret resolves elsewhere and isn't exempt) is
        # read-scope exempt: running it is benign and the bare leading
        # invocation is already allowed (command token unchecked). See
        # INTERPRETER_BIN_RE.
        if INTERPRETER_BIN_RE.match(norm):
            continue
        return False, norm
    return True, None


def check_paths(
    paths: list,
    scope_root: str,
    cwd: str,
    scope_name: str,
) -> tuple:
    """Return (ok, blocked_path) for a list of path tokens.

    `ok` is True iff every path resolves under scope_root. On failure
    `blocked_path` is the canonical path of the first violation.
    """
    for p in paths:
        if p in PATH_EXEMPT:
            continue
        norm = normalize(p, cwd)
        if not is_under(norm, scope_root):
            return False, norm
    return True, None


def main() -> int:
    raw = sys.stdin.read()
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"[sandbox] HOOK ERROR: bad stdin JSON: {exc}\n")
        append_denial_record({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "error": f"bad stdin JSON: {exc}",
        })
        return 2

    tool_name = envelope.get("tool_name", "")
    tool_input = envelope.get("tool_input", {}) or {}
    cwd = envelope.get("cwd") or os.getcwd()

    experiment_dir_env = os.environ.get("ROBOPHD_EXPERIMENT_DIR")
    if not experiment_dir_env:
        sys.stderr.write("[sandbox] HOOK ERROR: ROBOPHD_EXPERIMENT_DIR not set\n")
        append_denial_record({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "error": "ROBOPHD_EXPERIMENT_DIR not set",
            "tool": tool_name,
        })
        return 2

    experiment_dir = os.path.realpath(experiment_dir_env)
    cwd_real = os.path.realpath(cwd)

    # Write-scope root: the harness-declared iteration root if set,
    # else the runtime cwd (legacy behavior). The iteration-rooted
    # form lets the agent edit <iter>/agent.py from any subdir it has
    # `cd`'d into for probing. The security boundary is unchanged —
    # iteration dirs are under experiment_dir, sibling iterations /
    # runs / repo / ~/.claude are not under THIS iteration's root.
    write_root_env = os.environ.get("ROBOPHD_EVOLUTION_ITERATION_DIR")
    write_root = os.path.realpath(write_root_env) if write_root_env else cwd_real
    # Threaded into every deny_message so the agent sees the right
    # explanation: "your evolution iteration directory" when the
    # harness has declared one, generic "only under <path>" in the
    # cwd-fallback case. Tests above pin both modes.
    write_root_is_iteration = bool(write_root_env)

    extra_read_roots = parse_extra_read_paths(sys.argv)

    # Run-scoped auto-memory carve-out. Read is granted via the normal
    # extra-read mechanism (covers Read/Glob/Grep and Bash read
    # commands, so the agent can recall its own within-run memory).
    # Write is granted ONLY to the WRITE_TOOLS branch below — the Bash
    # write path deliberately does NOT honor these dirs, so an injection
    # using `bash -c '... > ~/.claude/.../memory/x'` stays denied; only
    # Claude's legitimate Write-tool auto-memory is relocated-friendly.
    #
    # Dual slug (raw env value AND its realpath): we do NOT know whether
    # Claude Code slugs the literal cwd-derived path or a realpath'd one
    # (undocumented). On a symlinked deployment root the two diverge; a
    # single-form guess would silently degrade the carve-out to no-op
    # (fail-safe, but the feature just wouldn't work). Whitelisting both
    # plausible forms covers the two realistic behaviors at trivial
    # cost; anything more exotic still fails safe + logged. Both forms
    # are this run's own timestamp-unique slug, so scope is unchanged.
    memory_carveouts = []
    for cand in (experiment_dir_env, experiment_dir):
        d = auto_memory_dir(cand)
        if d and d not in memory_carveouts:
            memory_carveouts.append(d)
    extra_read_roots = extra_read_roots + memory_carveouts

    # Session-project READ carve-out: the Claude CLI spills large tool
    # outputs and stores transcripts under <config-dir>/projects/
    # <slug(cwd)>/. The model must Read its own spilled output back; the
    # raw experiment-dir read scope would deny. See auto_session_dirs.
    # READ-ONLY by design — NOT added to memory_carveouts (write),
    # so an injection writing to ~/.claude/... via Write or Bash stays
    # denied.
    session_carveouts = auto_session_dirs(cwd)
    for d in session_carveouts:
        if d not in extra_read_roots:
            extra_read_roots = extra_read_roots + [d]

    # Cwd must itself be under the experiment dir; otherwise the write
    # scope is broken before we start.
    if not is_under(cwd_real, experiment_dir):
        emit_decision(
            "deny",
            deny_message(experiment_dir, write_root, cwd_real, "write",
                         extra_read_roots,
                         write_root_is_iteration=write_root_is_iteration),
        )
        append_denial_record({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "tool": tool_name,
            "scope": "write",
            "blocked_path": cwd_real,
            "reason": "cwd is outside experiment_dir",
            "command": tool_input.get("command", ""),
            "cwd": cwd_real,
        })
        return 0

    # Classify the tool.
    if tool_name in READ_TOOLS:
        path = tool_input.get("file_path") or tool_input.get("path") or ""
        if not path:
            # Glob/Grep with no path — search runs from cwd, which is
            # already inside experiment_dir. Allow.
            return 0
        ok, blocked = check_read_paths([path], experiment_dir, extra_read_roots, cwd_real)
        if ok:
            return 0
        emit_decision("deny", deny_message(experiment_dir, write_root, blocked, "read",
                                           extra_read_roots,
                                           write_root_is_iteration=write_root_is_iteration))
        append_denial_record({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "tool": tool_name,
            "scope": "read",
            "blocked_path": blocked,
            "command": "",
            "cwd": cwd_real,
        })
        return 0

    if tool_name in WRITE_TOOLS:
        path = tool_input.get("file_path") or tool_input.get("path") or ""
        if not path:
            sys.stderr.write(f"[sandbox] HOOK ERROR: {tool_name} with no file_path\n")
            append_denial_record({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "error": f"{tool_name} with no file_path",
                "tool": tool_name,
            })
            return 2
        ok, blocked = check_paths([path], write_root, cwd_real, "write")
        # Run-scoped auto-memory carve-out (Write-tool path only). A
        # write outside write_root is still allowed iff it lands in
        # exactly this run's ~/.claude/projects/<slug>/memory/ dir
        # (raw-env or realpath slug form). Different slug (sibling
        # run / source repo / drifted anchor) -> not under any
        # carve-out -> denied + logged. Containment holds because
        # experiment-dir names are timestamp-unique and RoboPhD owns
        # EXPERIMENT_DIR (an injection can't induce a colliding EXP);
        # see auto_memory_dir for why the lossy-slug collision
        # boundary is inherited from Claude, not introduced here.
        if not ok and any(is_under(blocked, mc) for mc in memory_carveouts):
            ok, blocked = True, None
        if ok:
            return 0
        emit_decision("deny", deny_message(experiment_dir, write_root, blocked, "write",
                                           extra_read_roots,
                                           write_root_is_iteration=write_root_is_iteration))
        append_denial_record({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "tool": tool_name,
            "scope": "write",
            "blocked_path": blocked,
            "command": "",
            "cwd": cwd_real,
        })
        return 0

    if tool_name == "Bash":
        command = tool_input.get("command", "")
        if not command:
            return 0  # nothing to check

        # Decision is made by evaluate_bash (the same pure pipeline the
        # catalog replays): strip heredoc bodies (opaque stdin data that
        # would false-positive the `(`/`)` punctuation check), collapse
        # `$((expr))` (else split_compound breaks on `((`/`))`), make
        # newlines explicit statement separators, then per-segment
        # classify + scope-check. find -exec / xargs / $(...) / subshells
        # / unknown commands all flow their visible path tokens into the
        # read check (no fail-closed subprocess_bypass branch). Here we
        # only translate the verdict into the emit + log IO.
        decision, scope, blocked = evaluate_bash(
            command, cwd_real, experiment_dir, write_root, extra_read_roots,
        )
        if decision == "allow":
            return 0

        if scope == "parse":
            # shlex couldn't parse — likely heredoc, unbalanced quotes,
            # or a construct we don't model. Fail closed. Log the
            # transformed command (matches prior behavior).
            emit_decision(
                "deny",
                "Sandbox denied. Could not parse this Bash command (heredoc, "
                "unbalanced quotes, or unsupported construct). Use Read/Edit/"
                "Write tools instead, or simplify the command.",
            )
            append_denial_record({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "tool": "Bash",
                "scope": "parse",
                "blocked_path": "",
                "command": split_statement_newlines(
                    collapse_arithmetic_expansions(
                        strip_heredoc_bodies(command))),
                "cwd": cwd_real,
                "reason": "shlex.split failed",
            })
            return 0

        emit_decision(
            "deny",
            deny_message(experiment_dir, write_root, blocked, scope,
                         extra_read_roots,
                         write_root_is_iteration=write_root_is_iteration),
        )
        append_denial_record({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "tool": "Bash",
            "scope": scope,
            "blocked_path": blocked,
            "command": command,
            "cwd": cwd_real,
        })
        return 0

    # Unknown tool — passthrough. Claude CLI may add tools we don't
    # know; we don't want to break unrelated functionality.
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.stderr.write(f"[sandbox] HOOK ERROR:\n{traceback.format_exc()}\n")
        try:
            append_denial_record({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "error": traceback.format_exc(),
            })
        except Exception:
            pass
        sys.exit(2)
