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
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

# tree-sitter-bash is the Bash front-end: it parses the real shell
# grammar (quoting, escapes, heredocs, arithmetic/command/process
# substitution, redirects, control flow), replacing the former
# hand-rolled preprocessor stack whose partial quote/escape tracking
# was a recurring source of false-positive denials. Imported
# defensively: if the dependency is missing, evaluate_bash fails CLOSED
# (deny) rather than run unprotected — a loud, safe failure. It is a
# hard dependency; see requirements.txt.
try:
    import tree_sitter_bash as _ts_bash
    from tree_sitter import Language as _TSLanguage, Parser as _TSParser
    _TS_LANGUAGE = _TSLanguage(_ts_bash.language())
    _TS_PARSER = _TSParser(_TS_LANGUAGE)
    _TS_IMPORT_ERROR = None
except Exception as _ts_exc:  # pragma: no cover - dependency missing at runtime
    _TS_PARSER = None
    # Capture WHY, so the fail-closed denial can name the missing
    # dependency instead of blaming the command (see main()'s parse
    # branch). Without this, a missing install looks identical to a
    # malformed command — a loud but misleading failure.
    _TS_IMPORT_ERROR = repr(_ts_exc)

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
    # `usr` covers `/usr/local/...` too via the `(.*/)?` middle, so no
    # separate `usr/local` alternative is needed.
    r"^/(opt/anaconda3|opt/miniconda3|opt/homebrew|usr)/"
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
    #   * value-taking option flags (`-m N`, `-A N`, ...) must not have
    #     their value misread as a pattern/path. We skip a known set of
    #     flags that take a SEPARATED value; `--long=val` forms carry the
    #     value inline so need no lookahead.
    #
    # CRITICAL: only flags that ALWAYS take a separated value belong in
    # grep_value_flags. An OPTIONAL-value flag (`--color[=WHEN]`,
    # `--colour`) takes its value ONLY attached (`--color=auto`), never as
    # the next token — listing it would consume the real PATTERN as a
    # phantom value, leaving the trailing FILE to be misread as the
    # pattern and NEVER scope-checked (a read-scope escape:
    # `grep --color foo /sibling/x` would open /sibling/x). The
    # wrong-direction error is the only unsafe one here, so when unsure
    # whether a flag separates its value, leave it OUT (a wrongly-kept
    # token at worst over-classifies a pattern as a path -> spurious deny,
    # never an escape).
    if cmd in BASH_GREP_COMMANDS:
        have_pattern = False
        expect_pattern_file = False       # token after -f is a pattern FILE (read)
        expect_skip_value = False         # token after a value-flag: skip
        # Flags that ALWAYS take a SEPARATED value (the next token). NOT
        # included: optional-value flags like --color/--colour (attached
        # `=WHEN` only) — see the CRITICAL note above.
        grep_value_flags = {
            "-e", "--regexp", "-m", "--max-count", "-A", "--after-context",
            "-B", "--before-context", "-C", "--context", "-d", "--directories",
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


# ---------------------------------------------------------------------
# Bash front-end (tree-sitter-bash)
#
# Parse the command with tree-sitter-bash and walk the concrete syntax
# tree, routing each path-bearing node to the read or write scope. This
# replaces the former hand-rolled stack (strip_heredoc_bodies,
# collapse_arithmetic_expansions, split_statement_newlines,
# split_compound) plus their manual quote/escape tracking, which was the
# recurring source of false-positive denials (a legit in-scope command
# mis-tokenized and blocked). The per-command classifier
# classify_bash_segment (which arg tokens are reads vs writes, by command
# name) is UNCHANGED — it is fed a clean ``[name, arg, ...]`` token list
# extracted from each ``command`` node.
#
# Fail-closed: a parse with any error node (unbalanced quotes, a
# construct the grammar rejects) denies with scope "parse", matching the
# old shlex-ValueError behavior. A missing tree-sitter dependency does
# the same, so the sandbox never silently runs unprotected.
# ---------------------------------------------------------------------

# Subtrees that are opaque data, not shell to descend into for path
# tokens. A heredoc BODY is stdin content (usually a file being
# written); its same-line ``> target`` redirect is a SIBLING node and is
# still scope-checked. Skipping the body mirrors the old
# strip_heredoc_bodies: opaque body, visible redirect.
_TS_OPAQUE_NODES = {"heredoc_body"}

# Node types that introduce their own nested command(s) — the walk
# descends into them so an inner out-of-scope path (e.g. ``cat
# $(cat /sibling)``) is scope-checked, but the OUTER command does not
# treat them as a path token (their runtime value is unknown; the inner
# command is what actually reads).
_TS_SUBSTITUTION_NODES = {"command_substitution", "process_substitution"}


def _strip_line_continuations(command: str) -> str:
    r"""Remove bash line-continuations (``\<newline>`` -> joined).

    bash strips a backslash-newline before tokenizing, so
    ``/exp\<nl>_escape/secret`` is the single word
    ``/exp_escape/secret``. tree-sitter-bash does NOT perform this join
    (it yields two words), which would let a path split at an in-scope
    prefix boundary slip the scope check. Joining here, before parsing,
    restores bash semantics.

    Quote-insensitive by design: inside single quotes bash keeps the
    literal backslash-newline, but that content is opaque data we never
    scope-check by value, and a mis-join only ever MERGES two tokens
    into one — a narrowing, the safe direction (a merged out-of-scope
    token denies; it can never widen scope).
    """
    return command.replace("\\\n", "")


def _elide_heredocs(command: str) -> str:
    """Remove heredoc operators AND bodies, preserving the rest of each
    start line (other redirects, pipes, args).

    tree-sitter-bash cannot parse some heredoc + redirect + pipe
    combinations — ``cmd <<EOF ... EOF 2>&1 | tail`` errors even though
    ``<<EOF | tail`` and ``<<EOF 2>&1`` each parse alone. This is a
    FALLBACK, used only when the primary parse fails and a heredoc is
    present (see evaluate_bash): a heredoc body is opaque stdin we never
    scope-check, so eliding the whole heredoc lets tree-sitter parse the
    surrounding command. A same-line write target (``<<EOF > out``) is
    preserved and still scope-checked — only the ``<<DELIM`` operator and
    the body lines are dropped. This is deliberately NOT the old
    strip_heredoc_bodies (which kept ``<<DELIM`` and would leave an
    unterminated heredoc tree-sitter also rejects); it removes the
    operator too.

    Heredoc bodies are opaque here exactly as they are on the primary
    path (the walk skips ``heredoc_body`` nodes), so this introduces no
    new blind spot — an unquoted-heredoc body that bash would expand was
    already out of scope-checking by design.
    """
    out: list = []
    pos = 0
    while pos < len(command):
        m = HEREDOC_START_RE.search(command, pos)
        if not m:
            out.append(command[pos:])
            break
        delim = m.group(1) or m.group(2) or m.group(3)
        # Keep the start line with the `<<DELIM` operator excised, so any
        # same-line redirect/pipe/arg survives.
        out.append(command[pos:m.start()])
        nl = command.find("\n", m.end())
        if nl < 0:
            # No newline after the heredoc start -> no body. Keep the rest
            # of the line (operator already dropped) and finish.
            out.append(command[m.end():])
            break
        out.append(command[m.end():nl + 1])  # rest of start line + newline
        # Drop the body up to and including the closing delimiter line.
        end_pattern = re.compile(rf"^[ \t]*{re.escape(delim)}\s*$",
                                 re.MULTILINE)
        em = end_pattern.search(command, nl + 1)
        if not em:
            # Unterminated body (e.g. a truncated-as-logged command):
            # drop everything after the start line.
            break
        line_end = command.find("\n", em.end())
        pos = em.end() if line_end < 0 else line_end + 1
    return "".join(out)


def _ts_word(node) -> str:
    """Reconstruct the shell-word string a value node contributes to
    path classification.

    Quotes are resolved by the grammar. An embedded arithmetic expansion
    collapses to a digit (faithful in shape — bash substitutes the
    integer result, and the enclosing token's path-ness is unchanged). A
    command/process substitution keeps its raw text ONLY when embedded in
    a larger token (concatenation/string), so the surrounding path keeps
    its shape; as a standalone argument it is dropped by
    ``_command_tokens`` (the inner command is scope-checked by the walk).
    Returns "" for nodes carrying no static token.
    """
    t = node.type
    if t in ("word", "number", "variable_name", "simple_expansion",
             "expansion", "regex"):
        return node.text.decode(errors="replace")
    if t == "arithmetic_expansion":
        return "0"
    if t in _TS_SUBSTITUTION_NODES:
        return node.text.decode(errors="replace")
    if t == "raw_string":  # '...' : literal; strip the single quotes
        s = node.text.decode(errors="replace")
        return s[1:-1] if len(s) >= 2 and s[0] == "'" else s
    if t == "ansi_c_string":  # $'...'
        s = node.text.decode(errors="replace")
        return s[2:-1] if s.startswith("$'") and s.endswith("'") else s
    if t in ("string", "translated_string"):  # "..." : literal + inline exp
        parts: list = []
        for c in node.children:
            if c.type == "string_content":
                parts.append(c.text.decode(errors="replace"))
            elif c.type == "arithmetic_expansion":
                parts.append("0")
            elif c.type in ("simple_expansion", "expansion",
                            "command_substitution"):
                parts.append(c.text.decode(errors="replace"))
        return "".join(parts)
    if t == "concatenation":
        return "".join(_ts_word(c) for c in node.children)
    return ""


def _command_tokens(cmd_node) -> list:
    """Extract a ``[name, arg, ...]`` token list from a ``command`` node
    for classify_bash_segment.

    Variable assignments, redirects, and (standalone) substitutions are
    excluded here — the tree walk handles them structurally — so the
    classifier receives exactly the command name and its literal
    argument words, the shape its per-command grammar expects.
    """
    tokens: list = []
    for child in cmd_node.children:
        ct = child.type
        if ct == "command_name":
            tokens.append(child.text.decode(errors="replace"))
        elif ct in ("variable_assignment", "file_redirect",
                    "heredoc_redirect") or ct in _TS_SUBSTITUTION_NODES:
            continue  # handled by the walk, not a literal arg token
        else:
            tok = _ts_word(child)
            if tok:
                tokens.append(tok)
    return tokens


def _assignment_value(node) -> str:
    """The right-hand value of a ``variable_assignment`` node as a word
    token (``x=/a`` -> ``/a``), or "" if not a static path-shaped value."""
    for c in node.children:
        if not c.is_named or c.type == "variable_name":
            continue
        return _ts_word(c)
    return ""


def _redirect_parts(node) -> tuple:
    """Return ``(operator, target)`` for a ``file_redirect`` node.

    The operator carries the direction (``>``/``>>``/``&>``/``2>`` write,
    ``<`` read); the target is the file word. A leading
    ``file_descriptor`` child (the ``2`` in ``2>``) is not a target.
    """
    op = ""
    target = ""
    for c in node.children:
        if not c.is_named:
            txt = c.text.decode(errors="replace")
            if "<" in txt or ">" in txt:
                op = txt
        elif c.type == "file_descriptor":
            continue
        else:
            tok = _ts_word(c)
            if tok:
                target = tok
    return op, target


def _for_list_words(node) -> list:
    """Loop-list element words of a ``for_statement`` (``for p in A B; ...``
    -> ``[A, B]``). These are read operands: ``for p in /sibling/x; do
    cat $p; done`` must deny on the out-of-scope list element, matching
    the old classifier which routed the stripped loop body's leading
    tokens through the read check. The do-group body's commands are
    visited separately by the walk.
    """
    out: list = []
    for c in node.children:
        if not c.is_named or c.type in ("variable_name", "do_group",
                                        "compound_statement", "comment"):
            continue
        tok = _ts_word(c)
        if tok:
            out.append(tok)
    return out


def _test_command_reads(node) -> list:
    """Path-shaped operand words inside a ``[ ... ]`` / ``[[ ... ]]`` test.

    Mirrors the old parser, which fed ``[`` to the unknown-command branch
    and read-checked its path tokens (so ``[ -f /sibling/x ]`` — an
    existence/metadata probe of an out-of-scope path — was denied). Test
    operators (``-f``, ``-s``, ``=``, ...) are their own ``test_operator``
    nodes, not words, so they are not collected. Command/process
    substitutions are skipped here — the main walk reaches their inner
    ``command`` nodes and scope-checks those directly, so a
    ``[[ $(cat /sibling) ]]`` content read is still caught (and not
    double-counted). A ``concatenation`` is taken as one token rather than
    descending into its parts.
    """
    out: list = []

    def visit(n):
        t = n.type
        if t in _TS_SUBSTITUTION_NODES or t == "command":
            return  # handled by the main walk
        if t in ("word", "string", "raw_string", "ansi_c_string",
                 "concatenation"):
            tok = _ts_word(n)
            if tok and looks_like_path(tok):
                out.append(tok)
            return  # a token is atomic; don't descend into its parts
        for c in n.children:
            visit(c)

    for c in node.children:
        visit(c)
    return out


# The effective, non-reconstructable scope inputs for THIS hook
# invocation. main() populates this (set_scope_context) once the inputs
# are known; append_denial_record stamps them into every denial record so
# offline replay (replay_denial_record) can faithfully rebuild the
# decision. These two inputs — the iteration write-root and the
# config-provided extra read roots (e.g. BIRD_DATA_DIR) — are the ones
# replay CANNOT re-derive from the record alone (it can rebuild the auto
# memory/session carve-outs from experiment_dir/cwd, but not these).
#
# LOGGING ONLY. Nothing in the enforcement path ever reads this — the
# allow/deny decision is computed from the live arguments in main(), and
# this context is consulted solely when writing the post-decision record.
_SCOPE_CONTEXT: dict = {}


def set_scope_context(write_root=None, extra_read_roots=None) -> None:
    """Record the effective scope inputs for denial logging (see
    _SCOPE_CONTEXT). Best-effort and never raises: a failure here must not
    perturb the decision path that calls it."""
    try:
        if write_root is not None:
            _SCOPE_CONTEXT["write_root"] = str(write_root)
        if extra_read_roots is not None:
            _SCOPE_CONTEXT["extra_read_roots"] = [str(r) for r in extra_read_roots]
    except Exception:  # pragma: no cover - defensive; logging must not fail
        pass


def append_denial_record(record: dict) -> None:
    """Append a JSON line to $ROBOPHD_EXPERIMENT_DIR/sandbox_denials.jsonl.

    Stamps the effective scope inputs (set_scope_context) into the record
    when absent, so replay can reconstruct the decision. Best-effort: if
    the env var or the directory is missing, write a fallback record to
    /tmp/robophd_sandbox_denials.jsonl so the failure isn't completely
    silent.
    """
    # Stamp scope inputs without clobbering anything a call site set
    # explicitly. setdefault on a plain dict can't raise, but keep the
    # whole block defensive — logging must never break the deny path.
    try:
        for k, v in _SCOPE_CONTEXT.items():
            record.setdefault(k, v)
    except Exception:  # pragma: no cover - defensive
        pass
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


def auto_scratch_dirs(cwd: str) -> list:
    """READ-ONLY carve-outs for the Claude CLI's per-session scratch root.

    Besides ``<config-dir>/projects/`` (see auto_session_dirs), the CLI
    keeps a per-session scratch tree under
    ``<tmp>/claude-<uid>/<slug(cwd)>/<session-id>/`` — an advertised
    ``scratchpad/`` plus ``tasks/<id>.output`` spills, where a
    background task's full output lands for the model to Read back
    when it exceeds the inline limit. Without this carve-out that Read
    is denied (outside EXPERIMENT_DIR): observed in autoresearch run
    sudoku_20260709_215531, where a val-eval score was lost to exactly
    this denial and the session journaled a sentinel string instead of
    the number.

    Same key (slug of the runtime cwd), same READ-ONLY rationale (the
    CLI writes these files itself; any model write to ``/tmp/...``
    stays denied), and same fragile-but-fail-safe profile as
    auto_session_dirs: slug or layout drift => ordinary deny + logged
    record, never an escape. Which tmp base the CLI uses is
    undocumented — the literal ``/tmp`` (observed on macOS, realpathing
    to /private/tmp) and ``tempfile.gettempdir()`` are both covered;
    each form is still scoped to this run's own cwd slug.
    """
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        return []
    leaf = f"claude-{getuid()}"
    bases = ["/tmp"]
    tmpdir = tempfile.gettempdir()
    if tmpdir and tmpdir not in bases:
        bases.append(tmpdir)
    out: list = []
    seen_slugs: set = set()
    for cand in (cwd, os.path.realpath(cwd)):
        slug = project_slug(cand)
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        for base in bases:
            p = os.path.realpath(os.path.join(base, leaf, slug))
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

    Parses ``command`` with tree-sitter-bash and walks the parse tree,
    routing every path-bearing node to the read or write scope. NO IO
    (no emit, no append, no env reads). Returns ``(decision, scope,
    blocked)`` where:

      * ``("deny", "parse", "")``   — unparseable (error node / bad quotes)
      * ``("deny", "read", path)``  — a read token outside read scope
      * ``("deny", "write", path)`` — a write token outside write scope
      * ``("allow", None, None)``   — every path cleared

    Extracting this lets ``catalog_sandbox_denials.py`` replay a logged
    denial through the LIVE policy ("does HEAD still deny this?") using
    the exact code main() runs, so the catalog's FP-FIXED verdict can
    never drift from the hook's real behavior.

    All reads are checked before any writes: on a command that violates
    both, the read violation is reported (matching the old per-segment
    read-before-write ordering).
    """
    if not command:
        return "allow", None, None
    if _TS_PARSER is None:
        # Security-critical dependency missing — fail closed so the
        # sandbox never runs unprotected (see the import block).
        return "deny", "parse", ""

    command = _strip_line_continuations(command)
    root = _TS_PARSER.parse(command.encode()).root_node
    if root.has_error and "<<" in command:
        # tree-sitter-bash rejects some heredoc + redirect + pipe combos
        # (e.g. `cmd <<EOF ... EOF 2>&1 | tail`), a grammar limitation.
        # The heredoc body is opaque stdin we never scope-check, so retry
        # with heredocs elided (operator + body removed, other redirects/
        # pipe preserved). If the elided form parses clean, walk it; else
        # fall through to fail-closed. Narrow fallback — the happy path is
        # untouched (this only runs after a real parse error with a `<<`).
        reparsed = _TS_PARSER.parse(_elide_heredocs(command).encode()).root_node
        if not reparsed.has_error:
            root = reparsed
    if root.has_error:
        # Unbalanced quotes / a construct the grammar rejects — fail
        # closed, same direction as the old shlex-ValueError branch.
        return "deny", "parse", ""

    read_paths: list = []
    write_paths: list = []
    stack = [root]
    while stack:
        node = stack.pop()
        ntype = node.type
        if ntype in _TS_OPAQUE_NODES:
            continue  # heredoc body: opaque stdin data, don't descend
        if ntype == "command":
            rp, wp, _ = classify_bash_segment(_command_tokens(node))
            read_paths.extend(rp)
            write_paths.extend(wp)
        elif ntype == "variable_assignment":
            val = _assignment_value(node)
            if val and looks_like_path(val):
                read_paths.append(val)
        elif ntype == "file_redirect":
            op, target = _redirect_parts(node)
            if target and looks_like_path(target):
                if "<" in op and ">" in op:
                    # `<>` opens the file read-write: check BOTH scopes so
                    # correctness doesn't rest on the (currently true, but
                    # implicit) write-scope ⊆ read-scope invariant.
                    read_paths.append(target)
                    write_paths.append(target)
                elif ">" in op:
                    write_paths.append(target)
                else:
                    read_paths.append(target)
        elif ntype == "for_statement":
            for word in _for_list_words(node):
                if looks_like_path(word):
                    read_paths.append(word)
        elif ntype == "test_command":
            # `[ -f /p ]` / `[[ -f /p ]]` parse as test_command (NOT a
            # `command` node), so their path operands would otherwise skip
            # scope-checking — a silent widening vs. the old parser, which
            # fed `[` to the unknown-command read-default. Route path-shaped
            # operands to the read check to restore that behavior.
            read_paths.extend(_test_command_reads(node))
        stack.extend(node.children)

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
    ``sandbox_denials.jsonl`` the record came from).

    ``write_root`` and ``extra_read_roots`` default (when the caller
    passes None) to the values stamped into the record at denial time
    (set_scope_context) — the iteration write-root and the
    config-provided read roots (e.g. BIRD_DATA_DIR) that replay cannot
    otherwise see. Records written BEFORE that stamping landed lack the
    fields and fall back to the record's ``cwd`` (write) and ``[]``
    (read). Those fallbacks are the legacy replay blind spots: a
    cwd-rooted write replay can only ever be MORE likely to say "still
    denies" (safe for an FP-FIXED claim — we never falsely declare a
    write fixed), and an empty read-root replay may phantom-deny a read
    the live hook would have allowed via a config read root. New records
    carry the inputs and replay faithfully.

    Replay is best-effort: a record we can't faithfully reconstruct
    (missing fields, unreadable shape) raises, and the caller falls back
    to pattern classification.
    """
    tool = record.get("tool", "")
    cwd = record.get("cwd") or experiment_dir
    if extra_read_roots is None:
        # Prefer the scope inputs stamped into the record at denial time
        # (set_scope_context). Records written before that field existed
        # lack it -> fall back to [], the legacy blind spot where replay
        # can't see config-provided read roots (e.g. BIRD_DATA_DIR) and so
        # may phantom-deny a read the live hook would have allowed.
        extra_read_roots = record.get("extra_read_roots") or []
    if write_root is None:
        # Likewise prefer the recorded iteration write-root; older records
        # fall back to cwd (the narrowest, safe-direction default).
        write_root = record.get("write_root") or cwd

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
    for d in auto_session_dirs(cwd_real) + auto_scratch_dirs(cwd_real):
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

    # Stamp the non-reconstructable scope inputs into any denial we log
    # this invocation (logging only — see set_scope_context). We capture
    # the RAW extra-read list here, BEFORE the memory/session carve-outs
    # are appended below, because replay re-derives those carve-outs
    # itself from the record's experiment_dir/cwd; it only needs the
    # config-provided roots (e.g. BIRD_DATA_DIR) and the iteration
    # write-root, which it cannot otherwise see.
    set_scope_context(write_root=write_root, extra_read_roots=extra_read_roots)

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

    # Session-project READ carve-outs: the Claude CLI spills large tool
    # outputs and stores transcripts under <config-dir>/projects/
    # <slug(cwd)>/, and background-task outputs + scratchpad under
    # <tmp>/claude-<uid>/<slug(cwd)>/. The model must Read its own
    # spilled output back; the raw experiment-dir read scope would
    # deny. See auto_session_dirs / auto_scratch_dirs.
    # READ-ONLY by design — NOT added to memory_carveouts (write),
    # so an injection writing to ~/.claude/... or /tmp/... via Write
    # or Bash stays denied.
    session_carveouts = auto_session_dirs(cwd) + auto_scratch_dirs(cwd)
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
        # catalog replays): parse with tree-sitter-bash and walk the
        # tree, routing each command's path tokens through the per-command
        # classifier and each redirect / assignment / loop-list operand to
        # the right scope. find -exec / xargs / $(...) / subshells / unknown
        # commands all flow their visible path tokens into the read check
        # (no fail-closed subprocess_bypass branch). Here we only translate
        # the verdict into the emit + log IO.
        decision, scope, blocked = evaluate_bash(
            command, cwd_real, experiment_dir, write_root, extra_read_roots,
        )
        if decision == "allow":
            return 0

        if scope == "parse":
            # Fail closed. Two distinct causes share this branch; name
            # which one so a missing install doesn't masquerade as a
            # malformed command (both deny, but the fix differs).
            if _TS_PARSER is None:
                reason = (f"tree-sitter dependency missing "
                          f"({_TS_IMPORT_ERROR}) — Bash cannot be "
                          f"scope-checked; install with "
                          f"'pip install -r requirements.txt'")
                sys.stderr.write(f"[sandbox] {reason}\n")
                message = (
                    "Sandbox denied. The Bash command parser dependency "
                    "(tree-sitter / tree-sitter-bash) is not installed, so "
                    "Bash commands cannot be scope-checked and are denied. "
                    "Install it: pip install -r requirements.txt. "
                    "(Read/Edit/Write tools still work.)"
                )
            else:
                reason = "tree-sitter parse error"
                message = (
                    "Sandbox denied. Could not parse this Bash command "
                    "(heredoc, unbalanced quotes, or unsupported construct). "
                    "Use Read/Edit/Write tools instead, or simplify the "
                    "command."
                )
            emit_decision("deny", message)
            append_denial_record({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "tool": "Bash",
                "scope": "parse",
                "blocked_path": "",
                "command": command,
                "cwd": cwd_real,
                "reason": reason,
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
