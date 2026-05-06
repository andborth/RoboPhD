#!/usr/bin/env python3
"""PreToolUse hook: enforce experiment-dir read scope and cwd write scope.

Invoked once per tool call by Claude CLI when the per-experiment
.claude/settings.local.json wires it in. Reads JSON from stdin
({"tool_name", "tool_input", "cwd", ...}), writes JSON decision to
stdout, and on denial appends a structured record to
$ROBOPHD_EXPERIMENT_DIR/sandbox_denials.jsonl. researcher.py tails
that file in a daemon thread and re-emits each denial via its
standard logger, so denials surface to the run's normal log output.

Policy:
  - Read scope: anywhere under $ROBOPHD_EXPERIMENT_DIR.
  - Write scope: only under the tool's cwd (i.e., the iteration's
    working directory).

Fail-closed: if the hook can't classify a command or hits an internal
error, the tool call is denied (exit 2) so a misclassified Bash
command never silently leaks reads.
"""

import json
import os
import os.path
import shlex
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Tools whose tool_input describes a single read.
READ_TOOLS = {"Read", "Glob", "Grep"}

# Tools whose tool_input describes a single write.
WRITE_TOOLS = {"Edit", "Write", "NotebookEdit", "MultiEdit"}

# Bash command names that read files. Path tokens map to the read scope.
BASH_READ_COMMANDS = {
    "cat", "head", "tail", "less", "more", "bat",
    "grep", "rg", "ag", "find", "fd",
    "awk", "jq", "ls", "du", "wc",
    "file", "stat", "diff", "comm", "cmp",
    "xxd", "od", "strings", "tree",
}

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

# Special path strings that are never scope-checked.
PATH_EXEMPT = {"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty", "-"}


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

    Returns (read_paths, write_paths, unknown_command_with_paths)
    where unknown_command_with_paths is True iff the leading command
    is not in any classifier table AND there are path-shaped tokens
    we can't reliably attribute to read-vs-write.

    Handles redirects (`>`, `>>`, `&>`, `2>`, `<`) by extracting the
    redirect target out of `tokens` before command classification.
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

    if not remaining:
        return read_paths, write_paths, False

    cmd = remaining[0]
    args = remaining[1:]

    # `sed -i ...` is a write; otherwise sed is a read.
    if cmd == "sed":
        if any(a == "-i" or a.startswith("-i") for a in args):
            for a in args:
                if not a.startswith("-") and looks_like_path(a):
                    write_paths.append(a)
            return read_paths, write_paths, False
        for a in args:
            if not a.startswith("-") and looks_like_path(a):
                read_paths.append(a)
        return read_paths, write_paths, False

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
        return read_paths, write_paths, False

    # `find` is a read command, BUT `-exec`/`-execdir`/`-ok`/`-okdir`
    # invokes arbitrary commands per-match. Those subprocesses run
    # outside the hook's view — bypass for the read scope. Treat as
    # unknown-with-paths (fail-closed). Plain find without -exec is
    # fine and falls through to the BASH_READ_COMMANDS branch below.
    if cmd == "find" and any(
        a in ("-exec", "-execdir", "-ok", "-okdir") for a in args
    ):
        return read_paths, write_paths, True

    # `xargs <command>` invokes a command per-line of stdin, taking
    # paths from a piped predecessor. Same shape as find -exec — paths
    # bypass per-command classification. Fail closed.
    if cmd == "xargs":
        return read_paths, write_paths, True

    if cmd in BASH_WRITE_LAST_POSITIONAL:
        positional = [a for a in args if not a.startswith("-")]
        if positional:
            write_paths.append(positional[-1])
            for a in positional[:-1]:
                read_paths.append(a)
        return read_paths, write_paths, False

    if cmd in BASH_WRITE_ALL_POSITIONAL:
        for a in args:
            if not a.startswith("-") and looks_like_path(a):
                write_paths.append(a)
        return read_paths, write_paths, False

    if cmd in BASH_READ_COMMANDS:
        for a in args:
            if not a.startswith("-") and looks_like_path(a):
                read_paths.append(a)
        return read_paths, write_paths, False

    if cmd in BASH_PASSTHROUGH:
        for a in args:
            if looks_like_path(a):
                read_paths.append(a)
        return read_paths, write_paths, False

    # Unknown command. If it has any path-shaped tokens, fail closed —
    # we don't know whether they're sources, sinks, regex patterns, or
    # something else.
    has_path_token = any(looks_like_path(a) for a in args)
    return read_paths, write_paths, has_path_token


def split_compound(command: str) -> list:
    """Split a Bash command on top-level pipes/semicolons/&&/|| boundaries.

    Returns a list of token-lists, one per segment. Tokens are produced
    via shlex.split with posix=True. If shlex fails (unbalanced quotes,
    heredocs we don't model), returns None to signal "unparseable —
    fail closed."
    """
    try:
        all_tokens = shlex.split(command, posix=True)
    except ValueError:
        return None

    segments: list = []
    current: list = []
    for tok in all_tokens:
        if tok in ("|", "||", "&&", ";", "&"):
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


def deny_message(
    experiment_dir: str,
    cwd: str,
    blocked_path: str,
    scope: str,
    extra_read_roots: list = None,
) -> str:
    """Format the human-readable denial reason shown to the agent."""
    extra_lines = ""
    if extra_read_roots:
        bullets = "\n".join(f"      {p}" for p in extra_read_roots)
        extra_lines = f"    plus task-specific resource roots:\n{bullets}\n"
    return (
        "Sandbox denied. This Claude CLI session has two scopes:\n"
        f"  • Read (Read/Glob/Grep/Bash file inputs): anywhere under\n"
        f"    {experiment_dir}\n"
        f"{extra_lines}"
        f"  • Write (Edit/Write/Bash redirects, cp/mv/rm/mkdir/touch/sed -i targets): only\n"
        f"    {cwd}\n"
        "\n"
        f"Blocked: {blocked_path} — outside {scope} scope. Sibling experiment runs and the source repo are out of scope by policy."
    )


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

    experiment_dir = os.environ.get("ROBOPHD_EXPERIMENT_DIR")
    if not experiment_dir:
        sys.stderr.write("[sandbox] HOOK ERROR: ROBOPHD_EXPERIMENT_DIR not set\n")
        append_denial_record({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "error": "ROBOPHD_EXPERIMENT_DIR not set",
            "tool": tool_name,
        })
        return 2

    experiment_dir = os.path.realpath(experiment_dir)
    cwd_real = os.path.realpath(cwd)
    extra_read_roots = parse_extra_read_paths(sys.argv)

    # Cwd must itself be under the experiment dir; otherwise the write
    # scope is broken before we start.
    if not is_under(cwd_real, experiment_dir):
        emit_decision(
            "deny",
            deny_message(experiment_dir, cwd_real, cwd_real, "write",
                         extra_read_roots),
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
        emit_decision("deny", deny_message(experiment_dir, cwd_real, blocked, "read",
                                           extra_read_roots))
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
        ok, blocked = check_paths([path], cwd_real, cwd_real, "write")
        if ok:
            return 0
        emit_decision("deny", deny_message(experiment_dir, cwd_real, blocked, "write",
                                           extra_read_roots))
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

        segments = split_compound(command)
        if segments is None:
            # shlex couldn't parse — likely heredoc, unbalanced quotes,
            # or a construct we don't model. Fail closed.
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
                "command": command,
                "cwd": cwd_real,
                "reason": "shlex.split failed",
            })
            return 0

        for tokens in segments:
            if not tokens:
                continue
            read_paths, write_paths, unknown_with_paths = classify_bash_segment(tokens)

            if unknown_with_paths:
                emit_decision(
                    "deny",
                    "Sandbox denied. Bash command not in classifier and "
                    "contains path-shaped tokens — refusing to allow without "
                    "knowing whether they're sources or sinks. Use Read/Edit/"
                    "Write tools instead, or extend "
                    "utilities/sandbox_hook.py to teach the classifier this "
                    "command.\n"
                    f"Command: {' '.join(tokens)}",
                )
                append_denial_record({
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "tool": "Bash",
                    "scope": "classifier",
                    "blocked_path": "",
                    "command": command,
                    "cwd": cwd_real,
                    "reason": f"unknown command with path tokens: {tokens[0]}",
                })
                return 0

            ok, blocked = check_read_paths(read_paths, experiment_dir,
                                            extra_read_roots, cwd_real)
            if not ok:
                emit_decision(
                    "deny",
                    deny_message(experiment_dir, cwd_real, blocked, "read",
                                 extra_read_roots),
                )
                append_denial_record({
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "tool": "Bash",
                    "scope": "read",
                    "blocked_path": blocked,
                    "command": command,
                    "cwd": cwd_real,
                })
                return 0

            ok, blocked = check_paths(write_paths, cwd_real, cwd_real, "write")
            if not ok:
                emit_decision(
                    "deny",
                    deny_message(experiment_dir, cwd_real, blocked, "write",
                                 extra_read_roots),
                )
                append_denial_record({
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "tool": "Bash",
                    "scope": "write",
                    "blocked_path": blocked,
                    "command": command,
                    "cwd": cwd_real,
                })
                return 0

        return 0  # all segments cleared

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
