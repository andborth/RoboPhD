"""
Summarize a Claude Code session transcript (.jsonl or .jsonl.gz) into a readable markdown file.

Usage:
    python RoboPhD/utilities/transcript_summarizer.py <transcript_path> [output_path]

Also importable:
    from RoboPhD.utilities.transcript_summarizer import summarize_transcript, find_transcript
"""

import gzip
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _parse_timestamp(ts_str):
    """Parse ISO 8601 timestamp string to datetime."""
    if not ts_str:
        return None
    # Handle 'Z' suffix and fractional seconds
    ts_str = ts_str.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None


def _format_duration(seconds):
    """Format seconds as 'Xh Ym Zs' or 'Ym Zs' or 'Zs'."""
    if seconds < 0:
        return "0s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _format_tokens(n):
    """Format token count with commas."""
    return f"{n:,}"


# Session-flow label budgets. Labels feed Round-2 / meta-evolution prompts,
# so they stay bounded — but normalization below strips the run-root path
# noise first, so the budget is spent on command payload, not prefixes.
BASH_LABEL_MAX = 200
BASH_LABEL_TAIL = 40
EDIT_SNIPPET_MAX = 40

# Absolute interpreter paths (e.g. /opt/anaconda3/envs/x/bin/python) carry no
# information the reader needs; collapse to the bare program name.
_INTERP_RE = re.compile(r"\S+/bin/(python[0-9.]*|pip[0-9.]*)\b")


def _derive_run_root(output_path):
    """Infer the experiment/run root from where the summary is written.

    Summaries land in <run>/evolution_output/iteration_NNN/ (or the
    meta_evolution_output twin), so the component before that marker is the
    run root. Returns None when the layout doesn't match — normalization is
    then skipped, never wrong.
    """
    parts = Path(output_path).resolve().parts
    for marker in ('evolution_output', 'meta_evolution_output'):
        if marker in parts:
            idx = parts.index(marker)
            if idx > 0:
                return str(Path(*parts[:idx]))
    return None


def _shorten_path(path, run_root):
    """Rewrite an absolute path under run_root as $RUN/..."""
    if run_root and isinstance(path, str) and path.startswith(run_root):
        rest = path[len(run_root):].lstrip('/')
        return f"$RUN/{rest}" if rest else "$RUN"
    return path


def _normalize_command(cmd, run_root):
    """Strip noise from a command before truncation: run-root prefixes
    become $RUN, absolute interpreter paths become bare program names."""
    if run_root:
        cmd = cmd.replace(run_root, '$RUN')
    return _INTERP_RE.sub(lambda m: m.group(1), cmd)


def _truncate_head_tail(text, limit, tail):
    """Keep the head and tail of an over-limit label — command tails carry
    filenames and pipe targets, usually the part worth keeping."""
    if len(text) <= limit:
        return text
    head = limit - tail - 3
    return text[:head] + " … " + text[-tail:]


def _clip(line, limit):
    line = line.strip()
    if len(line) > limit:
        line = line[:limit - 1] + "…"
    return line


def _edit_signature(old, new, limit):
    """Pick the first differing line pair from an edit's old/new strings —
    identical leading lines (shared context) would make every signature in
    a block of edits look the same."""
    old_lines = [l for l in old.strip().split('\n')]
    new_lines = [l for l in new.strip().split('\n')]
    for o, n in zip(old_lines, new_lines):
        if o != n:
            return _clip(o, limit), _clip(n, limit)
    # One side is a prefix of the other (pure insertion/deletion) or equal
    o = old_lines[len(new_lines)] if len(old_lines) > len(new_lines) else old_lines[0]
    n = new_lines[len(old_lines)] if len(new_lines) > len(old_lines) else new_lines[0]
    return _clip(o, limit), _clip(n, limit)


def _tool_one_liner(name, inp, run_root=None):
    """Produce a one-line summary for a tool call."""
    if name == 'Read':
        label = f"→ Read {_shorten_path(inp.get('file_path', '?'), run_root)}"
        offset = inp.get('offset')
        limit = inp.get('limit')
        if offset or limit:
            start = offset or 1
            span = f"lines {start}–{start + limit - 1}" if limit else f"from line {start}"
            label += f" ({span})"
        return label
    if name == 'Write':
        return f"→ Write {_shorten_path(inp.get('file_path', '?'), run_root)}"
    if name == 'Edit':
        label = f"→ Edit {_shorten_path(inp.get('file_path', '?'), run_root)}"
        old = inp.get('old_string')
        new = inp.get('new_string')
        if old is not None and new is not None:
            o, n = _edit_signature(old, new, EDIT_SNIPPET_MAX)
            label += f': "{o}" → "{n}"'
        return label
    if name == 'Bash':
        desc = inp.get('description', '')
        if desc:
            label = desc
        else:
            label = _normalize_command(inp.get('command', '?'), run_root)
        label = _truncate_head_tail(label, BASH_LABEL_MAX, BASH_LABEL_TAIL)
        return f"→ Bash: {label}"
    if name == 'Glob':
        return f"→ Glob {inp.get('pattern', '?')}"
    if name == 'Grep':
        pattern = inp.get('pattern', '?')
        path = inp.get('path', '')
        if path:
            return f'→ Grep "{pattern}" in {_shorten_path(path, run_root)}'
        return f'→ Grep "{pattern}"'
    if name == 'Task':
        desc = inp.get('description', '?')
        return f"→ Task: {desc}"
    # Skip internal bookkeeping tools
    if name in ('TodoWrite', 'TaskCreate', 'TaskUpdate', 'TaskList', 'TaskGet'):
        return None
    # Fallback for unknown tools
    return f"→ {name}"


def find_transcript(working_dir, session_id):
    """
    Locate a Claude Code session transcript JSONL file.

    Claude CLI stores transcripts at ~/.claude/projects/[sanitized_path]/[session_id].jsonl
    where the path is sanitized by replacing / and _ with -.

    Args:
        working_dir: The working directory used when launching Claude CLI.
        session_id: The Claude Code session ID.

    Returns:
        Path to the transcript file, or None if not found.
    """
    # Check CLAUDE_CONFIG_DIR from environment first, then from .envrc in the
    # working directory tree (direnv sets it for subprocesses but not the parent).
    claude_config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if not claude_config_dir:
        from utilities.claude_cli import parse_envrc_exports
        envrc_vars = parse_envrc_exports(Path(working_dir))
        claude_config_dir = envrc_vars.get("CLAUDE_CONFIG_DIR")
    claude_config = Path(claude_config_dir) if claude_config_dir else Path.home() / ".claude"

    project_dir_name = str(Path(working_dir).resolve()).replace('/', '-').replace('_', '-')
    chat_file = claude_config / "projects" / project_dir_name / f"{session_id}.jsonl"
    return chat_file if chat_file.exists() else None


def _read_jsonl(path):
    """Read a JSONL file (gzipped or plain) into a list of dicts."""
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    messages = []
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


def summarize_transcript(transcript_path, output_path=None, run_root=None):
    """
    Read a session transcript and write a session_summary.md.

    Args:
        transcript_path: Path to .jsonl or .jsonl.gz file
        output_path: Where to write the summary. Defaults to
                     session_summary.md in the same directory.
        run_root: Experiment/run root used to abbreviate paths as $RUN in
                  the summary. Defaults to deriving it from output_path's
                  evolution_output / meta_evolution_output component; when
                  neither is given nor derivable, paths are left untouched.

    Returns:
        Path to the written summary file.
    """
    transcript_path = Path(transcript_path)
    if output_path is None:
        output_path = transcript_path.parent / "session_summary.md"
    else:
        output_path = Path(output_path)

    run_root = str(run_root) if run_root else _derive_run_root(output_path)
    messages = _read_jsonl(transcript_path)

    # --- First pass: collect stats ---
    model = None
    first_ts = None
    last_ts = None
    assistant_count = 0
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_write = 0
    tool_counts = Counter()
    files_read = []
    files_written = []
    files_edited = []

    for msg in messages:
        if msg.get('type') == 'queue-operation':
            continue

        # Track wall-clock timestamps from all message types
        ts = _parse_timestamp(msg.get('timestamp'))
        if ts:
            if first_ts is None:
                first_ts = ts
            last_ts = ts

        if msg.get('type') == 'assistant':
            assistant_count += 1

            m = msg.get('message', {})
            if not model and isinstance(m, dict):
                model = m.get('model')

            usage = m.get('usage', {}) if isinstance(m, dict) else {}
            total_input += usage.get('input_tokens', 0)
            total_output += usage.get('output_tokens', 0)
            total_cache_read += usage.get('cache_read_input_tokens', 0)
            total_cache_write += usage.get('cache_creation_input_tokens', 0)

            content = m.get('content', []) if isinstance(m, dict) else []
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'tool_use':
                        name = block['name']
                        tool_counts[name] += 1
                        inp = block.get('input', {})
                        if name == 'Read':
                            fp = inp.get('file_path')
                            if fp and fp not in files_read:
                                files_read.append(fp)
                        elif name == 'Write':
                            fp = inp.get('file_path')
                            if fp and fp not in files_written:
                                files_written.append(fp)
                        elif name == 'Edit':
                            fp = inp.get('file_path')
                            if fp and fp not in files_edited:
                                files_edited.append(fp)

    # --- Build output ---
    lines = []
    lines.append("# Session Summary\n")
    lines.append("## Overview")

    if model:
        lines.append(f"- **Model**: {model}")

    if first_ts and last_ts:
        duration_s = (last_ts - first_ts).total_seconds()
        start_str = first_ts.strftime("%H:%M:%S")
        end_str = last_ts.strftime("%H:%M:%S")
        lines.append(f"- **Duration**: {_format_duration(duration_s)} ({start_str} → {end_str} UTC)")

    lines.append(f"- **Turns**: {assistant_count} assistant responses")

    # Token line
    token_parts = [_format_tokens(total_input) + " input"]
    cache_detail = []
    if total_cache_read:
        cache_detail.append(f"{_format_tokens(total_cache_read)} cache read")
    if total_cache_write:
        cache_detail.append(f"{_format_tokens(total_cache_write)} cache write")
    if cache_detail:
        token_parts[0] += f" ({', '.join(cache_detail)})"
    token_parts.append(f"{_format_tokens(total_output)} output")
    lines.append(f"- **Tokens**: {' → '.join(token_parts)}")

    # Tool summary line
    if tool_counts:
        # Skip bookkeeping tools from the summary line
        display_tools = {k: v for k, v in tool_counts.items()
                         if k not in ('TodoWrite', 'TaskCreate', 'TaskUpdate', 'TaskList', 'TaskGet')}
        if display_tools:
            tool_strs = [f"{name} ×{count}" for name, count in
                         sorted(display_tools.items(), key=lambda x: -x[1])]
            lines.append(f"- **Tools**: {', '.join(tool_strs)}")

    # Legend for the $RUN shorthand used throughout the sections below
    if run_root:
        lines.append(f"- **$RUN**: {run_root}")

    # Files sections
    if files_read:
        lines.append("\n## Files Read")
        for fp in files_read:
            lines.append(f"- {_shorten_path(fp, run_root)}")

    if files_written or files_edited:
        lines.append("\n## Files Written")
        seen = set()
        for fp in files_written:
            lines.append(f"- {_shorten_path(fp, run_root)}")
            seen.add(fp)
        for fp in files_edited:
            if fp not in seen:
                lines.append(f"- {_shorten_path(fp, run_root)} (edited)")

    # --- Second pass: session flow ---
    lines.append("\n## Session Flow\n")

    turn_number = 0  # Track turn boundaries from new user prompts

    for msg in messages:
        if msg.get('type') == 'queue-operation':
            continue

        # Detect new user prompts (str content = new turn; list content = tool results)
        if msg.get('type') == 'user':
            content = msg.get('message', {}).get('content') if isinstance(msg.get('message'), dict) else None
            if isinstance(content, str):
                turn_number += 1
                if turn_number > 1:
                    lines.append(f"---\n")
                    lines.append(f"**Turn {turn_number}**\n")
            continue

        if msg.get('type') == 'assistant':
            ts = _parse_timestamp(msg.get('timestamp'))
            m = msg.get('message', {})
            content = m.get('content', []) if isinstance(m, dict) else []
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict):
                    continue

                if block.get('type') == 'text':
                    text = block['text'].strip()
                    if not text:
                        continue
                    ts_prefix = f"[{ts.strftime('%H:%M:%S')}] " if ts else ""
                    lines.append(f"{ts_prefix}{text}\n")

                elif block.get('type') == 'tool_use':
                    one_liner = _tool_one_liner(block['name'], block.get('input', {}),
                                                run_root=run_root)
                    if one_liner is not None:
                        lines.append(f"  {one_liner}\n")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    return output_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <transcript_path> [output_path]", file=sys.stderr)
        sys.exit(1)

    transcript = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    result = summarize_transcript(transcript, output)
    print(f"Wrote summary to {result}")
