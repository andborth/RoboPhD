"""
Summarize a Claude Code session transcript (.jsonl or .jsonl.gz) into a readable markdown file.

Usage:
    python RoboPhD/utilities/transcript_summarizer.py <transcript_path> [output_path]

Also importable:
    from RoboPhD.utilities.transcript_summarizer import summarize_transcript
"""

import gzip
import json
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


def _tool_one_liner(name, inp):
    """Produce a one-line summary for a tool call."""
    if name == 'Read':
        return f"→ Read {inp.get('file_path', '?')}"
    if name == 'Write':
        return f"→ Write {inp.get('file_path', '?')}"
    if name == 'Edit':
        return f"→ Edit {inp.get('file_path', '?')}"
    if name == 'Bash':
        cmd = inp.get('command', '?')
        desc = inp.get('description', '')
        if desc:
            label = desc
        else:
            label = cmd
        if len(label) > 120:
            label = label[:117] + "..."
        return f"→ Bash: {label}"
    if name == 'Glob':
        return f"→ Glob {inp.get('pattern', '?')}"
    if name == 'Grep':
        pattern = inp.get('pattern', '?')
        path = inp.get('path', '')
        if path:
            return f'→ Grep "{pattern}" in {path}'
        return f'→ Grep "{pattern}"'
    if name == 'Task':
        desc = inp.get('description', '?')
        return f"→ Task: {desc}"
    # Skip internal bookkeeping tools
    if name in ('TodoWrite', 'TaskCreate', 'TaskUpdate', 'TaskList', 'TaskGet'):
        return None
    # Fallback for unknown tools
    return f"→ {name}"


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


def summarize_transcript(transcript_path, output_path=None):
    """
    Read a session transcript and write a session_summary.md.

    Args:
        transcript_path: Path to .jsonl or .jsonl.gz file
        output_path: Where to write the summary. Defaults to
                     session_summary.md in the same directory.

    Returns:
        Path to the written summary file.
    """
    transcript_path = Path(transcript_path)
    if output_path is None:
        output_path = transcript_path.parent / "session_summary.md"
    else:
        output_path = Path(output_path)

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

        if msg.get('type') == 'assistant':
            assistant_count += 1
            ts = _parse_timestamp(msg.get('timestamp'))
            if ts:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

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

    # Files sections
    if files_read:
        lines.append("\n## Files Read")
        for fp in files_read:
            lines.append(f"- {fp}")

    if files_written or files_edited:
        lines.append("\n## Files Written")
        seen = set()
        for fp in files_written:
            lines.append(f"- {fp}")
            seen.add(fp)
        for fp in files_edited:
            if fp not in seen:
                lines.append(f"- {fp} (edited)")

    # --- Second pass: session flow ---
    lines.append("\n## Session Flow\n")

    for msg in messages:
        if msg.get('type') == 'queue-operation':
            continue

        # Skip user messages (tool results / system prompts)
        if msg.get('type') == 'user':
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
                    one_liner = _tool_one_liner(block['name'], block.get('input', {}))
                    if one_liner:
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
