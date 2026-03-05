"""
Shared debug logging for LLM calls.

Probabilistically writes JSON files capturing the full prompt, response, and metadata
for each LLM call. Used by CostTrackingLM (GEPA reflection) and evaluators (AIME, Text2SQL).
"""

import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _slugify(text: str, max_len: int = 40) -> str:
    """Convert text to a safe filename slug."""
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', text.strip().lower())
    return slug[:max_len].rstrip('_')


def _truncate_message(content: str, max_chars: int = 400) -> str:
    """Truncate long message content, keeping first and last 200 chars."""
    if len(content) <= max_chars:
        return content
    kept = max_chars // 2
    truncated = len(content) - max_chars
    return f"{content[:kept]}\n\n... [TRUNCATED {truncated} chars] ...\n\n{content[-kept:]}"


def maybe_debug_log(
    debug_log_probability: float,
    debug_log_dir: Path | str | None,
    call_type: str,
    model: str,
    messages: list[dict] | str,
    response_text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Probabilistically write a debug log JSON file.

    Creates debug_log_dir if needed. Silent on failure (never breaks the run).

    Args:
        debug_log_probability: Probability (0-1.0) of actually writing the log.
        debug_log_dir: Directory to write log files to. If None, no-op.
        call_type: Type of call (e.g. "reflection", "generation", "verification").
        model: Model name/ID.
        messages: The messages sent to the LLM (list of dicts or raw prompt string).
        response_text: The LLM response text.
        metadata: Optional extra data (token usage, cost, question_id, etc.).
    """
    if not debug_log_dir or debug_log_probability <= 0:
        return
    if random.random() >= debug_log_probability:
        return

    try:
        debug_log_dir = Path(debug_log_dir)
        debug_log_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        # Extract prompt slug from user message
        if isinstance(messages, str):
            prompt_slug = _slugify(messages)
        elif isinstance(messages, list) and messages:
            # Use last user message for slug
            user_msgs = [m for m in messages if m.get("role") == "user"]
            slug_source = user_msgs[-1]["content"] if user_msgs else messages[-1].get("content", "")
            prompt_slug = _slugify(slug_source)
        else:
            prompt_slug = "unknown"

        filename = f"{call_type}_{timestamp}_{prompt_slug}.json"

        # Truncate long messages for the log
        if isinstance(messages, list):
            logged_messages = []
            for msg in messages:
                logged_msg = dict(msg)
                if isinstance(logged_msg.get("content"), str) and len(logged_msg["content"]) > 400:
                    logged_msg["content"] = _truncate_message(logged_msg["content"])
                logged_messages.append(logged_msg)
        else:
            logged_messages = _truncate_message(messages) if len(messages) > 400 else messages

        entry = {
            "timestamp": now.isoformat(),
            "call_type": call_type,
            "model": model,
            "messages": logged_messages,
            "response": _truncate_message(response_text) if len(response_text) > 400 else response_text,
        }
        if metadata:
            entry["metadata"] = metadata

        (debug_log_dir / filename).write_text(json.dumps(entry, indent=2, default=str))
    except Exception:
        pass  # Never break the run
