#!/usr/bin/env python3
"""
Utility for finding and managing Claude CLI executable path.
Includes rate limit handling for long-running experiments.
"""

import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo


class RateLimitExceeded(Exception):
    """Raised when Claude CLI rate limit wait time exceeds max threshold."""
    pass


def find_claude_cli() -> Optional[str]:
    """Find the Claude CLI executable."""
    # Check common locations (prioritize the actual path over aliases)
    possible_paths = [
        '/Users/andrew/.claude/local/claude',  # Original dev's installation (prioritized)
        os.path.expanduser('~/.claude/local/claude'),
        os.path.expanduser('~/node_modules/.bin/claude'),  # npm global install
        '/usr/local/bin/claude',
        'claude'  # Try system PATH last
    ]

    for path in possible_paths:
        # Expand user path
        expanded_path = os.path.expanduser(path)

        # Check if file exists and is executable first
        if os.path.exists(expanded_path) and os.access(expanded_path, os.X_OK):
            return expanded_path

        # Then try using which (for system PATH)
        if path == 'claude':
            try:
                result = subprocess.run(['which', 'claude'], capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    found_path = result.stdout.strip()
                    if os.path.exists(found_path) and os.access(found_path, os.X_OK):
                        return found_path
            except Exception:
                pass

    print("  ⚠️  Claude CLI not found in any expected location")
    return None


def _parse_rate_limit_reset_time(error_result: str) -> Optional[datetime]:
    """
    Parse reset time from Claude CLI rate limit error message.

    Args:
        error_result: The 'result' field from Claude CLI JSON output
                     e.g., "You've hit your limit · resets 1am (America/Los_Angeles)"

    Returns:
        datetime of reset time in local timezone, or None if not parseable
    """
    # Pattern: "resets Xam/pm" with timezone in parentheses
    time_match = re.search(r'resets\s+(\d{1,2})(am|pm)', error_result, re.IGNORECASE)
    if not time_match:
        return None

    hour = int(time_match.group(1))
    is_pm = time_match.group(2).lower() == 'pm'

    # Convert to 24-hour format
    if is_pm and hour != 12:
        hour += 12
    elif not is_pm and hour == 12:
        hour = 0

    # Extract timezone from parentheses (e.g., "America/Los_Angeles")
    tz_match = re.search(r'\(([A-Za-z_/]+)\)', error_result)
    if tz_match:
        try:
            msg_tz = ZoneInfo(tz_match.group(1))
        except Exception:
            # Invalid timezone, fall back to local time assumption
            msg_tz = None
    else:
        msg_tz = None

    if msg_tz:
        # Build reset time in the message's timezone
        now_in_msg_tz = datetime.now(msg_tz)
        reset_time = now_in_msg_tz.replace(hour=hour, minute=0, second=0, microsecond=0)

        # If reset time is in the past, it's tomorrow
        if reset_time <= now_in_msg_tz:
            reset_time += timedelta(days=1)

        # Convert to local timezone for display and comparison
        reset_time = reset_time.astimezone()
    else:
        # No timezone info, assume local time
        now = datetime.now()
        reset_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)

        if reset_time <= now:
            reset_time += timedelta(days=1)

    return reset_time


def _is_rate_limit_error(result: subprocess.CompletedProcess) -> Optional[str]:
    """
    Check if the subprocess result indicates a rate limit error.

    Args:
        result: CompletedProcess from subprocess.run()

    Returns:
        The error result string if rate limited, None otherwise
    """
    if result.returncode == 0:
        return None

    try:
        output = json.loads(result.stdout)
        if output.get('is_error') and 'hit your limit' in output.get('result', '').lower():
            return output.get('result', '')
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    return None


def call_claude_cli(
    cmd: List[str],
    cwd: Path,
    timeout: int,
    max_wait_hours: float = 4.0,
    update_interval_min: int = 5,
    logger: Optional[logging.Logger] = None
) -> subprocess.CompletedProcess:
    """
    Call Claude CLI with automatic rate limit handling.

    If rate limit is hit and wait time is within max_wait_hours, displays
    a countdown and retries automatically after reset. Each thread handles
    its own waiting independently.

    Args:
        cmd: Command list to execute (e.g., ['claude', '--model', 'opus', ...])
        cwd: Working directory for the command
        timeout: Timeout in seconds for each attempt
        max_wait_hours: Maximum hours to wait for rate limit reset (default 4.0)
        update_interval_min: Minutes between countdown updates (default 5)
        logger: Optional logger for debug messages

    Returns:
        subprocess.CompletedProcess on success

    Raises:
        RateLimitExceeded: If wait time exceeds max_wait_hours
        subprocess.TimeoutExpired: If command times out
        Other subprocess exceptions on failure
    """
    log = logger or logging.getLogger(__name__)

    while True:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Check for rate limit error
        rate_limit_msg = _is_rate_limit_error(result)
        if not rate_limit_msg:
            return result

        # Parse reset time
        reset_time = _parse_rate_limit_reset_time(rate_limit_msg)
        if not reset_time:
            log.warning(f"Rate limit hit but couldn't parse reset time: {rate_limit_msg}")
            return result  # Return the error result if we can't parse

        # Calculate wait time (use timezone-aware now if reset_time is aware)
        if reset_time.tzinfo is not None:
            now = datetime.now(reset_time.tzinfo)
        else:
            now = datetime.now()
        wait_seconds = (reset_time - now).total_seconds()
        wait_hours = wait_seconds / 3600

        if wait_hours > max_wait_hours:
            raise RateLimitExceeded(
                f"Rate limit reset in {wait_hours:.1f} hours exceeds max wait of {max_wait_hours} hours. "
                f"Reset at: {reset_time.strftime('%I:%M%p %Z')}"
            )

        # Display countdown and wait
        reset_time_str = reset_time.strftime('%I:%M%p').lstrip('0').lower()

        def get_now():
            return datetime.now(reset_time.tzinfo) if reset_time.tzinfo else datetime.now()

        while get_now() < reset_time:
            remaining = reset_time - get_now()
            remaining_min = int(remaining.total_seconds() / 60)

            if remaining_min > 60:
                remaining_str = f"{remaining_min // 60}h {remaining_min % 60}m"
            else:
                remaining_str = f"{remaining_min} min"

            print(f"⏸️  Claude rate limit - resuming at {reset_time_str} ({remaining_str} remaining)")

            # Sleep for update interval or until reset, whichever is shorter
            sleep_seconds = min(update_interval_min * 60, remaining.total_seconds() + 1)
            time.sleep(sleep_seconds)

        print("✓ Resuming Claude Code call...")
        log.info("Rate limit wait complete, retrying Claude CLI call")
