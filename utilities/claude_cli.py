#!/usr/bin/env python3
"""
Utility for finding and managing Claude CLI executable path.
"""

import os
import subprocess
from typing import Optional


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