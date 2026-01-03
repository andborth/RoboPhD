#!/usr/bin/env python3
"""
Utility for finding and managing Claude CLI executable path.
"""

import os
import subprocess
from typing import Optional


def find_claude_cli() -> Optional[str]:
    """Find the Claude CLI executable."""
    # Check common locations
    possible_paths = [
        os.path.expanduser('~/.claude/local/claude'),
        '/usr/local/bin/claude',
        'claude'  # Try system PATH last
    ]
    
    for path in possible_paths:
        # Check if file exists and is executable first
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path

        # Then try using which (for system PATH)
        if path == 'claude':
            result = subprocess.run(['which', path], capture_output=True, text=True, shell=True)
            if result.returncode == 0 and result.stdout.strip():
                found_path = result.stdout.strip()
                return found_path
    
    print("  ⚠️  Claude CLI not found in any expected location")
    return None