"""Unit tests for utilities.claude_cli.claude_cli_settings.

Catches accidental refactor breakage of the rendered settings JSON. Does NOT
test that Claude CLI itself honors the deny rule — that requires invoking
the CLI in a test, which is heavy and flaky. Catching DSL changes in Claude
CLI is left to the smoke-test verification in the original sandbox commit
(63fb02d): re-run a meta-evo firing and confirm the agent's reflexive Read
of repo files now fails. Worth re-running periodically as Claude CLI
versions advance.
"""

import json
from pathlib import Path

import pytest

from utilities.claude_cli import REPO_ROOT, claude_cli_settings


def test_settings_is_valid_json():
    parsed = json.loads(claude_cli_settings())
    assert isinstance(parsed, dict)


def test_settings_has_autocompact():
    parsed = json.loads(claude_cli_settings())
    assert parsed.get("autoCompact") is True


def test_settings_denies_repo_read():
    parsed = json.loads(claude_cli_settings())
    deny = parsed.get("permissions", {}).get("deny", [])
    assert len(deny) == 1, f"Expected exactly one deny rule, got: {deny}"
    rule = deny[0]
    # The rule must be a Read deny scoped to the repo.
    assert rule.startswith("Read("), f"Expected Read() rule, got: {rule}"
    assert rule.endswith("/**)"), f"Expected glob-suffixed rule, got: {rule}"
    # The repo path inside the rule must use the // absolute-path prefix
    # required by Claude CLI's permission DSL.
    assert "(//" in rule, f"Expected // absolute-path prefix, got: {rule}"


def test_repo_root_is_real_directory():
    """REPO_ROOT must point at the actual repository root, not somewhere stale."""
    assert REPO_ROOT.is_dir(), f"REPO_ROOT does not exist: {REPO_ROOT}"
    # Repo-root sentinel: contains both the package dir and a top-level marker.
    assert (REPO_ROOT / "RoboPhD").is_dir(), f"REPO_ROOT missing RoboPhD/: {REPO_ROOT}"
    assert (REPO_ROOT / "utilities").is_dir(), f"REPO_ROOT missing utilities/: {REPO_ROOT}"


def test_deny_rule_contains_repo_root():
    """The deny rule's path must match REPO_ROOT exactly (not a stale or wrong path)."""
    parsed = json.loads(claude_cli_settings())
    rule = parsed["permissions"]["deny"][0]
    expected = f"Read(/{REPO_ROOT}/**)"
    assert rule == expected, f"Expected {expected!r}, got {rule!r}"
