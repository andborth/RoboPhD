"""LM Studio routing and evolution-CLI env plumbing.

Non-Anthropic model names are served by LM Studio's Anthropic-compatible
endpoint, which the evolution CLI reaches via env overrides rather than a
separate code path. These tests cover the override plumbing;
``test_model_registry`` covers which names take that path at all.

Recovered from an untracked ``tests/`` copy that had rotted unnoticed
(it referenced ``RoboPhD.tools.run_critic_evaluation``, deleted long ago,
and ``subprocess.run``, which ``call_claude_cli`` no longer uses). Kept
here, under the collected test paths, so the next such drift fails a
routine ``pytest`` run instead of sitting silent.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from RoboPhD.config import LMSTUDIO_DEFAULT_BASE_URL, get_lmstudio_env


class TestLmStudioEnv:
    """``get_lmstudio_env`` override contents."""

    def test_lmstudio_model_gets_base_url_and_token(self):
        env = get_lmstudio_env("qwen/qwen3-coder-30b")
        assert env["ANTHROPIC_BASE_URL"] == LMSTUDIO_DEFAULT_BASE_URL
        assert env["ANTHROPIC_AUTH_TOKEN"] == "lmstudio"

    def test_custom_base_url_passes_through(self):
        env = get_lmstudio_env("qwen/qwen3-coder-30b", base_url="http://myhost:5555")
        assert env["ANTHROPIC_BASE_URL"] == "http://myhost:5555"

    def test_default_url_comes_from_the_constant(self):
        """Guards against the default arg drifting to a stale literal."""
        assert get_lmstudio_env("local/model")["ANTHROPIC_BASE_URL"] == LMSTUDIO_DEFAULT_BASE_URL

    def test_constant_is_expected_value(self):
        assert LMSTUDIO_DEFAULT_BASE_URL == "http://localhost:1234"


class TestCallClaudeCliExtraEnv:
    """``call_claude_cli`` merges extra_env on top of os.environ.

    Patches ``subprocess.Popen``: call_claude_cli moved off
    ``subprocess.run`` because run() discards stdout/stderr on
    TimeoutExpired (utilities/claude_cli.py).
    """

    @staticmethod
    def _popen_mock():
        proc = MagicMock()
        proc.communicate.return_value = ('{"result": "ok"}', "")
        proc.returncode = 0
        return proc

    @patch("utilities.claude_cli.subprocess.Popen")
    def test_extra_env_merged_over_os_environ(self, mock_popen):
        from utilities.claude_cli import call_claude_cli

        mock_popen.return_value = self._popen_mock()
        extra = {
            "ANTHROPIC_BASE_URL": "http://localhost:1234",
            "ANTHROPIC_AUTH_TOKEN": "lmstudio",
        }
        call_claude_cli(cmd=["claude", "--version"], cwd=Path("."), timeout=30, extra_env=extra)

        env = mock_popen.call_args.kwargs["env"]
        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:1234"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "lmstudio"
        assert "PATH" in env, "extra_env must extend os.environ, not replace it"

    @patch("utilities.claude_cli.subprocess.Popen")
    def test_without_extra_env_no_lmstudio_override(self, mock_popen):
        """env is always an explicit dict (os.environ + .envrc exports), so
        the check that matters is that no LM Studio override leaks in."""
        from utilities.claude_cli import call_claude_cli

        mock_popen.return_value = self._popen_mock()
        call_claude_cli(cmd=["claude", "--version"], cwd=Path("."), timeout=30)

        env = mock_popen.call_args.kwargs["env"]
        assert env.get("ANTHROPIC_AUTH_TOKEN") != "lmstudio"
        assert "PATH" in env
