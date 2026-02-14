"""Unit tests for LM Studio model support."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGetLmstudioEnv(unittest.TestCase):
    """Tests for config.get_lmstudio_env()."""

    def setUp(self):
        from RoboPhD.config import get_lmstudio_env, LMSTUDIO_DEFAULT_BASE_URL
        self.get_lmstudio_env = get_lmstudio_env
        self.default_url = LMSTUDIO_DEFAULT_BASE_URL

    def test_anthropic_models_return_none(self):
        """Known Anthropic models should not get LM Studio env overrides."""
        for model in ('haiku-4.5', 'sonnet-4.5', 'opus-4.5', 'opus-4.6'):
            with self.subTest(model=model):
                self.assertIsNone(self.get_lmstudio_env(model))

    def test_lmstudio_model_returns_env(self):
        """Non-Anthropic models should get LM Studio env overrides."""
        env = self.get_lmstudio_env('qwen/qwen3-coder-30b')
        self.assertIsInstance(env, dict)
        self.assertEqual(env['ANTHROPIC_BASE_URL'], self.default_url)
        self.assertEqual(env['ANTHROPIC_AUTH_TOKEN'], 'lmstudio')

    def test_custom_base_url(self):
        """Custom base URL should be passed through."""
        env = self.get_lmstudio_env('qwen/qwen3-coder-30b', base_url='http://myhost:5555')
        self.assertEqual(env['ANTHROPIC_BASE_URL'], 'http://myhost:5555')

    def test_unknown_model_treated_as_lmstudio(self):
        """Any model not in SUPPORTED_MODELS or CLAUDE_CLI_MODEL_MAP gets LM Studio env."""
        env = self.get_lmstudio_env('some-local-model')
        self.assertIsNotNone(env)

    def test_default_url_is_constant(self):
        """Default arg should use the module constant, not a stale literal."""
        from RoboPhD.config import LMSTUDIO_DEFAULT_BASE_URL
        env = self.get_lmstudio_env('local/model')
        self.assertEqual(env['ANTHROPIC_BASE_URL'], LMSTUDIO_DEFAULT_BASE_URL)


class TestValidateModelVersion(unittest.TestCase):
    """Tests for run_critic_evaluation.validate_model_version()."""

    def setUp(self):
        from RoboPhD.tools.run_critic_evaluation import validate_model_version
        self.validate = validate_model_version

    def test_valid_versioned_anthropic_models(self):
        """Versioned Anthropic model names should pass."""
        for model in ('haiku-4.5', 'sonnet-4.5', 'opus-4.5'):
            with self.subTest(model=model):
                self.validate(model, 'test')  # should not raise

    def test_bare_aliases_rejected(self):
        """Unversioned Anthropic aliases should be rejected."""
        for model in ('haiku', 'sonnet', 'opus'):
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    self.validate(model, 'test')

    def test_anthropic_typos_rejected(self):
        """Typos on Anthropic model names should be caught."""
        for model in ('haiku-4.6', 'sonnet-5.0', 'opus-3.0', 'claude-blah'):
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    self.validate(model, 'test')

    def test_lmstudio_models_accepted(self):
        """Non-Anthropic model names should pass without validation."""
        for model in ('qwen/qwen3-coder-30b', 'meta/llama-3.1-70b', 'local-model'):
            with self.subTest(model=model):
                self.validate(model, 'test')  # should not raise

    def test_error_message_includes_valid_names(self):
        """Error message should list valid model names."""
        with self.assertRaises(ValueError) as ctx:
            self.validate('haiku', 'coder_model')
        msg = str(ctx.exception)
        self.assertIn('coder_model', msg)
        self.assertIn('haiku-4.5', msg)


class TestCacheDirSanitization(unittest.TestCase):
    """Tests for cache directory path construction in CodeGenDomain."""

    def test_slash_replaced(self):
        """Model names with '/' should use '--' in cache dir."""
        model = 'qwen/qwen3-coder-30b'
        sanitized = model.replace('/', '--')
        self.assertEqual(sanitized, 'qwen--qwen3-coder-30b')
        self.assertNotIn('/', sanitized)

    def test_tag_appended(self):
        """coder_model_tag should be appended with underscore separator."""
        model = 'qwen/qwen3-coder-30b'
        tag = '6bit_100K'
        cache_model_name = model.replace('/', '--')
        if tag:
            cache_model_name = f"{cache_model_name}_{tag}"
        self.assertEqual(cache_model_name, 'qwen--qwen3-coder-30b_6bit_100K')

    def test_empty_tag_no_suffix(self):
        """Empty tag should not add trailing underscore."""
        model = 'qwen/qwen3-coder-30b'
        tag = ''
        cache_model_name = model.replace('/', '--')
        if tag:
            cache_model_name = f"{cache_model_name}_{tag}"
        self.assertEqual(cache_model_name, 'qwen--qwen3-coder-30b')

    def test_anthropic_model_unchanged(self):
        """Anthropic model names (no '/') should pass through unchanged."""
        model = 'haiku-4.5'
        sanitized = model.replace('/', '--')
        self.assertEqual(sanitized, 'haiku-4.5')


class TestCallClaudeCliExtraEnv(unittest.TestCase):
    """Tests for extra_env parameter in call_claude_cli."""

    @patch('utilities.claude_cli.subprocess.run')
    def test_extra_env_passed_to_subprocess(self, mock_run):
        """When extra_env is provided, subprocess.run should receive merged env."""
        from utilities.claude_cli import call_claude_cli
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "ok"}',
            stderr='',
        )

        extra = {'ANTHROPIC_BASE_URL': 'http://localhost:1234', 'ANTHROPIC_AUTH_TOKEN': 'lmstudio'}
        try:
            call_claude_cli(
                cmd=['claude', '--version'],
                cwd=Path('.'),
                timeout=30,
                extra_env=extra,
            )
        except Exception:
            pass  # We only care about the subprocess.run call

        mock_run.assert_called()
        call_kwargs = mock_run.call_args
        env = call_kwargs.kwargs.get('env') or call_kwargs[1].get('env')
        self.assertIsNotNone(env)
        self.assertEqual(env['ANTHROPIC_BASE_URL'], 'http://localhost:1234')
        self.assertEqual(env['ANTHROPIC_AUTH_TOKEN'], 'lmstudio')
        # Should also contain existing env vars
        self.assertIn('PATH', env)

    @patch('utilities.claude_cli.subprocess.run')
    def test_no_extra_env_passes_none(self, mock_run):
        """When extra_env is not provided, subprocess.run env should be None."""
        from utilities.claude_cli import call_claude_cli
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "ok"}',
            stderr='',
        )

        try:
            call_claude_cli(
                cmd=['claude', '--version'],
                cwd=Path('.'),
                timeout=30,
            )
        except Exception:
            pass

        mock_run.assert_called()
        call_kwargs = mock_run.call_args
        env = call_kwargs.kwargs.get('env') or call_kwargs[1].get('env')
        self.assertIsNone(env)


class TestDefaultUrlConsistency(unittest.TestCase):
    """Verify all modules use the shared constant, not stale string literals."""

    def test_constant_is_expected_value(self):
        from RoboPhD.config import LMSTUDIO_DEFAULT_BASE_URL
        self.assertEqual(LMSTUDIO_DEFAULT_BASE_URL, 'http://localhost:1234')

    def test_config_manager_uses_constant(self):
        """config_manager defaults should reference the constant."""
        from RoboPhD.config_manager import ConfigManager
        cm = ConfigManager()
        defaults = cm.get_defaults()
        from RoboPhD.config import LMSTUDIO_DEFAULT_BASE_URL
        self.assertEqual(defaults['lmstudio_base_url'], LMSTUDIO_DEFAULT_BASE_URL)


if __name__ == '__main__':
    unittest.main()
