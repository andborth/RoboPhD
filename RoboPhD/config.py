"""
Configuration settings for RoboPhD Text-to-SQL Research System
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union

# Model configuration with pricing from official Anthropic pricing
SUPPORTED_MODELS = {
    'fable-5': {
        'name': 'claude-fable-5',  # plain ID for pricing/display; the 1M [1m] variant lives in CLAUDE_CLI_MODEL_MAP
        'pricing': {
            'input': 10.00,       # $10/MTok (base input)
            'output': 50.00,      # $50/MTok
            'cache_write': 12.50,  # $12.50/MTok (5m cache writes)
            'cache_read': 1.00     # $1/MTok (cache hits & refreshes)
        }
    },
    'opus-4.8': {
        'name': 'claude-opus-4-8',  # plain ID for pricing/display; the 1M [1m] variant lives in CLAUDE_CLI_MODEL_MAP
        'pricing': {                 # official opus-4.8 rates (same as 4.7). Evolution-CLI cost comes from total_cost_usd; GEPA may register these with litellm (runner_utils._ensure_litellm_pricing)
            'input': 5.00,       # $5/MTok (base input)
            'output': 25.00,     # $25/MTok
            'cache_write': 6.25,  # $6.25/MTok (5m cache writes)
            'cache_read': 0.50    # $0.50/MTok (cache hits & refreshes)
        }
    },
    'opus-4.7': {
        'name': 'claude-opus-4-7',
        'pricing': {
            'input': 5.00,       # $5/MTok (base input)
            'output': 25.00,     # $25/MTok
            'cache_write': 6.25,  # $6.25/MTok (5m cache writes)
            'cache_read': 0.50    # $0.50/MTok (cache hits & refreshes)
        }
    },
    'opus-4.6': {
        'name': 'claude-opus-4-6',
        'pricing': {
            'input': 5.00,       # $5/MTok (base input)
            'output': 25.00,     # $25/MTok
            'cache_write': 6.25,  # $6.25/MTok (5m cache writes)
            'cache_read': 0.50    # $0.50/MTok (cache hits & refreshes)
        }
    },
    'sonnet-5': {
        'name': 'claude-sonnet-5',  # plain ID for pricing/display; the 1M [1m] variant lives in CLAUDE_CLI_MODEL_MAP
        'pricing': {                 # sticker rates; intro pricing ($2 in / $10 out) applies through 2026-08-31 — litellm's bundled registry tracks the live rate for GEPA cost accounting
            'input': 3.00,       # $3/MTok
            'output': 15.00,     # $15/MTok
            'cache_write': 3.75,  # $3.75/MTok
            'cache_read': 0.30    # $0.30/MTok
        }
    },
    'sonnet-4.6': {
        'name': 'claude-sonnet-4-6',
        'pricing': {
            'input': 3.00,       # $3/MTok
            'output': 15.00,     # $15/MTok
            'cache_write': 3.75,  # $3.75/MTok
            'cache_read': 0.30    # $0.30/MTok
        }
    },
    'sonnet-4.5': {
        'name': 'claude-sonnet-4-5-20250929',
        'pricing': {
            'input': 3.00,       # $3/MTok
            'output': 15.00,     # $15/MTok
            'cache_write': 3.75,  # $3.75/MTok
            'cache_read': 0.30    # $0.30/MTok
        }
    },
    'haiku-4.5': {
        'name': 'claude-haiku-4-5-20251001',
        'pricing': {
            'input': 1.00,       # $1/MTok
            'output': 5.00,      # $5/MTok
            'cache_write': 1.25,  # $1.25/MTok
            'cache_read': 0.10    # $0.10/MTok
        }
    }
}

# Model to Claude CLI name mapping
# Maps our short names to full pinned model IDs for cache isolation.
# Claude CLI accepts both aliases ('haiku') and full IDs; full IDs are
# preferred so cache keys stay stable when aliases roll forward.
CLAUDE_CLI_MODEL_MAP = {
    'fable-5': 'claude-fable-5[1m]',    # [1m] = explicit 1M-context request for the evolution CLI session
    'opus-4.8': 'claude-opus-4-8[1m]',
    'opus-4.7': 'claude-opus-4-7',
    'opus-4.6': 'claude-opus-4-6[1m]',
    'sonnet-5': 'claude-sonnet-5[1m]',  # [1m] = explicit 1M-context request for the evolution CLI session
    'sonnet-4.6': 'claude-sonnet-4-6',
    'sonnet-4.5': 'claude-sonnet-4-5-20250929',
    'haiku-4.5': 'claude-haiku-4-5-20251001',
}

# Default model settings
DEFAULT_MODEL = 'sonnet-4.5'
MODEL_NAME = SUPPORTED_MODELS[DEFAULT_MODEL]['name']
MAX_TOKENS = 1000
DEFAULT_MAX_CONCURRENT_DBS = 10

# API settings
API_KEY_ENV_VAR = "ANTHROPIC_API_KEY_FOR_ROBOPHD"

# File paths
DEFAULT_DEV_DATA = "benchmark_resources/datasets/dev/dev_20240627/dev.json"
DEFAULT_DB_ROOT = "benchmark_resources/datasets/dev/dev_20240627/dev_databases"
DEFAULT_DEV_SQL = "benchmark_resources/datasets/dev/dev_20240627/dev.sql"

# Dataset paths - single source of truth for all dataset locations
DATASET_PATHS = {
    'train': {
        'questions': 'benchmark_resources/datasets/train/train/train.json',
        'db_root': 'benchmark_resources/datasets/train/train/train_databases'
    },
    'train-filtered': {
        'questions': 'benchmark_resources/datasets/train-filtered/train_filtered.json',
        'db_root': 'benchmark_resources/datasets/train/train/train_databases'
    },
    'train-no-evidence': {
        'questions': 'benchmark_resources/datasets/train-no-evidence/train_filtered_no_evidence.json',
        'db_root': 'benchmark_resources/datasets/train/train/train_databases'
    },
    'test': {
        'questions': 'benchmark_resources/datasets/test/test/test.json',
        'db_root': 'benchmark_resources/datasets/test/test/test_databases'
    },
    'dev': {
        'questions': 'benchmark_resources/datasets/dev/dev_20240627/dev.json',
        'db_root': 'benchmark_resources/datasets/dev/dev_20240627/dev_databases'
    },
    'dev-no-evidence': {
        'questions': 'benchmark_resources/datasets/dev-no-evidence/dev_no_evidence.json',
        'db_root': 'benchmark_resources/datasets/dev/dev_20240627/dev_databases'
    }
}

# Evaluation settings
EVALUATION_TIMEOUT = 5  # seconds per SQL query

# Subprocess timeout settings
ANALYZER_TIMEOUT = 3600  # 60 minutes for database analysis (large databases need time)
FALLBACK_SQL = "SELECT 1;"

# LM Studio integration
LMSTUDIO_DEFAULT_BASE_URL = "http://localhost:1234"


def get_api_key() -> Optional[str]:
    """Get the Anthropic API key without mutating the environment.

    Checks ANTHROPIC_API_KEY first, falls back to ANTHROPIC_API_KEY_FOR_ROBOPHD.
    """
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(API_KEY_ENV_VAR)


def get_api_key_for_model(resolved_model: str) -> Optional[str]:
    """Get the appropriate API key for a model, or None to let litellm resolve it.

    Only returns the Anthropic key for Anthropic-provider models. For other
    providers (OpenAI, OpenRouter, etc.), returns None so litellm picks up
    the correct key from environment variables (OPENAI_API_KEY, etc.).
    """
    try:
        import litellm
        _, provider, _, _ = litellm.get_llm_provider(resolved_model)
        if provider == "anthropic":
            return get_api_key()
    except Exception:
        # Fallback: if litellm can't identify the provider, don't pass a key
        pass
    return None


def resolve_model_name(model: str) -> str:
    """Resolve short model names (e.g., 'haiku-4.5') to full litellm-compatible names.

    Returns the input unchanged if not found in SUPPORTED_MODELS.
    """
    if model in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model]["name"]
    return model


def get_lmstudio_env(model: str, base_url: str = LMSTUDIO_DEFAULT_BASE_URL) -> Optional[Dict[str, str]]:
    """Return env overrides for LM Studio models, or None for Anthropic models.

    Anthropic models (those in CLAUDE_CLI_MODEL_MAP or SUPPORTED_MODELS) use the
    default Anthropic API and need no env overrides. All other model names are
    assumed to be served by LM Studio's Anthropic-compatible endpoint.

    Args:
        model: Model identifier (e.g., 'haiku-4.5' or 'qwen/qwen3-coder-30b')
        base_url: LM Studio server URL (default: http://localhost:1234)

    Returns:
        Dict with ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN for LM Studio models,
        or None for Anthropic models.
    """
    if model in CLAUDE_CLI_MODEL_MAP or model in SUPPORTED_MODELS:
        return None
    return {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_AUTH_TOKEN": "lmstudio",
    }


def build_evolution_env(
    model: str,
    experiment_dir: Optional[Union[str, Path]],
    iteration_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """Build the env dict for an evolution Claude CLI invocation.

    Combines three concerns historically tangled at every call site:
      * LM Studio routing for non-Anthropic models (None for Anthropic).
      * Sandbox env var ROBOPHD_EXPERIMENT_DIR (must be ABSOLUTE — the
        sandbox hook resolves it against its own cwd, which is the
        iteration dir, so a relative path would point at a bogus
        location and silently break the cwd-under-experiment-dir gate).
      * Sandbox env var ROBOPHD_EVOLUTION_ITERATION_DIR (optional, also absolute):
        declares the iteration's writable root. Write scope is anchored
        on this dir, not on the runtime cwd, so an agent can edit
        ``<iteration_dir>/agent.py`` regardless of whether it has
        ``cd``'d into a nested test subdir. When unset (legacy /
        non-evolution callers) the hook falls back to cwd-rooted write
        scope (the historical behavior).

    Centralizing avoids the regression mode where a fix to one call
    site doesn't reach the other two (e.g., commits f75228e + aa548a4,
    each of which had to patch three identical blocks).
    """
    env: Dict[str, str] = get_lmstudio_env(model) or {}
    if experiment_dir is not None:
        env["ROBOPHD_EXPERIMENT_DIR"] = str(Path(experiment_dir).resolve())
    if iteration_dir is not None:
        env["ROBOPHD_EVOLUTION_ITERATION_DIR"] = str(Path(iteration_dir).resolve())
    return env
