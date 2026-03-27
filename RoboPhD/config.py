"""
Configuration settings for RoboPhD Text-to-SQL Research System
"""

import os
from typing import Dict, Optional

# Model configuration with pricing from official Anthropic pricing
SUPPORTED_MODELS = {
    'opus-4.6': {
        'name': 'claude-opus-4-6',
        'pricing': {
            'input': 5.00,       # $5/MTok (base input)
            'output': 25.00,     # $25/MTok
            'cache_write': 6.25,  # $6.25/MTok (5m cache writes)
            'cache_read': 0.50    # $0.50/MTok (cache hits & refreshes)
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
    'opus-4.6': 'claude-opus-4-6[1m]',
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
