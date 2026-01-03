"""
Configuration settings for RoboPhD Text-to-SQL Research System
"""

# Model configuration with pricing from official Anthropic pricing
SUPPORTED_MODELS = {
    'opus-4.5': {
        'name': 'claude-opus-4-5-20251101',
        'pricing': {
            'input': 5.00,       # $5/MTok (base input)
            'output': 25.00,     # $25/MTok
            'cache_write': 6.25,  # $6.25/MTok (5m cache writes)
            'cache_read': 0.50    # $0.50/MTok (cache hits & refreshes)
        }
    },
    'opus-4.1': {
        'name': 'claude-opus-4-1-20250805',
        'pricing': {
            'input': 15.00,      # $15/MTok
            'output': 75.00,     # $75/MTok
            'cache_write': 18.75, # $18.75/MTok
            'cache_read': 1.50    # $1.50/MTok
        }
    },
    'sonnet-4.5': {
        'name': 'claude-sonnet-4-5-20250929',  # Sonnet 4.5
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
# Claude CLI expects simple aliases: 'opus', 'sonnet', 'haiku'
CLAUDE_CLI_MODEL_MAP = {
    'opus-4.5': 'opus',      # CLI expects 'opus' not 'opus-4.5'
    'opus-4.1': 'opus',      # CLI expects 'opus' not 'opus-4.1'
    'sonnet-4.5': 'sonnet',  # CLI expects 'sonnet' not 'sonnet-4.5'
    'haiku-4.5': 'haiku'     # CLI expects 'haiku' not 'haiku-4.5'
}

# Model fallback configuration for Claude CLI
MODEL_FALLBACKS = {
    'opus-4.5': 'sonnet',    # opus-4.5 falls back to sonnet
    'opus-4.1': 'sonnet',    # opus-4.1 falls back to sonnet
    'sonnet-4.5': 'haiku',   # sonnet falls back to haiku
    'haiku-4.5': None        # haiku has no fallback
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

# Evaluation settings
EVALUATION_TIMEOUT = 5  # seconds per SQL query

# Subprocess timeout settings
ANALYZER_TIMEOUT = 3600  # 60 minutes for database analysis (large databases need time)
FALLBACK_SQL = "SELECT 1;"