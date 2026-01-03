"""
LLM Provider abstraction for multi-provider support.

Supports:
- Anthropic Claude models (with prompt caching)
- Ollama local models via litellm
- LM Studio local models via litellm (MLX-optimized for Apple Silicon)
- Any other provider supported by litellm

Usage:
    from llm_providers import get_provider, parse_model_spec

    # Anthropic (default)
    provider = get_provider("haiku-4.5", api_key="...")
    response = provider.generate(system_prompt, user_prompt)

    # Ollama (llama.cpp + Metal)
    provider = get_provider("ollama/qwen2.5-coder:32b")
    response = provider.generate(system_prompt, user_prompt)

    # LM Studio (MLX - faster on Apple Silicon)
    provider = get_provider("lmstudio/qwen2.5-coder-32b-instruct")
    response = provider.generate(system_prompt, user_prompt)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)

# Ollama default endpoint
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# LM Studio default endpoint (MLX-optimized for Apple Silicon)
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")


@dataclass
class LLMResponse:
    """Unified response format across providers."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    # Anthropic-specific (optional)
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    # Raw response for debugging
    raw_response: Optional[Any] = field(default=None, repr=False)


def parse_model_spec(model_spec: str) -> Tuple[str, str]:
    """
    Parse 'provider/model' format or return ('anthropic', model_spec) for default.

    Examples:
        "haiku-4.5" -> ("anthropic", "haiku-4.5")
        "sonnet-4.5" -> ("anthropic", "sonnet-4.5")
        "ollama/sqlcoder:latest" -> ("ollama", "sqlcoder:latest")
        "ollama/qwen2.5-coder:32b" -> ("ollama", "qwen2.5-coder:32b")
        "lmstudio/qwen2.5-coder-32b" -> ("lmstudio", "qwen2.5-coder-32b")
    """
    if '/' in model_spec:
        provider, model = model_spec.split('/', 1)
        return (provider.lower(), model)
    return ('anthropic', model_spec)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0,
        max_tokens: int = 1000
    ) -> LLMResponse:
        """Generate a response given system and user prompts."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name for logging/tracking."""
        pass

    @property
    def supports_caching(self) -> bool:
        """Whether this provider supports prompt caching."""
        return False


class AnthropicProvider(LLMProvider):
    """
    Direct Anthropic API provider with prompt caching support.

    Preserves the existing Anthropic-specific features:
    - cache_control for cost optimization
    - Detailed token tracking including cache tokens
    """

    def __init__(self, model_key: str, api_key: Optional[str] = None):
        """
        Initialize Anthropic provider.

        Args:
            model_key: Short model key like 'haiku-4.5' or full name
            api_key: Optional API key (defaults to env var)
        """
        import anthropic
        from .config import SUPPORTED_MODELS, API_KEY_ENV_VAR

        self.model_key = model_key

        # Resolve model name
        if model_key in SUPPORTED_MODELS:
            self._model_name = SUPPORTED_MODELS[model_key]['name']
            self.pricing = SUPPORTED_MODELS[model_key]['pricing']
        else:
            # Assume it's a full model name
            self._model_name = model_key
            self.pricing = {'input': 0, 'output': 0, 'cache_write': 0, 'cache_read': 0}

        # Initialize client
        api_key = api_key or os.getenv(API_KEY_ENV_VAR)
        if not api_key:
            raise ValueError(f"API key required. Set {API_KEY_ENV_VAR} or pass api_key")

        self.client = anthropic.Anthropic(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supports_caching(self) -> bool:
        return True

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0,
        max_tokens: int = 1000,
        use_cache: bool = True
    ) -> LLMResponse:
        """
        Generate SQL using Anthropic API with optional prompt caching.

        Args:
            system_prompt: System/database analysis prompt
            user_prompt: User question prompt
            temperature: Sampling temperature (default 0 for deterministic)
            max_tokens: Maximum tokens in response
            use_cache: Whether to use prompt caching (default True)
        """
        # Build message content with optional caching
        if use_cache:
            content = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
        else:
            content = [
                {"type": "text", "text": system_prompt},
                {"type": "text", "text": user_prompt}
            ]

        response = self.client.messages.create(
            model=self._model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": content}]
        )

        # Extract token usage
        input_tokens = 0
        output_tokens = 0
        cache_creation_tokens = 0
        cache_read_tokens = 0

        if hasattr(response, 'usage'):
            usage = response.usage
            input_tokens = getattr(usage, 'input_tokens', 0)
            output_tokens = getattr(usage, 'output_tokens', 0)
            cache_creation_tokens = getattr(usage, 'cache_creation_input_tokens', 0)
            cache_read_tokens = getattr(usage, 'cache_read_input_tokens', 0)

        return LLMResponse(
            text=response.content[0].text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            raw_response=response
        )


class LiteLLMProvider(LLMProvider):
    """
    Provider using litellm for Ollama, vLLM, and other local/cloud models.

    litellm provides a unified interface across 100+ providers.
    """

    def __init__(self, model: str, base_url: Optional[str] = None, num_ctx: int = 16384, api_key: Optional[str] = None, default_timeout: int = 120):
        """
        Initialize LiteLLM provider.

        Args:
            model: Model name in litellm format (e.g., "ollama/sqlcoder:latest")
            base_url: Optional API base URL (for Ollama, defaults to localhost:11434)
            num_ctx: Context window size for Ollama models (default 16384)
            api_key: Optional API key (required for OpenAI-compatible endpoints like LM Studio)
            default_timeout: Default timeout in seconds for LLM calls (default 120)
        """
        self._model = model
        self.base_url = base_url
        self.num_ctx = num_ctx
        self.api_key = api_key
        self.default_timeout = default_timeout

        # Auto-detect Ollama and set base URL
        if model.startswith("ollama/") and base_url is None:
            self.base_url = OLLAMA_BASE_URL

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0,
        max_tokens: int = 1000,
        timeout: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate response using litellm.

        Uses standard OpenAI-compatible message format.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Sampling temperature (default 0)
            max_tokens: Max tokens in response (default 1000)
            timeout: Per-call timeout in seconds (defaults to self.default_timeout)
        """
        import litellm

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        effective_timeout = timeout if timeout is not None else self.default_timeout

        kwargs = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": effective_timeout,
        }

        if self.base_url:
            kwargs["api_base"] = self.base_url

        if self.api_key:
            kwargs["api_key"] = self.api_key

        # For Ollama models, pass num_ctx to extend context window
        if self._model.startswith("ollama/"):
            kwargs["num_ctx"] = self.num_ctx

        response = litellm.completion(**kwargs)

        # Extract token usage
        input_tokens = 0
        output_tokens = 0

        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'prompt_tokens', 0)
            output_tokens = getattr(response.usage, 'completion_tokens', 0)

        return LLMResponse(
            text=response.choices[0].message.content.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response=response
        )


def get_provider(model_spec: str, api_key: Optional[str] = None, validate: bool = True, default_timeout: int = 120) -> LLMProvider:
    """
    Factory function to get the appropriate provider for a model spec.

    Args:
        model_spec: Model specification (e.g., "haiku-4.5", "ollama/sqlcoder:latest")
        api_key: Optional API key (for Anthropic)
        validate: Whether to validate local model availability (default True)
        default_timeout: Default timeout in seconds for LLM calls (default 120, for LiteLLM providers)

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If validate=True and local model is not available

    Examples:
        provider = get_provider("haiku-4.5")  # Anthropic
        provider = get_provider("ollama/sqlcoder:latest")  # Ollama via litellm
        provider = get_provider("lmstudio/qwen2.5-coder-32b")  # LM Studio (MLX) via litellm
        provider = get_provider("lmstudio/qwen", default_timeout=60)  # 60s timeout
    """
    provider_name, model = parse_model_spec(model_spec)

    if provider_name == 'anthropic':
        return AnthropicProvider(model, api_key)
    elif provider_name == 'ollama':
        if validate:
            available, error = check_ollama_available(model)
            if not available:
                raise ValueError(f"Ollama model validation failed: {error}")
        return LiteLLMProvider(f"ollama/{model}", OLLAMA_BASE_URL, default_timeout=default_timeout)
    elif provider_name == 'lmstudio':
        if validate:
            available, error = check_lmstudio_available(model)
            if not available:
                raise ValueError(f"LM Studio model validation failed: {error}")
        # LM Studio exposes OpenAI-compatible API, MLX-optimized for Apple Silicon
        # Requires dummy API key for litellm's OpenAI client (LM Studio ignores it)
        return LiteLLMProvider(f"openai/{model}", LMSTUDIO_BASE_URL, api_key="lm-studio", default_timeout=default_timeout)
    else:
        # Assume litellm can handle other providers
        return LiteLLMProvider(model_spec, default_timeout=default_timeout)


def check_ollama_available(model: str = "sqlcoder:latest") -> Tuple[bool, Optional[str]]:
    """
    Check if Ollama is running and the specified model is available.

    Returns:
        Tuple of (is_available, error_message)
    """
    try:
        import requests
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.ok:
            available_models = [m['name'] for m in resp.json().get('models', [])]

            # Check exact match or base name match (without tag)
            model_base = model.split(':')[0]
            for m in available_models:
                if m == model or m.split(':')[0] == model_base:
                    return True, None

            return False, f"Model '{model}' not found. Available: {available_models}"
        else:
            return False, f"Ollama API error: {resp.status_code}"

    except Exception as e:
        if "ConnectionError" in str(type(e).__name__) or "Connection refused" in str(e):
            return False, f"Ollama not running at {OLLAMA_BASE_URL}. Start with: ollama serve"
        return False, f"Error checking Ollama: {e}"


def check_lmstudio_available(model: str) -> Tuple[bool, Optional[str]]:
    """
    Check if LM Studio is running and the specified model is loaded.

    Args:
        model: Model name to check (e.g., "qwen/qwen3-30b-a3b-2507")

    Returns:
        Tuple of (is_available, error_message)
    """
    try:
        import requests
        resp = requests.get(f"{LMSTUDIO_BASE_URL}/models", timeout=5)
        if resp.ok:
            data = resp.json()
            loaded_models = [m['id'] for m in data.get('data', [])]

            # Check if requested model is in the loaded models
            if model in loaded_models:
                return True, None

            # Also check for partial matches (model name without org prefix)
            model_base = model.split('/')[-1] if '/' in model else model
            for m in loaded_models:
                m_base = m.split('/')[-1] if '/' in m else m
                if m_base == model_base:
                    return True, None

            return False, f"Model '{model}' not loaded in LM Studio. Loaded models: {loaded_models}"
        else:
            return False, f"LM Studio API error: {resp.status_code}"

    except Exception as e:
        if "ConnectionError" in str(type(e).__name__) or "Connection refused" in str(e):
            return False, f"LM Studio not running at {LMSTUDIO_BASE_URL}. Start the Local Server in LM Studio."
        return False, f"Error checking LM Studio: {e}"


# Pricing for local models (all zeros)
LOCAL_MODEL_PRICING = {
    'input': 0.0,
    'output': 0.0,
    'cache_write': 0.0,
    'cache_read': 0.0
}


def get_model_pricing(model_spec: str) -> Dict[str, float]:
    """
    Get pricing for a model spec.

    Returns LOCAL_MODEL_PRICING for non-Anthropic models.
    """
    from .config import SUPPORTED_MODELS

    provider_name, model = parse_model_spec(model_spec)

    if provider_name == 'anthropic' and model in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model]['pricing']

    return LOCAL_MODEL_PRICING
