"""
AEGIS LLM Provider Integrations

Provider integrations for various LLM services including LM Studio, OpenAI,
Anthropic, and Ollama.
"""

import logging

# Always import base components
from .base import BaseLLMProvider
from .lmstudio import LMStudioProvider
from .factory import (
    create_provider,
    create_and_initialize_provider,
    get_available_providers,
    is_provider_available,
    get_provider_info,
    get_all_provider_info,
    register_provider,
    initialize_all_providers
)

logger = logging.getLogger(__name__)

# Optional provider imports - these will be registered by factory if available
_optional_providers = {}

# Try to import optional providers
try:
    from .openai import OpenAIProvider
    _optional_providers["openai"] = OpenAIProvider
except ImportError:
    logger.debug("OpenAI provider not available")

try:
    from .anthropic import AnthropicProvider
    _optional_providers["anthropic"] = AnthropicProvider
except ImportError:
    logger.debug("Anthropic provider not available")

try:
    from .ollama import OllamaProvider
    _optional_providers["ollama"] = OllamaProvider
except ImportError:
    logger.debug("Ollama provider not available")


def get_provider_status() -> dict:
    """Get status of all provider integrations."""
    available_providers = get_available_providers()
    return {
        "total_providers": len(available_providers),
        "available_providers": [p.value for p in available_providers],
        "lm_studio_available": True,  # Always available
        "optional_providers": list(_optional_providers.keys()),
        "provider_info": get_all_provider_info()
    }


# Export all available components
__all__ = [
    # Base classes
    "BaseLLMProvider",
    
    # Core providers (always available)
    "LMStudioProvider",
    
    # Factory functions
    "create_provider",
    "create_and_initialize_provider",
    "get_available_providers",
    "is_provider_available",
    "get_provider_info",
    "get_all_provider_info",
    "register_provider",
    "initialize_all_providers",
    
    # Utility functions
    "get_provider_status",
]

# Add optional providers to exports if available
__all__.extend(_optional_providers.keys())