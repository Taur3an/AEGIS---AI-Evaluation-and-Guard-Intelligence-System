"""
Provider factory for creating and managing LLM provider instances.

This module provides factory functions for creating provider instances
and managing provider availability.
"""

import logging
from typing import Any, Dict, List, Optional, Type

from .base import BaseLLMProvider
from .lmstudio import LMStudioProvider
from ..utils.data_structures import LLMProvider


# Provider class registry
PROVIDER_REGISTRY: Dict[LLMProvider, Type[BaseLLMProvider]] = {
    LLMProvider.LM_STUDIO: LMStudioProvider,
}


def register_provider(provider_type: LLMProvider, provider_class: Type[BaseLLMProvider]) -> None:
    """
    Register a new provider class.
    
    Args:
        provider_type: The LLMProvider enum value
        provider_class: The provider class to register
    """
    PROVIDER_REGISTRY[provider_type] = provider_class


def create_provider(provider_type: LLMProvider, config: Dict[str, Any], 
                   logger: Optional[logging.Logger] = None) -> BaseLLMProvider:
    """
    Create a provider instance based on type and configuration.
    
    Args:
        provider_type: The type of provider to create
        config: Configuration dictionary for the provider
        logger: Optional logger instance
        
    Returns:
        Initialized provider instance
        
    Raises:
        ValueError: If provider type is not supported
        ImportError: If required dependencies are not installed
    """
    if provider_type not in PROVIDER_REGISTRY:
        available_providers = list(PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unsupported provider type: {provider_type}. Available: {available_providers}")
    
    provider_class = PROVIDER_REGISTRY[provider_type]
    
    try:
        provider = provider_class(config, logger)
        return provider
    except ImportError as e:
        raise ImportError(f"Failed to create {provider_type.value} provider. Missing dependencies: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to create {provider_type.value} provider: {e}")


async def create_and_initialize_provider(provider_type: LLMProvider, config: Dict[str, Any],
                                       logger: Optional[logging.Logger] = None) -> BaseLLMProvider:
    """
    Create and initialize a provider instance.
    
    Args:
        provider_type: The type of provider to create
        config: Configuration dictionary for the provider
        logger: Optional logger instance
        
    Returns:
        Initialized and ready-to-use provider instance
    """
    provider = create_provider(provider_type, config, logger)
    await provider.initialize()
    return provider


def get_available_providers() -> List[LLMProvider]:
    """
    Get list of available provider types.
    
    Returns:
        List of available LLMProvider enum values
    """
    return list(PROVIDER_REGISTRY.keys())


def is_provider_available(provider_type: LLMProvider) -> bool:
    """
    Check if a provider type is available.
    
    Args:
        provider_type: The provider type to check
        
    Returns:
        True if provider is available, False otherwise
    """
    return provider_type in PROVIDER_REGISTRY


def get_provider_info(provider_type: LLMProvider) -> Dict[str, Any]:
    """
    Get information about a provider type.
    
    Args:
        provider_type: The provider type to get info for
        
    Returns:
        Dictionary with provider information
    """
    if provider_type not in PROVIDER_REGISTRY:
        return {"available": False, "error": f"Provider {provider_type.value} not registered"}
    
    provider_class = PROVIDER_REGISTRY[provider_type]
    
    info = {
        "available": True,
        "provider_type": provider_type.value,
        "class_name": provider_class.__name__,
        "module": provider_class.__module__,
        "description": provider_class.__doc__.split('\n')[0] if provider_class.__doc__ else "No description available"
    }
    
    # Add provider-specific information if available
    if hasattr(provider_class, 'get_recommended_models'):
        info["recommended_models"] = provider_class.get_recommended_models()
    
    if hasattr(provider_class, 'get_setup_guide'):
        info["has_setup_guide"] = True
    
    return info


def get_all_provider_info() -> Dict[str, Any]:
    """
    Get information about all available providers.
    
    Returns:
        Dictionary mapping provider names to their information
    """
    return {
        provider_type.value: get_provider_info(provider_type)
        for provider_type in get_available_providers()
    }


# Lazy loading support for optional providers
def _try_import_openai_provider():
    """Try to import and register OpenAI provider if available."""
    try:
        from .openai import OpenAIProvider
        register_provider(LLMProvider.OPENAI, OpenAIProvider)
        return True
    except ImportError:
        return False


def _try_import_anthropic_provider():
    """Try to import and register Anthropic provider if available."""
    try:
        from .anthropic import AnthropicProvider
        register_provider(LLMProvider.ANTHROPIC, AnthropicProvider)
        return True
    except ImportError:
        return False


def _try_import_ollama_provider():
    """Try to import and register Ollama provider if available."""
    try:
        from .ollama import OllamaProvider
        register_provider(LLMProvider.OLLAMA, OllamaProvider)
        return True
    except ImportError:
        return False


def initialize_all_providers() -> Dict[str, bool]:
    """
    Try to import and register all available providers.
    
    Returns:
        Dictionary mapping provider names to their import success status
    """
    results = {
        "lm_studio": True,  # Always available since we implemented it
        "openai": _try_import_openai_provider(),
        "anthropic": _try_import_anthropic_provider(),
        "ollama": _try_import_ollama_provider(),
    }
    
    logger = logging.getLogger(__name__)
    for provider_name, success in results.items():
        if success:
            logger.info(f"Provider {provider_name} registered successfully")
        else:
            logger.warning(f"Provider {provider_name} not available (missing dependencies)")
    
    return results


# Initialize providers on module import
_initialization_results = initialize_all_providers()