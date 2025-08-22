"""
Base provider functionality for AEGIS LLM integrations.

This module provides the abstract base class for all LLM provider integrations.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..utils.data_structures import LLMProvider, ConversationMessage, LLMResponse


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers in the AEGIS system.
    
    This class defines the common interface that all provider implementations
    must follow for integration with the core LLM components.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the provider with configuration.
        
        Args:
            config: Provider-specific configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._client = None
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider client and connections."""
        pass
    
    @abstractmethod
    async def generate_response(self, messages: List[ConversationMessage], **kwargs) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse object with the generated response
        """
        pass
    
    @abstractmethod
    async def check_health(self) -> bool:
        """
        Check if the provider is available and healthy.
        
        Returns:
            True if the provider is healthy, False otherwise
        """
        pass
    
    @abstractmethod
    def get_provider_type(self) -> LLMProvider:
        """
        Get the provider type enum.
        
        Returns:
            LLMProvider enum value
        """
        pass
    
    async def cleanup(self) -> None:
        """Clean up provider resources."""
        self._initialized = False
        if hasattr(self._client, 'close') and callable(self._client.close):
            try:
                await self._client.close()
            except Exception as e:
                self.logger.warning(f"Error during cleanup: {e}")
    
    def is_initialized(self) -> bool:
        """Check if the provider is initialized."""
        return self._initialized
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "provider": self.get_provider_type().value,
            "model": self.config.get("model_name", "unknown"),
            "base_url": self.config.get("base_url"),
            "status": "initialized" if self._initialized else "not_initialized"
        }
    
    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}(model={self.config.get('model_name', 'unknown')})"