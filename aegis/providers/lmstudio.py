"""
LM Studio provider implementation for AEGIS.

This module provides the LM Studio provider for running uncensored local models
via LM Studio's OpenAI-compatible API.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from .base import BaseLLMProvider
from ..utils.data_structures import LLMProvider, ConversationMessage, LLMResponse


class LMStudioProvider(BaseLLMProvider):
    """
    LM Studio provider for running uncensored models locally.
    
    LM Studio provides an OpenAI-compatible API for local model inference,
    making it ideal for red teaming with uncensored models that don't have
    built-in safety filters.
    
    Recommended models for red teaming:
    - WizardLM-Uncensored variants
    - Dolphin models (uncensored)
    - CodeLlama-Instruct variants
    - Mixtral-Instruct uncensored variants
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize LM Studio provider.
        
        Args:
            config: Configuration including host, port, model_name, etc.
            logger: Optional logger instance
        """
        super().__init__(config, logger)
        
        # LM Studio specific configuration
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 1234)
        self.model_name = config.get("model_name", "local-model")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.timeout = config.get("timeout", 30)
        
        # Build base URL
        self.base_url = f"http://{self.host}:{self.port}/v1"
        
        # Client will be initialized in initialize() method
        self._client = None
    
    async def initialize(self) -> None:
        """Initialize the LM Studio client."""
        try:
            import openai
            
            # LM Studio uses OpenAI-compatible API
            self._client = openai.AsyncOpenAI(
                api_key="lm-studio",  # LM Studio doesn't require real API key
                base_url=self.base_url,
                timeout=self.timeout
            )
            
            # Test connection
            health_check = await self.check_health()
            if health_check:
                self._initialized = True
                self.logger.info(f"LM Studio provider initialized successfully at {self.base_url}")
            else:
                raise ConnectionError("LM Studio health check failed")
                
        except ImportError:
            raise ImportError("OpenAI package required for LM Studio. Install with: pip install openai")
        except Exception as e:
            self.logger.error(f"Failed to initialize LM Studio provider: {e}")
            raise
    
    async def generate_response(self, messages: List[ConversationMessage], **kwargs) -> LLMResponse:
        """
        Generate a response using LM Studio.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            LLMResponse with the generated content
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            start_time = time.time()
            
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # Prepare request parameters
            request_params = {
                "model": self.model_name,
                "messages": openai_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "top_p": kwargs.get("top_p", self.top_p),
                "stream": False
            }
            
            # Add any additional custom parameters
            custom_params = kwargs.get("custom_params", {})
            request_params.update(custom_params)
            
            response = await self._client.chat.completions.create(**request_params)
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000
            
            # Extract usage information if available
            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0)
                }
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider=LLMProvider.LM_STUDIO,
                model=self.model_name,
                usage=usage,
                response_time_ms=response_time,
                metadata={
                    "finish_reason": getattr(response.choices[0], 'finish_reason', 'unknown'),
                    "model": getattr(response, 'model', self.model_name),
                    "base_url": self.base_url
                }
            )
            
        except Exception as e:
            self.logger.error(f"LM Studio generation failed: {e}")
            raise
    
    async def check_health(self) -> bool:
        """
        Check if LM Studio server is available and responsive.
        
        Returns:
            True if LM Studio is healthy, False otherwise
        """
        try:
            if not self._client:
                return False
            
            # Try to list models as a health check
            models_response = await asyncio.wait_for(
                self._client.models.list(),
                timeout=5.0
            )
            
            # Check if our model is available
            if hasattr(models_response, 'data'):
                available_models = [model.id for model in models_response.data]
                self.logger.debug(f"Available models: {available_models}")
                
                # If specific model not found, log warning but consider healthy
                if self.model_name not in available_models and available_models:
                    self.logger.warning(f"Model '{self.model_name}' not found. Available: {available_models}")
            
            return True
            
        except asyncio.TimeoutError:
            self.logger.warning("LM Studio health check timed out")
            return False
        except Exception as e:
            self.logger.warning(f"LM Studio health check failed: {e}")
            return False
    
    async def list_models(self) -> List[str]:
        """
        List available models in LM Studio.
        
        Returns:
            List of available model names
        """
        try:
            if not self._client:
                await self.initialize()
            
            response = await self._client.models.list()
            
            if hasattr(response, 'data'):
                return [model.id for model in response.data]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to list LM Studio models: {e}")
            return []
    
    def get_provider_type(self) -> LLMProvider:
        """Get the provider type."""
        return LLMProvider.LM_STUDIO
    
    async def generate_streaming_response(self, messages: List[ConversationMessage], **kwargs):
        """
        Generate a streaming response using LM Studio.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters
            
        Yields:
            Response chunks as they are generated
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # Prepare request parameters for streaming
            request_params = {
                "model": self.model_name,
                "messages": openai_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "top_p": kwargs.get("top_p", self.top_p),
                "stream": True
            }
            
            # Create streaming request
            stream = await self._client.chat.completions.create(**request_params)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error(f"LM Studio streaming failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the LM Studio configuration."""
        base_info = super().get_model_info()
        base_info.update({
            "host": self.host,
            "port": self.port,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.timeout,
            "provider_specific": {
                "supports_streaming": True,
                "supports_custom_models": True,
                "requires_api_key": False,
                "local_inference": True
            }
        })
        return base_info
    
    @staticmethod
    def get_recommended_models() -> Dict[str, List[str]]:
        """
        Get recommended uncensored models for different use cases.
        
        Returns:
            Dictionary of use cases and recommended models
        """
        return {
            "red_teaming": [
                "WizardLM-13B-Uncensored",
                "WizardLM-30B-Uncensored",
                "dolphin-2.2.1-mistral-7b",
                "dolphin-2.6-mixtral-8x7b",
                "CodeLlama-13B-Instruct",
                "Mixtral-8x7B-Instruct-v0.1-uncensored"
            ],
            "reasoning": [
                "OpenHermes-2.5-Mistral-7B",
                "Neural-Chat-7B-v3-1",
                "Starling-LM-7B-alpha"
            ],
            "creative": [
                "MythoMax-L2-13B",
                "Airoboros-L2-13B-2.1"
            ]
        }
    
    @staticmethod
    def get_setup_guide() -> str:
        """
        Get setup instructions for LM Studio with AEGIS.
        
        Returns:
            Formatted setup guide string
        """
        return """
# LM Studio Setup Guide for AEGIS

## 1. Download and Install LM Studio
- Visit: https://lmstudio.ai/
- Download for your platform (Windows, macOS, Linux)
- Install and launch LM Studio

## 2. Download Recommended Models
For red teaming, download these uncensored models:

### High Priority (Start Here):
- **WizardLM-13B-Uncensored** - Excellent for adversarial prompts
- **dolphin-2.2.1-mistral-7b** - Fast and effective for attacks
- **Mixtral-8x7B-Instruct-uncensored** - Powerful reasoning for complex attacks

### Advanced Models:
- **WizardLM-30B-Uncensored** - More capable but requires more VRAM
- **CodeLlama-13B-Instruct** - Great for code-based attacks

## 3. Configure LM Studio
1. Load your chosen model in LM Studio
2. Go to "Local Server" tab
3. Start the server (default: localhost:1234)
4. Ensure "Cross-Origin-Resource-Sharing (CORS)" is enabled

## 4. AEGIS Configuration
```python
from aegis.core.base import LLMConfig
from aegis.utils.data_structures import LLMProvider

config = LLMConfig(
    provider=LLMProvider.LM_STUDIO,
    model_name="WizardLM-13B-Uncensored",  # Use your loaded model name
    base_url="http://localhost:1234/v1",
    temperature=0.8,  # Higher for creative attacks
    max_tokens=2048
)
```

## 5. Optimal Settings for Red Teaming
- **Temperature**: 0.7-0.9 (higher for more creative attacks)
- **Top-p**: 0.9 (allows diverse token selection)
- **Max Tokens**: 2048-4096 (longer responses for complex attacks)

## 6. Security Considerations
- Run LM Studio on isolated networks for sensitive testing
- Monitor resource usage - uncensored models can be computationally intensive
- Keep attack logs secure and follow responsible disclosure practices
"""


def create_lmstudio_config(
    host: str = "localhost",
    port: int = 1234,
    model_name: str = "local-model",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a configuration dictionary for LM Studio provider.
    
    Args:
        host: LM Studio server host
        port: LM Studio server port
        model_name: Name of the model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration dictionary for LM Studio provider
    """
    config = {
        "host": host,
        "port": port,
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "base_url": f"http://{host}:{port}/v1",
        **kwargs
    }
    
    return config