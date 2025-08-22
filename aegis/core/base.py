"""
Base LLM functionality for the AEGIS system.

This module provides the foundational BaseLLM class that serves as the abstract base
for all LLM components in the system, including AttackerLLM, DefenderLLM, and JudgeLLM.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
import json

# Import data structures from Phase 1
from ..utils.data_structures import (
    LLMProvider, ConversationMessage, LLMResponse,
    AttackResult, DefenseResult, JudgeResult, RedTeamSession
)


@dataclass
class LLMConfig:
    """Configuration for LLM instances."""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    custom_headers: Optional[Dict[str, str]] = None
    custom_params: Optional[Dict[str, Any]] = None


class BaseLLM(ABC):
    """
    Abstract base class for all LLM components in the AEGIS system.
    
    This class provides common functionality for:
    - Provider initialization and configuration
    - Async response generation
    - Error handling and retry logic
    - Logging and monitoring
    - Rate limiting and resource management
    """
    
    def __init__(self, config: LLMConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the BaseLLM with configuration.
        
        Args:
            config: LLM configuration settings
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Initialize provider-specific client
        self._client = None
        self._initialize_client()
        
        # Rate limiting and monitoring
        self._request_count = 0
        self._last_request_time = 0.0
        self._rate_limit_delay = 0.1  # Minimum delay between requests
        
        self.logger.info(f"Initialized {self.__class__.__name__} with provider {config.provider.value}")
    
    def _initialize_client(self) -> None:
        """Initialize the provider-specific client."""
        try:
            if self.config.provider == LLMProvider.OPENAI:
                self._initialize_openai_client()
            elif self.config.provider == LLMProvider.ANTHROPIC:
                self._initialize_anthropic_client()
            elif self.config.provider == LLMProvider.OLLAMA:
                self._initialize_ollama_client()
            elif self.config.provider == LLMProvider.LM_STUDIO:
                self._initialize_lm_studio_client()
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.config.provider.value} client: {e}")
            raise
    
    def _initialize_openai_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            import openai
            self._client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    def _initialize_anthropic_client(self) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
        except ImportError:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
    
    def _initialize_ollama_client(self) -> None:
        """Initialize Ollama client."""
        try:
            import ollama
            self._client = ollama.AsyncClient(
                host=self.config.base_url or "http://localhost:11434"
            )
        except ImportError:
            raise ImportError("Ollama package not installed. Install with: pip install ollama")
    
    def _initialize_lm_studio_client(self) -> None:
        """Initialize LM Studio client (OpenAI-compatible)."""
        try:
            import openai
            # LM Studio uses OpenAI-compatible API
            base_url = self.config.base_url or "http://localhost:1234/v1"
            self._client = openai.AsyncOpenAI(
                api_key=self.config.api_key or "lm-studio",  # LM Studio doesn't require real API key
                base_url=base_url,
                timeout=self.config.timeout
            )
            self.logger.info(f"Initialized LM Studio client with base_url: {base_url}")
        except ImportError:
            raise ImportError("OpenAI package required for LM Studio. Install with: pip install openai")
    
    async def _rate_limit_check(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - time_since_last)
        
        self._last_request_time = time.time()
        self._request_count += 1
    
    async def _make_request_with_retry(self, request_func, *args, **kwargs) -> Any:
        """
        Make a request with retry logic and error handling.
        
        Args:
            request_func: The async function to call
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            The response from the request function
        """
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                await self._rate_limit_check()
                
                # Log the attempt
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt + 1}/{self.config.retry_attempts}")
                
                result = await request_func(*args, **kwargs)
                
                # Log successful request
                self.logger.debug(f"Request successful on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Request failed on attempt {attempt + 1}: {str(e)}")
                
                if attempt < self.config.retry_attempts - 1:
                    delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All {self.config.retry_attempts} attempts failed")
        
        # If we get here, all attempts failed
        raise last_exception
    
    async def _generate_openai_response(self, messages: List[ConversationMessage]) -> LLMResponse:
        """Generate response using OpenAI API."""
        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            response = await self._client.chat.completions.create(
                model=self.config.model_name,
                messages=openai_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **(self.config.custom_params or {})
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider=self.config.provider,
                model=self.config.model_name,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else {}
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _generate_anthropic_response(self, messages: List[ConversationMessage]) -> LLMResponse:
        """Generate response using Anthropic API."""
        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            system_message = None
            
            for msg in messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    anthropic_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            kwargs = {
                "model": self.config.model_name,
                "messages": anthropic_messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                **(self.config.custom_params or {})
            }
            
            if system_message:
                kwargs["system"] = system_message
            
            response = await self._client.messages.create(**kwargs)
            
            return LLMResponse(
                content=response.content[0].text,
                provider=self.config.provider,
                model=self.config.model_name,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                } if response.usage else {}
            )
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise
    
    async def _generate_ollama_response(self, messages: List[ConversationMessage]) -> LLMResponse:
        """Generate response using Ollama API."""
        try:
            # Convert messages to Ollama format
            ollama_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            response = await self._client.chat(
                model=self.config.model_name,
                messages=ollama_messages,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    **(self.config.custom_params or {})
                }
            )
            
            return LLMResponse(
                content=response["message"]["content"],
                provider=self.config.provider,
                model=self.config.model_name,
                usage={
                    "prompt_eval_count": response.get("prompt_eval_count", 0),
                    "eval_count": response.get("eval_count", 0),
                    "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Ollama API error: {e}")
            raise
    
    async def _generate_lm_studio_response(self, messages: List[ConversationMessage]) -> LLMResponse:
        """Generate response using LM Studio API (OpenAI-compatible)."""
        try:
            # LM Studio uses OpenAI-compatible format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            response = await self._client.chat.completions.create(
                model=self.config.model_name,
                messages=openai_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **(self.config.custom_params or {})
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider=self.config.provider,
                model=self.config.model_name,
                usage={
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0,
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0) if response.usage else 0,
                    "total_tokens": getattr(response.usage, 'total_tokens', 0) if response.usage else 0
                }
            )
            
        except Exception as e:
            self.logger.error(f"LM Studio API error: {e}")
            raise
    
    async def generate_response(self, messages: List[ConversationMessage]) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            LLMResponse object with the generated response
        """
        if not messages:
            raise ValueError("Messages cannot be empty")
        
        start_time = time.time()
        
        # Route to appropriate provider
        if self.config.provider == LLMProvider.OPENAI:
            request_func = self._generate_openai_response
        elif self.config.provider == LLMProvider.ANTHROPIC:
            request_func = self._generate_anthropic_response
        elif self.config.provider == LLMProvider.OLLAMA:
            request_func = self._generate_ollama_response
        elif self.config.provider == LLMProvider.LM_STUDIO:
            request_func = self._generate_lm_studio_response
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
        
        # Make request with retry logic
        response = await self._make_request_with_retry(request_func, messages)
        
        # Calculate response time
        response.response_time_ms = (time.time() - start_time) * 1000
        
        # Log the response
        self.logger.debug(f"Generated response: {len(response.content)} characters in {response.response_time_ms:.2f}ms")
        
        return response
    
    async def generate_streaming_response(self, messages: List[ConversationMessage]) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            messages: List of conversation messages
            
        Yields:
            Chunks of the response as they are generated
        """
        # This is a basic implementation - specific providers may override for true streaming
        response = await self.generate_response(messages)
        
        # Simulate streaming by yielding chunks
        content = response.content
        chunk_size = 50
        
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            yield chunk
            await asyncio.sleep(0.05)  # Small delay to simulate streaming
    
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a specific request type for this LLM component.
        
        This method should be implemented by subclasses to handle their specific
        request types (attack generation, defense, judgment, etc.)
        
        Args:
            request: The request to process
            
        Returns:
            The processed result
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this LLM instance."""
        return {
            "provider": self.config.provider.value,
            "model": self.config.model_name,
            "request_count": self._request_count,
            "last_request_time": self._last_request_time,
            "config": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "timeout": self.config.timeout
            }
        }
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the LLM provider.
        
        Returns:
            True if the provider is healthy, False otherwise
        """
        try:
            test_messages = [
                ConversationMessage(role="user", content="Hello, this is a health check.")
            ]
            
            response = await self.generate_response(test_messages)
            return bool(response.content)
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation of the LLM instance."""
        return f"{self.__class__.__name__}(provider={self.config.provider.value}, model={self.config.model_name})"