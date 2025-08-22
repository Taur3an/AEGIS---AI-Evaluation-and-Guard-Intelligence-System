"""
AEGIS Configuration Management

Configuration system for AEGIS including settings classes, YAML configuration loading,
and environment management.
"""

try:
    from .settings import AegisConfig, LLMConfig, EvaluationConfig, LoggingConfig
    from .loader import load_config, save_config, get_default_config
except ImportError:
    # During package setup phase, create placeholder classes
    import logging
    from typing import Dict, Any, Optional
    from dataclasses import dataclass
    
    logger = logging.getLogger(__name__)
    
    @dataclass
    class LLMConfig:
        """Configuration for LLM providers."""
        provider: str = "ollama"
        model_name: str = "llama2:latest"
        api_key: Optional[str] = None
        base_url: Optional[str] = None
        temperature: float = 0.7
        max_tokens: int = 2048
        timeout: int = 30
    
    @dataclass
    class EvaluationConfig:
        """Configuration for evaluation settings."""
        enabled_categories: list = None
        max_iterations: int = 10
        timeout_seconds: int = 300
        parallel_execution: bool = False
        
        def __post_init__(self):
            if self.enabled_categories is None:
                self.enabled_categories = ["reward_hacking", "deception"]
    
    @dataclass
    class LoggingConfig:
        """Configuration for logging."""
        level: str = "INFO"
        format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handlers: list = None
        retention_days: int = 30
        
        def __post_init__(self):
            if self.handlers is None:
                self.handlers = ["console", "file"]
    
    @dataclass
    class AegisConfig:
        """Main AEGIS configuration."""
        system_name: str = "AEGIS"
        version: str = "1.0.0"
        environment: str = "development"
        
        # LLM configurations
        attacker_llm: LLMConfig = None
        defender_llm: LLMConfig = None
        judge_llm: LLMConfig = None
        
        # Evaluation configuration
        evaluation: EvaluationConfig = None
        
        # Logging configuration
        logging: LoggingConfig = None
        
        # Safety settings
        safety: Dict[str, Any] = None
        
        def __post_init__(self):
            if self.attacker_llm is None:
                self.attacker_llm = LLMConfig()
            if self.defender_llm is None:
                self.defender_llm = LLMConfig()
            if self.judge_llm is None:
                self.judge_llm = LLMConfig()
            if self.evaluation is None:
                self.evaluation = EvaluationConfig()
            if self.logging is None:
                self.logging = LoggingConfig()
            if self.safety is None:
                self.safety = {
                    "max_concurrent_attacks": 5,
                    "attack_timeout_seconds": 120,
                    "content_filtering_enabled": True,
                    "rate_limiting_enabled": True
                }
    
    def load_config(config_path: Optional[str] = None) -> AegisConfig:
        """Load configuration from file or return default."""
        logger.warning("Configuration loading not yet implemented. Returning default config.")
        return get_default_config()
    
    def save_config(config: AegisConfig, config_path: str) -> None:
        """Save configuration to file."""
        logger.warning("Configuration saving not yet implemented.")
    
    def get_default_config() -> AegisConfig:
        """Get default AEGIS configuration."""
        return AegisConfig()

__all__ = [
    "AegisConfig",
    "LLMConfig", 
    "EvaluationConfig",
    "LoggingConfig",
    "load_config",
    "save_config",
    "get_default_config",
]