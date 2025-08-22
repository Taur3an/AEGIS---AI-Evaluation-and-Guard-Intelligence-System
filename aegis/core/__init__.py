"""
AEGIS Core LLM Components

Core components for the AEGIS red teaming system including BaseLLM,
AttackerLLM, DefenderLLM, JudgeLLM, and RedTeamOrchestrator.

This module provides the complete suite of LLM components for conducting
sophisticated AI safety evaluations and red teaming exercises.
"""

import logging

logger = logging.getLogger(__name__)

# Import core components
from .base import BaseLLM, LLMConfig
from .attacker import AttackerLLM, AttackStrategy
from .defender import DefenderLLM, SafetyConfig, DefenseMetrics
from .judge import JudgeLLM, JudgmentCriteria, ScoringMetrics
from .orchestrator import (
    RedTeamOrchestrator,
    OrchestrationConfig,
    SessionMetrics,
    SessionStatus,
    ExecutionMode
)

logger.info("AEGIS Core LLM Components loaded successfully")


def create_basic_attacker(provider, model_name, **kwargs):
    """
    Create a basic AttackerLLM instance with LM Studio provider.
    
    Args:
        provider: LLMProvider enum value
        model_name: Name of the model to use
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured AttackerLLM instance
    """
    config = LLMConfig(
        provider=provider,
        model_name=model_name,
        **kwargs
    )
    return AttackerLLM(config)


def create_basic_defender(provider, model_name, **kwargs):
    """
    Create a basic DefenderLLM instance.
    
    Args:
        provider: LLMProvider enum value
        model_name: Name of the model to use
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured DefenderLLM instance
    """
    config = LLMConfig(
        provider=provider,
        model_name=model_name,
        **kwargs
    )
    return DefenderLLM(config)


def create_basic_judge(provider, model_name, **kwargs):
    """
    Create a basic JudgeLLM instance.
    
    Args:
        provider: LLMProvider enum value
        model_name: Name of the model to use
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured JudgeLLM instance
    """
    config = LLMConfig(
        provider=provider,
        model_name=model_name,
        **kwargs
    )
    return JudgeLLM(config)


def create_red_team_orchestrator(attacker, defender, judge, **config_kwargs):
    """
    Create a RedTeamOrchestrator with the provided LLM components.
    
    Args:
        attacker: AttackerLLM instance
        defender: DefenderLLM instance
        judge: JudgeLLM instance
        **config_kwargs: Additional orchestration configuration
        
    Returns:
        Configured RedTeamOrchestrator instance
    """
    orchestration_config = OrchestrationConfig(**config_kwargs)
    return RedTeamOrchestrator(attacker, defender, judge, orchestration_config)


def create_complete_red_team_setup(provider, attacker_model, defender_model, judge_model, **kwargs):
    """
    Create a complete red teaming setup with all components.
    
    Args:
        provider: LLMProvider enum value
        attacker_model: Model name for attacker
        defender_model: Model name for defender
        judge_model: Model name for judge
        **kwargs: Additional configuration parameters
        
    Returns:
        Tuple of (attacker, defender, judge, orchestrator)
    """
    # Create components
    attacker = create_basic_attacker(provider, attacker_model, **kwargs)
    defender = create_basic_defender(provider, defender_model, **kwargs)
    judge = create_basic_judge(provider, judge_model, **kwargs)
    
    # Create orchestrator
    orchestrator = create_red_team_orchestrator(attacker, defender, judge)
    
    return attacker, defender, judge, orchestrator


__all__ = [
    # Core classes
    "BaseLLM",
    "LLMConfig",
    "AttackerLLM",
    "DefenderLLM",
    "JudgeLLM",
    "RedTeamOrchestrator",
    
    # Configuration classes
    "AttackStrategy",
    "SafetyConfig",
    "DefenseMetrics",
    "JudgmentCriteria",
    "ScoringMetrics",
    "OrchestrationConfig",
    "SessionMetrics",
    
    # Enums
    "SessionStatus",
    "ExecutionMode",
    
    # Factory functions
    "create_basic_attacker",
    "create_basic_defender",
    "create_basic_judge",
    "create_red_team_orchestrator",
    "create_complete_red_team_setup",
]