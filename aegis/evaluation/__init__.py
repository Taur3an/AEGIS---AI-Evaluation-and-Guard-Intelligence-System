"""
AEGIS Evaluation System

Comprehensive evaluation modules for assessing AI alignment risks including:
- BaseEvaluationTarget: Abstract base class for all risk evaluations
- AttackVectorLibrary: Library of 45+ research-based attack patterns
- RiskEvaluator: Multi-dimensional risk scoring and evaluation
- TestScenarioGenerator: Dynamic test scenario creation
- 9 Risk-specific evaluation targets for comprehensive AI safety assessment

Academic Foundation:
Based on peer-reviewed research from Anthropic, OpenAI, MIRI, FHI, and other
leading AI safety institutions.
"""

# Core evaluation framework
from .base import (
    BaseEvaluationTarget,
    EvaluationResult,
    TestScenario,
    EvaluationTargetRegistry
)

# Attack vector library and evaluation components
from .library import (
    AttackVectorLibrary,
    AttackVectorDefinition,
    attack_library  # Pre-instantiated library
)

from .evaluator import RiskEvaluator
from .generator import TestScenarioGenerator

# All risk-specific evaluation targets
from .targets import (
    RewardHackingTarget,
    DeceptionTarget,
    HiddenMotivationsTarget,
    SabotageTarget,
    InappropriateToolUseTarget,
    DataExfiltrationTarget,
    SandbaggingTarget,
    EvaluationAwarenessTarget,
    ChainOfThoughtIssuesTarget,
    RISK_TARGET_MAP,
    get_target_for_category,
    get_all_targets
)

__all__ = [
    # Base framework
    "BaseEvaluationTarget",
    "EvaluationResult",
    "TestScenario",
    "EvaluationTargetRegistry",
    
    # Core components
    "AttackVectorLibrary",
    "AttackVectorDefinition",
    "attack_library",
    "RiskEvaluator",
    "TestScenarioGenerator",
    
    # Risk-specific targets
    "RewardHackingTarget",
    "DeceptionTarget",
    "HiddenMotivationsTarget",
    "SabotageTarget",
    "InappropriateToolUseTarget",
    "DataExfiltrationTarget",
    "SandbaggingTarget",
    "EvaluationAwarenessTarget",
    "ChainOfThoughtIssuesTarget",
    
    # Utility functions
    "RISK_TARGET_MAP",
    "get_target_for_category",
    "get_all_targets",
]