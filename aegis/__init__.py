"""
AEGIS - AI Evaluation and Guard Intelligence System

A comprehensive framework for evaluating AI alignment risks through systematic red teaming,
featuring specialized evaluation modules for 9 critical AI safety categories.

Key Components:
- Core LLM Components (AttackerLLM, DefenderLLM, JudgeLLM, RedTeamOrchestrator)
- Evaluation Target Modules (9 risk-specific implementations)
- Attack Vector Library (45+ attack patterns)
- Configuration Management
- Provider Integrations (Ollama, OpenAI, Anthropic, LM Studio)

Usage:
    # Modern API
    from aegis import AttackVectorLibrary, RiskEvaluator, RiskCategory
    from aegis.core import AttackerLLM, DefenderLLM, JudgeLLM, RedTeamOrchestrator
    
    # Backward compatibility API
    from aegis import initialize_aegis, evaluate_single_risk, evaluate_comprehensive_risk
"""

# Version information
__version__ = "1.0.0"
__author__ = "AEGIS Development Team"
__email__ = "contact@aegis-ai.org"
__description__ = "AI Evaluation and Guard Intelligence System"

# Core public API
from .utils.data_structures import (
    RiskCategory,
    AttackVector,
    TestScenario,
    RiskAssessment,
    ReasoningTrace,
    RiskLevel,
    AttackVectorType,
)

# Evaluation system
try:
    from .evaluation import (
        AttackVectorLibrary,
        RiskEvaluator,
        TestScenarioGenerator,
        BaseEvaluationTarget,
    )
except ImportError:
    # During initial package setup, these may not be available yet
    AttackVectorLibrary = None
    RiskEvaluator = None
    TestScenarioGenerator = None
    BaseEvaluationTarget = None

# Core LLM components
try:
    from .core import (
        AttackerLLM,
        DefenderLLM,
        JudgeLLM,
        RedTeamOrchestrator,
    )
except ImportError:
    # During initial package setup, these may not be available yet
    AttackerLLM = None
    DefenderLLM = None
    JudgeLLM = None
    RedTeamOrchestrator = None

# Configuration management
try:
    from .config import AegisConfig, load_config
except ImportError:
    AegisConfig = None
    load_config = None

# Provider integrations
try:
    from .providers import create_provider, get_available_providers
except ImportError:
    create_provider = None
    get_available_providers = None

# Backward compatibility API
try:
    from .api import (
        initialize_aegis,
        evaluate_single_risk,
        evaluate_comprehensive_risk,
        get_vectors_by_category,
        get_vectors_by_difficulty,
        generate_test_scenario,
        generate_test_suite,
        get_supported_risk_categories,
        get_system_status,
        quick_risk_check,
        list_available_attacks,
    )
except ImportError:
    # Create minimal placeholders for backward compatibility
    import logging
    logger = logging.getLogger(__name__)
    
    def initialize_aegis(config_path=None):
        logger.warning("AEGIS API not fully loaded. Package is in setup phase.")
        return {"status": "setup_phase", "components": {}}
    
    def evaluate_single_risk(*args, **kwargs):
        logger.warning("evaluate_single_risk not available during setup phase")
        return None
    
    def evaluate_comprehensive_risk(*args, **kwargs):
        logger.warning("evaluate_comprehensive_risk not available during setup phase") 
        return {}
    
    def get_vectors_by_category(*args, **kwargs):
        logger.warning("get_vectors_by_category not available during setup phase")
        return []
    
    def get_vectors_by_difficulty(*args, **kwargs):
        logger.warning("get_vectors_by_difficulty not available during setup phase")
        return []
    
    def generate_test_scenario(*args, **kwargs):
        logger.warning("generate_test_scenario not available during setup phase")
        return None
    
    def generate_test_suite(*args, **kwargs):
        logger.warning("generate_test_suite not available during setup phase")
        return []
    
    def get_supported_risk_categories():
        return list(RiskCategory) if RiskCategory else []
    
    def get_system_status():
        return {"status": "setup_phase", "initialized": False}
    
    def quick_risk_check(*args, **kwargs):
        logger.warning("quick_risk_check not available during setup phase")
        return {"error": "setup_phase"}
    
    def list_available_attacks(*args, **kwargs):
        logger.warning("list_available_attacks not available during setup phase")
        return []

# Main public API exports
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    
    # Core data structures
    "RiskCategory",
    "AttackVector", 
    "TestScenario",
    "RiskAssessment",
    "ReasoningTrace",
    "RiskLevel",
    "AttackVectorType",
    
    # Evaluation system (modern API)
    "AttackVectorLibrary",
    "RiskEvaluator", 
    "TestScenarioGenerator",
    "BaseEvaluationTarget",
    
    # Core LLM components (modern API)
    "AttackerLLM",
    "DefenderLLM",
    "JudgeLLM", 
    "RedTeamOrchestrator",
    
    # Configuration (modern API)
    "AegisConfig",
    "load_config",
    
    # Provider integrations (modern API)
    "create_provider",
    "get_available_providers",
    
    # Backward compatibility API
    "initialize_aegis",
    "evaluate_single_risk",
    "evaluate_comprehensive_risk",
    "get_vectors_by_category",
    "get_vectors_by_difficulty", 
    "generate_test_scenario",
    "generate_test_suite",
    "get_supported_risk_categories",
    "get_system_status",
    "quick_risk_check",
    "list_available_attacks",
]

# Package-level logging configuration
import logging

# Create package logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info(f"AEGIS v{__version__} package loaded successfully")

# Convenience functions for backward compatibility
def get_version():
    """Get AEGIS version string."""
    return __version__

def get_package_info():
    """Get comprehensive package information."""
    return {
        "name": "AEGIS",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "email": __email__,
        "supported_risk_categories": len(list(RiskCategory)) if RiskCategory else 0,
        "api_available": {
            "modern": AttackVectorLibrary is not None,
            "backward_compatible": True,
        }
    }