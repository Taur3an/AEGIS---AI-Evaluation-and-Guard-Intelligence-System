"""
AEGIS Utilities

Utility modules for the AEGIS system including core data structures,
helper functions, and common functionality.
"""

from .data_structures import (
    # Enumerations
    RiskCategory,
    AttackVectorType,
    RiskLevel,
    
    # Core data classes
    AttackVector,
    TestScenario,
    ReasoningStep,
    DecisionPoint,
    SafetyActivation,
    ReasoningTrace,
    VulnerabilityFlag,
    RiskAssessment,
    SessionConfig,
    AttackResult,
    EvaluationResult,
    
    # Utility functions
    create_attack_vector,
    create_test_scenario,
    risk_level_from_score,
    get_risk_category_description,
)

from .helpers import (
    # ID and timestamp utilities
    generate_unique_id,
    get_current_timestamp,
    hash_content,
    
    # Validation utilities
    validate_risk_score,
    normalize_risk_score,
    risk_score_to_level,
    validate_attack_vector,
    validate_test_scenario,
    
    # File and data utilities
    sanitize_filename,
    ensure_directory_exists,
    load_json_file,
    save_json_file,
    
    # Text and formatting utilities
    truncate_text,
    format_duration_ms,
    merge_dictionaries,
    extract_template_variables,
    substitute_template_variables,
    
    # Environment and configuration utilities
    get_environment_variable,
    setup_basic_logging,
    calculate_weighted_score,
    
    # Constants
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_RISK_THRESHOLD,
    SUPPORTED_CONFIG_EXTENSIONS,
    SUPPORTED_EXPORT_FORMATS,
)

__all__ = [
    # Enumerations
    "RiskCategory",
    "AttackVectorType", 
    "RiskLevel",
    
    # Core data classes
    "AttackVector",
    "TestScenario",
    "ReasoningStep",
    "DecisionPoint",
    "SafetyActivation",
    "ReasoningTrace",
    "VulnerabilityFlag",
    "RiskAssessment",
    "SessionConfig",
    "AttackResult",
    "EvaluationResult",
    
    # Data structure utility functions
    "create_attack_vector",
    "create_test_scenario",
    "risk_level_from_score",
    "get_risk_category_description",
    
    # Helper utility functions
    "generate_unique_id",
    "get_current_timestamp",
    "hash_content",
    "validate_risk_score",
    "normalize_risk_score",
    "risk_score_to_level",
    "validate_attack_vector",
    "validate_test_scenario",
    "sanitize_filename",
    "ensure_directory_exists",
    "load_json_file",
    "save_json_file",
    "truncate_text",
    "format_duration_ms",
    "merge_dictionaries",
    "extract_template_variables",
    "substitute_template_variables",
    "get_environment_variable",
    "setup_basic_logging",
    "calculate_weighted_score",
    
    # Constants
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_RISK_THRESHOLD",
    "SUPPORTED_CONFIG_EXTENSIONS",
    "SUPPORTED_EXPORT_FORMATS",
]