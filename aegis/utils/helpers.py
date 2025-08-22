"""
AEGIS Helper Functions

Utility functions for common operations, validation, and data processing
used throughout the AEGIS system.
"""

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from .data_structures import RiskCategory, RiskLevel, AttackVector, TestScenario


logger = logging.getLogger(__name__)


def generate_unique_id(prefix: str = "") -> str:
    """Generate a unique identifier with optional prefix."""
    unique_id = str(uuid.uuid4())
    if prefix:
        return f"{prefix}-{unique_id}"
    return unique_id


def get_current_timestamp() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def hash_content(content: str) -> str:
    """Generate SHA-256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()


def validate_risk_score(score: float) -> bool:
    """Validate that a risk score is in the valid range [0.0, 1.0]."""
    return isinstance(score, (int, float)) and 0.0 <= score <= 1.0


def normalize_risk_score(score: float) -> float:
    """Normalize a risk score to the range [0.0, 1.0]."""
    if not isinstance(score, (int, float)):
        logger.warning(f"Invalid score type: {type(score)}, defaulting to 0.0")
        return 0.0
    return max(0.0, min(1.0, float(score)))


def risk_score_to_level(score: float) -> RiskLevel:
    """Convert a numeric risk score to a RiskLevel enum."""
    normalized_score = normalize_risk_score(score)
    
    if normalized_score >= 0.9:
        return RiskLevel.CRITICAL
    elif normalized_score >= 0.7:
        return RiskLevel.HIGH
    elif normalized_score >= 0.5:
        return RiskLevel.MEDIUM
    elif normalized_score >= 0.3:
        return RiskLevel.LOW
    else:
        return RiskLevel.MINIMAL


def validate_attack_vector(vector: AttackVector) -> List[str]:
    """Validate an attack vector and return list of validation errors."""
    errors = []
    
    if not vector.id:
        errors.append("Attack vector ID is required")
    
    if not vector.name or not vector.name.strip():
        errors.append("Attack vector name is required")
    
    if not vector.description or not vector.description.strip():
        errors.append("Attack vector description is required")
    
    if not vector.prompt_template or not vector.prompt_template.strip():
        errors.append("Attack vector prompt template is required")
    
    if not isinstance(vector.risk_category, RiskCategory):
        errors.append("Invalid risk category")
    
    if vector.difficulty not in ["easy", "medium", "hard"]:
        errors.append("Difficulty must be 'easy', 'medium', or 'hard'")
    
    return errors


def validate_test_scenario(scenario: TestScenario) -> List[str]:
    """Validate a test scenario and return list of validation errors."""
    errors = []
    
    if not scenario.id:
        errors.append("Test scenario ID is required")
    
    if not scenario.name or not scenario.name.strip():
        errors.append("Test scenario name is required")
    
    if not scenario.description or not scenario.description.strip():
        errors.append("Test scenario description is required")
    
    if not isinstance(scenario.risk_category, RiskCategory):
        errors.append("Invalid risk category")
    
    if not scenario.target_prompt or not scenario.target_prompt.strip():
        errors.append("Target prompt is required")
    
    if scenario.difficulty not in ["easy", "medium", "hard"]:
        errors.append("Difficulty must be 'easy', 'medium', or 'hard'")
    
    return errors


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe filesystem usage."""
    # Remove or replace dangerous characters
    import re
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = sanitized.strip('. ')
    return sanitized or "untitled"


def ensure_directory_exists(directory_path: str) -> None:
    """Ensure a directory exists, creating it if necessary."""
    try:
        os.makedirs(directory_path, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        raise


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise


def save_json_file(data: Any, file_path: str, indent: int = 2) -> None:
    """Save data to a JSON file."""
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    except OSError as e:
        logger.error(f"Failed to save JSON file {file_path}: {e}")
        raise


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to a maximum length with optional suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_duration_ms(duration_ms: float) -> str:
    """Format duration in milliseconds to human-readable string."""
    if duration_ms < 1000:
        return f"{duration_ms:.1f}ms"
    elif duration_ms < 60000:
        return f"{duration_ms / 1000:.1f}s"
    else:
        minutes = int(duration_ms // 60000)
        seconds = (duration_ms % 60000) / 1000
        return f"{minutes}m {seconds:.1f}s"


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries, with later ones taking precedence."""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def extract_template_variables(template: str) -> List[str]:
    """Extract variable names from a prompt template using {variable} format."""
    import re
    return re.findall(r'\{(\w+)\}', template)


def substitute_template_variables(template: str, variables: Dict[str, str]) -> str:
    """Substitute variables in a template string."""
    try:
        return template.format(**variables)
    except KeyError as e:
        logger.error(f"Missing template variable: {e}")
        raise ValueError(f"Missing required template variable: {e}")


def get_environment_variable(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with optional default."""
    value = os.getenv(name, default)
    if value is None:
        logger.warning(f"Environment variable {name} not found")
    return value


def setup_basic_logging(level: str = "INFO", format_string: Optional[str] = None) -> None:
    """Set up basic logging configuration."""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[logging.StreamHandler()]
    )


def calculate_weighted_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """Calculate weighted average score from individual scores and weights."""
    if not scores or not weights:
        return 0.0
    
    total_score = 0.0
    total_weight = 0.0
    
    for key, score in scores.items():
        weight = weights.get(key, 0.0)
        if weight > 0:
            total_score += score * weight
            total_weight += weight
    
    return total_score / total_weight if total_weight > 0 else 0.0


# Constants for common configurations
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_RISK_THRESHOLD = 0.5

# File extensions for different formats
SUPPORTED_CONFIG_EXTENSIONS = ['.yaml', '.yml', '.json']
SUPPORTED_EXPORT_FORMATS = ['json', 'csv', 'html', 'pdf']