"""
AEGIS Risk-Specific Evaluation Targets

Individual evaluation target modules for each of the 9 AI alignment risk categories.
Each module implements specialized attack vectors and evaluation logic for its
specific risk category.
"""

# Import RiskCategory at module level for RISK_TARGET_MAP
from ...utils.data_structures import RiskCategory

try:
    from .reward_hacking import RewardHackingTarget
    from .deception import DeceptionTarget
    from .hidden_motivations import HiddenMotivationsTarget
    from .sabotage import SabotageTarget
    from .tool_use import InappropriateToolUseTarget
    from .data_exfiltration import DataExfiltrationTarget
    from .sandbagging import SandbaggingTarget
    from .evaluation_awareness import EvaluationAwarenessTarget
    from .chain_of_thought import ChainOfThoughtIssuesTarget
except ImportError:
    # During package setup phase, create placeholder classes
    import logging
    from ...utils.data_structures import RiskCategory
    from ..base import BaseEvaluationTarget
    
    logger = logging.getLogger(__name__)
    
    class RewardHackingTarget(BaseEvaluationTarget):
        """Evaluation target for reward hacking risks."""
        def __init__(self, config=None):
            super().__init__(RiskCategory.REWARD_HACKING, config)
            logger.warning("RewardHackingTarget is not yet implemented. This is a placeholder.")
    
    class DeceptionTarget(BaseEvaluationTarget):
        """Evaluation target for deception risks."""
        def __init__(self, config=None):
            super().__init__(RiskCategory.DECEPTION, config)
            logger.warning("DeceptionTarget is not yet implemented. This is a placeholder.")
    
    class HiddenMotivationsTarget(BaseEvaluationTarget):
        """Evaluation target for hidden motivations/deceptive alignment risks."""
        def __init__(self, config=None):
            super().__init__(RiskCategory.HIDDEN_MOTIVATIONS, config)
            logger.warning("HiddenMotivationsTarget is not yet implemented. This is a placeholder.")
    
    class SabotageTarget(BaseEvaluationTarget):
        """Evaluation target for sabotage risks."""
        def __init__(self, config=None):
            super().__init__(RiskCategory.SABOTAGE, config)
            logger.warning("SabotageTarget is not yet implemented. This is a placeholder.")
    
    class InappropriateToolUseTarget(BaseEvaluationTarget):
        """Evaluation target for inappropriate tool use risks."""
        def __init__(self, config=None):
            super().__init__(RiskCategory.INAPPROPRIATE_TOOL_USE, config)
            logger.warning("InappropriateToolUseTarget is not yet implemented. This is a placeholder.")
    
    class DataExfiltrationTarget(BaseEvaluationTarget):
        """Evaluation target for data exfiltration risks."""
        def __init__(self, config=None):
            super().__init__(RiskCategory.DATA_EXFILTRATION, config)
            logger.warning("DataExfiltrationTarget is not yet implemented. This is a placeholder.")
    
    class SandbaggingTarget(BaseEvaluationTarget):
        """Evaluation target for sandbagging risks."""
        def __init__(self, config=None):
            super().__init__(RiskCategory.SANDBAGGING, config)
            logger.warning("SandbaggingTarget is not yet implemented. This is a placeholder.")
    
    class EvaluationAwarenessTarget(BaseEvaluationTarget):
        """Evaluation target for evaluation awareness risks."""
        def __init__(self, config=None):
            super().__init__(RiskCategory.EVALUATION_AWARENESS, config)
            logger.warning("EvaluationAwarenessTarget is not yet implemented. This is a placeholder.")
    
    class ChainOfThoughtIssuesTarget(BaseEvaluationTarget):
        """Evaluation target for chain-of-thought issues."""
        def __init__(self, config=None):
            super().__init__(RiskCategory.CHAIN_OF_THOUGHT_ISSUES, config)
            logger.warning("ChainOfThoughtIssuesTarget is not yet implemented. This is a placeholder.")

# Map of risk categories to their corresponding target classes
RISK_TARGET_MAP = {
    RiskCategory.REWARD_HACKING: RewardHackingTarget,
    RiskCategory.DECEPTION: DeceptionTarget,
    RiskCategory.HIDDEN_MOTIVATIONS: HiddenMotivationsTarget,
    RiskCategory.SABOTAGE: SabotageTarget,
    RiskCategory.INAPPROPRIATE_TOOL_USE: InappropriateToolUseTarget,
    RiskCategory.DATA_EXFILTRATION: DataExfiltrationTarget,
    RiskCategory.SANDBAGGING: SandbaggingTarget,
    RiskCategory.EVALUATION_AWARENESS: EvaluationAwarenessTarget,
    RiskCategory.CHAIN_OF_THOUGHT_ISSUES: ChainOfThoughtIssuesTarget,
}

def get_target_for_category(risk_category: RiskCategory, config=None):
    """Get the appropriate evaluation target for a risk category."""
    target_class = RISK_TARGET_MAP.get(risk_category)
    if target_class:
        return target_class(config)
    else:
        raise ValueError(f"No evaluation target found for risk category: {risk_category}")

def get_all_targets(config=None):
    """Get instances of all evaluation targets."""
    return [target_class(config) for target_class in RISK_TARGET_MAP.values()]

__all__ = [
    "RewardHackingTarget",
    "DeceptionTarget", 
    "HiddenMotivationsTarget",
    "SabotageTarget",
    "InappropriateToolUseTarget",
    "DataExfiltrationTarget",
    "SandbaggingTarget",
    "EvaluationAwarenessTarget",
    "ChainOfThoughtIssuesTarget",
    "RISK_TARGET_MAP",
    "get_target_for_category",
    "get_all_targets",
]