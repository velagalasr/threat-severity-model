"""
API Utility Functions
"""

import numpy as np
from typing import Tuple
from src.config import RISK_LEVEL_THRESHOLDS, RECOMMENDED_ACTIONS


def get_risk_level(threat_score: float) -> str:
    """
    Determine risk level from threat score.
    
    Args:
        threat_score: Threat severity score (0-10)
        
    Returns:
        Risk level string (low/medium/high/critical)
    """
    for level, (min_score, max_score) in RISK_LEVEL_THRESHOLDS.items():
        if min_score <= threat_score < max_score:
            return level
    
    # Default to critical if >= 10
    return 'critical'


def get_recommended_action(risk_level: str) -> str:
    """
    Get recommended action for risk level.
    
    Args:
        risk_level: Risk level (low/medium/high/critical)
        
    Returns:
        Recommended action string
    """
    return RECOMMENDED_ACTIONS.get(risk_level, "Review and escalate as needed")


def validate_features(features: list) -> Tuple[bool, str]:
    """
    Validate feature input.
    
    Args:
        features: List of feature values
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(features, list):
        return False, "Features must be a list"
    
    if len(features) != 41:
        return False, f"Expected 41 features, got {len(features)}"
    
    try:
        features_array = np.array(features, dtype=float)
        
        if np.any(np.isnan(features_array)):
            return False, "Features contain NaN values"
        
        if np.any(np.isinf(features_array)):
            return False, "Features contain infinite values"
        
    except (ValueError, TypeError) as e:
        return False, f"Invalid feature values: {str(e)}"
    
    return True, ""
