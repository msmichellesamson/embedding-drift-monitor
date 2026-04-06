from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DriftSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DriftScore:
    overall_score: float
    severity: DriftSeverity
    component_scores: Dict[str, float]
    confidence: float
    timestamp: float

class DriftScorer:
    """Combines multiple drift metrics into unified drift score."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'statistical': 0.4,
            'cosine_similarity': 0.3,
            'dimensional': 0.2,
            'anomaly': 0.1
        }
        self._validate_weights()
    
    def _validate_weights(self) -> None:
        """Ensure weights sum to 1.0."""
        total = sum(self.weights.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    def calculate_drift_score(
        self,
        statistical_drift: float,
        similarity_drift: float,
        dimensional_drift: float,
        anomaly_score: float,
        sample_size: int = 100
    ) -> DriftScore:
        """Calculate weighted drift score from component metrics."""
        try:
            # Normalize all inputs to [0, 1]
            components = {
                'statistical': min(statistical_drift, 1.0),
                'cosine_similarity': min(similarity_drift, 1.0),
                'dimensional': min(dimensional_drift, 1.0),
                'anomaly': min(anomaly_score, 1.0)
            }
            
            # Calculate weighted score
            overall = sum(
                components[key] * self.weights[key]
                for key in components.keys()
            )
            
            # Adjust confidence based on sample size
            confidence = min(1.0, sample_size / 1000.0)
            
            severity = self._determine_severity(overall, confidence)
            
            return DriftScore(
                overall_score=overall,
                severity=severity,
                component_scores=components,
                confidence=confidence,
                timestamp=np.datetime64('now').astype(float)
            )
            
        except Exception as e:
            logger.error(f"Error calculating drift score: {e}")
            raise
    
    def _determine_severity(self, score: float, confidence: float) -> DriftSeverity:
        """Determine drift severity based on score and confidence."""
        adjusted_score = score * confidence
        
        if adjusted_score >= 0.8:
            return DriftSeverity.CRITICAL
        elif adjusted_score >= 0.6:
            return DriftSeverity.HIGH
        elif adjusted_score >= 0.3:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW