"""Embedding quality degradation detection.

Detects when embedding quality degrades based on internal consistency
and distributional properties.
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Quality metrics for embeddings."""
    avg_magnitude: float
    std_magnitude: float
    cosine_coherence: float  # How similar embeddings are to expected patterns
    dimension_variance: float  # Variance across dimensions
    timestamp: datetime

class QualityDegradationDetector:
    """Detects embedding quality degradation over time."""
    
    def __init__(
        self,
        magnitude_threshold: float = 0.3,
        coherence_threshold: float = 0.2,
        variance_threshold: float = 0.4,
        history_window: int = 100
    ):
        self.magnitude_threshold = magnitude_threshold
        self.coherence_threshold = coherence_threshold 
        self.variance_threshold = variance_threshold
        self.history_window = history_window
        self.baseline_metrics: Optional[QualityMetrics] = None
        self.metrics_history: List[QualityMetrics] = []
        
    def compute_quality_metrics(self, embeddings: np.ndarray) -> QualityMetrics:
        """Compute quality metrics for a batch of embeddings."""
        if embeddings.size == 0:
            raise ValueError("Empty embeddings array")
            
        magnitudes = np.linalg.norm(embeddings, axis=1)
        avg_magnitude = float(np.mean(magnitudes))
        std_magnitude = float(np.std(magnitudes))
        
        # Compute pairwise cosine similarity for coherence
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarities = np.dot(normalized, normalized.T)
        # Use upper triangle excluding diagonal
        upper_tri = np.triu(similarities, k=1)
        coherence = float(np.mean(upper_tri[upper_tri != 0]))
        
        # Dimension-wise variance
        dim_variance = float(np.mean(np.var(embeddings, axis=0)))
        
        return QualityMetrics(
            avg_magnitude=avg_magnitude,
            std_magnitude=std_magnitude,
            cosine_coherence=coherence,
            dimension_variance=dim_variance,
            timestamp=datetime.utcnow()
        )
    
    def set_baseline(self, embeddings: np.ndarray) -> None:
        """Set baseline quality metrics."""
        self.baseline_metrics = self.compute_quality_metrics(embeddings)
        logger.info(f"Baseline quality metrics set: magnitude={self.baseline_metrics.avg_magnitude:.3f}")
    
    def detect_degradation(self, embeddings: np.ndarray) -> Dict[str, any]:
        """Detect quality degradation in current embeddings."""
        if self.baseline_metrics is None:
            raise ValueError("No baseline metrics set")
            
        current_metrics = self.compute_quality_metrics(embeddings)
        self.metrics_history.append(current_metrics)
        
        # Keep history window
        if len(self.metrics_history) > self.history_window:
            self.metrics_history = self.metrics_history[-self.history_window:]
        
        # Calculate degradation scores
        magnitude_change = abs(current_metrics.avg_magnitude - self.baseline_metrics.avg_magnitude) / self.baseline_metrics.avg_magnitude
        coherence_change = abs(current_metrics.cosine_coherence - self.baseline_metrics.cosine_coherence)
        variance_change = abs(current_metrics.dimension_variance - self.baseline_metrics.dimension_variance) / self.baseline_metrics.dimension_variance
        
        # Determine if degraded
        is_degraded = (
            magnitude_change > self.magnitude_threshold or
            coherence_change > self.coherence_threshold or
            variance_change > self.variance_threshold
        )
        
        return {
            "is_degraded": is_degraded,
            "degradation_score": max(magnitude_change, coherence_change, variance_change),
            "metrics": {
                "magnitude_change": magnitude_change,
                "coherence_change": coherence_change, 
                "variance_change": variance_change
            },
            "current_quality": current_metrics,
            "baseline_quality": self.baseline_metrics
        }
