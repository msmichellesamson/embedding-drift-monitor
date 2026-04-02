"""Vector health monitoring for embedding quality assessment."""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class VectorHealthMetrics:
    """Health metrics for embedding vectors."""
    timestamp: datetime
    vector_dimension: int
    magnitude_mean: float
    magnitude_std: float
    sparsity_ratio: float  # Percentage of near-zero values
    norm_distribution_skew: float
    health_score: float  # 0-1 score, 1 = healthy


class VectorHealthChecker:
    """Monitors embedding vector quality and detects degradation."""
    
    def __init__(self, 
                 dimension_threshold: float = 0.1,
                 sparsity_threshold: float = 0.8,
                 magnitude_threshold: float = 0.01):
        self.dimension_threshold = dimension_threshold
        self.sparsity_threshold = sparsity_threshold  
        self.magnitude_threshold = magnitude_threshold
        self.baseline_metrics: Optional[VectorHealthMetrics] = None
        
    def analyze_vectors(self, vectors: np.ndarray) -> VectorHealthMetrics:
        """Analyze health of embedding vectors.
        
        Args:
            vectors: Array of shape (n_vectors, dimension)
            
        Returns:
            VectorHealthMetrics with quality assessment
        """
        if vectors.size == 0:
            raise ValueError("Cannot analyze empty vector array")
            
        # Calculate vector magnitudes
        magnitudes = np.linalg.norm(vectors, axis=1)
        
        # Sparsity: percentage of values close to zero
        near_zero = np.abs(vectors) < 1e-6
        sparsity_ratio = np.mean(near_zero)
        
        # Distribution skew of vector norms
        from scipy.stats import skew
        norm_skew = skew(magnitudes)
        
        metrics = VectorHealthMetrics(
            timestamp=datetime.utcnow(),
            vector_dimension=vectors.shape[1],
            magnitude_mean=float(np.mean(magnitudes)),
            magnitude_std=float(np.std(magnitudes)),
            sparsity_ratio=float(sparsity_ratio),
            norm_distribution_skew=float(norm_skew),
            health_score=self._calculate_health_score(
                magnitudes, sparsity_ratio, norm_skew
            )
        )
        
        logger.info(f"Vector health analysis: {metrics.health_score:.3f} score")
        return metrics
        
    def _calculate_health_score(self, 
                              magnitudes: np.ndarray,
                              sparsity_ratio: float, 
                              norm_skew: float) -> float:
        """Calculate overall health score (0-1, higher is better)."""
        score = 1.0
        
        # Penalize high sparsity
        if sparsity_ratio > self.sparsity_threshold:
            score *= (1 - sparsity_ratio)
            
        # Penalize very low or very high magnitudes
        mean_mag = np.mean(magnitudes)
        if mean_mag < self.magnitude_threshold or mean_mag > 10:
            score *= 0.5
            
        # Penalize extreme skew in norm distribution
        if abs(norm_skew) > 2.0:
            score *= 0.7
            
        return max(0.0, min(1.0, score))
        
    def set_baseline(self, vectors: np.ndarray) -> None:
        """Set baseline metrics for drift detection."""
        self.baseline_metrics = self.analyze_vectors(vectors)
        logger.info("Baseline vector health metrics established")
        
    def detect_degradation(self, vectors: np.ndarray) -> Dict[str, float]:
        """Detect vector quality degradation vs baseline.
        
        Returns:
            Dict with degradation scores for different metrics
        """
        if self.baseline_metrics is None:
            raise ValueError("No baseline set - call set_baseline() first")
            
        current = self.analyze_vectors(vectors)
        
        degradation = {
            'health_score_drop': self.baseline_metrics.health_score - current.health_score,
            'sparsity_increase': current.sparsity_ratio - self.baseline_metrics.sparsity_ratio,
            'magnitude_drift': abs(current.magnitude_mean - self.baseline_metrics.magnitude_mean),
            'std_change': abs(current.magnitude_std - self.baseline_metrics.magnitude_std)
        }
        
        return degradation