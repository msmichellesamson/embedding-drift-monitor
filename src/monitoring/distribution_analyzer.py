"""Embedding distribution analysis for drift detection."""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class DistributionStats:
    """Statistics for embedding distribution."""
    mean: np.ndarray
    std: np.ndarray
    skewness: float
    kurtosis: float
    percentiles: Dict[int, np.ndarray]
    timestamp: datetime

class DistributionAnalyzer:
    """Analyzes embedding distributions for drift detection."""
    
    def __init__(self, percentiles: List[int] = None):
        self.percentiles = percentiles or [25, 50, 75, 90, 95]
        self.baseline_stats: Optional[DistributionStats] = None
    
    def compute_stats(self, embeddings: np.ndarray) -> DistributionStats:
        """Compute distribution statistics for embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, n_dimensions)
            
        Returns:
            Distribution statistics
        """
        if embeddings.size == 0:
            raise ValueError("Cannot compute stats for empty embeddings")
        
        try:
            mean = np.mean(embeddings, axis=0)
            std = np.std(embeddings, axis=0)
            
            # Flatten for skewness/kurtosis (scalar metrics)
            flat_embeddings = embeddings.flatten()
            skewness = float(stats.skew(flat_embeddings))
            kurtosis = float(stats.kurtosis(flat_embeddings))
            
            # Compute percentiles per dimension
            percentile_dict = {}
            for p in self.percentiles:
                percentile_dict[p] = np.percentile(embeddings, p, axis=0)
            
            return DistributionStats(
                mean=mean,
                std=std,
                skewness=skewness,
                kurtosis=kurtosis,
                percentiles=percentile_dict,
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Failed to compute distribution stats: {e}")
            raise
    
    def set_baseline(self, embeddings: np.ndarray) -> None:
        """Set baseline distribution for comparison."""
        self.baseline_stats = self.compute_stats(embeddings)
        logger.info(f"Baseline set with {embeddings.shape[0]} samples")
    
    def detect_drift(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Detect distribution drift compared to baseline.
        
        Returns:
            Dictionary with drift metrics
        """
        if self.baseline_stats is None:
            raise ValueError("Baseline not set. Call set_baseline() first")
        
        current_stats = self.compute_stats(embeddings)
        
        # Mean shift (Euclidean distance)
        mean_shift = float(np.linalg.norm(
            current_stats.mean - self.baseline_stats.mean
        ))
        
        # Standard deviation change
        std_ratio = float(np.mean(
            current_stats.std / (self.baseline_stats.std + 1e-8)
        ))
        
        # Shape changes
        skewness_diff = abs(current_stats.skewness - self.baseline_stats.skewness)
        kurtosis_diff = abs(current_stats.kurtosis - self.baseline_stats.kurtosis)
        
        return {
            'mean_shift': mean_shift,
            'std_ratio': std_ratio,
            'skewness_drift': skewness_diff,
            'kurtosis_drift': kurtosis_diff,
            'overall_drift': mean_shift + abs(std_ratio - 1.0) + skewness_diff * 0.1
        }