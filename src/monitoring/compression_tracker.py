"""Tracks embedding compression ratios to detect model output changes."""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class CompressionMetrics:
    """Compression analysis results."""
    explained_variance_ratio: float
    compression_ratio: float
    information_loss: float
    effective_dimensions: int
    variance_concentration: float


class CompressionTracker:
    """Monitors embedding compression characteristics for drift detection."""
    
    def __init__(self, variance_threshold: float = 0.95, window_size: int = 1000):
        self.variance_threshold = variance_threshold
        self.window_size = window_size
        self.baseline_metrics: Optional[CompressionMetrics] = None
        self.scaler = StandardScaler()
        
    def analyze_compression(self, embeddings: np.ndarray) -> CompressionMetrics:
        """Analyze compression characteristics of embeddings."""
        if len(embeddings) < 10:
            raise ValueError("Need at least 10 embeddings for analysis")
            
        # Standardize embeddings
        scaled_embeddings = self.scaler.fit_transform(embeddings)
        
        # Fit PCA
        pca = PCA()
        pca.fit(scaled_embeddings)
        
        # Calculate metrics
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        effective_dims = np.argmax(cumsum_variance >= self.variance_threshold) + 1
        
        compression_ratio = effective_dims / embeddings.shape[1]
        information_loss = 1.0 - cumsum_variance[effective_dims - 1]
        variance_concentration = pca.explained_variance_ratio_[0]
        
        return CompressionMetrics(
            explained_variance_ratio=cumsum_variance[effective_dims - 1],
            compression_ratio=compression_ratio,
            information_loss=information_loss,
            effective_dimensions=effective_dims,
            variance_concentration=variance_concentration
        )
    
    def set_baseline(self, embeddings: np.ndarray) -> None:
        """Set baseline compression metrics."""
        self.baseline_metrics = self.analyze_compression(embeddings)
        logger.info(f"Baseline set: {self.baseline_metrics.effective_dimensions} effective dims")
    
    def detect_compression_drift(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Detect drift in compression characteristics."""
        if self.baseline_metrics is None:
            raise ValueError("Baseline not set. Call set_baseline() first")
            
        current_metrics = self.analyze_compression(embeddings)
        
        # Calculate drift scores
        dim_drift = abs(current_metrics.effective_dimensions - self.baseline_metrics.effective_dimensions)
        compression_drift = abs(current_metrics.compression_ratio - self.baseline_metrics.compression_ratio)
        concentration_drift = abs(current_metrics.variance_concentration - self.baseline_metrics.variance_concentration)
        
        return {
            "dimension_drift": dim_drift,
            "compression_drift": compression_drift,
            "concentration_drift": concentration_drift,
            "overall_drift": (dim_drift + compression_drift + concentration_drift) / 3
        }
