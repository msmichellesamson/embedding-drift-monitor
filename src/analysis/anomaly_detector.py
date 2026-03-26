"""Anomaly detection for embedding distributions using isolation forest."""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import IsolationForest
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    is_anomaly: bool
    anomaly_score: float
    timestamp: datetime
    confidence: float
    feature_contributions: Dict[str, float]

class EmbeddingAnomalyDetector:
    """Detects anomalies in embedding distributions using isolation forest."""
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        """Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers in the data
            n_estimators: Number of base estimators in the ensemble
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model: Optional[IsolationForest] = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        
    def fit(self, embeddings: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """Fit the anomaly detector on baseline embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            feature_names: Optional names for embedding dimensions
        """
        try:
            if embeddings.size == 0:
                raise ValueError("Cannot fit on empty embeddings")
                
            self.model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(embeddings)
            self.is_fitted = True
            
            # Store feature names for interpretability
            if feature_names:
                self.feature_names = feature_names
            else:
                self.feature_names = [f"dim_{i}" for i in range(embeddings.shape[1])]
                
            logger.info(f"Fitted anomaly detector on {embeddings.shape[0]} samples")
            
        except Exception as e:
            logger.error(f"Failed to fit anomaly detector: {e}")
            raise
            
    def detect_anomalies(self, embeddings: np.ndarray) -> List[AnomalyResult]:
        """Detect anomalies in new embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            List of anomaly detection results
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Detector must be fitted before detecting anomalies")
            
        try:
            # Predict anomalies (-1 for outlier, 1 for inlier)
            predictions = self.model.predict(embeddings)
            # Get anomaly scores (lower means more anomalous)
            scores = self.model.decision_function(embeddings)
            
            results = []
            timestamp = datetime.now(timezone.utc)
            
            for i, (pred, score) in enumerate(zip(predictions, scores)):
                is_anomaly = pred == -1
                # Convert score to 0-1 confidence (higher = more confident it's normal)
                confidence = min(max((score + 0.5) / 1.0, 0.0), 1.0)
                
                # Calculate feature contributions (simplified)
                embedding = embeddings[i]
                feature_contributions = self._calculate_feature_contributions(embedding)
                
                results.append(AnomalyResult(
                    is_anomaly=is_anomaly,
                    anomaly_score=float(score),
                    timestamp=timestamp,
                    confidence=confidence,
                    feature_contributions=feature_contributions
                ))
                
            logger.debug(f"Detected {sum(1 for r in results if r.is_anomaly)} anomalies in {len(results)} samples")
            return results
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            raise
            
    def _calculate_feature_contributions(self, embedding: np.ndarray) -> Dict[str, float]:
        """Calculate which features contribute most to anomaly score.
        
        Args:
            embedding: Single embedding vector
            
        Returns:
            Dictionary mapping feature names to contribution scores
        """
        # Simplified feature importance based on deviation from mean
        contributions = {}
        
        for i, (feature_name, value) in enumerate(zip(self.feature_names, embedding)):
            # Simple heuristic: larger absolute values contribute more
            contribution = abs(float(value)) / (np.linalg.norm(embedding) + 1e-8)
            contributions[feature_name] = contribution
            
        return contributions
