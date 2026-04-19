"""Model performance tracking and accuracy monitoring."""

import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np
from prometheus_client import Histogram, Counter, Gauge


@dataclass
class PredictionResult:
    """Represents a single prediction with ground truth."""
    embedding_id: str
    prediction: float
    ground_truth: Optional[float]
    timestamp: float
    model_version: str


class ModelPerformanceTracker:
    """Tracks model performance metrics and accuracy over time."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions: deque = deque(maxlen=window_size)
        self.logger = logging.getLogger(__name__)
        
        # Prometheus metrics
        self.accuracy_gauge = Gauge(
            'embedding_model_accuracy',
            'Current model accuracy',
            ['model_version']
        )
        self.prediction_counter = Counter(
            'embedding_predictions_total',
            'Total predictions made',
            ['model_version', 'status']
        )
        self.mae_gauge = Gauge(
            'embedding_model_mae',
            'Mean absolute error',
            ['model_version']
        )
        
    def record_prediction(
        self,
        embedding_id: str,
        prediction: float,
        ground_truth: Optional[float] = None,
        model_version: str = "default"
    ) -> None:
        """Record a prediction result."""
        result = PredictionResult(
            embedding_id=embedding_id,
            prediction=prediction,
            ground_truth=ground_truth,
            timestamp=time.time(),
            model_version=model_version
        )
        
        self.predictions.append(result)
        
        # Update counters
        status = "with_truth" if ground_truth is not None else "prediction_only"
        self.prediction_counter.labels(
            model_version=model_version,
            status=status
        ).inc()
        
        # Update metrics if we have ground truth
        if ground_truth is not None:
            self._update_metrics(model_version)
            
    def _update_metrics(self, model_version: str) -> None:
        """Update performance metrics for a specific model version."""
        # Get predictions with ground truth for this model
        predictions_with_truth = [
            p for p in self.predictions
            if p.ground_truth is not None and p.model_version == model_version
        ]
        
        if len(predictions_with_truth) < 10:  # Need minimum samples
            return
            
        # Calculate metrics
        accuracy = self._calculate_accuracy(predictions_with_truth)
        mae = self._calculate_mae(predictions_with_truth)
        
        # Update Prometheus metrics
        self.accuracy_gauge.labels(model_version=model_version).set(accuracy)
        self.mae_gauge.labels(model_version=model_version).set(mae)
        
        self.logger.info(
            f"Updated metrics for {model_version}: accuracy={accuracy:.3f}, mae={mae:.3f}"
        )
    
    def _calculate_accuracy(self, predictions: List[PredictionResult]) -> float:
        """Calculate accuracy (for binary classification tasks)."""
        if not predictions:
            return 0.0
            
        correct = 0
        for pred in predictions:
            # Assume binary classification with 0.5 threshold
            predicted_class = 1 if pred.prediction > 0.5 else 0
            actual_class = 1 if pred.ground_truth > 0.5 else 0
            if predicted_class == actual_class:
                correct += 1
                
        return correct / len(predictions)
    
    def _calculate_mae(self, predictions: List[PredictionResult]) -> float:
        """Calculate mean absolute error."""
        if not predictions:
            return 0.0
            
        errors = [
            abs(p.prediction - p.ground_truth)
            for p in predictions
        ]
        return np.mean(errors)
    
    def get_performance_summary(self, model_version: str = "default") -> Dict:
        """Get current performance summary for a model."""
        version_predictions = [
            p for p in self.predictions
            if p.model_version == model_version
        ]
        
        with_truth = [p for p in version_predictions if p.ground_truth is not None]
        
        if not with_truth:
            return {
                "model_version": model_version,
                "total_predictions": len(version_predictions),
                "predictions_with_truth": 0,
                "accuracy": None,
                "mae": None
            }
            
        return {
            "model_version": model_version,
            "total_predictions": len(version_predictions),
            "predictions_with_truth": len(with_truth),
            "accuracy": self._calculate_accuracy(with_truth),
            "mae": self._calculate_mae(with_truth),
            "latest_prediction": version_predictions[-1].timestamp if version_predictions else None
        }
