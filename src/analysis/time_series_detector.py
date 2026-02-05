from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class TimeSeriesAnomaly:
    timestamp: datetime
    value: float
    expected_range: Tuple[float, float]
    severity: str  # 'low', 'medium', 'high'
    metric_name: str

class TimeSeriesAnomalyDetector:
    """Detects anomalies in time-series embedding metrics using moving averages and std dev."""
    
    def __init__(self, window_size: int = 50, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity  # std deviations for anomaly threshold
        self._history: deque = deque(maxlen=window_size)
        
    def add_datapoint(self, timestamp: datetime, value: float) -> Optional[TimeSeriesAnomaly]:
        """Add new datapoint and check for anomalies."""
        self._history.append((timestamp, value))
        
        if len(self._history) < self.window_size:
            return None
            
        return self._detect_anomaly(timestamp, value)
    
    def _detect_anomaly(self, timestamp: datetime, current_value: float) -> Optional[TimeSeriesAnomaly]:
        """Detect if current value is anomalous based on historical data."""
        # Get historical values (excluding current)
        historical_values = [v for _, v in list(self._history)[:-1]]
        
        if len(historical_values) == 0:
            return None
            
        mean_val = np.mean(historical_values)
        std_val = np.std(historical_values)
        
        if std_val == 0:  # No variance in data
            return None
            
        # Calculate z-score
        z_score = abs((current_value - mean_val) / std_val)
        
        if z_score <= self.sensitivity:
            return None
            
        # Determine severity based on z-score
        if z_score > 4.0:
            severity = 'high'
        elif z_score > 3.0:
            severity = 'medium'
        else:
            severity = 'low'
            
        expected_range = (
            mean_val - (self.sensitivity * std_val),
            mean_val + (self.sensitivity * std_val)
        )
        
        logger.warning(
            f"Time series anomaly detected: value={current_value:.4f}, "
            f"expected_range=({expected_range[0]:.4f}, {expected_range[1]:.4f}), "
            f"z_score={z_score:.2f}, severity={severity}"
        )
        
        return TimeSeriesAnomaly(
            timestamp=timestamp,
            value=current_value,
            expected_range=expected_range,
            severity=severity,
            metric_name="embedding_similarity"
        )
    
    def get_recent_anomalies(self, hours: int = 24) -> List[TimeSeriesAnomaly]:
        """Get anomalies from the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            anomaly for anomaly in self._recent_anomalies 
            if anomaly.timestamp >= cutoff
        ]
