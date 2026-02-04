from typing import Dict, List, Optional, Tuple
import time
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3
    
    def __post_init__(self):
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED
        self.half_open_calls = 0

    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        else:  # HALF_OPEN
            return self.half_open_calls < self.half_open_max_calls

    def record_success(self):
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
        self.half_open_calls = 0

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1

class SimilarityTracker:
    def __init__(self, embedding_store, drift_detector):
        self.embedding_store = embedding_store
        self.drift_detector = drift_detector
        self.circuit_breaker = CircuitBreaker()
        
    def track_similarity(self, embeddings: List[List[float]]) -> Optional[Dict]:
        """Track embedding similarities with circuit breaker protection."""
        if not self.circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker is {self.circuit_breaker.state.value}, skipping similarity tracking")
            return None
            
        try:
            similarities = self._compute_similarities(embeddings)
            drift_score = self.drift_detector.detect_drift(similarities)
            
            result = {
                "similarities": similarities,
                "drift_score": drift_score,
                "timestamp": time.time(),
                "circuit_state": self.circuit_breaker.state.value
            }
            
            self.circuit_breaker.record_success()
            return result
            
        except Exception as e:
            logger.error(f"Similarity tracking failed: {e}")
            self.circuit_breaker.record_failure()
            return None
    
    def _compute_similarities(self, embeddings: List[List[float]]) -> List[float]:
        """Compute pairwise cosine similarities."""
        import numpy as np
        
        if len(embeddings) < 2:
            return []
            
        embeddings_array = np.array(embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        normalized = embeddings_array / (norms + 1e-8)
        
        similarities = []
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                sim = np.dot(normalized[i], normalized[j])
                similarities.append(float(sim))
                
        return similarities
    
    def get_health_status(self) -> Dict:
        """Get circuit breaker health status."""
        return {
            "circuit_state": self.circuit_breaker.state.value,
            "failure_count": self.circuit_breaker.failure_count,
            "last_failure_time": self.circuit_breaker.last_failure_time,
            "healthy": self.circuit_breaker.state == CircuitState.CLOSED
        }