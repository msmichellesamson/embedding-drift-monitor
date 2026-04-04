from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class DimensionViolation:
    """Represents a dimension consistency violation."""
    model_id: str
    expected_dim: int
    actual_dim: int
    timestamp: datetime
    severity: str

class DimensionValidator:
    """Validates embedding dimensions for consistency across models."""
    
    def __init__(self, tolerance_window: int = 300):
        self.tolerance_window = tolerance_window  # seconds
        self._model_dimensions: Dict[str, int] = {}
        self._violations: List[DimensionViolation] = []
    
    def register_model_dimension(self, model_id: str, dimension: int) -> None:
        """Register expected dimension for a model."""
        if model_id in self._model_dimensions:
            if self._model_dimensions[model_id] != dimension:
                logger.warning(f"Dimension changed for {model_id}: {self._model_dimensions[model_id]} -> {dimension}")
        
        self._model_dimensions[model_id] = dimension
        logger.info(f"Registered model {model_id} with dimension {dimension}")
    
    def validate_embedding(self, model_id: str, embedding: List[float]) -> Optional[DimensionViolation]:
        """Validate embedding dimension against registered model."""
        if model_id not in self._model_dimensions:
            logger.warning(f"Model {model_id} not registered, skipping validation")
            return None
        
        expected_dim = self._model_dimensions[model_id]
        actual_dim = len(embedding)
        
        if expected_dim != actual_dim:
            violation = DimensionViolation(
                model_id=model_id,
                expected_dim=expected_dim,
                actual_dim=actual_dim,
                timestamp=datetime.utcnow(),
                severity="high" if abs(expected_dim - actual_dim) > 10 else "medium"
            )
            
            self._violations.append(violation)
            logger.error(f"Dimension violation: {model_id} expected {expected_dim}, got {actual_dim}")
            return violation
        
        return None
    
    def get_recent_violations(self, minutes: int = 60) -> List[DimensionViolation]:
        """Get violations from the last N minutes."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [v for v in self._violations if v.timestamp > cutoff]
    
    def get_violation_summary(self) -> Dict[str, int]:
        """Get summary of violations by model."""
        summary = {}
        for violation in self._violations:
            summary[violation.model_id] = summary.get(violation.model_id, 0) + 1
        return summary
