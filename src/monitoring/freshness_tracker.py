"""Embedding freshness monitoring with TTL tracking."""
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class FreshnessMetric:
    """Embedding freshness metrics."""
    embedding_id: str
    last_updated: datetime
    age_hours: float
    ttl_hours: Optional[float]
    is_stale: bool
    staleness_severity: str  # "low", "medium", "high", "critical"

class EmbeddingFreshnessTracker:
    """Tracks embedding age and staleness across the system."""
    
    def __init__(self, default_ttl_hours: float = 24.0):
        self.default_ttl_hours = default_ttl_hours
        self.embedding_timestamps: Dict[str, datetime] = {}
        self.ttl_overrides: Dict[str, float] = {}
        
    def register_embedding(self, embedding_id: str, ttl_hours: Optional[float] = None) -> None:
        """Register a new embedding with optional TTL override."""
        self.embedding_timestamps[embedding_id] = datetime.utcnow()
        if ttl_hours is not None:
            self.ttl_overrides[embedding_id] = ttl_hours
        logger.debug(f"Registered embedding {embedding_id} with TTL {ttl_hours or self.default_ttl_hours}h")
    
    def update_embedding(self, embedding_id: str) -> None:
        """Update embedding timestamp when it's refreshed."""
        self.embedding_timestamps[embedding_id] = datetime.utcnow()
        logger.debug(f"Updated timestamp for embedding {embedding_id}")
    
    def get_freshness_metric(self, embedding_id: str) -> Optional[FreshnessMetric]:
        """Get freshness metrics for a specific embedding."""
        if embedding_id not in self.embedding_timestamps:
            return None
            
        last_updated = self.embedding_timestamps[embedding_id]
        ttl_hours = self.ttl_overrides.get(embedding_id, self.default_ttl_hours)
        age_hours = (datetime.utcnow() - last_updated).total_seconds() / 3600
        
        is_stale = age_hours > ttl_hours
        severity = self._calculate_staleness_severity(age_hours, ttl_hours)
        
        return FreshnessMetric(
            embedding_id=embedding_id,
            last_updated=last_updated,
            age_hours=age_hours,
            ttl_hours=ttl_hours,
            is_stale=is_stale,
            staleness_severity=severity
        )
    
    def get_stale_embeddings(self, severity_threshold: str = "medium") -> List[FreshnessMetric]:
        """Get all embeddings that exceed staleness threshold."""
        severity_levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        threshold_level = severity_levels.get(severity_threshold, 1)
        
        stale_embeddings = []
        for embedding_id in self.embedding_timestamps:
            metric = self.get_freshness_metric(embedding_id)
            if metric and severity_levels.get(metric.staleness_severity, 0) >= threshold_level:
                stale_embeddings.append(metric)
                
        return sorted(stale_embeddings, key=lambda x: x.age_hours, reverse=True)
    
    def _calculate_staleness_severity(self, age_hours: float, ttl_hours: float) -> str:
        """Calculate staleness severity based on age vs TTL."""
        ratio = age_hours / ttl_hours
        
        if ratio <= 1.0:
            return "low"
        elif ratio <= 2.0:
            return "medium"
        elif ratio <= 4.0:
            return "high"
        else:
            return "critical"