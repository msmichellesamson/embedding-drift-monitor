from typing import Optional, List, Dict, Any
import redis
import pickle
import logging
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ttl_seconds: int = 3600  # 1 hour default
    max_connections: int = 10

class EmbeddingCache:
    """Redis-based cache for embedding vectors and drift metrics."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis = redis.ConnectionPool(
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            max_connections=config.max_connections,
            decode_responses=False
        )
        self.client = redis.Redis(connection_pool=self.redis)
        
    def get_embedding(self, key: str) -> Optional[np.ndarray]:
        """Retrieve embedding vector from cache."""
        try:
            data = self.client.get(f"embedding:{key}")
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.warning(f"Cache read failed for {key}: {e}")
            return None
    
    def set_embedding(self, key: str, embedding: np.ndarray) -> bool:
        """Store embedding vector in cache with TTL."""
        try:
            data = pickle.dumps(embedding)
            return self.client.setex(
                f"embedding:{key}", 
                self.config.ttl_seconds, 
                data
            )
        except Exception as e:
            logger.error(f"Cache write failed for {key}: {e}")
            return False
    
    def get_drift_metrics(self, window_key: str) -> Optional[Dict[str, float]]:
        """Get cached drift metrics for a time window."""
        try:
            data = self.client.get(f"drift:{window_key}")
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.warning(f"Drift metrics cache read failed: {e}")
            return None
    
    def set_drift_metrics(self, window_key: str, metrics: Dict[str, float]) -> bool:
        """Cache drift metrics with shorter TTL."""
        try:
            data = pickle.dumps(metrics)
            return self.client.setex(
                f"drift:{window_key}",
                min(300, self.config.ttl_seconds),  # 5min max for metrics
                data
            )
        except Exception as e:
            logger.error(f"Drift metrics cache write failed: {e}")
            return False
    
    def health_check(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            return self.client.ping()
        except Exception:
            return False