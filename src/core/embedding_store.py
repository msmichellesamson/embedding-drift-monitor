import logging
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
from .redis_cache import EmbeddingCache, CacheConfig

logger = logging.getLogger(__name__)

class EmbeddingStore:
    """PostgreSQL-backed embedding storage with Redis caching."""
    
    def __init__(self, db_config: dict, cache_config: Optional[CacheConfig] = None):
        self.db_config = db_config
        self.cache = EmbeddingCache(cache_config) if cache_config else None
        self._connection = None
    
    def _get_connection(self):
        """Get database connection with retry logic."""
        if not self._connection or self._connection.closed:
            try:
                self._connection = psycopg2.connect(**self.db_config)
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                raise
        return self._connection
    
    def store_embedding(self, model_id: str, embedding: np.ndarray, 
                       timestamp: Optional[datetime] = None) -> str:
        """Store embedding vector with caching."""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        embedding_id = f"{model_id}:{timestamp.isoformat()}"
        
        # Store in database
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO embeddings (id, model_id, vector, created_at) VALUES (%s, %s, %s, %s)",
                (embedding_id, model_id, embedding.tolist(), timestamp)
            )
        conn.commit()
        
        # Cache the embedding
        if self.cache and self.cache.health_check():
            self.cache.set_embedding(embedding_id, embedding)
            
        logger.info(f"Stored embedding {embedding_id}")
        return embedding_id
    
    def get_embeddings_window(self, model_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get embeddings from time window with cache fallback."""
        cache_key = f"{model_id}:{hours}h"
        
        # Try cache first
        if self.cache and self.cache.health_check():
            cached = self.cache.get_drift_metrics(cache_key)
            if cached:
                return cached.get('embeddings', [])
        
        # Fallback to database
        since = datetime.utcnow() - timedelta(hours=hours)
        conn = self._get_connection()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, vector, created_at FROM embeddings WHERE model_id = %s AND created_at >= %s ORDER BY created_at DESC",
                (model_id, since)
            )
            results = [dict(row) for row in cur.fetchall()]
            
        # Convert lists back to numpy arrays
        for result in results:
            result['vector'] = np.array(result['vector'])
            
        return results