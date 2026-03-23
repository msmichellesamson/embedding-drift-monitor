"""Embedding storage with circuit breaker protection."""
import asyncpg
import numpy as np
from typing import List, Optional, Dict, Any
import logging
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig

logger = logging.getLogger(__name__)

class EmbeddingStore:
    """PostgreSQL-backed embedding storage with reliability patterns."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool: Optional[asyncpg.Pool] = None
        
        # Circuit breaker for database operations
        self.circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30,
                expected_exception=asyncpg.PostgreSQLError
            )
        )
    
    async def initialize(self):
        """Initialize connection pool and create tables."""
        try:
            self.pool = await self.circuit_breaker.call(
                asyncpg.create_pool,
                self.connection_string,
                min_size=2,
                max_size=10
            )
            await self._create_tables()
            logger.info("EmbeddingStore initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingStore: {e}")
            raise
    
    async def store_embedding(self, model_name: str, embedding: np.ndarray, metadata: Dict[str, Any]):
        """Store embedding with circuit breaker protection."""
        async def _store():
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO embeddings (model_name, embedding, metadata, created_at)
                    VALUES ($1, $2, $3, NOW())
                    """,
                    model_name, embedding.tobytes(), metadata
                )
        
        return await self.circuit_breaker.call(_store)
    
    async def get_recent_embeddings(self, model_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent embeddings with error handling."""
        async def _get():
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT embedding, metadata, created_at 
                    FROM embeddings 
                    WHERE model_name = $1 
                    ORDER BY created_at DESC 
                    LIMIT $2
                    """,
                    model_name, limit
                )
                return [{
                    'embedding': np.frombuffer(row['embedding'], dtype=np.float32),
                    'metadata': row['metadata'],
                    'created_at': row['created_at']
                } for row in rows]
        
        return await self.circuit_breaker.call(_get)
    
    async def _create_tables(self):
        """Create required database tables."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(255) NOT NULL,
                    embedding BYTEA NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    INDEX(model_name, created_at)
                )
                """
            )
