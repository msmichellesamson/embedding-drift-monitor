import asyncio
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import asyncpg
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class ConnectionConfig:
    host: str
    port: int = 5432
    database: str = "embeddings"
    user: str = "postgres"
    password: str = ""
    min_size: int = 5
    max_size: int = 20
    command_timeout: int = 30


class EmbeddingStore:
    """Production-ready embedding store with connection pooling."""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=self.config.min_size,
                max_size=self.config.max_size,
                command_timeout=self.config.command_timeout,
            )
            self.logger.info(f"Connection pool initialized: {self.config.min_size}-{self.config.max_size}")
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def store_embedding(self, model_id: str, embedding: np.ndarray, 
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store embedding with retry logic."""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                embedding_id = await conn.fetchval(
                    """
                    INSERT INTO embeddings (model_id, vector, metadata, created_at)
                    VALUES ($1, $2, $3, NOW())
                    RETURNING id
                    """,
                    model_id,
                    embedding.tobytes(),
                    metadata or {}
                )
                self.logger.debug(f"Stored embedding {embedding_id} for model {model_id}")
                return str(embedding_id)
        except Exception as e:
            self.logger.error(f"Failed to store embedding: {e}")
            raise
    
    async def close(self) -> None:
        """Close connection pool gracefully."""
        if self.pool:
            await self.pool.close()
            self.logger.info("Connection pool closed")
