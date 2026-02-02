import asyncio
import json
import struct
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import asyncpg
import redis.asyncio as redis
import numpy as np
import structlog
from pydantic import BaseModel, Field

from ..config import settings
from ..exceptions import EmbeddingStoreError, EmbeddingNotFoundError

logger = structlog.get_logger(__name__)

class EmbeddingRecord(BaseModel):
    """Embedding record with metadata"""
    id: str
    embedding: List[float]
    model_name: str
    model_version: str
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    drift_score: Optional[float] = None

class EmbeddingBatch(BaseModel):
    """Batch of embeddings for efficient processing"""
    records: List[EmbeddingRecord]
    batch_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class EmbeddingStore:
    """Production-grade hybrid embedding storage using PostgreSQL + Redis"""
    
    def __init__(
        self,
        postgres_dsn: str,
        redis_url: str,
        cache_ttl: int = 3600,
        batch_size: int = 1000
    ) -> None:
        self.postgres_dsn = postgres_dsn
        self.redis_url = redis_url
        self.cache_ttl = cache_ttl
        self.batch_size = batch_size
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize database connections and schema"""
        if self._initialized:
            return
            
        try:
            # Initialize PostgreSQL connection pool
            self.pg_pool = await asyncpg.create_pool(
                self.postgres_dsn,
                min_size=5,
                max_size=20,
                command_timeout=30,
                server_settings={
                    'application_name': 'embedding_drift_monitor',
                    'timezone': 'UTC'
                }
            )
            
            # Initialize Redis connection
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We handle binary data
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connections
            async with self.pg_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                
            await self.redis_client.ping()
            
            # Setup database schema
            await self._setup_schema()
            
            self._initialized = True
            logger.info("EmbeddingStore initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize EmbeddingStore", error=str(e))
            raise EmbeddingStoreError(f"Initialization failed: {e}")
    
    async def _setup_schema(self) -> None:
        """Setup PostgreSQL schema with proper indexing"""
        schema_sql = """
        -- Enable vector extension if available
        CREATE EXTENSION IF NOT EXISTS vector;
        
        -- Create embeddings table with partitioning by date
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            embedding_data BYTEA NOT NULL,
            embedding_dim INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            metadata JSONB DEFAULT '{}',
            drift_score FLOAT8,
            CONSTRAINT valid_drift_score CHECK (drift_score IS NULL OR drift_score >= 0)
        ) PARTITION BY RANGE (created_at);
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_embeddings_model 
            ON embeddings (model_name, model_version);
        CREATE INDEX IF NOT EXISTS idx_embeddings_created_at 
            ON embeddings (created_at);
        CREATE INDEX IF NOT EXISTS idx_embeddings_drift_score 
            ON embeddings (drift_score) WHERE drift_score IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_embeddings_metadata 
            ON embeddings USING GIN (metadata);
            
        -- Create current month partition
        DO $$
        DECLARE
            partition_name TEXT;
            start_date DATE;
            end_date DATE;
        BEGIN
            start_date := DATE_TRUNC('month', CURRENT_DATE);
            end_date := start_date + INTERVAL '1 month';
            partition_name := 'embeddings_' || TO_CHAR(start_date, 'YYYY_MM');
            
            EXECUTE format('
                CREATE TABLE IF NOT EXISTS %I 
                PARTITION OF embeddings 
                FOR VALUES FROM (%L) TO (%L)',
                partition_name, start_date, end_date
            );
        END $$;
        
        -- Create drift statistics table
        CREATE TABLE IF NOT EXISTS drift_statistics (
            id SERIAL PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            window_start TIMESTAMP WITH TIME ZONE NOT NULL,
            window_end TIMESTAMP WITH TIME ZONE NOT NULL,
            sample_count INTEGER NOT NULL,
            mean_drift_score FLOAT8 NOT NULL,
            max_drift_score FLOAT8 NOT NULL,
            drift_threshold FLOAT8 NOT NULL,
            alert_triggered BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE(model_name, model_version, window_start)
        );
        
        CREATE INDEX IF NOT EXISTS idx_drift_stats_model_time 
            ON drift_statistics (model_name, model_version, window_start DESC);
        """
        
        async with self.pg_pool.acquire() as conn:
            await conn.execute(schema_sql)
            
    @staticmethod
    def _serialize_embedding(embedding: List[float]) -> bytes:
        """Serialize embedding to binary format for efficient storage"""
        return struct.pack(f'{len(embedding)}f', *embedding)
    
    @staticmethod
    def _deserialize_embedding(data: bytes) -> List[float]:
        """Deserialize embedding from binary format"""
        size = len(data) // 4  # 4 bytes per float
        return list(struct.unpack(f'{size}f', data))
    
    def _cache_key(self, embedding_id: str) -> str:
        """Generate Redis cache key"""
        return f"embedding:{embedding_id}"
    
    def _batch_cache_key(self, batch_id: str) -> str:
        """Generate Redis batch cache key"""
        return f"batch:{batch_id}"
    
    async def store_embedding(self, record: EmbeddingRecord) -> None:
        """Store single embedding in PostgreSQL and cache in Redis"""
        if not self._initialized:
            raise EmbeddingStoreError("Store not initialized")
            
        try:
            # Serialize embedding
            embedding_data = self._serialize_embedding(record.embedding)
            
            # Store in PostgreSQL
            async with self.pg_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO embeddings (
                        id, embedding_data, embedding_dim, model_name, 
                        model_version, created_at, metadata, drift_score
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (id) DO UPDATE SET
                        embedding_data = EXCLUDED.embedding_data,
                        embedding_dim = EXCLUDED.embedding_dim,
                        model_name = EXCLUDED.model_name,
                        model_version = EXCLUDED.model_version,
                        created_at = EXCLUDED.created_at,
                        metadata = EXCLUDED.metadata,
                        drift_score = EXCLUDED.drift_score
                    """,
                    record.id,
                    embedding_data,
                    len(record.embedding),
                    record.model_name,
                    record.model_version,
                    record.timestamp,
                    json.dumps(record.metadata),
                    record.drift_score
                )
            
            # Cache in Redis
            cache_data = {
                'embedding_data': embedding_data,
                'model_name': record.model_name,
                'model_version': record.model_version,
                'timestamp': record.timestamp.isoformat(),
                'metadata': json.dumps(record.metadata),
                'drift_score': record.drift_score
            }
            
            pipe = self.redis_client.pipeline()
            pipe.hset(self._cache_key(record.id), mapping=cache_data)
            pipe.expire(self._cache_key(record.id), self.cache_ttl)
            await pipe.execute()
            
            logger.debug("Stored embedding", embedding_id=record.id, model=record.model_name)
            
        except Exception as e:
            logger.error("Failed to store embedding", 
                        embedding_id=record.id, error=str(e))
            raise EmbeddingStoreError(f"Failed to store embedding {record.id}: {e}")
    
    async def store_batch(self, batch: EmbeddingBatch) -> None:
        """Store batch of embeddings efficiently"""
        if not self._initialized:
            raise EmbeddingStoreError("Store not initialized")
            
        if not batch.records:
            return
            
        try:
            # Prepare batch data for PostgreSQL
            batch_data = []
            cache_operations = []
            
            for record in batch.records:
                embedding_data = self._serialize_embedding(record.embedding)
                
                batch_data.append((
                    record.id,
                    embedding_data,
                    len(record.embedding),
                    record.model_name,
                    record.model_version,
                    record.timestamp,
                    json.dumps(record.metadata),
                    record.drift_score
                ))
                
                # Prepare cache operation
                cache_data = {
                    'embedding_data': embedding_data,
                    'model_name': record.model_name,
                    'model_version': record.model_version,
                    'timestamp': record.timestamp.isoformat(),
                    'metadata': json.dumps(record.metadata),
                    'drift_score': record.drift_score
                }
                cache_operations.append((record.id, cache_data))
            
            # Batch insert to PostgreSQL
            async with self.pg_pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO embeddings (
                        id, embedding_data, embedding_dim, model_name, 
                        model_version, created_at, metadata, drift_score
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (id) DO UPDATE SET
                        embedding_data = EXCLUDED.embedding_data,
                        embedding_dim = EXCLUDED.embedding_dim,
                        model_name = EXCLUDED.model_name,
                        model_version = EXCLUDED.model_version,
                        created_at = EXCLUDED.created_at,
                        metadata = EXCLUDED.metadata,
                        drift_score = EXCLUDED.drift_score
                    """,
                    batch_data
                )
            
            # Batch cache operations
            pipe = self.redis_client.pipeline()
            for embedding_id, cache_data in cache_operations:
                pipe.hset(self._cache_key(embedding_id), mapping=cache_data)
                pipe.expire(self._cache_key(embedding_id), self.cache_ttl)
            
            # Cache batch metadata
            batch_metadata = {
                'count': len(batch.records),
                'created_at': batch.created_at.isoformat(),
                'models': list(set(r.model_name for r in batch.records))
            }
            pipe.hset(self._batch_cache_key(batch.batch_id), mapping=batch_metadata)
            pipe.expire(self._batch_cache_key(batch.batch_id), self.cache_ttl)
            
            await pipe.execute()
            
            logger.info("Stored embedding batch", 
                       batch_id=batch.batch_id, count=len(batch.records))
            
        except Exception as e:
            logger.error("Failed to store embedding batch", 
                        batch_id=batch.batch_id, error=str(e))
            raise EmbeddingStoreError(f"Failed to store batch {batch.batch_id}: {e}")
    
    async def get_embedding(self, embedding_id: str) -> EmbeddingRecord:
        """Retrieve embedding by ID, checking cache first"""
        if not self._initialized:
            raise EmbeddingStoreError("Store not initialized")
            
        try:
            # Try cache first
            cache_data = await self.redis_client.hgetall(self._cache_key(embedding_id))
            
            if cache_data:
                embedding = self._deserialize_embedding(cache_data[b'embedding_data'])
                return EmbeddingRecord(
                    id=embedding_id,
                    embedding=embedding,
                    model_name=cache_data[b'model_name'].decode(),
                    model_version=cache_data[b'model_version'].decode(),
                    timestamp=datetime.fromisoformat(cache_data[b'timestamp'].decode()),
                    metadata=json.loads(cache_data[b'metadata'].decode()),
                    drift_score=float(cache_data[b'drift_score']) if cache_data.get(b'drift_score') else None
                )
            
            # Fallback to PostgreSQL
            async with self.pg_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, embedding_data, model_name, model_version, 
                           created_at, metadata, drift_score
                    FROM embeddings 
                    WHERE id = $1
                    """,
                    embedding_id
                )
                
                if not row:
                    raise EmbeddingNotFoundError(f"Embedding {embedding_id} not found")
                
                embedding = self._deserialize_embedding(row['embedding_data'])
                
                record = EmbeddingRecord(
                    id=row['id'],
                    embedding=embedding,
                    model_name=row['model_name'],
                    model_version=row['model_version'],
                    timestamp=row['created_at'],
                    metadata=json.loads(row['metadata']),
                    drift_score=row['drift_score']
                )
                
                # Update cache
                await self._cache_embedding(record)
                
                return record
                
        except EmbeddingNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to retrieve embedding", 
                        embedding_id=embedding_id, error=str(e))
            raise EmbeddingStoreError(f"Failed to retrieve embedding {embedding_id}: {e}")
    
    async def _cache_embedding(self, record: EmbeddingRecord) -> None:
        """Cache embedding record in Redis"""
        cache_data = {
            'embedding_data': self._serialize_embedding(record.embedding),
            'model_name': record.model_name,
            'model_version': record.model_version,
            'timestamp': record.timestamp.isoformat(),
            'metadata': json.dumps(record.metadata),
            'drift_score': record.drift_score
        }
        
        pipe = self.redis_client.pipeline()
        pipe.hset(self._cache_key(record.id), mapping=cache_data)
        pipe.expire(self._cache_key(record.id), self.cache_ttl)
        await pipe.execute()
    
    async def get_recent_embeddings(
        self,
        model_name: str,
        model_version: str,
        limit: int = 1000,
        hours_back: int = 24
    ) -> List[EmbeddingRecord]:
        """Get recent embeddings for drift analysis"""
        if not self._initialized:
            raise EmbeddingStoreError("Store not initialized")
            
        try:
            async with self.pg_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, embedding_data, model_name, model_version, 
                           created_at, metadata, drift_score
                    FROM embeddings 
                    WHERE model_name = $1 
                      AND model_version = $2
                      AND created_at >= NOW() - INTERVAL '%s hours'
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    model_name, model_version, hours_back, limit
                )
                
                records = []
                for row in rows:
                    embedding = self._deserialize_embedding(row['embedding_data'])
                    records.append(EmbeddingRecord(
                        id=row['id'],
                        embedding=embedding,
                        model_name=row['model_name'],
                        model_version=row['model_version'],
                        timestamp=row['created_at'],
                        metadata=json.loads(row['metadata']),
                        drift_score=row['drift_score']
                    ))
                
                logger.debug("Retrieved recent embeddings", 
                           model=model_name, count=len(records))
                return records
                
        except Exception as e:
            logger.error("Failed to retrieve recent embeddings", 
                        model=model_name, error=str(e))
            raise EmbeddingStoreError(f"Failed to retrieve recent embeddings: {e}")
    
    async def update_drift_scores(self, scores: Dict[str, float]) -> None:
        """Batch update drift scores for embeddings"""
        if not self._initialized or not scores:
            return
            
        try:
            # Update PostgreSQL
            async with self.pg_pool.acquire() as conn:
                update_data = [(embedding_id, score) for embedding_id, score in scores.items()]
                await conn.executemany(
                    "UPDATE embeddings SET drift_score = $2 WHERE id = $1",
                    update_data
                )
            
            # Update cache
            pipe = self.redis_client.pipeline()
            for embedding_id, score in scores.items():
                pipe.hset(self._cache_key(embedding_id), 'drift_score', score)
            await pipe.execute()
            
            logger.info("Updated drift scores", count=len(scores))
            
        except Exception as e:
            logger.error("Failed to update drift scores", error=str(e))
            raise EmbeddingStoreError(f"Failed to update drift scores: {e}")
    
    async def store_drift_statistics(
        self,
        model_name: str,
        model_version: str,
        window_start: datetime,
        window_end: datetime,
        stats: Dict[str, Any]
    ) -> None:
        """Store drift analysis statistics"""
        if not self._initialized:
            raise EmbeddingStoreError("Store not initialized")
            
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO drift_statistics (
                        model_name, model_version, window_start, window_end,
                        sample_count, mean_drift_score, max_drift_score,
                        drift_threshold, alert_triggered
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (model_name, model_version, window_start)
                    DO UPDATE SET
                        window_end = EXCLUDED.window_end,
                        sample_count = EXCLUDED.sample_count,
                        mean_drift_score = EXCLUDED.mean_drift_score,
                        max_drift_score = EXCLUDED.max_drift_score,
                        drift_threshold = EXCLUDED.drift_threshold,
                        alert_triggered = EXCLUDED.alert_triggered
                    """,
                    model_name, model_version, window_start, window_end,
                    stats['sample_count'], stats['mean_drift_score'],
                    stats['max_drift_score'], stats['drift_threshold'],
                    stats['alert_triggered']
                )
                
            logger.debug("Stored drift statistics", 
                        model=model_name, window_start=window_start)
            
        except Exception as e:
            logger.error("Failed to store drift statistics", 
                        model=model_name, error=str(e))
            raise EmbeddingStoreError(f"Failed to store drift statistics: {e}")
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Clean up old embeddings and statistics"""
        if not self._initialized:
            raise EmbeddingStoreError("Store not initialized")
            
        try:
            async with self.pg_pool.acquire() as conn:
                # Clean up old embeddings
                result = await conn.execute(
                    "DELETE FROM embeddings WHERE created_at < NOW() - INTERVAL '%s days'",
                    days_to_keep
                )
                
                # Clean up old statistics
                await conn.execute(
                    "DELETE FROM drift_statistics WHERE created_at < NOW() - INTERVAL '%s days'",
                    days_to_keep
                )
                
                # Extract count from result string like "DELETE 123"
                deleted_count = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
                
                logger.info("Cleaned up old data", 
                           days_to_keep=days_to_keep, deleted_count=deleted_count)
                
                return deleted_count
                
        except Exception as e:
            logger.error("Failed to cleanup old data", error=str(e))
            raise EmbeddingStoreError(f"Failed to cleanup old data: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get storage health status for monitoring"""
        if not self._initialized:
            return {'status': 'not_initialized', 'healthy': False}
            
        try:
            # Test PostgreSQL
            async with self.pg_pool.acquire() as conn:
                pg_result = await conn.fetchval("SELECT COUNT(*) FROM embeddings")
                
            # Test Redis
            redis_info = await self.redis_client.info('memory')
            
            return {
                'status': 'healthy',
                'healthy': True,
                'postgres': {
                    'connected': True,
                    'embedding_count': pg_result,
                    'pool_size': self.pg_pool.get_size()
                },
                'redis': {
                    'connected': True,
                    'memory_used': redis_info.get('used_memory_human', 'unknown')
                }
            }
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                'status': 'unhealthy',
                'healthy': False,
                'error': str(e)
            }
    
    async def close(self) -> None:
        """Clean shutdown of all connections"""
        try:
            if self.pg_pool:
                await self.pg_pool.close()
            if self.redis_client:
                await self.redis_client.close()
            logger.info("EmbeddingStore connections closed")
        except Exception as e:
            logger.error("Error closing connections", error=str(e))