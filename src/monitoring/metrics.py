import asyncio
import logging
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from redis.asyncio import Redis, ConnectionPool
from ..core.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Enhanced metrics collector with connection pooling and graceful shutdown."""
    
    def __init__(self, redis_url: str, max_connections: int = 20):
        self.registry = CollectorRegistry()
        self._setup_metrics()
        
        # Connection pooling for better resource management
        self.pool = ConnectionPool.from_url(
            redis_url,
            max_connections=max_connections,
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={}
        )
        self.redis: Optional[Redis] = None
        self.circuit_breaker = CircuitBreaker(failure_threshold=3)
        self._shutdown_event = asyncio.Event()
        
    def _setup_metrics(self):
        """Initialize Prometheus metrics."""
        self.drift_score_gauge = Gauge(
            'embedding_drift_score',
            'Current embedding drift score',
            ['model_version', 'dimension'],
            registry=self.registry
        )
        self.vector_operations_counter = Counter(
            'vector_operations_total',
            'Total vector operations',
            ['operation', 'status'],
            registry=self.registry
        )
        self.processing_duration = Histogram(
            'embedding_processing_seconds',
            'Time spent processing embeddings',
            registry=self.registry
        )
        
    async def start(self):
        """Initialize Redis connection with proper error handling."""
        try:
            self.redis = Redis(connection_pool=self.pool)
            await self.redis.ping()
            logger.info("Metrics collector started successfully")
        except Exception as e:
            logger.error(f"Failed to start metrics collector: {e}")
            raise
            
    async def shutdown(self):
        """Graceful shutdown with connection cleanup."""
        logger.info("Shutting down metrics collector...")
        self._shutdown_event.set()
        
        if self.redis:
            try:
                await self.redis.close()
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")
                
        if self.pool:
            try:
                await self.pool.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting pool: {e}")
                
        logger.info("Metrics collector shutdown complete")
    
    @asynccontextmanager
    async def _redis_operation(self):
        """Context manager for Redis operations with circuit breaker."""
        if self._shutdown_event.is_set():
            raise RuntimeError("Metrics collector is shutting down")
            
        try:
            await self.circuit_breaker.call(self._check_redis_health)
            yield self.redis
        except Exception as e:
            logger.error(f"Redis operation failed: {e}")
            raise
            
    async def _check_redis_health(self):
        """Health check for Redis connection."""
        if not self.redis:
            raise ConnectionError("Redis not initialized")
        await self.redis.ping()
        
    async def record_drift_score(self, score: float, model_version: str, dimension: int):
        """Record drift score with enhanced error handling."""
        try:
            async with self._redis_operation():
                self.drift_score_gauge.labels(
                    model_version=model_version,
                    dimension=str(dimension)
                ).set(score)
                
                # Store in Redis for historical tracking
                key = f"drift:{model_version}:{dimension}"
                await self.redis.lpush(key, f"{score}:{asyncio.get_event_loop().time()}")
                await self.redis.ltrim(key, 0, 999)  # Keep last 1000 entries
                
        except Exception as e:
            logger.error(f"Failed to record drift score: {e}")
            self.vector_operations_counter.labels(
                operation='drift_record', status='error'
            ).inc()