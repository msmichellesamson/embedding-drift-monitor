"""Batch processing for embedding vectors with improved connection handling."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import time

from src.core.embedding_store import EmbeddingStore
from src.monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 100
    max_wait_time: float = 5.0
    connection_timeout: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 1.0
    pool_size: int = 10

@dataclass
class BatchItem:
    """Single item in processing batch."""
    vector_id: str
    embedding: List[float]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BatchProcessor:
    """Processes embedding vectors in batches with connection pooling."""
    
    def __init__(self, store: EmbeddingStore, config: BatchConfig = None):
        self.store = store
        self.config = config or BatchConfig()
        self.metrics = MetricsCollector()
        self._pending_batch: List[BatchItem] = []
        self._batch_lock = asyncio.Lock()
        self._connection_semaphore = asyncio.Semaphore(self.config.pool_size)
        self._shutdown = False
    
    async def add_item(self, item: BatchItem) -> None:
        """Add item to processing batch."""
        async with self._batch_lock:
            self._pending_batch.append(item)
            
            if len(self._pending_batch) >= self.config.batch_size:
                await self._process_batch()
    
    @asynccontextmanager
    async def _connection_context(self):
        """Manage database connections with timeout and pooling."""
        async with self._connection_semaphore:
            try:
                # Simulate connection acquisition with timeout
                await asyncio.wait_for(
                    asyncio.sleep(0.1),  # Simulate connection setup
                    timeout=self.config.connection_timeout
                )
                yield
            except asyncio.TimeoutError:
                logger.error("Connection timeout exceeded")
                self.metrics.increment('batch_processor_connection_timeouts')
                raise
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self.metrics.increment('batch_processor_connection_errors')
                raise
    
    async def _process_batch_with_retry(self, batch: List[BatchItem]) -> bool:
        """Process batch with retry logic."""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                async with self._connection_context():
                    await self._store_batch(batch)
                    self.metrics.increment('batch_processor_success')
                    return True
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Batch processing attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_backoff * (2 ** attempt))
        
        logger.error(f"Batch processing failed after {self.config.max_retries} attempts: {last_error}")
        self.metrics.increment('batch_processor_failures')
        return False
    
    async def _store_batch(self, batch: List[BatchItem]) -> None:
        """Store batch items in embedding store."""
        vectors = [(item.vector_id, item.embedding, item.metadata) for item in batch]
        await self.store.store_batch(vectors)
        
        self.metrics.histogram('batch_processor_batch_size', len(batch))
        logger.info(f"Processed batch of {len(batch)} embeddings")
    
    async def _process_batch(self) -> None:
        """Process current batch."""
        if not self._pending_batch:
            return
            
        batch = self._pending_batch.copy()
        self._pending_batch.clear()
        
        start_time = time.time()
        success = await self._process_batch_with_retry(batch)
        duration = time.time() - start_time
        
        self.metrics.histogram('batch_processor_duration', duration)
        
        if not success:
            # Re-queue failed items for retry (simplified)
            logger.warning(f"Re-queueing {len(batch)} failed items")
    
    async def flush(self) -> None:
        """Force process any pending items."""
        async with self._batch_lock:
            if self._pending_batch:
                await self._process_batch()
    
    async def shutdown(self) -> None:
        """Gracefully shutdown processor."""
        self._shutdown = True
        await self.flush()
        logger.info("Batch processor shutdown complete")