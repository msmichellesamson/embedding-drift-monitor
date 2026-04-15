"""Batch processor for efficient vector health metrics collection."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class MetricsBatch:
    """Batch of metrics to process."""
    vectors: List[Dict[str, Any]]
    timestamp: float
    batch_id: str
    metadata: Dict[str, Any]


class BatchProcessor:
    """Process vector metrics in batches for better performance."""
    
    def __init__(
        self,
        batch_size: int = 100,
        flush_interval: float = 30.0,
        max_batches: int = 10
    ):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_batches = max_batches
        self._batches: List[MetricsBatch] = []
        self._current_batch: List[Dict[str, Any]] = []
        self._metrics_buffer = defaultdict(list)
        self._logger = logging.getLogger(__name__)
        self._processing = False
    
    async def add_vector(self, vector_data: Dict[str, Any]) -> None:
        """Add vector to current batch."""
        try:
            self._current_batch.append(vector_data)
            
            if len(self._current_batch) >= self.batch_size:
                await self._flush_current_batch()
        except Exception as e:
            self._logger.error(f"Failed to add vector to batch: {e}")
            raise
    
    async def _flush_current_batch(self) -> None:
        """Flush current batch to processing queue."""
        if not self._current_batch:
            return
        
        import time
        import uuid
        
        batch = MetricsBatch(
            vectors=self._current_batch.copy(),
            timestamp=time.time(),
            batch_id=str(uuid.uuid4())[:8],
            metadata={"size": len(self._current_batch)}
        )
        
        self._batches.append(batch)
        self._current_batch.clear()
        
        # Keep only recent batches
        if len(self._batches) > self.max_batches:
            self._batches = self._batches[-self.max_batches:]
        
        self._logger.debug(f"Flushed batch {batch.batch_id} with {len(batch.vectors)} vectors")
    
    async def process_batches(self) -> Dict[str, Any]:
        """Process all pending batches and return aggregated metrics."""
        if self._processing:
            self._logger.warning("Batch processing already in progress")
            return {}
        
        self._processing = True
        try:
            # Flush any pending vectors
            await self._flush_current_batch()
            
            if not self._batches:
                return {}
            
            # Process all batches
            results = await self._process_all_batches()
            
            # Clear processed batches
            self._batches.clear()
            
            return results
        finally:
            self._processing = False
    
    async def _process_all_batches(self) -> Dict[str, Any]:
        """Process all batches and aggregate results."""
        total_vectors = 0
        dimension_counts = defaultdict(int)
        
        for batch in self._batches:
            total_vectors += len(batch.vectors)
            
            for vector_data in batch.vectors:
                if 'dimensions' in vector_data:
                    dimension_counts[vector_data['dimensions']] += 1
        
        return {
            'total_vectors_processed': total_vectors,
            'batches_processed': len(self._batches),
            'dimension_distribution': dict(dimension_counts),
            'processing_timestamp': asyncio.get_event_loop().time()
        }