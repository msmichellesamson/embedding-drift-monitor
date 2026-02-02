from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
import numpy as np

@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0

class EmbeddingStore:
    def __init__(self, redis_client, retry_config: Optional[RetryConfig] = None):
        self.redis = redis_client
        self.retry_config = retry_config or RetryConfig()
        self.logger = logging.getLogger(__name__)
    
    async def _retry_with_backoff(self, operation, *args, **kwargs):
        """Execute operation with exponential backoff retry."""
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt == self.retry_config.max_retries:
                    break
                
                delay = min(
                    self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
                    self.retry_config.max_delay
                )
                
                self.logger.warning(
                    f"Operation failed (attempt {attempt + 1}/{self.retry_config.max_retries + 1}): {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                await asyncio.sleep(delay)
        
        raise last_exception
    
    async def store_embeddings(self, model_id: str, embeddings: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Store embeddings with retry logic."""
        async def _store_operation():
            timestamp = datetime.now(timezone.utc).isoformat()
            key = f"embeddings:{model_id}:{timestamp}"
            
            data = {
                'embeddings': embeddings.tobytes(),
                'shape': embeddings.shape,
                'dtype': str(embeddings.dtype),
                'metadata': metadata,
                'timestamp': timestamp
            }
            
            await self.redis.hset(key, mapping=data)
            await self.redis.expire(key, 86400 * 7)  # 7 days TTL
            return True
        
        try:
            return await self._retry_with_backoff(_store_operation)
        except Exception as e:
            self.logger.error(f"Failed to store embeddings for {model_id}: {e}")
            return False
    
    async def get_recent_embeddings(self, model_id: str, hours: int = 24) -> Optional[List[Dict]]:
        """Get recent embeddings with retry logic."""
        async def _get_operation():
            pattern = f"embeddings:{model_id}:*"
            keys = await self.redis.keys(pattern)
            
            if not keys:
                return []
            
            # Sort by timestamp (newest first)
            keys.sort(reverse=True)
            
            embeddings = []
            for key in keys[:100]:  # Limit to recent 100
                data = await self.redis.hgetall(key)
                if data:
                    embeddings.append({
                        'key': key,
                        'timestamp': data.get('timestamp'),
                        'metadata': data.get('metadata', {}),
                        'shape': eval(data.get('shape', '(0,)')),
                    })
            
            return embeddings
        
        try:
            return await self._retry_with_backoff(_get_operation)
        except Exception as e:
            self.logger.error(f"Failed to get embeddings for {model_id}: {e}")
            return None