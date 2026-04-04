import redis
import redis.connection
import logging
import time
from typing import Optional, List, Any
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    max_connections: int = 20
    retry_attempts: int = 3
    retry_delay: float = 0.5
    socket_timeout: int = 30
    health_check_interval: int = 30


class RedisCache:
    def __init__(self, config: RedisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.pool = redis.ConnectionPool(
            host=config.host,
            port=config.port,
            db=config.db,
            max_connections=config.max_connections,
            socket_timeout=config.socket_timeout,
            health_check_interval=config.health_check_interval,
            retry_on_timeout=True
        )
        
        self.client = redis.Redis(connection_pool=self.pool)
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test Redis connection on startup"""
        try:
            self.client.ping()
            self.logger.info("Redis connection established")
        except redis.RedisError as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _retry_operation(self, operation, *args, **kwargs) -> Any:
        """Retry Redis operations with exponential backoff"""
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                return operation(*args, **kwargs)
            except (redis.ConnectionError, redis.TimeoutError) as e:
                last_exception = e
                if attempt < self.config.retry_attempts - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Redis operation failed (attempt {attempt + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"Redis operation failed after {self.config.retry_attempts} attempts")
        
        raise last_exception
    
    def get(self, key: str) -> Optional[bytes]:
        """Get value with retry logic"""
        return self._retry_operation(self.client.get, key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with retry logic"""
        return self._retry_operation(self.client.set, key, value, ex=ttl)
    
    def delete(self, key: str) -> int:
        """Delete key with retry logic"""
        return self._retry_operation(self.client.delete, key)
    
    @contextmanager
    def pipeline(self):
        """Context manager for Redis pipeline with retry"""
        pipe = None
        try:
            pipe = self.client.pipeline()
            yield pipe
            self._retry_operation(pipe.execute)
        except Exception as e:
            if pipe:
                pipe.reset()
            raise e
    
    def health_check(self) -> bool:
        """Check if Redis is healthy"""
        try:
            self.client.ping()
            return True
        except redis.RedisError:
            return False
    
    def close(self) -> None:
        """Close connection pool"""
        self.pool.disconnect()
        self.logger.info("Redis connection pool closed")