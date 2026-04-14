import redis
import redis.connection
from typing import Optional, Any, Dict
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class RedisCacheError(Exception):
    """Redis cache operation error"""
    pass

class RedisCache:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        max_connections: int = 10,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30
    ):
        self.connection_pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            retry_on_timeout=retry_on_timeout,
            health_check_interval=health_check_interval
        )
        self.client = redis.Redis(connection_pool=self.connection_pool)
        self._last_health_check = 0
        self._health_check_interval = 30

    @contextmanager
    def _handle_redis_errors(self, operation: str):
        """Context manager for consistent Redis error handling"""
        try:
            yield
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error during {operation}: {e}")
            raise RedisCacheError(f"Connection failed: {e}")
        except redis.TimeoutError as e:
            logger.error(f"Redis timeout during {operation}: {e}")
            raise RedisCacheError(f"Operation timed out: {e}")
        except redis.RedisError as e:
            logger.error(f"Redis error during {operation}: {e}")
            raise RedisCacheError(f"Redis operation failed: {e}")

    def get(self, key: str) -> Optional[bytes]:
        """Get value with error handling and connection validation"""
        with self._handle_redis_errors(f"GET {key}"):
            self._check_health()
            return self.client.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL and error handling"""
        with self._handle_redis_errors(f"SET {key}"):
            self._check_health()
            if ttl:
                return self.client.setex(key, ttl, value)
            return self.client.set(key, value)

    def delete(self, key: str) -> int:
        """Delete key with error handling"""
        with self._handle_redis_errors(f"DELETE {key}"):
            self._check_health()
            return self.client.delete(key)

    def _check_health(self) -> None:
        """Periodic health check to ensure connection is alive"""
        now = time.time()
        if now - self._last_health_check > self._health_check_interval:
            try:
                self.client.ping()
                self._last_health_check = now
            except redis.RedisError:
                logger.warning("Redis health check failed, connection may be stale")
                # Let the connection pool handle reconnection

    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        pool = self.connection_pool
        return {
            "created_connections": pool.created_connections,
            "available_connections": len(pool._available_connections),
            "in_use_connections": len(pool._in_use_connections),
            "max_connections": pool.max_connections
        }

    def close(self) -> None:
        """Clean up connection pool"""
        try:
            self.connection_pool.disconnect()
            logger.info("Redis connection pool closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection pool: {e}")
