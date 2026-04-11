import redis
import logging
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import time
from dataclasses import dataclass

@dataclass
class RedisConfig:
    host: str = 'localhost'
    port: int = 6379
    db: int = 0
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: int = 30

class RedisCache:
    def __init__(self, config: RedisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Connection pool with timeout settings
        self.pool = redis.ConnectionPool(
            host=config.host,
            port=config.port,
            db=config.db,
            max_connections=config.max_connections,
            socket_timeout=config.socket_timeout,
            socket_connect_timeout=config.socket_connect_timeout,
            retry_on_timeout=config.retry_on_timeout,
            health_check_interval=config.health_check_interval
        )
        self.client = redis.Redis(connection_pool=self.pool)
    
    @contextmanager
    def get_connection(self):
        """Get connection with proper error handling and cleanup"""
        conn = None
        try:
            conn = self.client
            yield conn
        except redis.TimeoutError as e:
            self.logger.error(f"Redis timeout: {e}")
            raise
        except redis.ConnectionError as e:
            self.logger.error(f"Redis connection error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected Redis error: {e}")
            raise
        finally:
            # Connection automatically returned to pool
            pass
    
    def get_embedding(self, key: str) -> Optional[List[float]]:
        """Get embedding with connection timeout handling"""
        try:
            with self.get_connection() as conn:
                data = conn.get(key)
                if data:
                    import json
                    return json.loads(data)
                return None
        except Exception as e:
            self.logger.error(f"Failed to get embedding {key}: {e}")
            return None
    
    def set_embedding(self, key: str, embedding: List[float], ttl: int = 3600) -> bool:
        """Set embedding with connection timeout handling"""
        try:
            with self.get_connection() as conn:
                import json
                return bool(conn.setex(key, ttl, json.dumps(embedding)))
        except Exception as e:
            self.logger.error(f"Failed to set embedding {key}: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Check Redis health with connection metrics"""
        try:
            start = time.time()
            with self.get_connection() as conn:
                conn.ping()
                latency = time.time() - start
                
                info = conn.info()
                return {
                    'status': 'healthy',
                    'latency_ms': round(latency * 1000, 2),
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory_human': info.get('used_memory_human', 'unknown')
                }
        except Exception as e:
            self.logger.error(f"Redis health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }