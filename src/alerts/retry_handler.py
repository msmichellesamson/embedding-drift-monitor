import asyncio
import logging
from typing import Callable, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)

class RetryConfig:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

class RetryHandler:
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        delay = self.config.base_delay * (self.config.backoff_factor ** attempt)
        # Add jitter to prevent thundering herd
        import random
        jitter = random.uniform(0.1, 0.3) * delay
        return min(delay + jitter, self.config.max_delay)
    
    async def retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """Retry async function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.config.max_retries:
                    logger.error(
                        f"Final retry failed for {func.__name__}: {e}",
                        extra={"attempt": attempt + 1, "error": str(e)}
                    )
                    break
                
                delay = self.calculate_delay(attempt)
                logger.warning(
                    f"Retry {attempt + 1}/{self.config.max_retries} for {func.__name__} in {delay:.2f}s: {e}"
                )
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def retry_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Retry sync function with exponential backoff."""
        import time
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.config.max_retries:
                    logger.error(
                        f"Final retry failed for {func.__name__}: {e}",
                        extra={"attempt": attempt + 1, "error": str(e)}
                    )
                    break
                
                delay = self.calculate_delay(attempt)
                logger.warning(
                    f"Retry {attempt + 1}/{self.config.max_retries} for {func.__name__} in {delay:.2f}s: {e}"
                )
                time.sleep(delay)
        
        raise last_exception

def with_retry(config: Optional[RetryConfig] = None):
    """Decorator for automatic retries with exponential backoff."""
    def decorator(func: Callable):
        handler = RetryHandler(config)
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await handler.retry_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return handler.retry_sync(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator