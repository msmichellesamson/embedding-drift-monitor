import asyncio
import logging
from typing import Callable, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)

class RetryConfig:
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

def with_exponential_backoff(config: Optional[RetryConfig] = None):
    """Decorator that adds exponential backoff retry logic to async functions."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == config.max_retries:
                        logger.error(f"Final retry attempt failed for {func.__name__}: {e}")
                        raise e
                    
                    # Calculate exponential backoff delay
                    delay = min(
                        config.base_delay * (2 ** attempt),
                        config.max_delay
                    )
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator

class RetryHandler:
    """Handles retry logic with exponential backoff for notification systems."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        decorated_func = with_exponential_backoff(self.config)(func)
        return await decorated_func(*args, **kwargs)
