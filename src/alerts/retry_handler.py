"""Retry mechanism for alert delivery with exponential backoff."""
import asyncio
import logging
import random
from typing import Any, Callable, Optional, Type
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RetryPolicy(Enum):
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"
    LINEAR_BACKOFF = "linear_backoff"

@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF
    jitter: bool = True
    backoff_multiplier: float = 2.0
    retryable_exceptions: tuple = (Exception,)

class RetryHandler:
    """Handles retry logic for alert delivery operations."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        
    async def retry_async(
        self, 
        func: Callable[..., Any], 
        *args, 
        **kwargs
    ) -> Any:
        """Execute function with async retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                logger.debug(f"Attempt {attempt}/{self.config.max_attempts} for {func.__name__}")
                return await func(*args, **kwargs)
                
            except self.config.retryable_exceptions as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt} failed for {func.__name__}: {str(e)}"
                )
                
                if attempt == self.config.max_attempts:
                    break
                    
                delay = self._calculate_delay(attempt)
                logger.info(f"Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
                
        logger.error(f"All retry attempts exhausted for {func.__name__}")
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay based on retry policy."""
        if self.config.policy == RetryPolicy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
        elif self.config.policy == RetryPolicy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        else:  # FIXED_DELAY
            delay = self.config.base_delay
            
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            jitter = delay * 0.1 * random.random()
            delay += jitter
            
        return delay