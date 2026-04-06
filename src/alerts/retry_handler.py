import asyncio
import logging
from typing import Callable, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RetryStrategy(Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"

@dataclass
class RetryConfig:
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 300  # 5 minutes

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class RetryHandler:
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.circuit_state = CircuitState.CLOSED
        
    async def execute_with_retry(
        self, 
        func: Callable, 
        *args, 
        **kwargs
    ) -> Any:
        """Execute function with retry logic and circuit breaker."""
        
        if self._is_circuit_open():
            raise Exception("Circuit breaker is OPEN - too many failures")
            
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
                
            except Exception as e:
                last_exception = e
                self._on_failure()
                
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_retries + 1} attempts failed")
                    
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay based on retry strategy."""
        if self.config.strategy == RetryStrategy.FIXED:
            return self.config.initial_delay
        elif self.config.strategy == RetryStrategy.LINEAR:
            return min(self.config.initial_delay * (attempt + 1), self.config.max_delay)
        else:  # EXPONENTIAL
            delay = self.config.initial_delay * (self.config.backoff_multiplier ** attempt)
            return min(delay, self.config.max_delay)
    
    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.circuit_state == CircuitState.CLOSED:
            return False
            
        if self.circuit_state == CircuitState.OPEN:
            if self.last_failure_time and \
               (datetime.now() - self.last_failure_time).seconds > self.config.circuit_breaker_timeout:
                self.circuit_state = CircuitState.HALF_OPEN
                return False
            return True
            
        return False  # HALF_OPEN
    
    def _on_success(self):
        """Reset circuit breaker on success."""
        self.failure_count = 0
        self.circuit_state = CircuitState.CLOSED
        self.last_failure_time = None
        
    def _on_failure(self):
        """Track failure and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.config.circuit_breaker_threshold:
            self.circuit_state = CircuitState.OPEN
            logger.error(f"Circuit breaker OPEN after {self.failure_count} failures")