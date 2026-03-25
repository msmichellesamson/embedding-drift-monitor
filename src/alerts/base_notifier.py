"""Base class for alert notifiers with retry support."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

from .retry_handler import RetryHandler, RetryConfig

logger = logging.getLogger(__name__)

class BaseNotifier(ABC):
    """Base class for all alert notifiers."""
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        self.retry_handler = RetryHandler(retry_config)
        
    @abstractmethod
    async def _send_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send alert implementation - must be implemented by subclasses."""
        pass
    
    async def send_alert_with_retry(self, alert_data: Dict[str, Any]) -> bool:
        """Send alert with retry mechanism."""
        try:
            result = await self.retry_handler.retry_async(
                self._send_alert, 
                alert_data
            )
            logger.info(f"Alert sent successfully via {self.__class__.__name__}")
            return result
        except Exception as e:
            logger.error(f"Failed to send alert via {self.__class__.__name__}: {str(e)}")
            return False
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate notifier configuration."""
        pass