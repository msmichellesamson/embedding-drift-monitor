import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AlertContext:
    alert_id: str
    timestamp: float
    severity: AlertSeverity
    service: str
    metadata: Dict[str, Any]
    retry_count: int = 0

class NotificationError(Exception):
    """Base exception for notification failures"""
    def __init__(self, message: str, context: Optional[AlertContext] = None, retry_after: Optional[int] = None):
        super().__init__(message)
        self.context = context
        self.retry_after = retry_after

class BaseNotifier(ABC):
    """Base class for all notification providers with structured logging"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"alerts.{name}")
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure structured logging for this notifier"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s %(levelname)s: %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def send_alert(self, context: AlertContext) -> bool:
        """Send alert with comprehensive error handling and logging"""
        start_time = time.time()
        
        self.logger.info(
            "Sending alert",
            extra={
                "alert_id": context.alert_id,
                "severity": context.severity.value,
                "service": context.service,
                "retry_count": context.retry_count,
                "notifier": self.name
            }
        )
        
        try:
            success = self._send_notification(context)
            duration = time.time() - start_time
            
            if success:
                self.logger.info(
                    "Alert sent successfully",
                    extra={
                        "alert_id": context.alert_id,
                        "duration_ms": round(duration * 1000, 2),
                        "notifier": self.name
                    }
                )
            else:
                self.logger.warning(
                    "Alert sending failed",
                    extra={
                        "alert_id": context.alert_id,
                        "duration_ms": round(duration * 1000, 2),
                        "notifier": self.name
                    }
                )
            
            return success
            
        except NotificationError as e:
            self.logger.error(
                "Notification error",
                extra={
                    "alert_id": context.alert_id,
                    "error": str(e),
                    "retry_after": e.retry_after,
                    "notifier": self.name
                },
                exc_info=True
            )
            return False
            
        except Exception as e:
            self.logger.error(
                "Unexpected error during notification",
                extra={
                    "alert_id": context.alert_id,
                    "error": str(e),
                    "notifier": self.name
                },
                exc_info=True
            )
            return False
    
    @abstractmethod
    def _send_notification(self, context: AlertContext) -> bool:
        """Implementation-specific notification logic"""
        pass
