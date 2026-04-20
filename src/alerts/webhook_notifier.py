"""Enhanced webhook notifier with circuit breaker for reliability."""

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_notifier import BaseNotifier
from ..core.circuit_breaker import CircuitBreaker


@dataclass
class WebhookConfig:
    url: str
    timeout: int = 30
    headers: Optional[Dict[str, str]] = None
    signature_key: Optional[str] = None


class WebhookNotifier(BaseNotifier):
    """Webhook notifier with circuit breaker and retry logic."""
    
    def __init__(self, config: WebhookConfig):
        super().__init__()
        self.config = config
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60,
            expected_exception=Exception
        )
        self.logger = logging.getLogger(__name__)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def send_notification(self, alert_data: Dict[str, Any]) -> bool:
        """Send webhook notification with circuit breaker protection."""
        try:
            return await self.circuit_breaker.call(self._send_webhook, alert_data)
        except Exception as e:
            self.logger.error(f"Webhook notification failed: {e}")
            return False
    
    async def _send_webhook(self, alert_data: Dict[str, Any]) -> bool:
        """Internal webhook sender."""
        payload = self._build_payload(alert_data)
        headers = self._build_headers(payload)
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.config.url,
                json=payload,
                headers=headers
            ) as response:
                if response.status >= 400:
                    raise aiohttp.ClientError(f"HTTP {response.status}: {await response.text()}")
                
                self.logger.info(f"Webhook sent successfully: {response.status}")
                return True
    
    def _build_payload(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build webhook payload."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "alert_type": alert_data.get("type", "unknown"),
            "severity": alert_data.get("severity", "medium"),
            "message": alert_data.get("message", ""),
            "metadata": alert_data.get("metadata", {}),
            "source": "embedding-drift-monitor"
        }
    
    def _build_headers(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """Build request headers with optional signature."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "embedding-drift-monitor/1.0"
        }
        
        if self.config.headers:
            headers.update(self.config.headers)
        
        if self.config.signature_key:
            import hmac
            import hashlib
            import json
            
            body = json.dumps(payload, sort_keys=True)
            signature = hmac.new(
                self.config.signature_key.encode(),
                body.encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-Signature-SHA256"] = f"sha256={signature}"
        
        return headers
    
    async def health_check(self) -> Dict[str, Any]:
        """Check webhook endpoint health."""
        return {
            "status": "healthy" if self.circuit_breaker.state == "closed" else "degraded",
            "circuit_breaker_state": self.circuit_breaker.state,
            "failure_count": self.circuit_breaker.failure_count,
            "last_failure": self.circuit_breaker.last_failure_time.isoformat() if self.circuit_breaker.last_failure_time else None
        }