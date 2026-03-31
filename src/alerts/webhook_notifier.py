from typing import Dict, Any, Optional
import asyncio
import aiohttp
import json
import logging
from urllib.parse import urlparse

from .base_notifier import BaseNotifier
from .retry_handler import RetryHandler

logger = logging.getLogger(__name__)

class WebhookNotifier(BaseNotifier):
    """Generic webhook notifier for custom integrations."""
    
    def __init__(self, webhook_url: str, secret_token: Optional[str] = None):
        super().__init__()
        self.webhook_url = webhook_url
        self.secret_token = secret_token
        self.retry_handler = RetryHandler(max_retries=3, base_delay=1.0)
        
        # Validate webhook URL
        parsed = urlparse(webhook_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid webhook URL: {webhook_url}")
    
    async def send_notification(self, alert_data: Dict[str, Any]) -> bool:
        """Send alert via webhook with retries."""
        payload = self._format_webhook_payload(alert_data)
        headers = self._get_headers()
        
        async def webhook_request():
            timeout = aiohttp.ClientTimeout(total=10.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status >= 400:
                        error_text = await response.text()
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"Webhook failed: {error_text}"
                        )
                    return response.status
        
        try:
            status_code = await self.retry_handler.execute_with_retry(webhook_request)
            logger.info(f"Webhook notification sent successfully: {status_code}")
            return True
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False
    
    def _format_webhook_payload(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format alert data for webhook payload."""
        return {
            "timestamp": alert_data.get("timestamp"),
            "alert_type": "embedding_drift",
            "severity": alert_data.get("drift_score", 0) > 0.8 and "high" or "medium",
            "message": f"Drift detected: {alert_data.get('drift_score', 0):.3f}",
            "details": {
                "drift_score": alert_data.get("drift_score"),
                "threshold": alert_data.get("threshold"),
                "embedding_model": alert_data.get("model_name"),
                "samples_analyzed": alert_data.get("samples_count")
            },
            "source": "embedding-drift-monitor"
        }
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for webhook request."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "embedding-drift-monitor/1.0"
        }
        
        if self.secret_token:
            headers["Authorization"] = f"Bearer {self.secret_token}"
        
        return headers
