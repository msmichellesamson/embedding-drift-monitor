from typing import Dict, Any, Optional
import aiohttp
import logging
from .base_notifier import BaseNotifier
from .retry_handler import RetryHandler

logger = logging.getLogger(__name__)

class DiscordNotifier(BaseNotifier):
    """Discord webhook notifier for development teams."""
    
    def __init__(self, webhook_url: str, username: Optional[str] = None):
        self.webhook_url = webhook_url
        self.username = username or "Embedding Drift Monitor"
        self.retry_handler = RetryHandler(max_retries=3)
        
    async def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert to Discord channel via webhook."""
        try:
            embed = self._create_embed(alert)
            payload = {
                "username": self.username,
                "embeds": [embed]
            }
            
            return await self.retry_handler.execute(
                self._send_webhook, payload
            )
        except Exception as e:
            logger.error(f"Discord notification failed: {e}")
            return False
            
    def _create_embed(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Create Discord embed from alert data."""
        severity = alert.get('severity', 'info')
        color_map = {
            'critical': 0xFF0000,  # Red
            'warning': 0xFFFF00,   # Yellow
            'info': 0x00FF00       # Green
        }
        
        return {
            "title": f"🚨 {alert.get('title', 'Drift Alert')}",
            "description": alert.get('message', ''),
            "color": color_map.get(severity, 0x808080),
            "fields": [
                {
                    "name": "Model",
                    "value": alert.get('model_id', 'unknown'),
                    "inline": True
                },
                {
                    "name": "Drift Score",
                    "value": f"{alert.get('drift_score', 0):.4f}",
                    "inline": True
                },
                {
                    "name": "Timestamp",
                    "value": alert.get('timestamp', ''),
                    "inline": False
                }
            ],
            "footer": {
                "text": "Embedding Drift Monitor"
            }
        }
        
    async def _send_webhook(self, payload: Dict[str, Any]) -> bool:
        """Send webhook to Discord."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 204:
                    logger.info("Discord alert sent successfully")
                    return True
                else:
                    logger.error(f"Discord webhook failed: {response.status}")
                    return False