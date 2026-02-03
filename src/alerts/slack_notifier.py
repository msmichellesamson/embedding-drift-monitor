import asyncio
import json
import logging
from typing import Dict, Optional

import aiohttp
from pydantic import BaseModel, HttpUrl

logger = logging.getLogger(__name__)


class DriftAlert(BaseModel):
    model_name: str
    drift_score: float
    threshold: float
    timestamp: str
    severity: str  # "warning" | "critical"
    details: Optional[Dict] = None


class SlackNotifier:
    """Sends drift alerts to Slack channels."""
    
    def __init__(self, webhook_url: HttpUrl, channel: str = "#ml-alerts"):
        self.webhook_url = str(webhook_url)
        self.channel = channel
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def send_drift_alert(self, alert: DriftAlert) -> bool:
        """Send drift alert to Slack channel."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        color = "danger" if alert.severity == "critical" else "warning"
        
        payload = {
            "channel": self.channel,
            "attachments": [{
                "color": color,
                "title": f"🚨 Model Drift Detected: {alert.model_name}",
                "fields": [
                    {"title": "Drift Score", "value": f"{alert.drift_score:.4f}", "short": True},
                    {"title": "Threshold", "value": f"{alert.threshold:.4f}", "short": True},
                    {"title": "Severity", "value": alert.severity.upper(), "short": True},
                    {"title": "Timestamp", "value": alert.timestamp, "short": True}
                ],
                "footer": "Embedding Drift Monitor",
                "ts": int(float(alert.timestamp))
            }]
        }
        
        try:
            async with self.session.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    logger.info(f"Drift alert sent to Slack for model {alert.model_name}")
                    return True
                else:
                    logger.error(f"Failed to send Slack alert: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False
