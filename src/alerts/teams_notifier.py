import asyncio
import json
from typing import Dict, Any, Optional
from datetime import datetime

import aiohttp
from pydantic import BaseModel, HttpUrl

from .base_notifier import BaseNotifier
from .retry_handler import RetryHandler


class TeamsAlert(BaseModel):
    """Teams alert message format"""
    summary: str
    severity: str
    metric: str
    threshold: float
    current_value: float
    timestamp: datetime
    additional_context: Optional[Dict[str, Any]] = None


class TeamsNotifier(BaseNotifier):
    """Microsoft Teams webhook notifier for drift alerts"""
    
    def __init__(self, webhook_url: HttpUrl, retry_handler: RetryHandler):
        super().__init__()
        self.webhook_url = str(webhook_url)
        self.retry_handler = retry_handler
        
    def _build_teams_card(self, alert: TeamsAlert) -> Dict[str, Any]:
        """Build Teams Adaptive Card format"""
        color = {
            "critical": "attention",
            "warning": "warning", 
            "info": "good"
        }.get(alert.severity.lower(), "default")
        
        facts = [
            {"title": "Metric", "value": alert.metric},
            {"title": "Current Value", "value": f"{alert.current_value:.4f}"},
            {"title": "Threshold", "value": f"{alert.threshold:.4f}"},
            {"title": "Severity", "value": alert.severity.upper()},
            {"title": "Time", "value": alert.timestamp.isoformat()}
        ]
        
        return {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "summary": alert.summary,
            "themeColor": color,
            "sections": [{
                "activityTitle": "🚨 Embedding Drift Alert",
                "activitySubtitle": alert.summary,
                "facts": facts
            }]
        }
        
    async def send_alert(self, alert: TeamsAlert) -> bool:
        """Send alert to Teams channel"""
        try:
            payload = self._build_teams_card(alert)
            
            async with aiohttp.ClientSession() as session:
                response = await self.retry_handler.execute(
                    self._send_request,
                    session,
                    payload
                )
                
            self._log_success(alert)
            return True
            
        except Exception as e:
            self._log_error(alert, e)
            return False
            
    async def _send_request(self, session: aiohttp.ClientSession, payload: Dict) -> aiohttp.ClientResponse:
        """Send HTTP request to Teams webhook"""
        async with session.post(
            self.webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            response.raise_for_status()
            return response
            
    def _log_success(self, alert: TeamsAlert) -> None:
        """Log successful alert delivery"""
        self.logger.info(
            "Teams alert sent successfully",
            extra={
                "metric": alert.metric,
                "severity": alert.severity,
                "webhook_url": self.webhook_url[:50] + "..."
            }
        )
        
    def _log_error(self, alert: TeamsAlert, error: Exception) -> None:
        """Log alert delivery failure"""
        self.logger.error(
            "Failed to send Teams alert",
            extra={
                "metric": alert.metric,
                "severity": alert.severity,
                "error": str(error)
            }
        )