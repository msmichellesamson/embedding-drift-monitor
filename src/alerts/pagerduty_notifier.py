from typing import Dict, Any, Optional
import httpx
import logging
from datetime import datetime
from dataclasses import dataclass


@dataclass
class PagerDutyConfig:
    integration_key: str
    service_name: str = "Embedding Drift Monitor"
    component: str = "ML Pipeline"


class PagerDutyNotifier:
    """Sends critical drift alerts to PagerDuty for immediate response"""
    
    def __init__(self, config: PagerDutyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.api_url = "https://events.pagerduty.com/v2/enqueue"
        
    async def trigger_alert(self, drift_data: Dict[str, Any]) -> bool:
        """Trigger a PagerDuty incident for critical drift"""
        try:
            payload = self._build_payload(drift_data, "trigger")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    json=payload,
                    timeout=10.0
                )
                response.raise_for_status()
                
            self.logger.info(f"PagerDuty alert triggered: {drift_data.get('model_name')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send PagerDuty alert: {e}")
            return False
    
    async def resolve_alert(self, dedup_key: str) -> bool:
        """Resolve a PagerDuty incident when drift returns to normal"""
        try:
            payload = {
                "routing_key": self.config.integration_key,
                "event_action": "resolve",
                "dedup_key": dedup_key
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    json=payload,
                    timeout=10.0
                )
                response.raise_for_status()
                
            self.logger.info(f"PagerDuty alert resolved: {dedup_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to resolve PagerDuty alert: {e}")
            return False
    
    def _build_payload(self, drift_data: Dict[str, Any], action: str) -> Dict[str, Any]:
        """Build PagerDuty event payload"""
        model_name = drift_data.get("model_name", "unknown")
        drift_score = drift_data.get("drift_score", 0.0)
        threshold = drift_data.get("threshold", 0.0)
        
        dedup_key = f"drift-{model_name}-{drift_data.get('timestamp', '')}"
        
        return {
            "routing_key": self.config.integration_key,
            "event_action": action,
            "dedup_key": dedup_key,
            "payload": {
                "summary": f"Critical embedding drift detected in {model_name}",
                "severity": "critical" if drift_score > threshold * 1.5 else "error",
                "source": self.config.service_name,
                "component": self.config.component,
                "group": "ml-monitoring",
                "class": "drift-detection",
                "custom_details": {
                    "model_name": model_name,
                    "drift_score": drift_score,
                    "threshold": threshold,
                    "deviation": f"{((drift_score / threshold - 1) * 100):.1f}%",
                    "timestamp": datetime.now().isoformat(),
                    "environment": drift_data.get("environment", "production")
                }
            }
        }