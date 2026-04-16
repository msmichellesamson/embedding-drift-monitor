from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class EscalationRule:
    severity: AlertSeverity
    initial_delay: int  # seconds
    max_attempts: int
    escalate_after: int  # seconds
    next_severity: Optional[AlertSeverity] = None

class SeverityManager:
    def __init__(self):
        self.escalation_rules = {
            AlertSeverity.LOW: EscalationRule(
                severity=AlertSeverity.LOW,
                initial_delay=300,  # 5 minutes
                max_attempts=3,
                escalate_after=1800,  # 30 minutes
                next_severity=AlertSeverity.MEDIUM
            ),
            AlertSeverity.MEDIUM: EscalationRule(
                severity=AlertSeverity.MEDIUM,
                initial_delay=120,  # 2 minutes
                max_attempts=5,
                escalate_after=900,  # 15 minutes
                next_severity=AlertSeverity.HIGH
            ),
            AlertSeverity.HIGH: EscalationRule(
                severity=AlertSeverity.HIGH,
                initial_delay=60,  # 1 minute
                max_attempts=10,
                escalate_after=600,  # 10 minutes
                next_severity=AlertSeverity.CRITICAL
            ),
            AlertSeverity.CRITICAL: EscalationRule(
                severity=AlertSeverity.CRITICAL,
                initial_delay=0,  # immediate
                max_attempts=20,
                escalate_after=0  # no escalation
            )
        }
        self.active_alerts: Dict[str, Dict] = {}
    
    def calculate_severity(self, drift_score: float, vector_health: float) -> AlertSeverity:
        """Calculate alert severity based on drift score and vector health."""
        if drift_score > 0.9 or vector_health < 0.1:
            return AlertSeverity.CRITICAL
        elif drift_score > 0.7 or vector_health < 0.3:
            return AlertSeverity.HIGH
        elif drift_score > 0.5 or vector_health < 0.5:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def should_escalate(self, alert_id: str) -> Optional[AlertSeverity]:
        """Check if alert should be escalated to higher severity."""
        if alert_id not in self.active_alerts:
            return None
            
        alert_data = self.active_alerts[alert_id]
        current_severity = alert_data['severity']
        first_seen = alert_data['first_seen']
        
        rule = self.escalation_rules[current_severity]
        if rule.next_severity is None:
            return None
            
        time_elapsed = (datetime.now() - first_seen).total_seconds()
        if time_elapsed >= rule.escalate_after:
            logger.warning(f"Escalating alert {alert_id} from {current_severity.value} to {rule.next_severity.value}")
            return rule.next_severity
            
        return None
    
    def register_alert(self, alert_id: str, severity: AlertSeverity) -> None:
        """Register a new alert or update existing one."""
        if alert_id not in self.active_alerts:
            self.active_alerts[alert_id] = {
                'severity': severity,
                'first_seen': datetime.now(),
                'attempt_count': 0
            }
            logger.info(f"Registered new alert {alert_id} with severity {severity.value}")
        else:
            # Update severity if escalated
            self.active_alerts[alert_id]['severity'] = severity
    
    def get_retry_delay(self, alert_id: str) -> int:
        """Get retry delay for alert based on severity and attempt count."""
        if alert_id not in self.active_alerts:
            return 60  # default
            
        alert_data = self.active_alerts[alert_id]
        severity = alert_data['severity']
        attempt_count = alert_data['attempt_count']
        
        rule = self.escalation_rules[severity]
        
        # Exponential backoff with severity-based initial delay
        base_delay = rule.initial_delay
        return max(base_delay, base_delay * (2 ** min(attempt_count, 4)))
    
    def clear_alert(self, alert_id: str) -> None:
        """Clear resolved alert."""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            logger.info(f"Cleared resolved alert {alert_id}")
