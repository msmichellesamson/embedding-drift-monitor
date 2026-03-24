#!/usr/bin/env python3
"""Email notification integration for drift alerts."""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EmailConfig:
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    from_email: str
    use_tls: bool = True

class EmailNotifier:
    """Email notifier for drift detection alerts."""
    
    def __init__(self, config: EmailConfig):
        self.config = config
        
    def send_alert(
        self, 
        drift_score: float, 
        threshold: float, 
        model_name: str,
        recipients: List[str],
        severity: str = "warning"
    ) -> bool:
        """Send email alert for drift detection."""
        try:
            subject = f"[{severity.upper()}] Embedding Drift Alert - {model_name}"
            
            body = self._create_alert_body(
                drift_score, threshold, model_name, severity
            )
            
            return self._send_email(subject, body, recipients)
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _create_alert_body(self, drift_score: float, threshold: float, 
                          model_name: str, severity: str) -> str:
        """Create formatted alert email body."""
        return f"""
Embedding Drift Alert

Model: {model_name}
Severity: {severity}
Drift Score: {drift_score:.4f}
Threshold: {threshold:.4f}

The model embeddings have drifted beyond the configured threshold.
Please review the model performance and consider retraining.

Timestamp: {self._get_timestamp()}
"""
    
    def _send_email(self, subject: str, body: str, recipients: List[str]) -> bool:
        """Send email using SMTP."""
        msg = MIMEMultipart()
        msg['From'] = self.config.from_email
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        context = ssl.create_default_context() if self.config.use_tls else None
        
        with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
            if self.config.use_tls:
                server.starttls(context=context)
            server.login(self.config.username, self.config.password)
            server.send_message(msg)
            
        logger.info(f"Email alert sent to {len(recipients)} recipients")
        return True
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for alerts."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"