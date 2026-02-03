# Embedding Drift Monitor

Production ML monitoring system that detects embedding drift and model degradation in real-time.

## Features

- **Real-time Drift Detection**: Statistical analysis of embedding distributions
- **Multiple Algorithms**: KS test, Jensen-Shannon divergence, Wasserstein distance
- **Scalable Storage**: Redis-backed embedding store with TTL management
- **Production Metrics**: Prometheus metrics for monitoring and alerting
- **Slack Integration**: Real-time notifications for critical drift events
- **Cloud Native**: Kubernetes deployment with Terraform infrastructure

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ML Models     │───▶│ Drift Detector   │───▶│ Alert Manager   │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Embedding Store  │    │ Slack Notifier  │
                       │ (Redis)          │    │                 │
                       └──────────────────┘    └─────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export REDIS_URL="redis://localhost:6379"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."

# Run the monitor
python -m src.main
```

## Configuration

```python
from src.alerts.slack_notifier import SlackNotifier, DriftAlert

# Configure Slack alerts
async with SlackNotifier(webhook_url=webhook_url) as notifier:
    alert = DriftAlert(
        model_name="recommendation-model",
        drift_score=0.85,
        threshold=0.7,
        timestamp="1234567890",
        severity="critical"
    )
    await notifier.send_drift_alert(alert)
```

## Deployment

```bash
# Deploy infrastructure
cd terraform && terraform apply

# Deploy to Kubernetes
kubectl apply -f k8s/
```

## Skills Demonstrated

- **ML/AI**: Embedding analysis, drift detection algorithms, model monitoring
- **Backend**: Async Python, Redis integration, gRPC APIs
- **Infrastructure**: Terraform (GCP), Kubernetes, Docker
- **SRE**: Prometheus metrics, alerting, observability
- **DevOps**: CI/CD pipelines, containerization
- **Data Engineering**: Real-time data processing, statistical analysis