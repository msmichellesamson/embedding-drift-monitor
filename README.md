# Embedding Drift Monitor

Production ML monitoring system that detects embedding drift and model degradation in real-time.

## Features

### Core Monitoring
- **Drift Detection**: Statistical tests for embedding distribution changes
- **Similarity Tracking**: Monitor embedding similarity patterns over time
- **Vector Health**: Track embedding quality metrics and outliers
- **Dimension Validation**: Ensure embedding dimensions remain consistent across models
- **Anomaly Detection**: Real-time detection of unusual embedding patterns

### Infrastructure
- **Redis Cache**: High-performance embedding storage with TTL
- **PostgreSQL**: Persistent storage for drift metrics and alerts
- **Circuit Breaker**: Fault tolerance for external dependencies
- **Prometheus Integration**: Comprehensive metrics collection

### Alerting
- **Multi-Channel Notifications**: Slack, Discord, PagerDuty, Email, Webhook
- **Intelligent Retry Logic**: Exponential backoff with circuit breaking
- **Alert Severity Levels**: Configurable thresholds and escalation

### API & Analysis
- **REST API**: Compare embeddings and retrieve drift metrics
- **Time Series Analysis**: Trend detection and forecasting
- **Statistical Testing**: Kolmogorov-Smirnov, Jensen-Shannon divergence

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   ML Models     │───▶│   API Layer  │───▶│  Drift Engine   │
└─────────────────┘    └──────────────┘    └─────────────────┘
                                                     │
                       ┌─────────────────┐          ▼
                       │   Alerting      │    ┌─────────────────┐
                       │   System        │◀───│  Redis Cache    │
                       └─────────────────┘    └─────────────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │   PostgreSQL    │
                                            │   (Metrics)     │
                                            └─────────────────┘
```

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start services
docker-compose up -d redis postgres

# Run the service
python -m src.main
```

### Production Deployment
```bash
# Deploy infrastructure
cd terraform
terraform init && terraform apply

# Deploy to Kubernetes
kubectl apply -k k8s/
```

## Configuration

```yaml
drift_detection:
  window_size: 1000
  similarity_threshold: 0.85
  statistical_significance: 0.05

monitoring:
  metrics_port: 8080
  health_check_interval: 30
  dimension_validation: true

alerting:
  slack_webhook: ${SLACK_WEBHOOK_URL}
  pagerduty_key: ${PAGERDUTY_INTEGRATION_KEY}
  email_smtp: ${SMTP_SERVER}
```

## API Usage

```python
import requests

# Register model dimension
response = requests.post('/api/v1/models/bert-base/dimension', 
                        json={'dimension': 768})

# Compare embeddings
response = requests.post('/api/v1/compare', json={
    'model_id': 'bert-base',
    'embedding': [0.1, 0.2, ...],  # 768-dim vector
    'reference_set': 'production-baseline'
})

print(response.json())
# {
#   "drift_score": 0.23,
#   "is_anomaly": false,
#   "confidence": 0.95,
#   "dimension_valid": true
# }
```

## Monitoring & Observability

### Prometheus Metrics
- `embedding_drift_score` - Current drift score by model
- `embedding_dimensions_violations_total` - Dimension consistency violations
- `embedding_similarity_avg` - Average similarity in time window
- `drift_alerts_fired_total` - Count of drift alerts by severity

### Health Checks
- `/health` - Service health status
- `/health/redis` - Redis connectivity
- `/health/postgres` - Database connectivity
- `/metrics` - Prometheus metrics endpoint

## Technology Stack

- **Python 3.11+**: Core application
- **FastAPI**: REST API framework
- **Redis**: High-performance caching
- **PostgreSQL**: Persistent storage
- **Terraform**: Infrastructure as code
- **Kubernetes**: Container orchestration
- **Prometheus**: Metrics collection
- **Docker**: Containerization

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/

# Integration tests
pytest tests/integration/
```

## License

MIT License - see LICENSE file for details.