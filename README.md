# Embedding Drift Monitor

Production ML monitoring system that detects embedding drift and model degradation in real-time.

## Features

- **Real-time Drift Detection**: Statistical tests and time-series analysis
- **Production Ready**: Connection pooling, retry logic, comprehensive error handling
- **Multi-Model Support**: Track embeddings across different models
- **Alerting**: Slack notifications with configurable thresholds
- **Observability**: Prometheus metrics and Grafana dashboards
- **Cloud Native**: Kubernetes deployment with Terraform IaC

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  ML Models      │───▶│ Embedding Store  │───▶│ Drift Detector  │
│ (via REST API)  │    │ (PostgreSQL)     │    │ (Statistical)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                         │
                                ▼                         ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Prometheus     │    │ Slack Alerts    │
                       │   Metrics        │    │ & Notifications │
                       └──────────────────┘    └─────────────────┘
```

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL
docker run -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:15

# Run migrations
python -m src.migrations.init_db

# Start monitoring
python -m src.main
```

### Production Deployment

```bash
# Deploy infrastructure
cd terraform && terraform init && terraform apply

# Deploy to Kubernetes
kubectl apply -f k8s/
```

## Configuration

```yaml
# config.yaml
database:
  connection_string: "postgresql://user:pass@localhost:5432/embeddings"
  pool_size: 20

monitoring:
  check_interval: 300
  drift_threshold: 0.15
  
alerting:
  slack_webhook: "https://hooks.slack.com/..."
  alert_threshold: 0.20
```

## API Usage

```python
# Store embeddings
POST /embeddings
{
  "model_name": "sentence-transformer",
  "embedding": [0.1, 0.2, ...],
  "metadata": {"text": "sample text", "timestamp": 1703980800}
}

# Get drift metrics
GET /drift/{model_name}
```

## Monitoring Stack

- **PostgreSQL**: Embedding storage with vector similarity search
- **Prometheus**: Metrics collection and alerting rules
- **Grafana**: Visualization dashboards
- **Kubernetes**: Container orchestration
- **Terraform**: Infrastructure as Code

## Drift Detection Methods

1. **Statistical Tests**: KS test, Wasserstein distance
2. **Similarity Tracking**: Cosine similarity trends over time
3. **Time Series Analysis**: Seasonal decomposition and anomaly detection

## Skills Demonstrated

- **ML/AI**: Embedding analysis, drift detection, model monitoring
- **Infrastructure**: Terraform, Kubernetes, cloud deployment
- **Backend**: REST APIs, database design, microservices
- **Database**: PostgreSQL, connection pooling, query optimization
- **DevOps**: Docker, CI/CD, monitoring and alerting
- **SRE**: Production reliability, error handling, observability

## Recent Updates

- ✅ Added connection pooling and retry logic for database reliability
- ✅ Implemented comprehensive error handling and logging
- ✅ Added Prometheus metrics for observability
- ✅ Kubernetes deployment configuration
- ✅ Terraform infrastructure setup
