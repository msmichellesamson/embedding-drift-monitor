# Embedding Drift Monitor

Production ML monitoring system that detects embedding drift and model degradation in real-time.

## Features

- **Real-time Drift Detection**: Statistical tests (KS, MMD, Wasserstein) for embedding distribution changes
- **Anomaly Detection**: Time-series analysis with seasonal decomposition and outlier detection
- **Circuit Breaker Pattern**: Fault tolerance for external dependencies
- **Multi-channel Alerting**: Slack, PagerDuty, and email notifications with exponential backoff retry
- **Prometheus Integration**: Custom metrics and alerting rules
- **Production Ready**: Kubernetes deployment, Terraform infrastructure, comprehensive testing

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   Embeddings    │───▶│ Drift        │───▶│  Alerting   │
│   Ingestion     │    │ Detection    │    │  System     │
└─────────────────┘    └──────────────┘    └─────────────┘
         │                      │                   │
         ▼                      ▼                   ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│  Embedding      │    │ Statistical  │    │ Retry       │
│  Store          │    │ Analysis     │    │ Handler     │
└─────────────────┘    └──────────────┘    └─────────────┘
```

## Quick Start

```bash
# Run with Docker
docker build -f docker/Dockerfile -t embedding-drift-monitor .
docker run -p 8080:8080 embedding-drift-monitor

# Deploy to Kubernetes
kubectl apply -f k8s/deployment.yaml

# Setup infrastructure
cd terraform && terraform init && terraform apply
```

## API Usage

```python
import httpx

# Submit embeddings for monitoring
response = httpx.post(
    "http://localhost:8080/embeddings",
    json={"embeddings": [[0.1, 0.2, 0.3]], "model_id": "bert-base"}
)

# Check drift status
drift_status = httpx.get("http://localhost:8080/drift/bert-base")
```

## Configuration

```python
from src.core.drift_detector import DriftConfig
from src.alerts.retry_handler import RetryConfig

config = DriftConfig(
    window_size=1000,
    drift_threshold=0.05,
    alert_threshold=3
)

retry_config = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0
)
```

## Monitoring

- **Prometheus metrics**: `http://localhost:8080/metrics`
- **Health check**: `http://localhost:8080/health`
- **OpenAPI docs**: `http://localhost:8080/docs`

## Tech Stack

- **Backend**: Python, FastAPI, asyncio
- **ML**: NumPy, SciPy, scikit-learn
- **Infrastructure**: Terraform (GCP), Kubernetes, Docker
- **Monitoring**: Prometheus, custom metrics
- **Database**: Redis for embedding storage
- **Alerting**: Slack, PagerDuty, SMTP with exponential backoff

## Testing

```bash
pytest tests/ -v --cov=src/
```

## Recent Updates

- ✅ Added exponential backoff retry logic for alert notifications
- ✅ Implemented circuit breaker pattern for external dependencies
- ✅ Added comprehensive statistical drift detection
- ✅ Kubernetes deployment with health checks
- ✅ Prometheus integration with custom metrics