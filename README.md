# Embedding Drift Monitor

> Production ML monitoring system that detects embedding drift and model degradation in real-time

## Architecture

```
Embeddings → Drift Detection → Statistical Analysis → Alerts → Monitoring
     ↓              ↓                   ↓             ↓          ↓
   Store       Circuit Breaker    Time Series    Slack/PD   Prometheus
```

## Features

- **Real-time drift detection** with configurable thresholds
- **Statistical analysis** (KS test, Wasserstein distance, PCA drift)
- **Circuit breaker** pattern for system resilience
- **Multi-channel alerting** (Slack, PagerDuty, Email)
- **Prometheus metrics** and Grafana dashboards
- **REST API** for embedding comparison and monitoring
- **Kubernetes deployment** with Istio service mesh

## Quick Start

```bash
# Local development
docker build -f docker/Dockerfile -t embedding-drift-monitor .
docker run -p 8000:8000 embedding-drift-monitor

# Test drift detection
curl -X POST http://localhost:8000/api/v1/comparison/detect \
  -H "Content-Type: application/json" \
  -d '{
    "baseline_embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    "current_embeddings": [[0.8, 0.9, 1.0], [1.1, 1.2, 1.3]],
    "model_name": "test-model",
    "threshold": 0.1
  }'
```

## API Endpoints

- `POST /api/v1/comparison/detect` - Compare embeddings for drift
- `GET /api/v1/comparison/health` - Service health check
- `GET /metrics` - Prometheus metrics

## Infrastructure

- **Terraform**: GCP infrastructure provisioning
- **Kubernetes**: Container orchestration with Istio
- **Prometheus**: Metrics collection and alerting
- **Circuit breaker**: Resilience patterns

## Skills Demonstrated

- **ML/AI**: Embedding drift detection, statistical analysis
- **Backend**: FastAPI, REST APIs, error handling
- **Infrastructure**: Terraform, Kubernetes, Istio
- **SRE**: Prometheus, alerting, circuit breakers
- **DevOps**: Docker, CI/CD, monitoring
- **Database**: Time series storage, metrics collection

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Apply infrastructure
cd terraform && terraform apply

# Deploy to Kubernetes
kubectl apply -f k8s/
```

## Configuration

Environment variables:
- `DRIFT_THRESHOLD`: Default drift detection threshold (0.1)
- `PROMETHEUS_PORT`: Metrics server port (9090)
- `LOG_LEVEL`: Logging level (INFO)
- `SLACK_WEBHOOK_URL`: Slack notifications
- `PAGERDUTY_API_KEY`: PagerDuty integration