# Embedding Drift Monitor

Production ML monitoring system that detects embedding drift and model degradation in real-time using statistical analysis and time-series anomaly detection.

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   ML Models     │───▶│ Drift Monitor│───▶│   Alerting      │
│                 │    │              │    │                 │
│ • Embeddings    │    │ • Statistical│    │ • Slack         │
│ • Predictions   │    │ • Time Series│    │ • PagerDuty     │
│ • Features      │    │ • Similarity │    │ • Email         │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │
                       ┌──────────────┐
                       │   Storage    │
                       │              │
                       │ • PostgreSQL │
                       │ • Redis      │
                       └──────────────┘
```

## Skills Demonstrated

**AI/ML**: Embedding drift detection, similarity analysis, model degradation monitoring
**Backend**: FastAPI REST API, async processing, database integration
**Database**: PostgreSQL for metrics storage, Redis for caching and circuit breaker state
**SRE**: Prometheus metrics, Grafana dashboards, health checks, circuit breaker pattern
**Infrastructure**: Terraform for GCP deployment, Kubernetes manifests, Istio service mesh
**DevOps**: CI/CD pipeline, Docker containerization, automated testing
**Data**: Real-time stream processing, statistical analysis, time-series anomaly detection

## Quick Start

```bash
# Start with Docker Compose
docker-compose up -d

# Submit embeddings
curl -X POST http://localhost:8000/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model_name": "test-model", "embeddings": [[0.1, 0.2, 0.3]]}'

# Check drift metrics
curl http://localhost:8000/metrics/test-model
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DRIFT_THRESHOLD` | Drift alert threshold | `0.5` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `DATABASE_URL` | PostgreSQL connection | `postgresql://localhost/drift` |
| `ALERT_COOLDOWN` | Minutes between alerts | `60` |

## Deployment

```bash
# Deploy to GCP with Terraform
cd terraform
terraform init
terraform apply

# Deploy to Kubernetes
kubectl apply -f k8s/
```

## Monitoring

- **Prometheus**: `/metrics` endpoint for drift scores and system metrics
- **Grafana**: Import dashboard from `monitoring/grafana-dashboard.json`
- **Alerts**: Configured via `monitoring/alert_rules.yml`

## Documentation

- [API Reference](docs/API.md) - Complete API documentation
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions