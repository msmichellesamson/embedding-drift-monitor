# Embedding Drift Monitor

Production ML monitoring system that detects embedding drift and model degradation in real-time using statistical analysis and observability.

## Architecture

- **Core Engine**: Real-time drift detection with circuit breaker patterns
- **Storage**: PostgreSQL + Redis caching with connection pooling
- **Monitoring**: Prometheus metrics + Grafana dashboards
- **Alerts**: Multi-channel notifications (Slack, PagerDuty, Discord, Email)
- **Infrastructure**: Kubernetes deployment with Istio service mesh

## Skills Demonstrated

- **AI/ML**: Embedding drift detection, statistical anomaly detection, vector similarity analysis
- **Backend**: FastAPI microservice, gRPC, circuit breaker patterns
- **Database**: PostgreSQL optimization, Redis caching, connection pooling
- **Infrastructure**: Terraform (GCP), Kubernetes manifests, Istio configuration
- **SRE**: Prometheus monitoring, alerting rules, health checks
- **DevOps**: Docker containerization, CI/CD pipeline, GitOps
- **Data**: Real-time streaming analysis, time series processing

## Quick Start

```bash
# Deploy infrastructure
cd terraform && terraform apply

# Deploy to Kubernetes
kubectl apply -f k8s/

# Local development
docker build -f docker/Dockerfile -t embedding-drift-monitor .
docker run -p 8080:8080 embedding-drift-monitor
```

## API Endpoints

- `POST /api/v1/embeddings/store` - Store reference embeddings
- `POST /api/v1/embeddings/compare` - Compare against baseline
- `GET /api/v1/similarity/{embedding_id}` - Get similarity metrics
- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics

## Infrastructure

- **Kubernetes**: HPA, network policies, Istio service mesh
- **Monitoring**: Prometheus + Grafana with custom dashboards
- **Database**: PostgreSQL with connection pooling
- **Cache**: Redis cluster for performance
- **Security**: Network policies, RBAC, secret management

## Testing

```bash
pytest tests/ -v
pytest tests/integration/ -v --integration
```

## Monitoring

Prometheus metrics available at `/metrics`:
- `drift_detection_total` - Total drift detections
- `embedding_processing_duration` - Processing latency
- `similarity_score_histogram` - Score distribution
- `circuit_breaker_state` - Circuit breaker status