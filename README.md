# Embedding Drift Monitor

Production ML monitoring system that detects embedding drift and model degradation in real-time.

## Architecture

**Core Components:**
- **Drift Detection**: Real-time embedding drift analysis with statistical tests
- **Vector Health**: Dimension validation and cluster health monitoring
- **Quality Tracking**: Model performance degradation detection
- **Alert System**: Multi-channel notifications (Slack, PagerDuty, Teams, Email, Discord, Webhooks)
- **Circuit Breaker**: Auto-failover with configurable thresholds

**Infrastructure:**
- **Database**: PostgreSQL for metadata, Redis for caching
- **Kubernetes**: Full production deployment with RBAC, HPA, network policies
- **Observability**: Prometheus metrics, Grafana dashboards, structured logging
- **Security**: Service accounts, network policies, Istio service mesh

## Skills Demonstrated

- **ML/AI**: Embedding drift detection, vector similarity analysis, statistical testing
- **Infrastructure**: Terraform (GCP), Kubernetes, Redis, PostgreSQL
- **SRE**: Prometheus monitoring, circuit breakers, health checks, alerting
- **Backend**: FastAPI, async processing, distributed architecture
- **Database**: PostgreSQL optimization, Redis caching strategies
- **DevOps**: K8s manifests, CI/CD pipeline, container orchestration
- **Data**: Real-time processing, batch analysis, time series detection

## Quick Start

```bash
# Deploy infrastructure
cd terraform && terraform apply

# Deploy to Kubernetes
kubectl apply -k k8s/

# Local development
docker build -f docker/Dockerfile -t drift-monitor .
docker run -p 8000:8000 drift-monitor
```

## API Endpoints

- `POST /embeddings/compare` - Compare embedding batches for drift
- `POST /embeddings/validate` - Validate embedding quality
- `GET /health/vector/{model_id}` - Get vector health metrics
- `GET /metrics` - Prometheus metrics endpoint

## Monitoring

- **Metrics**: Custom Prometheus metrics for drift detection
- **Alerts**: Configurable thresholds with severity levels
- **Health Checks**: Kubernetes readiness/liveness probes
- **Circuit Breaker**: Automatic failover on service degradation

## Production Features

- Async batch processing with configurable concurrency
- Multi-algorithm drift detection (KL divergence, Wasserstein, KS test)
- Automatic model degradation alerts
- High availability with circuit breakers
- Comprehensive observability and logging