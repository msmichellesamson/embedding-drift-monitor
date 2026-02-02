# Embedding Drift Monitor

[![CI/CD](https://github.com/michellesamson/embedding-drift-monitor/workflows/CI/badge.svg)](https://github.com/michellesamson/embedding-drift-monitor/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production ML monitoring system that detects embedding drift and model degradation in real-time using statistical analysis and observability patterns. Provides automated alerting when model performance degrades due to data distribution shifts.

## Skills Demonstrated

- **AI/ML Engineering**: Embedding drift detection with KL divergence, cosine similarity, and statistical hypothesis testing
- **SRE/Observability**: Prometheus metrics, Grafana dashboards, automated alerting with PagerDuty integration
- **Backend Engineering**: High-performance async FastAPI with batched processing and circuit breakers
- **Database Engineering**: Hybrid PostgreSQL + Redis architecture with optimized vector similarity queries
- **Infrastructure**: Terraform-managed GCP deployment (Cloud SQL, Memorystore, GKE) with auto-scaling
- **DevOps**: GitOps CI/CD pipeline with model validation tests and blue-green deployments

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Model     │───▶│  Embedding   │───▶│  FastAPI    │
│ Inference   │    │   Capture    │    │  Service    │
│  Service    │    │              │    │             │
└─────────────┘    └──────────────┘    └─────────────┘
                            │                    │
                            ▼                    ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Redis     │◀───│ PostgreSQL   │───▶│ Prometheus  │
│ (Hot Cache) │    │ (Embeddings) │    │  Metrics    │
└─────────────┘    └──────────────┘    └─────────────┘
                            │                    │
                            ▼                    ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Drift      │◀───│ Statistical  │───▶│  Grafana    │
│ Detection   │    │  Analysis    │    │ Dashboard   │
│  Engine     │    │              │    │             │
└─────────────┘    └──────────────┘    └─────────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │  PagerDuty   │
                   │  Alerting    │
                   └──────────────┘
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/michellesamson/embedding-drift-monitor.git
cd embedding-drift-monitor

# Local development
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Start services (requires Docker)
docker-compose up -d postgres redis

# Run migrations
alembic upgrade head

# Start the service
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Test drift detection
curl -X POST "http://localhost:8000/embeddings/batch" \
  -H "Content-Type: application/json" \
  -d '{"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "model_id": "bert-base", "timestamp": "2024-01-01T00:00:00Z"}'
```

## Configuration

```bash
# Required environment variables
export DATABASE_URL="postgresql://user:pass@localhost:5432/drift_monitor"
export REDIS_URL="redis://localhost:6379/0"
export PROMETHEUS_GATEWAY="http://localhost:9091"

# Optional configuration
export DRIFT_THRESHOLD=0.15           # KL divergence threshold
export EMBEDDING_DIMENSION=768        # Model embedding size
export BATCH_SIZE=1000                # Processing batch size
export ALERT_COOLDOWN_MINUTES=30      # Alert suppression window
export HISTORICAL_WINDOW_DAYS=7       # Baseline comparison window
```

## Infrastructure Deployment

```bash
# Deploy to GCP with Terraform
cd terraform/
terraform init
terraform plan -var="project_id=your-gcp-project"
terraform apply

# Deploy to Kubernetes
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Verify deployment
kubectl get pods -n embedding-drift-monitor
kubectl logs -f deployment/drift-monitor -n embedding-drift-monitor
```

## API Usage

### Submit Embeddings for Monitoring

```python
import requests
import numpy as np

# Batch embedding submission
embeddings = np.random.rand(100, 768).tolist()
response = requests.post("http://localhost:8000/embeddings/batch", json={
    "embeddings": embeddings,
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "timestamp": "2024-01-01T12:00:00Z",
    "metadata": {"environment": "production", "version": "v1.2.3"}
})
```

### Check Drift Status

```python
# Get current drift metrics
response = requests.get("http://localhost:8000/drift/status/bert-base")
drift_data = response.json()

print(f"KL Divergence: {drift_data['kl_divergence']:.4f}")
print(f"Drift Detected: {drift_data['drift_detected']}")
print(f"Confidence: {drift_data['confidence_interval']}")
```

### Configure Alerts

```python
# Set up custom drift thresholds
requests.post("http://localhost:8000/models/bert-base/config", json={
    "drift_threshold": 0.12,
    "statistical_test": "ks_test",
    "alert_webhook": "https://hooks.slack.com/services/...",
    "baseline_refresh_hours": 24
})
```

## Development

```bash
# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Type checking
mypy src/

# Code formatting
black src/ tests/
isort src/ tests/

# Performance testing
locust -f tests/load_test.py --host=http://localhost:8000

# Database migrations
alembic revision --autogenerate -m "Add embedding index"
alembic upgrade head

# Monitoring stack (local)
docker-compose -f docker/monitoring-stack.yml up -d
```

### Key Metrics Monitored

- **Drift Score**: KL divergence between current and baseline embeddings
- **Processing Latency**: P95 latency for batch embedding analysis
- **Database Performance**: Query execution times and connection pool usage
- **Alert Frequency**: Rate of drift detection events per model
- **System Health**: CPU, memory, and disk usage across service instances

### Statistical Tests Implemented

- **Kolmogorov-Smirnov Test**: Distribution similarity testing
- **Anderson-Darling Test**: Goodness-of-fit with emphasis on tail behavior  
- **Maximum Mean Discrepancy**: Kernel-based two-sample test
- **Population Stability Index**: Industry-standard drift measurement

## License

MIT License - see [LICENSE](LICENSE) file for details.