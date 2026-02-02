import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import structlog
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
from starlette.responses import Response

from .database import DatabaseManager, get_db_manager
from .cache import CacheManager, get_cache_manager
from .drift_detector import DriftDetector, DriftResult
from .exceptions import (
    EmbeddingDriftError,
    ModelNotFoundError,
    InvalidEmbeddingError,
    DatabaseError
)

# Configure structured logging
logger = structlog.get_logger(__name__)

# Prometheus metrics
DRIFT_DETECTIONS = Counter(
    'embedding_drift_detections_total',
    'Total number of drift detections',
    ['model_id', 'drift_type', 'severity']
)

EMBEDDING_PROCESSING_TIME = Histogram(
    'embedding_processing_seconds',
    'Time spent processing embeddings',
    ['model_id', 'operation']
)

ACTIVE_MODELS = Gauge(
    'active_models_total',
    'Number of active models being monitored'
)

API_REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['endpoint', 'method', 'status_code']
)

EMBEDDING_DIMENSIONS = Gauge(
    'embedding_dimensions',
    'Embedding vector dimensions',
    ['model_id']
)


class EmbeddingInput(BaseModel):
    """Input model for embedding drift detection."""
    model_id: str = Field(..., min_length=1, max_length=100)
    embedding: List[float] = Field(..., min_items=1, max_items=4096)
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    @validator('embedding')
    def validate_embedding(cls, v):
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError('All embedding values must be numeric')
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError('Embedding contains NaN or infinite values')
        return v

    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.now(timezone.utc)


class ModelRegistration(BaseModel):
    """Model registration input."""
    model_id: str = Field(..., min_length=1, max_length=100)
    embedding_dim: int = Field(..., ge=1, le=4096)
    drift_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    window_size: int = Field(default=1000, ge=100, le=10000)
    baseline_size: int = Field(default=5000, ge=1000, le=50000)
    description: Optional[str] = Field(None, max_length=500)


class DriftResponse(BaseModel):
    """Drift detection response."""
    model_id: str
    drift_detected: bool
    drift_score: float
    drift_type: str
    severity: str
    timestamp: datetime
    details: Dict[str, Any]


class ModelStatus(BaseModel):
    """Model monitoring status."""
    model_id: str
    is_active: bool
    embedding_count: int
    last_drift_check: Optional[datetime]
    drift_threshold: float
    baseline_complete: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    database_healthy: bool
    cache_healthy: bool
    active_models: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting embedding drift monitor")
    
    # Initialize background tasks
    drift_check_task = asyncio.create_task(periodic_drift_check())
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    try:
        yield
    finally:
        logger.info("Shutting down embedding drift monitor")
        drift_check_task.cancel()
        cleanup_task.cancel()
        
        try:
            await asyncio.gather(drift_check_task, cleanup_task, return_exceptions=True)
        except Exception as e:
            logger.error("Error during shutdown", error=str(e))


app = FastAPI(
    title="Embedding Drift Monitor",
    description="Production ML monitoring system for embedding drift detection",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def prometheus_middleware(request, call_next):
    """Middleware to track API metrics."""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    API_REQUEST_DURATION.labels(
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code
    ).observe(duration)
    
    return response


@app.post("/models/register", response_model=Dict[str, str])
async def register_model(
    model: ModelRegistration,
    db: DatabaseManager = Depends(get_db_manager)
) -> Dict[str, str]:
    """Register a new model for drift monitoring."""
    try:
        logger.info("Registering model", model_id=model.model_id)
        
        # Check if model already exists
        existing = await db.get_model(model.model_id)
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model.model_id} already registered"
            )
        
        # Register model
        await db.create_model(
            model_id=model.model_id,
            embedding_dim=model.embedding_dim,
            drift_threshold=model.drift_threshold,
            window_size=model.window_size,
            baseline_size=model.baseline_size,
            description=model.description
        )
        
        ACTIVE_MODELS.inc()
        EMBEDDING_DIMENSIONS.labels(model_id=model.model_id).set(model.embedding_dim)
        
        logger.info("Model registered successfully", model_id=model.model_id)
        return {"message": f"Model {model.model_id} registered successfully"}
        
    except DatabaseError as e:
        logger.error("Database error during model registration", error=str(e))
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as e:
        logger.error("Unexpected error during model registration", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/embeddings/ingest", response_model=DriftResponse)
async def ingest_embedding(
    embedding_input: EmbeddingInput,
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_db_manager),
    cache: CacheManager = Depends(get_cache_manager)
) -> DriftResponse:
    """Ingest embedding and perform drift detection."""
    start_time = time.time()
    
    try:
        logger.info(
            "Ingesting embedding",
            model_id=embedding_input.model_id,
            embedding_dim=len(embedding_input.embedding)
        )
        
        # Validate model exists
        model_config = await db.get_model(embedding_input.model_id)
        if not model_config:
            raise ModelNotFoundError(f"Model {embedding_input.model_id} not found")
        
        # Validate embedding dimensions
        if len(embedding_input.embedding) != model_config['embedding_dim']:
            raise InvalidEmbeddingError(
                f"Expected {model_config['embedding_dim']} dimensions, "
                f"got {len(embedding_input.embedding)}"
            )
        
        # Store embedding
        await db.store_embedding(
            model_id=embedding_input.model_id,
            embedding=embedding_input.embedding,
            timestamp=embedding_input.timestamp,
            metadata=embedding_input.metadata
        )
        
        # Perform drift detection
        detector = DriftDetector(db, cache)
        drift_result = await detector.detect_drift(
            model_id=embedding_input.model_id,
            new_embedding=embedding_input.embedding
        )
        
        # Update metrics
        processing_time = time.time() - start_time
        EMBEDDING_PROCESSING_TIME.labels(
            model_id=embedding_input.model_id,
            operation="ingest"
        ).observe(processing_time)
        
        if drift_result.drift_detected:
            DRIFT_DETECTIONS.labels(
                model_id=embedding_input.model_id,
                drift_type=drift_result.drift_type,
                severity=drift_result.severity
            ).inc()
            
            # Schedule alert if significant drift
            if drift_result.severity in ['high', 'critical']:
                background_tasks.add_task(send_drift_alert, drift_result)
        
        logger.info(
            "Embedding processed",
            model_id=embedding_input.model_id,
            drift_detected=drift_result.drift_detected,
            drift_score=drift_result.drift_score,
            processing_time=processing_time
        )
        
        return DriftResponse(
            model_id=embedding_input.model_id,
            drift_detected=drift_result.drift_detected,
            drift_score=drift_result.drift_score,
            drift_type=drift_result.drift_type,
            severity=drift_result.severity,
            timestamp=embedding_input.timestamp,
            details=drift_result.details
        )
        
    except (ModelNotFoundError, InvalidEmbeddingError) as e:
        logger.warning("Validation error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except DatabaseError as e:
        logger.error("Database error during embedding ingestion", error=str(e))
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as e:
        logger.error("Unexpected error during embedding ingestion", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/models/{model_id}/status", response_model=ModelStatus)
async def get_model_status(
    model_id: str,
    db: DatabaseManager = Depends(get_db_manager)
) -> ModelStatus:
    """Get model monitoring status."""
    try:
        model_config = await db.get_model(model_id)
        if not model_config:
            raise HTTPException(status_code=404, detail="Model not found")
        
        stats = await db.get_model_stats(model_id)
        
        return ModelStatus(
            model_id=model_id,
            is_active=model_config['is_active'],
            embedding_count=stats['embedding_count'],
            last_drift_check=stats.get('last_drift_check'),
            drift_threshold=model_config['drift_threshold'],
            baseline_complete=stats['embedding_count'] >= model_config['baseline_size']
        )
        
    except DatabaseError as e:
        logger.error("Database error getting model status", error=str(e))
        raise HTTPException(status_code=500, detail="Database error")


@app.get("/models", response_model=List[ModelStatus])
async def list_models(
    db: DatabaseManager = Depends(get_db_manager)
) -> List[ModelStatus]:
    """List all registered models."""
    try:
        models = await db.list_models()
        
        model_statuses = []
        for model in models:
            stats = await db.get_model_stats(model['model_id'])
            model_statuses.append(ModelStatus(
                model_id=model['model_id'],
                is_active=model['is_active'],
                embedding_count=stats['embedding_count'],
                last_drift_check=stats.get('last_drift_check'),
                drift_threshold=model['drift_threshold'],
                baseline_complete=stats['embedding_count'] >= model['baseline_size']
            ))
        
        return model_statuses
        
    except DatabaseError as e:
        logger.error("Database error listing models", error=str(e))
        raise HTTPException(status_code=500, detail="Database error")


@app.get("/models/{model_id}/drift/history")
async def get_drift_history(
    model_id: str,
    limit: int = 100,
    db: DatabaseManager = Depends(get_db_manager)
) -> List[Dict[str, Any]]:
    """Get drift detection history for a model."""
    try:
        model_config = await db.get_model(model_id)
        if not model_config:
            raise HTTPException(status_code=404, detail="Model not found")
        
        history = await db.get_drift_history(model_id, limit)
        return history
        
    except DatabaseError as e:
        logger.error("Database error getting drift history", error=str(e))
        raise HTTPException(status_code=500, detail="Database error")


@app.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    db: DatabaseManager = Depends(get_db_manager),
    cache: CacheManager = Depends(get_cache_manager)
) -> Dict[str, str]:
    """Delete a model and all its data."""
    try:
        model_config = await db.get_model(model_id)
        if not model_config:
            raise HTTPException(status_code=404, detail="Model not found")
        
        await db.delete_model(model_id)
        await cache.clear_model_cache(model_id)
        
        ACTIVE_MODELS.dec()
        EMBEDDING_DIMENSIONS.labels(model_id=model_id).set(0)
        
        logger.info("Model deleted", model_id=model_id)
        return {"message": f"Model {model_id} deleted successfully"}
        
    except DatabaseError as e:
        logger.error("Database error deleting model", error=str(e))
        raise HTTPException(status_code=500, detail="Database error")


@app.get("/health", response_model=HealthResponse)
async def health_check(
    db: DatabaseManager = Depends(get_db_manager),
    cache: CacheManager = Depends(get_cache_manager)
) -> HealthResponse:
    """Health check endpoint."""
    try:
        # Check database health
        db_healthy = await db.health_check()
        
        # Check cache health
        cache_healthy = await cache.health_check()
        
        # Get active model count
        models = await db.list_models()
        active_count = sum(1 for m in models if m['is_active'])
        
        status = "healthy" if db_healthy and cache_healthy else "unhealthy"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now(timezone.utc),
            database_healthy=db_healthy,
            cache_healthy=cache_healthy,
            active_models=active_count
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(timezone.utc),
            database_healthy=False,
            cache_healthy=False,
            active_models=0
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def periodic_drift_check():
    """Background task for periodic drift checking."""
    while True:
        try:
            logger.debug("Running periodic drift check")
            
            # This would implement batch drift detection logic
            # for models that haven't been checked recently
            
            await asyncio.sleep(300)  # Check every 5 minutes
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Error in periodic drift check", error=str(e))
            await asyncio.sleep(60)  # Wait before retrying


async def periodic_cleanup():
    """Background task for data cleanup."""
    while True:
        try:
            logger.debug("Running periodic cleanup")
            
            # This would implement cleanup logic for old embeddings
            # and drift detection records
            
            await asyncio.sleep(3600)  # Cleanup every hour
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Error in periodic cleanup", error=str(e))
            await asyncio.sleep(300)  # Wait before retrying


async def send_drift_alert(drift_result: DriftResult):
    """Send alert for significant drift detection."""
    try:
        logger.warning(
            "Significant drift detected",
            model_id=drift_result.model_id,
            drift_score=drift_result.drift_score,
            severity=drift_result.severity
        )
        
        # This would implement actual alerting logic
        # (email, Slack, PagerDuty, etc.)
        
    except Exception as e:
        logger.error("Error sending drift alert", error=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)