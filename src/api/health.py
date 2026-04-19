from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
from ..monitoring.vector_health import VectorHealthMonitor
from ..core.embedding_store import EmbeddingStore
from ..core.redis_cache import RedisCache

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["health"])

@router.get("/embeddings")
async def get_embedding_health() -> Dict[str, Any]:
    """Get real-time embedding health metrics."""
    try:
        store = EmbeddingStore()
        cache = RedisCache()
        monitor = VectorHealthMonitor(store, cache)
        
        health_metrics = await monitor.get_health_summary()
        
        return {
            "status": "healthy" if health_metrics["overall_score"] > 0.7 else "degraded",
            "overall_score": health_metrics["overall_score"],
            "metrics": {
                "dimension_consistency": health_metrics["dimension_health"],
                "magnitude_stability": health_metrics["magnitude_health"],
                "cluster_coherence": health_metrics["cluster_health"],
                "null_ratio": health_metrics["null_ratio"]
            },
            "alerts": health_metrics.get("active_alerts", []),
            "last_updated": health_metrics["timestamp"]
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check unavailable")

@router.get("/embeddings/{model_id}")
async def get_model_health(model_id: str) -> Dict[str, Any]:
    """Get health metrics for specific model embeddings."""
    try:
        store = EmbeddingStore()
        cache = RedisCache()
        monitor = VectorHealthMonitor(store, cache)
        
        model_health = await monitor.get_model_health(model_id)
        if not model_health:
            raise HTTPException(status_code=404, detail="Model not found")
            
        return {
            "model_id": model_id,
            "status": "healthy" if model_health["score"] > 0.7 else "degraded",
            "score": model_health["score"],
            "drift_score": model_health["drift_score"],
            "quality_score": model_health["quality_score"],
            "last_embedding_time": model_health["last_updated"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        raise HTTPException(status_code=500, detail="Model health check unavailable")