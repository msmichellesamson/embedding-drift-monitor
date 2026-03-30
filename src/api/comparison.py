from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from ..core.drift_detector import DriftDetector
from ..monitoring.metrics import MetricsCollector

router = APIRouter(prefix="/api/v1/comparison", tags=["comparison"])

class EmbeddingComparisonRequest(BaseModel):
    baseline_embeddings: List[List[float]]
    current_embeddings: List[List[float]]
    threshold: float = 0.1
    model_name: str

class ComparisonResult(BaseModel):
    drift_detected: bool
    drift_score: float
    statistical_distance: float
    similarity_metrics: Dict[str, float]
    recommendations: List[str]

@router.post("/detect", response_model=ComparisonResult)
async def compare_embeddings(request: EmbeddingComparisonRequest):
    """Compare baseline and current embeddings to detect drift."""
    try:
        detector = DriftDetector(threshold=request.threshold)
        metrics = MetricsCollector()
        
        baseline = np.array(request.baseline_embeddings)
        current = np.array(request.current_embeddings)
        
        if baseline.shape[1] != current.shape[1]:
            raise HTTPException(400, "Embedding dimensions must match")
        
        drift_result = detector.detect_drift(baseline, current)
        
        # Calculate additional similarity metrics
        cosine_sim = np.mean([np.dot(b, c) / (np.linalg.norm(b) * np.linalg.norm(c)) 
                             for b, c in zip(baseline[:10], current[:10])])
        
        recommendations = []
        if drift_result.drift_detected:
            recommendations.extend([
                "Consider retraining the model",
                "Investigate data distribution changes",
                "Check for upstream data quality issues"
            ])
        
        # Record metrics
        metrics.record_drift_check(request.model_name, drift_result.drift_detected)
        
        return ComparisonResult(
            drift_detected=drift_result.drift_detected,
            drift_score=drift_result.drift_score,
            statistical_distance=drift_result.statistical_distance,
            similarity_metrics={"cosine_similarity": float(cosine_sim)},
            recommendations=recommendations
        )
    
    except Exception as e:
        raise HTTPException(500, f"Comparison failed: {str(e)}")

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "embedding-comparison"}
