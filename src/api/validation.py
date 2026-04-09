"""Embedding validation API endpoints."""

from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from ..monitoring.dimension_validator import DimensionValidator
from ..monitoring.vector_health import VectorHealthChecker

router = APIRouter(prefix="/validate", tags=["validation"])

class EmbeddingValidationRequest(BaseModel):
    embedding: List[float] = Field(..., description="Embedding vector to validate")
    model_name: str = Field(..., description="Model name for dimension validation")
    expected_dimension: Optional[int] = Field(None, description="Expected vector dimension")

class EmbeddingValidationResponse(BaseModel):
    is_valid: bool
    dimension_check: bool
    health_check: bool
    errors: List[str]
    warnings: List[str]
    vector_stats: Dict[str, float]

@router.post("/embedding", response_model=EmbeddingValidationResponse)
async def validate_embedding(request: EmbeddingValidationRequest):
    """Validate a single embedding vector."""
    try:
        vector = np.array(request.embedding)
        validator = DimensionValidator()
        health_checker = VectorHealthChecker()
        
        errors = []
        warnings = []
        
        # Dimension validation
        dim_valid = validator.validate_dimension(vector, request.expected_dimension)
        if not dim_valid and request.expected_dimension:
            errors.append(f"Dimension mismatch: got {len(vector)}, expected {request.expected_dimension}")
        
        # Vector health checks
        health_result = health_checker.check_vector_health(vector)
        health_valid = health_result["is_healthy"]
        
        if not health_valid:
            errors.extend(health_result["issues"])
        
        if health_result["warnings"]:
            warnings.extend(health_result["warnings"])
        
        # Calculate vector statistics
        stats = {
            "norm": float(np.linalg.norm(vector)),
            "mean": float(np.mean(vector)),
            "std": float(np.std(vector)),
            "min": float(np.min(vector)),
            "max": float(np.max(vector)),
            "zero_ratio": float(np.sum(vector == 0) / len(vector))
        }
        
        return EmbeddingValidationResponse(
            is_valid=dim_valid and health_valid,
            dimension_check=dim_valid,
            health_check=health_valid,
            errors=errors,
            warnings=warnings,
            vector_stats=stats
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")

@router.get("/health")
async def validation_health():
    """Health check for validation service."""
    return {"status": "healthy", "service": "embedding-validation"}