from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import numpy as np
from ..core.embedding_store import EmbeddingStore
from ..monitoring.similarity_tracker import SimilarityTracker
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/similarity", tags=["similarity"])

class SimilarityRequest(BaseModel):
    embeddings: List[List[float]] = Field(..., min_items=2, max_items=100)
    threshold: float = Field(0.8, ge=0.0, le=1.0)
    method: str = Field("cosine", regex="^(cosine|euclidean|dot_product)$")

class SimilarityResponse(BaseModel):
    similarity_matrix: List[List[float]]
    outliers: List[int]
    mean_similarity: float
    drift_detected: bool

async def get_embedding_store() -> EmbeddingStore:
    return EmbeddingStore()

@router.post("/compare", response_model=SimilarityResponse)
async def compare_embeddings(
    request: SimilarityRequest,
    store: EmbeddingStore = Depends(get_embedding_store)
) -> SimilarityResponse:
    """Compare multiple embeddings and detect outliers."""
    try:
        embeddings = np.array(request.embeddings)
        
        # Validate embedding dimensions
        if embeddings.shape[1] == 0:
            raise HTTPException(400, "Empty embeddings provided")
            
        tracker = SimilarityTracker()
        similarity_matrix = tracker.compute_similarity_matrix(
            embeddings, method=request.method
        )
        
        # Find outliers based on mean similarity
        mean_similarities = np.mean(similarity_matrix, axis=1)
        outliers = np.where(mean_similarities < request.threshold)[0].tolist()
        
        # Check for drift
        mean_sim = float(np.mean(similarity_matrix))
        drift_detected = mean_sim < request.threshold
        
        logger.info(f"Compared {len(embeddings)} embeddings, found {len(outliers)} outliers")
        
        return SimilarityResponse(
            similarity_matrix=similarity_matrix.tolist(),
            outliers=outliers,
            mean_similarity=mean_sim,
            drift_detected=drift_detected
        )
        
    except ValueError as e:
        logger.error(f"Invalid embedding data: {e}")
        raise HTTPException(400, f"Invalid embedding data: {str(e)}")
    except Exception as e:
        logger.error(f"Similarity comparison failed: {e}")
        raise HTTPException(500, "Internal server error")