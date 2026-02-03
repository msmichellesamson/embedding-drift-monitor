from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class SimilarityWindow:
    """Rolling window of embedding similarities"""
    timestamp: datetime
    mean_similarity: float
    std_similarity: float
    min_similarity: float
    max_similarity: float
    sample_count: int

class SimilarityTracker:
    """Track cosine similarity between consecutive embeddings"""
    
    def __init__(self, window_minutes: int = 60, max_windows: int = 24):
        self.window_minutes = window_minutes
        self.max_windows = max_windows
        self.windows: List[SimilarityWindow] = []
        self.current_embeddings: List[np.ndarray] = []
        self.last_window_time: Optional[datetime] = None
        
    def add_embedding(self, embedding: np.ndarray) -> Optional[float]:
        """Add new embedding and return similarity to previous batch"""
        now = datetime.utcnow()
        
        # Initialize first window
        if self.last_window_time is None:
            self.last_window_time = now
            self.current_embeddings.append(embedding)
            return None
            
        # Check if we need to create new window
        if now - self.last_window_time >= timedelta(minutes=self.window_minutes):
            self._close_current_window(now)
            
        self.current_embeddings.append(embedding)
        
        # Calculate similarity to previous embeddings in current window
        if len(self.current_embeddings) > 1:
            similarities = cosine_similarity(
                [embedding], 
                self.current_embeddings[:-1]
            )[0]
            return float(np.mean(similarities))
            
        return None
        
    def _close_current_window(self, current_time: datetime) -> None:
        """Close current window and calculate statistics"""
        if len(self.current_embeddings) < 2:
            logger.warning(f"Skipping window with {len(self.current_embeddings)} embeddings")
            self._reset_window(current_time)
            return
            
        # Calculate all pairwise similarities within window
        similarities = []
        for i in range(len(self.current_embeddings)):
            for j in range(i + 1, len(self.current_embeddings)):
                sim = cosine_similarity(
                    [self.current_embeddings[i]], 
                    [self.current_embeddings[j]]
                )[0][0]
                similarities.append(sim)
                
        if similarities:
            window = SimilarityWindow(
                timestamp=self.last_window_time,
                mean_similarity=float(np.mean(similarities)),
                std_similarity=float(np.std(similarities)),
                min_similarity=float(np.min(similarities)),
                max_similarity=float(np.max(similarities)),
                sample_count=len(self.current_embeddings)
            )
            
            self.windows.append(window)
            
            # Keep only recent windows
            if len(self.windows) > self.max_windows:
                self.windows = self.windows[-self.max_windows:]
                
            logger.info(f"Window closed: mean_sim={window.mean_similarity:.3f}, samples={window.sample_count}")
            
        self._reset_window(current_time)
        
    def _reset_window(self, current_time: datetime) -> None:
        """Reset for new window"""
        self.current_embeddings = []
        self.last_window_time = current_time
        
    def get_similarity_trend(self, hours: int = 6) -> Tuple[float, str]:
        """Get similarity trend over last N hours"""
        if len(self.windows) < 2:
            return 0.0, "insufficient_data"
            
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_windows = [w for w in self.windows if w.timestamp >= cutoff]
        
        if len(recent_windows) < 2:
            return 0.0, "insufficient_recent_data"
            
        # Simple linear trend
        similarities = [w.mean_similarity for w in recent_windows]
        x = np.arange(len(similarities))
        slope = np.polyfit(x, similarities, 1)[0]
        
        if slope < -0.01:
            return slope, "decreasing"
        elif slope > 0.01:
            return slope, "increasing"
        else:
            return slope, "stable"
            
    def get_current_stats(self) -> Dict[str, float]:
        """Get current window statistics"""
        if not self.windows:
            return {}
            
        latest = self.windows[-1]
        return {
            "mean_similarity": latest.mean_similarity,
            "std_similarity": latest.std_similarity,
            "min_similarity": latest.min_similarity,
            "max_similarity": latest.max_similarity,
            "sample_count": float(latest.sample_count)
        }