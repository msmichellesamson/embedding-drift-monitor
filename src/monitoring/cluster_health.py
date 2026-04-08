"""Embedding cluster health monitoring for detecting vector space degradation."""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ClusterHealth:
    silhouette_score: float
    cluster_count: int
    avg_intra_distance: float
    avg_inter_distance: float
    separation_ratio: float
    timestamp: float

class ClusterHealthMonitor:
    """Monitors embedding cluster health to detect vector space degradation."""
    
    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 10,
        silhouette_threshold: float = 0.3,
        separation_threshold: float = 1.5
    ):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.silhouette_threshold = silhouette_threshold
        self.separation_threshold = separation_threshold
        
    def analyze_cluster_health(
        self, 
        embeddings: np.ndarray,
        sample_size: int = 1000
    ) -> ClusterHealth:
        """Analyze embedding cluster health metrics."""
        if len(embeddings) < self.min_clusters:
            raise ValueError(f"Need at least {self.min_clusters} embeddings")
            
        # Sample for performance if dataset is large
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[indices]
        else:
            sample_embeddings = embeddings
            
        optimal_k = self._find_optimal_clusters(sample_embeddings)
        
        # Compute cluster metrics
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(sample_embeddings)
        
        silhouette = silhouette_score(sample_embeddings, labels)
        intra_dist = self._compute_intra_cluster_distance(sample_embeddings, labels, kmeans.cluster_centers_)
        inter_dist = self._compute_inter_cluster_distance(kmeans.cluster_centers_)
        separation_ratio = inter_dist / intra_dist if intra_dist > 0 else 0.0
        
        return ClusterHealth(
            silhouette_score=silhouette,
            cluster_count=optimal_k,
            avg_intra_distance=intra_dist,
            avg_inter_distance=inter_dist,
            separation_ratio=separation_ratio,
            timestamp=time.time()
        )
    
    def _find_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """Find optimal number of clusters using elbow method."""
        inertias = []
        k_range = range(self.min_clusters, min(self.max_clusters + 1, len(embeddings)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection
        if len(inertias) < 2:
            return self.min_clusters
            
        deltas = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
        optimal_idx = deltas.index(max(deltas))
        return k_range[optimal_idx + 1]
    
    def _compute_intra_cluster_distance(self, embeddings: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
        """Compute average intra-cluster distance."""
        distances = []
        for cluster_id in range(len(centers)):
            cluster_points = embeddings[labels == cluster_id]
            if len(cluster_points) > 0:
                cluster_distances = np.linalg.norm(cluster_points - centers[cluster_id], axis=1)
                distances.extend(cluster_distances)
        return np.mean(distances) if distances else 0.0
    
    def _compute_inter_cluster_distance(self, centers: np.ndarray) -> float:
        """Compute average inter-cluster distance."""
        if len(centers) < 2:
            return 0.0
        distances = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                distances.append(np.linalg.norm(centers[i] - centers[j]))
        return np.mean(distances)
    
    def is_healthy(self, health: ClusterHealth) -> Tuple[bool, List[str]]:
        """Check if cluster health is within acceptable thresholds."""
        issues = []
        
        if health.silhouette_score < self.silhouette_threshold:
            issues.append(f"Low silhouette score: {health.silhouette_score:.3f}")
            
        if health.separation_ratio < self.separation_threshold:
            issues.append(f"Poor cluster separation: {health.separation_ratio:.3f}")
            
        return len(issues) == 0, issues
