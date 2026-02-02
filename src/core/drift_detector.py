"""Core drift detection algorithms for embedding monitoring."""

import asyncio
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time

import numpy as np
import structlog
from scipy.spatial.distance import cosine, jensenshannon
from scipy.stats import entropy, ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import redis.asyncio as redis

from ..exceptions import DriftDetectionError, InsufficientDataError
from ..models.drift_result import DriftResult, DriftSeverity, DriftType


logger = structlog.get_logger(__name__)


class DriftAlgorithm(str, Enum):
    """Available drift detection algorithms."""
    
    KL_DIVERGENCE = "kl_divergence"
    JS_DIVERGENCE = "js_divergence"
    COSINE_SIMILARITY = "cosine_similarity"
    WASSERSTEIN = "wasserstein"
    KOLMOGOROV_SMIRNOV = "ks_test"
    PCA_RECONSTRUCTION = "pca_reconstruction"


@dataclass
class DriftThresholds:
    """Configurable thresholds for drift detection."""
    
    warning: float
    critical: float
    severe: float
    
    def __post_init__(self) -> None:
        """Validate threshold ordering."""
        if not (0 < self.warning < self.critical < self.severe):
            raise ValueError("Thresholds must be ordered: 0 < warning < critical < severe")


@dataclass
class DriftMetrics:
    """Container for drift detection metrics."""
    
    algorithm: DriftAlgorithm
    score: float
    baseline_size: int
    current_size: int
    computation_time_ms: float
    metadata: Optional[Dict[str, Union[str, float, int]]] = None


class EmbeddingDriftDetector:
    """Production-grade drift detection for embedding vectors."""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        default_thresholds: Optional[Dict[DriftAlgorithm, DriftThresholds]] = None,
        pca_components: int = 50,
        min_baseline_size: int = 1000,
        min_current_size: int = 100,
    ) -> None:
        """Initialize drift detector with configuration.
        
        Args:
            redis_client: Redis client for caching embeddings and results
            default_thresholds: Per-algorithm drift thresholds
            pca_components: Number of PCA components for dimensionality reduction
            min_baseline_size: Minimum baseline samples required
            min_current_size: Minimum current samples required
        """
        self.redis = redis_client
        self.pca_components = pca_components
        self.min_baseline_size = min_baseline_size
        self.min_current_size = min_current_size
        
        self.thresholds = default_thresholds or self._default_thresholds()
        self.logger = logger.bind(component="drift_detector")
        
        # Cache for fitted PCA models per embedding space
        self._pca_cache: Dict[str, PCA] = {}
    
    def _default_thresholds(self) -> Dict[DriftAlgorithm, DriftThresholds]:
        """Default drift thresholds per algorithm."""
        return {
            DriftAlgorithm.KL_DIVERGENCE: DriftThresholds(0.1, 0.3, 0.5),
            DriftAlgorithm.JS_DIVERGENCE: DriftThresholds(0.05, 0.15, 0.25),
            DriftAlgorithm.COSINE_SIMILARITY: DriftThresholds(0.1, 0.2, 0.3),
            DriftAlgorithm.WASSERSTEIN: DriftThresholds(0.1, 0.3, 0.5),
            DriftAlgorithm.KOLMOGOROV_SMIRNOV: DriftThresholds(0.05, 0.1, 0.2),
            DriftAlgorithm.PCA_RECONSTRUCTION: DriftThresholds(0.1, 0.25, 0.4),
        }
    
    async def detect_drift(
        self,
        baseline_embeddings: np.ndarray,
        current_embeddings: np.ndarray,
        embedding_space_id: str,
        algorithms: Optional[List[DriftAlgorithm]] = None,
    ) -> List[DriftResult]:
        """Detect drift across multiple algorithms.
        
        Args:
            baseline_embeddings: Reference embeddings (N, D)
            current_embeddings: Current embeddings to compare (M, D)
            embedding_space_id: Unique identifier for this embedding space
            algorithms: Algorithms to run (all if None)
            
        Returns:
            List of drift detection results
            
        Raises:
            InsufficientDataError: Not enough samples for analysis
            DriftDetectionError: Algorithm execution error
        """
        self._validate_inputs(baseline_embeddings, current_embeddings)
        
        if algorithms is None:
            algorithms = list(DriftAlgorithm)
        
        results = []
        tasks = []
        
        for algorithm in algorithms:
            task = self._run_algorithm(
                algorithm,
                baseline_embeddings,
                current_embeddings,
                embedding_space_id,
            )
            tasks.append(task)
        
        metrics_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        for algorithm, metrics_or_error in zip(algorithms, metrics_list):
            if isinstance(metrics_or_error, Exception):
                self.logger.error(
                    "Algorithm failed",
                    algorithm=algorithm.value,
                    error=str(metrics_or_error),
                    embedding_space_id=embedding_space_id,
                )
                continue
            
            metrics = metrics_or_error
            severity = self._determine_severity(algorithm, metrics.score)
            
            result = DriftResult(
                algorithm=algorithm,
                drift_type=self._determine_drift_type(algorithm, metrics.score),
                severity=severity,
                score=metrics.score,
                threshold_warning=self.thresholds[algorithm].warning,
                threshold_critical=self.thresholds[algorithm].critical,
                threshold_severe=self.thresholds[algorithm].severe,
                baseline_size=metrics.baseline_size,
                current_size=metrics.current_size,
                computation_time_ms=metrics.computation_time_ms,
                embedding_space_id=embedding_space_id,
                metadata=metrics.metadata or {},
            )
            
            results.append(result)
            
            # Cache result for monitoring trends
            await self._cache_result(result)
        
        self.logger.info(
            "Drift detection completed",
            embedding_space_id=embedding_space_id,
            algorithms_run=len(results),
            drift_detected=sum(1 for r in results if r.severity != DriftSeverity.NONE),
        )
        
        return results
    
    async def _run_algorithm(
        self,
        algorithm: DriftAlgorithm,
        baseline: np.ndarray,
        current: np.ndarray,
        embedding_space_id: str,
    ) -> DriftMetrics:
        """Execute a specific drift detection algorithm."""
        start_time = time.perf_counter()
        
        try:
            if algorithm == DriftAlgorithm.KL_DIVERGENCE:
                score, metadata = await self._kl_divergence(baseline, current)
            elif algorithm == DriftAlgorithm.JS_DIVERGENCE:
                score, metadata = await self._js_divergence(baseline, current)
            elif algorithm == DriftAlgorithm.COSINE_SIMILARITY:
                score, metadata = await self._cosine_similarity_drift(baseline, current)
            elif algorithm == DriftAlgorithm.WASSERSTEIN:
                score, metadata = await self._wasserstein_distance(baseline, current)
            elif algorithm == DriftAlgorithm.KOLMOGOROV_SMIRNOV:
                score, metadata = await self._ks_test(baseline, current)
            elif algorithm == DriftAlgorithm.PCA_RECONSTRUCTION:
                score, metadata = await self._pca_reconstruction_error(
                    baseline, current, embedding_space_id
                )
            else:
                raise DriftDetectionError(f"Unknown algorithm: {algorithm}")
            
            computation_time = (time.perf_counter() - start_time) * 1000
            
            return DriftMetrics(
                algorithm=algorithm,
                score=score,
                baseline_size=baseline.shape[0],
                current_size=current.shape[0],
                computation_time_ms=computation_time,
                metadata=metadata,
            )
            
        except Exception as e:
            raise DriftDetectionError(
                f"Algorithm {algorithm.value} failed: {str(e)}"
            ) from e
    
    async def _kl_divergence(
        self, baseline: np.ndarray, current: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate KL divergence between embedding distributions."""
        # Reduce dimensionality for histogram-based comparison
        baseline_pca = self._fit_or_get_pca(baseline, "temp_kl")
        current_pca = baseline_pca.transform(current)
        baseline_pca = baseline_pca.transform(baseline)
        
        # Calculate per-dimension KL divergence
        kl_scores = []
        
        for dim in range(min(10, baseline_pca.shape[1])):  # Top 10 components
            base_hist, base_bins = np.histogram(
                baseline_pca[:, dim], bins=50, density=True
            )
            curr_hist, _ = np.histogram(
                current_pca[:, dim], bins=base_bins, density=True
            )
            
            # Add small epsilon to avoid log(0)
            base_hist += 1e-10
            curr_hist += 1e-10
            
            # Normalize to probabilities
            base_hist /= np.sum(base_hist)
            curr_hist /= np.sum(curr_hist)
            
            kl_score = entropy(curr_hist, base_hist)
            if not np.isfinite(kl_score):
                kl_score = 10.0  # Maximum drift score
            
            kl_scores.append(kl_score)
        
        mean_kl = np.mean(kl_scores)
        max_kl = np.max(kl_scores)
        
        return mean_kl, {
            "mean_kl_divergence": mean_kl,
            "max_kl_divergence": max_kl,
            "dimensions_analyzed": len(kl_scores),
        }
    
    async def _js_divergence(
        self, baseline: np.ndarray, current: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate Jensen-Shannon divergence."""
        # Similar to KL but more symmetric
        baseline_pca = self._fit_or_get_pca(baseline, "temp_js")
        current_pca = baseline_pca.transform(current)
        baseline_pca = baseline_pca.transform(baseline)
        
        js_scores = []
        
        for dim in range(min(10, baseline_pca.shape[1])):
            base_hist, base_bins = np.histogram(
                baseline_pca[:, dim], bins=50, density=True
            )
            curr_hist, _ = np.histogram(
                current_pca[:, dim], bins=base_bins, density=True
            )
            
            base_hist += 1e-10
            curr_hist += 1e-10
            
            base_hist /= np.sum(base_hist)
            curr_hist /= np.sum(curr_hist)
            
            js_score = jensenshannon(base_hist, curr_hist)
            js_scores.append(js_score)
        
        return np.mean(js_scores), {
            "mean_js_divergence": np.mean(js_scores),
            "max_js_divergence": np.max(js_scores),
        }
    
    async def _cosine_similarity_drift(
        self, baseline: np.ndarray, current: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Detect drift using cosine similarity between centroids."""
        baseline_centroid = np.mean(baseline, axis=0).reshape(1, -1)
        current_centroid = np.mean(current, axis=0).reshape(1, -1)
        
        # Calculate similarity between centroids
        similarity = cosine_similarity(baseline_centroid, current_centroid)[0, 0]
        drift_score = 1 - similarity  # Convert to drift score (higher = more drift)
        
        # Additional metrics
        baseline_std = np.std(baseline, axis=0)
        current_std = np.std(current, axis=0)
        variance_drift = np.mean(np.abs(baseline_std - current_std))
        
        return drift_score, {
            "centroid_cosine_similarity": similarity,
            "centroid_drift_score": drift_score,
            "variance_drift": variance_drift,
            "baseline_mean_norm": float(np.linalg.norm(baseline_centroid)),
            "current_mean_norm": float(np.linalg.norm(current_centroid)),
        }
    
    async def _wasserstein_distance(
        self, baseline: np.ndarray, current: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate Wasserstein distance between distributions."""
        # Calculate per-dimension Wasserstein distances
        baseline_pca = self._fit_or_get_pca(baseline, "temp_wasserstein")
        current_pca = baseline_pca.transform(current)
        baseline_pca = baseline_pca.transform(baseline)
        
        distances = []
        
        for dim in range(min(10, baseline_pca.shape[1])):
            distance = wasserstein_distance(
                baseline_pca[:, dim], current_pca[:, dim]
            )
            distances.append(distance)
        
        return np.mean(distances), {
            "mean_wasserstein": np.mean(distances),
            "max_wasserstein": np.max(distances),
        }
    
    async def _ks_test(
        self, baseline: np.ndarray, current: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Kolmogorov-Smirnov test for distribution differences."""
        baseline_pca = self._fit_or_get_pca(baseline, "temp_ks")
        current_pca = baseline_pca.transform(current)
        baseline_pca = baseline_pca.transform(baseline)
        
        ks_stats = []
        p_values = []
        
        for dim in range(min(10, baseline_pca.shape[1])):
            ks_stat, p_value = ks_2samp(
                baseline_pca[:, dim], current_pca[:, dim]
            )
            ks_stats.append(ks_stat)
            p_values.append(p_value)
        
        # Use maximum KS statistic as drift score
        max_ks_stat = np.max(ks_stats)
        min_p_value = np.min(p_values)
        
        return max_ks_stat, {
            "max_ks_statistic": max_ks_stat,
            "min_p_value": min_p_value,
            "mean_ks_statistic": np.mean(ks_stats),
            "significant_dimensions": sum(1 for p in p_values if p < 0.05),
        }
    
    async def _pca_reconstruction_error(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        embedding_space_id: str,
    ) -> Tuple[float, Dict[str, float]]:
        """Detect drift using PCA reconstruction error."""
        # Fit PCA on baseline data
        pca = self._fit_or_get_pca(baseline, embedding_space_id)
        
        # Transform and reconstruct current data
        current_transformed = pca.transform(current)
        current_reconstructed = pca.inverse_transform(current_transformed)
        
        # Calculate reconstruction error
        reconstruction_errors = np.mean(
            (current - current_reconstructed) ** 2, axis=1
        )
        mean_error = np.mean(reconstruction_errors)
        
        # Compare with baseline reconstruction error
        baseline_transformed = pca.transform(baseline)
        baseline_reconstructed = pca.inverse_transform(baseline_transformed)
        baseline_errors = np.mean(
            (baseline - baseline_reconstructed) ** 2, axis=1
        )
        baseline_mean_error = np.mean(baseline_errors)
        
        # Drift score is ratio of current to baseline error
        drift_score = mean_error / (baseline_mean_error + 1e-8)
        
        return drift_score, {
            "current_reconstruction_error": mean_error,
            "baseline_reconstruction_error": baseline_mean_error,
            "error_ratio": drift_score,
            "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
            "n_components": self.pca_components,
        }
    
    def _fit_or_get_pca(self, data: np.ndarray, cache_key: str) -> PCA:
        """Fit PCA model or retrieve from cache."""
        if cache_key in self._pca_cache:
            return self._pca_cache[cache_key]
        
        n_components = min(self.pca_components, data.shape[1], data.shape[0])
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(data)
        
        self._pca_cache[cache_key] = pca
        return pca
    
    def _validate_inputs(
        self, baseline: np.ndarray, current: np.ndarray
    ) -> None:
        """Validate input embeddings."""
        if baseline.shape[0] < self.min_baseline_size:
            raise InsufficientDataError(
                f"Baseline too small: {baseline.shape[0]} < {self.min_baseline_size}"
            )
        
        if current.shape[0] < self.min_current_size:
            raise InsufficientDataError(
                f"Current too small: {current.shape[0]} < {self.min_current_size}"
            )
        
        if baseline.shape[1] != current.shape[1]:
            raise DriftDetectionError(
                f"Dimension mismatch: baseline={baseline.shape[1]}, "
                f"current={current.shape[1]}"
            )
        
        if np.any(~np.isfinite(baseline)) or np.any(~np.isfinite(current)):
            raise DriftDetectionError("Embeddings contain invalid values (NaN/inf)")
    
    def _determine_severity(
        self, algorithm: DriftAlgorithm, score: float
    ) -> DriftSeverity:
        """Determine drift severity based on score and thresholds."""
        thresholds = self.thresholds[algorithm]
        
        if score >= thresholds.severe:
            return DriftSeverity.SEVERE
        elif score >= thresholds.critical:
            return DriftSeverity.CRITICAL
        elif score >= thresholds.warning:
            return DriftSeverity.WARNING
        else:
            return DriftSeverity.NONE
    
    def _determine_drift_type(
        self, algorithm: DriftAlgorithm, score: float
    ) -> DriftType:
        """Determine type of drift based on algorithm."""
        if score < self.thresholds[algorithm].warning:
            return DriftType.NONE
        
        # For now, classify all as COVARIATE_SHIFT
        # In production, would use more sophisticated classification
        return DriftType.COVARIATE_SHIFT
    
    async def _cache_result(self, result: DriftResult) -> None:
        """Cache drift detection result in Redis."""
        try:
            key = f"drift_result:{result.embedding_space_id}:{result.algorithm.value}"
            await self.redis.setex(
                key,
                3600,  # 1 hour TTL
                result.model_dump_json(),
            )
        except Exception as e:
            self.logger.warning(
                "Failed to cache drift result",
                error=str(e),
                embedding_space_id=result.embedding_space_id,
            )