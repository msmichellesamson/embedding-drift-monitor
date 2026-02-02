"""
Statistical tests for embedding drift detection and analysis.

Implements Kolmogorov-Smirnov, Wasserstein distance, and population stability index
tests for detecting distribution drift in high-dimensional embedding spaces.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.spatial.distance import wasserstein_distance
import structlog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)


class DriftTestError(Exception):
    """Base exception for drift test errors."""
    pass


class InsufficientDataError(DriftTestError):
    """Raised when insufficient data for statistical testing."""
    pass


class DimensionalityError(DriftTestError):
    """Raised when embedding dimensions don't match."""
    pass


class TestType(Enum):
    """Statistical test types for drift detection."""
    KOLMOGOROV_SMIRNOV = "ks_test"
    WASSERSTEIN = "wasserstein"
    POPULATION_STABILITY_INDEX = "psi"
    MAXIMUM_MEAN_DISCREPANCY = "mmd"
    ENERGY_DISTANCE = "energy"


@dataclass
class StatisticalTestResult:
    """Result of a statistical drift test."""
    test_type: TestType
    statistic: float
    p_value: Optional[float]
    critical_value: Optional[float]
    is_drift: bool
    confidence_level: float
    sample_size_reference: int
    sample_size_current: int
    effect_size: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConfidenceInterval:
    """Confidence interval for test statistics."""
    lower_bound: float
    upper_bound: float
    confidence_level: float
    estimate: float


class StatisticalDriftTester:
    """Production-grade statistical drift detection for embeddings."""
    
    def __init__(
        self,
        min_sample_size: int = 100,
        max_dimensions_full: int = 100,
        pca_components: int = 50,
        random_state: int = 42
    ) -> None:
        """
        Initialize statistical drift tester.
        
        Args:
            min_sample_size: Minimum samples required for testing
            max_dimensions_full: Max dimensions for full testing (PCA above this)
            pca_components: PCA components for dimensionality reduction
            random_state: Random seed for reproducibility
        """
        self.min_sample_size = min_sample_size
        self.max_dimensions_full = max_dimensions_full
        self.pca_components = pca_components
        self.random_state = random_state
        self._pca_fitted = False
        self._pca = PCA(n_components=pca_components, random_state=random_state)
        self._scaler = StandardScaler()
        
        logger.info(
            "initialized_statistical_tester",
            min_sample_size=min_sample_size,
            max_dimensions_full=max_dimensions_full,
            pca_components=pca_components
        )

    def _validate_inputs(
        self,
        reference: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64]
    ) -> None:
        """Validate input arrays for statistical testing."""
        if reference.shape[0] < self.min_sample_size:
            raise InsufficientDataError(
                f"Reference data has {reference.shape[0]} samples, "
                f"minimum {self.min_sample_size} required"
            )
        
        if current.shape[0] < self.min_sample_size:
            raise InsufficientDataError(
                f"Current data has {current.shape[0]} samples, "
                f"minimum {self.min_sample_size} required"
            )
        
        if reference.shape[1] != current.shape[1]:
            raise DimensionalityError(
                f"Dimension mismatch: reference {reference.shape[1]}, "
                f"current {current.shape[1]}"
            )

    def _reduce_dimensionality(
        self,
        reference: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Apply PCA dimensionality reduction if needed."""
        if reference.shape[1] <= self.max_dimensions_full:
            return reference, current
        
        logger.info(
            "applying_pca_reduction",
            original_dims=reference.shape[1],
            target_dims=self.pca_components
        )
        
        # Fit PCA on reference data if not already fitted
        if not self._pca_fitted:
            combined_data = np.vstack([reference, current])
            scaled_data = self._scaler.fit_transform(combined_data)
            self._pca.fit(scaled_data)
            self._pca_fitted = True
        
        # Transform both datasets
        ref_scaled = self._scaler.transform(reference)
        cur_scaled = self._scaler.transform(current)
        
        ref_reduced = self._pca.transform(ref_scaled)
        cur_reduced = self._pca.transform(cur_scaled)
        
        return ref_reduced, cur_reduced

    async def kolmogorov_smirnov_test(
        self,
        reference: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64],
        alpha: float = 0.05
    ) -> StatisticalTestResult:
        """
        Perform multivariate KS test on embeddings.
        
        Uses energy distance as the test statistic for high-dimensional data.
        """
        self._validate_inputs(reference, current)
        ref_reduced, cur_reduced = self._reduce_dimensionality(reference, current)
        
        # For multivariate data, use energy distance
        statistic, p_value = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: stats.energy_distance(ref_reduced, cur_reduced)
        )
        
        # Bootstrap critical value
        critical_value = await self._bootstrap_critical_value(
            ref_reduced, cur_reduced, alpha, "energy"
        )
        
        is_drift = statistic > critical_value
        
        # Calculate effect size (Cohen's d approximation for energy distance)
        pooled_std = np.sqrt(
            (np.var(ref_reduced, ddof=1) + np.var(cur_reduced, ddof=1)) / 2
        )
        effect_size = statistic / pooled_std if pooled_std > 0 else 0.0
        
        logger.info(
            "ks_test_completed",
            statistic=statistic,
            critical_value=critical_value,
            is_drift=is_drift,
            effect_size=effect_size
        )
        
        return StatisticalTestResult(
            test_type=TestType.KOLMOGOROV_SMIRNOV,
            statistic=statistic,
            p_value=None,  # Bootstrap test doesn't provide p-value directly
            critical_value=critical_value,
            is_drift=is_drift,
            confidence_level=1 - alpha,
            sample_size_reference=reference.shape[0],
            sample_size_current=current.shape[0],
            effect_size=effect_size,
            metadata={"dimensions_tested": ref_reduced.shape[1]}
        )

    async def wasserstein_distance_test(
        self,
        reference: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64],
        alpha: float = 0.05
    ) -> StatisticalTestResult:
        """
        Perform Wasserstein distance test for distribution drift.
        
        Projects to 1D using PCA for computational efficiency.
        """
        self._validate_inputs(reference, current)
        ref_reduced, cur_reduced = self._reduce_dimensionality(reference, current)
        
        # Project to 1D using first principal component for Wasserstein
        ref_1d = ref_reduced[:, 0]
        cur_1d = cur_reduced[:, 0]
        
        statistic = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: wasserstein_distance(ref_1d, cur_1d)
        )
        
        # Bootstrap critical value
        critical_value = await self._bootstrap_critical_value(
            ref_1d.reshape(-1, 1), cur_1d.reshape(-1, 1), alpha, "wasserstein"
        )
        
        is_drift = statistic > critical_value
        
        # Effect size based on standard deviations
        pooled_std = np.sqrt((np.var(ref_1d) + np.var(cur_1d)) / 2)
        effect_size = statistic / pooled_std if pooled_std > 0 else 0.0
        
        logger.info(
            "wasserstein_test_completed",
            statistic=statistic,
            critical_value=critical_value,
            is_drift=is_drift,
            effect_size=effect_size
        )
        
        return StatisticalTestResult(
            test_type=TestType.WASSERSTEIN,
            statistic=statistic,
            p_value=None,
            critical_value=critical_value,
            is_drift=is_drift,
            confidence_level=1 - alpha,
            sample_size_reference=reference.shape[0],
            sample_size_current=current.shape[0],
            effect_size=effect_size,
            metadata={"projection_dimension": "first_pc"}
        )

    async def population_stability_index(
        self,
        reference: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64],
        n_bins: int = 10
    ) -> StatisticalTestResult:
        """
        Calculate Population Stability Index for drift detection.
        
        Uses quantile-based binning for robust bin edges.
        """
        self._validate_inputs(reference, current)
        ref_reduced, cur_reduced = self._reduce_dimensionality(reference, current)
        
        # Calculate PSI for each dimension and take the maximum
        psi_values = []
        
        for dim in range(ref_reduced.shape[1]):
            ref_dim = ref_reduced[:, dim]
            cur_dim = cur_reduced[:, dim]
            
            # Create bins based on reference quantiles
            bin_edges = np.quantile(
                ref_dim, 
                np.linspace(0, 1, n_bins + 1)
            )
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
            # Calculate distributions
            ref_dist = np.histogram(ref_dim, bins=bin_edges)[0]
            cur_dist = np.histogram(cur_dim, bins=bin_edges)[0]
            
            # Normalize to probabilities
            ref_prob = ref_dist / np.sum(ref_dist)
            cur_prob = cur_dist / np.sum(cur_dist)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            ref_prob = np.maximum(ref_prob, epsilon)
            cur_prob = np.maximum(cur_prob, epsilon)
            
            # Calculate PSI
            psi = np.sum((cur_prob - ref_prob) * np.log(cur_prob / ref_prob))
            psi_values.append(psi)
        
        statistic = max(psi_values)
        
        # PSI thresholds: <0.1 no drift, 0.1-0.2 moderate, >0.2 significant
        is_drift = statistic > 0.1
        
        logger.info(
            "psi_test_completed",
            statistic=statistic,
            max_psi_dimension=np.argmax(psi_values),
            is_drift=is_drift,
            all_psi_values=psi_values[:5]  # Log first 5 dimensions
        )
        
        return StatisticalTestResult(
            test_type=TestType.POPULATION_STABILITY_INDEX,
            statistic=statistic,
            p_value=None,
            critical_value=0.1,
            is_drift=is_drift,
            confidence_level=0.95,  # Standard PSI confidence
            sample_size_reference=reference.shape[0],
            sample_size_current=current.shape[0],
            metadata={
                "n_bins": n_bins,
                "max_psi_dimension": int(np.argmax(psi_values)),
                "psi_interpretation": self._interpret_psi(statistic)
            }
        )

    def _interpret_psi(self, psi_value: float) -> str:
        """Interpret PSI value."""
        if psi_value < 0.1:
            return "no_drift"
        elif psi_value < 0.2:
            return "moderate_drift"
        else:
            return "significant_drift"

    async def maximum_mean_discrepancy(
        self,
        reference: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64],
        alpha: float = 0.05,
        gamma: Optional[float] = None
    ) -> StatisticalTestResult:
        """
        Compute Maximum Mean Discrepancy with RBF kernel.
        
        Uses median heuristic for gamma if not provided.
        """
        self._validate_inputs(reference, current)
        ref_reduced, cur_reduced = self._reduce_dimensionality(reference, current)
        
        # Use median heuristic for gamma if not provided
        if gamma is None:
            combined = np.vstack([ref_reduced, cur_reduced])
            pairwise_dists = np.linalg.norm(
                combined[:, None] - combined[None, :], axis=2
            )
            gamma = 1.0 / (2 * np.median(pairwise_dists) ** 2)
        
        # Compute MMD
        def rbf_kernel(X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64]) -> float:
            """RBF kernel computation."""
            pairwise_sq_dists = np.linalg.norm(X[:, None] - Y[None, :], axis=2) ** 2
            return np.exp(-gamma * pairwise_sq_dists)
        
        # MMD computation
        K_xx = rbf_kernel(ref_reduced, ref_reduced)
        K_yy = rbf_kernel(cur_reduced, cur_reduced)
        K_xy = rbf_kernel(ref_reduced, cur_reduced)
        
        n = ref_reduced.shape[0]
        m = cur_reduced.shape[0]
        
        mmd_squared = (
            np.sum(K_xx) / (n * (n - 1)) +
            np.sum(K_yy) / (m * (m - 1)) -
            2 * np.sum(K_xy) / (n * m)
        )
        
        statistic = np.sqrt(max(0, mmd_squared))
        
        # Bootstrap critical value
        critical_value = await self._bootstrap_critical_value(
            ref_reduced, cur_reduced, alpha, "mmd", gamma=gamma
        )
        
        is_drift = statistic > critical_value
        
        logger.info(
            "mmd_test_completed",
            statistic=statistic,
            critical_value=critical_value,
            gamma=gamma,
            is_drift=is_drift
        )
        
        return StatisticalTestResult(
            test_type=TestType.MAXIMUM_MEAN_DISCREPANCY,
            statistic=statistic,
            p_value=None,
            critical_value=critical_value,
            is_drift=is_drift,
            confidence_level=1 - alpha,
            sample_size_reference=reference.shape[0],
            sample_size_current=current.shape[0],
            metadata={"gamma": gamma, "kernel": "rbf"}
        )

    async def _bootstrap_critical_value(
        self,
        reference: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64],
        alpha: float,
        test_type: str,
        n_bootstrap: int = 1000,
        **kwargs
    ) -> float:
        """Bootstrap critical value for drift tests."""
        combined = np.vstack([reference, current])
        n_ref = reference.shape[0]
        n_cur = current.shape[0]
        
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Permute combined data
            perm_indices = np.random.permutation(len(combined))
            perm_data = combined[perm_indices]
            
            # Split into bootstrap samples
            boot_ref = perm_data[:n_ref]
            boot_cur = perm_data[n_ref:n_ref + n_cur]
            
            # Compute test statistic
            if test_type == "energy":
                stat = stats.energy_distance(boot_ref, boot_cur)
            elif test_type == "wasserstein":
                stat = wasserstein_distance(boot_ref.flatten(), boot_cur.flatten())
            elif test_type == "mmd":
                gamma = kwargs.get("gamma", 1.0)
                # Simplified MMD for bootstrap
                stat = np.linalg.norm(np.mean(boot_ref, axis=0) - np.mean(boot_cur, axis=0))
            else:
                stat = 0.0
            
            bootstrap_stats.append(stat)
        
        # Return (1 - alpha) quantile
        critical_value = np.quantile(bootstrap_stats, 1 - alpha)
        
        logger.debug(
            "bootstrap_critical_value_computed",
            test_type=test_type,
            critical_value=critical_value,
            n_bootstrap=n_bootstrap,
            alpha=alpha
        )
        
        return critical_value

    async def compute_confidence_interval(
        self,
        data: npt.NDArray[np.float64],
        confidence_level: float = 0.95,
        method: str = "bootstrap"
    ) -> ConfidenceInterval:
        """
        Compute confidence interval for embedding statistics.
        
        Args:
            data: Embedding data
            confidence_level: Confidence level (0-1)
            method: CI computation method
        """
        if method == "bootstrap":
            return await self._bootstrap_confidence_interval(data, confidence_level)
        else:
            raise ValueError(f"Unsupported CI method: {method}")

    async def _bootstrap_confidence_interval(
        self,
        data: npt.NDArray[np.float64],
        confidence_level: float,
        n_bootstrap: int = 1000
    ) -> ConfidenceInterval:
        """Bootstrap confidence interval for mean embedding."""
        n_samples = data.shape[0]
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            boot_indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_sample = data[boot_indices]
            boot_mean = np.linalg.norm(np.mean(boot_sample, axis=0))
            bootstrap_means.append(boot_mean)
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)
        estimate = np.linalg.norm(np.mean(data, axis=0))
        
        return ConfidenceInterval(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            estimate=estimate
        )

    async def run_comprehensive_test_suite(
        self,
        reference: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64],
        alpha: float = 0.05
    ) -> Dict[TestType, StatisticalTestResult]:
        """Run comprehensive drift test suite."""
        logger.info(
            "starting_comprehensive_test_suite",
            reference_shape=reference.shape,
            current_shape=current.shape,
            alpha=alpha
        )
        
        results = {}
        
        # Run all tests concurrently
        test_tasks = [
            self.kolmogorov_smirnov_test(reference, current, alpha),
            self.wasserstein_distance_test(reference, current, alpha),
            self.population_stability_index(reference, current),
            self.maximum_mean_discrepancy(reference, current, alpha)
        ]
        
        test_results = await asyncio.gather(*test_tasks, return_exceptions=True)
        
        test_types = [
            TestType.KOLMOGOROV_SMIRNOV,
            TestType.WASSERSTEIN,
            TestType.POPULATION_STABILITY_INDEX,
            TestType.MAXIMUM_MEAN_DISCREPANCY
        ]
        
        for test_type, result in zip(test_types, test_results):
            if isinstance(result, Exception):
                logger.error(
                    "test_failed",
                    test_type=test_type.value,
                    error=str(result)
                )
                continue
            results[test_type] = result
        
        # Log summary
        drift_detected = sum(1 for r in results.values() if r.is_drift)
        logger.info(
            "comprehensive_test_completed",
            total_tests=len(results),
            drift_detected=drift_detected,
            consensus_drift=drift_detected >= len(results) // 2
        )
        
        return results