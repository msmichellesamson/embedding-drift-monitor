import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.monitoring.drift_scorer import DriftScorer


class TestDriftScorer:
    @pytest.fixture
    def scorer(self):
        return DriftScorer(threshold=0.15, window_size=100)

    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing"""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, (50, 384))
        drifted = np.random.normal(0.5, 1.2, (50, 384))  # Shifted distribution
        return baseline, drifted

    def test_calculate_kl_divergence(self, scorer, sample_embeddings):
        baseline, drifted = sample_embeddings
        
        # Same distribution should have low KL divergence
        kl_same = scorer._calculate_kl_divergence(baseline, baseline)
        assert kl_same < 0.1
        
        # Different distributions should have higher KL divergence
        kl_drift = scorer._calculate_kl_divergence(baseline, drifted)
        assert kl_drift > kl_same

    def test_calculate_wasserstein_distance(self, scorer, sample_embeddings):
        baseline, drifted = sample_embeddings
        
        # Same distribution should have low Wasserstein distance
        w_same = scorer._calculate_wasserstein_distance(baseline, baseline)
        assert w_same < 0.1
        
        # Different distributions should have higher distance
        w_drift = scorer._calculate_wasserstein_distance(baseline, drifted)
        assert w_drift > w_same

    def test_compute_drift_score(self, scorer, sample_embeddings):
        baseline, drifted = sample_embeddings
        
        # Test no drift scenario
        score_no_drift = scorer.compute_drift_score(baseline, baseline)
        assert 0 <= score_no_drift <= 1
        assert score_no_drift < scorer.threshold
        
        # Test drift scenario
        score_drift = scorer.compute_drift_score(baseline, drifted)
        assert 0 <= score_drift <= 1
        assert score_drift > score_no_drift

    def test_is_drift_detected(self, scorer, sample_embeddings):
        baseline, drifted = sample_embeddings
        
        # No drift case
        is_drift_same = scorer.is_drift_detected(baseline, baseline)
        assert is_drift_same is False
        
        # Drift case (may or may not trigger based on threshold)
        is_drift_different = scorer.is_drift_detected(baseline, drifted)
        assert isinstance(is_drift_different, bool)

    def test_empty_embeddings_handling(self, scorer):
        empty = np.array([]).reshape(0, 384)
        baseline = np.random.normal(0, 1, (10, 384))
        
        with pytest.raises(ValueError, match="Empty embedding arrays"):
            scorer.compute_drift_score(empty, baseline)
        
        with pytest.raises(ValueError, match="Empty embedding arrays"):
            scorer.compute_drift_score(baseline, empty)

    def test_dimension_mismatch_handling(self, scorer):
        baseline = np.random.normal(0, 1, (10, 384))
        mismatched = np.random.normal(0, 1, (10, 512))
        
        with pytest.raises(ValueError, match="Embedding dimensions must match"):
            scorer.compute_drift_score(baseline, mismatched)

    @pytest.mark.parametrize("threshold", [0.05, 0.1, 0.2, 0.5])
    def test_different_thresholds(self, threshold, sample_embeddings):
        scorer = DriftScorer(threshold=threshold, window_size=100)
        baseline, drifted = sample_embeddings
        
        score = scorer.compute_drift_score(baseline, drifted)
        is_drift = scorer.is_drift_detected(baseline, drifted)
        
        assert (score >= threshold) == is_drift

    def test_scorer_initialization(self):
        # Test default values
        scorer = DriftScorer()
        assert scorer.threshold == 0.1
        assert scorer.window_size == 1000
        
        # Test custom values
        scorer = DriftScorer(threshold=0.25, window_size=500)
        assert scorer.threshold == 0.25
        assert scorer.window_size == 500

    def test_invalid_initialization_parameters(self):
        with pytest.raises(ValueError, match="Threshold must be positive"):
            DriftScorer(threshold=-0.1)
        
        with pytest.raises(ValueError, match="Window size must be positive"):
            DriftScorer(window_size=0)