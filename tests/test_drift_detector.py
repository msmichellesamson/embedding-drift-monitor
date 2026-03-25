import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.core.drift_detector import DriftDetector, DriftAlert
from src.analysis.statistical_tests import StatisticalTests


class TestDriftDetector:
    @pytest.fixture
    def detector(self):
        mock_embedding_store = Mock()
        return DriftDetector(
            embedding_store=mock_embedding_store,
            drift_threshold=0.15,
            min_samples=100
        )
    
    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing"""
        np.random.seed(42)
        return {
            'baseline': np.random.normal(0, 1, (200, 384)),
            'current': np.random.normal(0.1, 1.2, (150, 384)),  # Slight drift
            'drifted': np.random.normal(0.5, 1.5, (120, 384))   # Clear drift
        }
    
    def test_check_drift_no_drift(self, detector, sample_embeddings):
        """Test drift detection when no significant drift exists"""
        with patch.object(detector.embedding_store, 'get_baseline_embeddings') as mock_baseline:
            with patch.object(detector.embedding_store, 'get_recent_embeddings') as mock_recent:
                mock_baseline.return_value = sample_embeddings['baseline']
                mock_recent.return_value = sample_embeddings['baseline'][:100]  # Same distribution
                
                result = detector.check_drift(model_name='test_model')
                
                assert result is None  # No drift detected
    
    def test_check_drift_detected(self, detector, sample_embeddings):
        """Test drift detection when drift exists"""
        with patch.object(detector.embedding_store, 'get_baseline_embeddings') as mock_baseline:
            with patch.object(detector.embedding_store, 'get_recent_embeddings') as mock_recent:
                mock_baseline.return_value = sample_embeddings['baseline']
                mock_recent.return_value = sample_embeddings['drifted']
                
                result = detector.check_drift(model_name='test_model')
                
                assert result is not None
                assert isinstance(result, DriftAlert)
                assert result.model_name == 'test_model'
                assert result.drift_score > detector.drift_threshold
    
    def test_insufficient_samples(self, detector):
        """Test behavior when insufficient samples available"""
        with patch.object(detector.embedding_store, 'get_baseline_embeddings') as mock_baseline:
            with patch.object(detector.embedding_store, 'get_recent_embeddings') as mock_recent:
                mock_baseline.return_value = np.random.normal(0, 1, (50, 384))  # Too few
                mock_recent.return_value = np.random.normal(0, 1, (30, 384))    # Too few
                
                result = detector.check_drift(model_name='test_model')
                
                assert result is None
    
    def test_drift_alert_creation(self):
        """Test DriftAlert creation and properties"""
        alert = DriftAlert(
            model_name='test_model',
            drift_score=0.25,
            baseline_count=200,
            current_count=150,
            timestamp=datetime.now()
        )
        
        assert alert.model_name == 'test_model'
        assert alert.drift_score == 0.25
        assert alert.baseline_count == 200
        assert alert.current_count == 150
        assert isinstance(alert.timestamp, datetime)
