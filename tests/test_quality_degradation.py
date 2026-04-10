"""Tests for quality degradation detector."""
import pytest
import numpy as np
from datetime import datetime
from src.monitoring.quality_degradation import QualityDegradationDetector, QualityMetrics

class TestQualityDegradationDetector:
    """Test quality degradation detection."""
    
    def test_compute_quality_metrics(self):
        """Test quality metrics computation."""
        detector = QualityDegradationDetector()
        
        # Create test embeddings
        embeddings = np.random.randn(100, 128)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize
        
        metrics = detector.compute_quality_metrics(embeddings)
        
        assert isinstance(metrics, QualityMetrics)
        assert 0.8 < metrics.avg_magnitude < 1.2  # Should be close to 1 for normalized
        assert 0 <= metrics.cosine_coherence <= 1
        assert metrics.dimension_variance > 0
        assert isinstance(metrics.timestamp, datetime)
    
    def test_empty_embeddings_error(self):
        """Test error handling for empty embeddings."""
        detector = QualityDegradationDetector()
        
        with pytest.raises(ValueError, match="Empty embeddings array"):
            detector.compute_quality_metrics(np.array([]))
    
    def test_set_baseline(self):
        """Test baseline setting."""
        detector = QualityDegradationDetector()
        embeddings = np.random.randn(50, 64)
        
        detector.set_baseline(embeddings)
        
        assert detector.baseline_metrics is not None
        assert detector.baseline_metrics.avg_magnitude > 0
    
    def test_detect_no_degradation(self):
        """Test detection with no degradation."""
        detector = QualityDegradationDetector()
        
        # Set baseline
        baseline_embeddings = np.random.randn(100, 64)
        detector.set_baseline(baseline_embeddings)
        
        # Test with similar embeddings
        test_embeddings = baseline_embeddings + np.random.randn(100, 64) * 0.01  # Small noise
        result = detector.detect_degradation(test_embeddings)
        
        assert not result["is_degraded"]
        assert result["degradation_score"] < 0.1
        assert "metrics" in result
    
    def test_detect_magnitude_degradation(self):
        """Test detection of magnitude degradation."""
        detector = QualityDegradationDetector(magnitude_threshold=0.2)
        
        # Set baseline with normal magnitudes
        baseline_embeddings = np.random.randn(50, 32)
        detector.set_baseline(baseline_embeddings)
        
        # Create degraded embeddings with much larger magnitudes
        degraded_embeddings = baseline_embeddings * 2.0
        result = detector.detect_degradation(degraded_embeddings)
        
        assert result["is_degraded"]
        assert result["metrics"]["magnitude_change"] > 0.2
    
    def test_detect_coherence_degradation(self):
        """Test detection of coherence degradation."""
        detector = QualityDegradationDetector(coherence_threshold=0.15)
        
        # Set baseline with coherent embeddings
        baseline = np.ones((50, 32)) + np.random.randn(50, 32) * 0.1
        detector.set_baseline(baseline)
        
        # Create incoherent embeddings
        degraded = np.random.randn(50, 32) * 5  # Random, no coherence
        result = detector.detect_degradation(degraded)
        
        assert result["is_degraded"]
    
    def test_no_baseline_error(self):
        """Test error when no baseline is set."""
        detector = QualityDegradationDetector()
        embeddings = np.random.randn(30, 16)
        
        with pytest.raises(ValueError, match="No baseline metrics set"):
            detector.detect_degradation(embeddings)
    
    def test_history_window_management(self):
        """Test that history window is maintained."""
        detector = QualityDegradationDetector(history_window=5)
        baseline = np.random.randn(20, 16)
        detector.set_baseline(baseline)
        
        # Add more metrics than window size
        for _ in range(10):
            test_embeddings = np.random.randn(20, 16)
            detector.detect_degradation(test_embeddings)
        
        assert len(detector.metrics_history) == 5  # Should be limited to window size
