import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.monitoring.vector_health import VectorHealthChecker


class TestVectorHealthChecker:
    def setup_method(self):
        self.health_checker = VectorHealthChecker(
            dimension_threshold=0.1,
            magnitude_threshold=2.0,
            nan_threshold=0.05
        )
    
    def test_check_dimension_consistency_valid(self):
        embeddings = [np.random.rand(128) for _ in range(10)]
        result = self.health_checker.check_dimension_consistency(embeddings)
        assert result['valid'] is True
        assert result['dimension'] == 128
    
    def test_check_dimension_consistency_mismatch(self):
        embeddings = [np.random.rand(128), np.random.rand(64)]
        result = self.health_checker.check_dimension_consistency(embeddings)
        assert result['valid'] is False
        assert 'dimension_mismatch' in result['issues']
    
    def test_check_magnitude_health_normal(self):
        embeddings = [np.random.normal(0, 1, 128) for _ in range(10)]
        result = self.health_checker.check_magnitude_health(embeddings)
        assert result['valid'] is True
        assert 'mean_magnitude' in result
    
    def test_check_magnitude_health_outlier(self):
        embeddings = [np.random.normal(0, 1, 128) for _ in range(9)]
        embeddings.append(np.ones(128) * 10)  # Large outlier
        result = self.health_checker.check_magnitude_health(embeddings)
        assert result['valid'] is False
        assert 'magnitude_outliers' in result['issues']
    
    def test_check_nan_inf_health_clean(self):
        embeddings = [np.random.rand(128) for _ in range(10)]
        result = self.health_checker.check_nan_inf_health(embeddings)
        assert result['valid'] is True
        assert result['nan_count'] == 0
    
    def test_check_nan_inf_health_with_nans(self):
        embeddings = [np.random.rand(128) for _ in range(8)]
        bad_embedding = np.random.rand(128)
        bad_embedding[0] = np.nan
        bad_embedding[1] = np.inf
        embeddings.extend([bad_embedding, bad_embedding])
        
        result = self.health_checker.check_nan_inf_health(embeddings)
        assert result['valid'] is False
        assert result['nan_count'] > 0
        assert 'nan_inf_detected' in result['issues']
    
    def test_comprehensive_health_check(self):
        # Valid embeddings
        embeddings = [np.random.normal(0, 1, 128) for _ in range(10)]
        result = self.health_checker.comprehensive_health_check(embeddings)
        
        assert result['overall_health'] is True
        assert 'dimension_check' in result
        assert 'magnitude_check' in result
        assert 'nan_check' in result
    
    def test_comprehensive_health_check_unhealthy(self):
        # Mix of problems
        embeddings = [np.random.rand(128), np.random.rand(64)]  # Dimension mismatch
        result = self.health_checker.comprehensive_health_check(embeddings)
        
        assert result['overall_health'] is False
        assert len(result['issues']) > 0