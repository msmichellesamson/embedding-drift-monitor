import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.analysis.statistical_tests import StatisticalTestRunner, TestResult, TestType


class TestStatisticalTestRunner:
    @pytest.fixture
    def test_runner(self):
        return StatisticalTestRunner()
    
    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings with known statistical properties."""
        baseline = np.random.normal(0, 1, (100, 512))
        current = np.random.normal(0.1, 1.1, (100, 512))  # Slight shift
        return baseline, current
    
    def test_ks_test_no_drift(self, test_runner):
        """Test KS test with identical distributions."""
        baseline = np.random.normal(0, 1, (100, 512))
        current = np.random.normal(0, 1, (100, 512))
        
        result = test_runner.run_ks_test(baseline, current)
        
        assert isinstance(result, TestResult)
        assert result.test_type == TestType.KS_TEST
        assert result.p_value > 0.05
        assert not result.is_significant
    
    def test_ks_test_with_drift(self, test_runner):
        """Test KS test detects significant drift."""
        baseline = np.random.normal(0, 1, (100, 512))
        current = np.random.normal(2, 1, (100, 512))  # Clear shift
        
        result = test_runner.run_ks_test(baseline, current)
        
        assert result.p_value < 0.05
        assert result.is_significant
        assert result.statistic > 0.1
    
    def test_mmd_test_calculation(self, test_runner, sample_embeddings):
        """Test MMD test computation."""
        baseline, current = sample_embeddings
        
        result = test_runner.run_mmd_test(baseline, current)
        
        assert result.test_type == TestType.MMD_TEST
        assert result.statistic >= 0
        assert 0 <= result.p_value <= 1
    
    def test_psi_calculation(self, test_runner):
        """Test PSI calculation with binned data."""
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.5, 1, 1000)  # Shifted
        
        result = test_runner.calculate_psi(baseline, current)
        
        assert result.test_type == TestType.PSI
        assert result.statistic > 0
        assert result.p_value is not None
    
    def test_batch_testing(self, test_runner, sample_embeddings):
        """Test running multiple statistical tests."""
        baseline, current = sample_embeddings
        
        results = test_runner.run_all_tests(baseline, current)
        
        assert len(results) >= 3
        assert all(isinstance(r, TestResult) for r in results)
        assert {r.test_type for r in results} == {TestType.KS_TEST, TestType.MMD_TEST, TestType.PSI}
    
    def test_empty_data_handling(self, test_runner):
        """Test graceful handling of empty data."""
        empty = np.array([])
        normal = np.random.normal(0, 1, (100, 512))
        
        with pytest.raises(ValueError, match="Empty embedding arrays"):
            test_runner.run_ks_test(empty, normal)
    
    def test_dimension_mismatch(self, test_runner):
        """Test handling of mismatched embedding dimensions."""
        baseline = np.random.normal(0, 1, (100, 512))
        current = np.random.normal(0, 1, (100, 256))
        
        with pytest.raises(ValueError, match="Embedding dimensions"):
            test_runner.run_mmd_test(baseline, current)
    
    def test_significance_threshold(self, test_runner):
        """Test custom significance threshold."""
        baseline = np.random.normal(0, 1, (100, 512))
        current = np.random.normal(0, 1, (100, 512))
        
        result = test_runner.run_ks_test(baseline, current, alpha=0.001)
        
        assert result.significance_threshold == 0.001