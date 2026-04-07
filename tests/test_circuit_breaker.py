import pytest
import time
from unittest.mock import AsyncMock, patch
from src.core.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    @pytest.fixture
    def circuit_breaker(self):
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            expected_exception=ValueError
        )

    @pytest.mark.asyncio
    async def test_success_keeps_circuit_closed(self, circuit_breaker):
        mock_func = AsyncMock(return_value="success")
        
        result = await circuit_breaker.call(mock_func)
        
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_failures_open_circuit(self, circuit_breaker):
        mock_func = AsyncMock(side_effect=ValueError("test error"))
        
        # First two failures should keep circuit closed
        for _ in range(2):
            with pytest.raises(ValueError):
                await circuit_breaker.call(mock_func)
            assert circuit_breaker.state == CircuitState.CLOSED
        
        # Third failure should open circuit
        with pytest.raises(ValueError):
            await circuit_breaker.call(mock_func)
        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self, circuit_breaker):
        # Force circuit open
        circuit_breaker.failure_count = 3
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.next_attempt = time.time() + 60
        
        mock_func = AsyncMock()
        
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await circuit_breaker.call(mock_func)
        
        mock_func.assert_not_called()

    @pytest.mark.asyncio
    async def test_half_open_recovery(self, circuit_breaker):
        # Force circuit to half-open state
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.next_attempt = time.time() - 1
        
        mock_func = AsyncMock(return_value="recovered")
        
        result = await circuit_breaker.call(mock_func)
        
        assert result == "recovered"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self, circuit_breaker):
        # Force circuit to half-open state
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.next_attempt = time.time() - 1
        
        mock_func = AsyncMock(side_effect=ValueError("still failing"))
        
        with pytest.raises(ValueError):
            await circuit_breaker.call(mock_func)
        
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.next_attempt > time.time()

    def test_reset_circuit_breaker(self, circuit_breaker):
        # Set up broken state
        circuit_breaker.failure_count = 5
        circuit_breaker.state = CircuitState.OPEN
        
        circuit_breaker.reset()
        
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.next_attempt == 0

    @pytest.mark.asyncio
    async def test_unexpected_exception_not_counted(self, circuit_breaker):
        mock_func = AsyncMock(side_effect=RuntimeError("unexpected"))
        
        with pytest.raises(RuntimeError):
            await circuit_breaker.call(mock_func)
        
        # Should not count towards failure threshold
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.state == CircuitState.CLOSED