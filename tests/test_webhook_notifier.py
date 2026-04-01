import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp
from src.alerts.webhook_notifier import WebhookNotifier
from src.core.drift_detector import DriftAlert


@pytest.fixture
def drift_alert():
    return DriftAlert(
        metric_name="embedding_similarity",
        threshold=0.85,
        current_value=0.75,
        severity="HIGH",
        timestamp="2024-01-15T10:30:00Z",
        message="Embedding similarity dropped below threshold"
    )


@pytest.fixture
def webhook_config():
    return {
        "url": "https://api.example.com/webhooks/alerts",
        "timeout": 30,
        "headers": {"Authorization": "Bearer test-token"}
    }


class TestWebhookNotifier:
    @pytest.mark.asyncio
    async def test_send_success(self, webhook_config, drift_alert):
        notifier = WebhookNotifier(webhook_config)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await notifier.send(drift_alert)
            
            assert result is True
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert kwargs['json']['severity'] == 'HIGH'
            assert kwargs['json']['metric_name'] == 'embedding_similarity'

    @pytest.mark.asyncio
    async def test_send_with_retry_on_500(self, webhook_config, drift_alert):
        notifier = WebhookNotifier(webhook_config)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # First call fails with 500, second succeeds
            mock_response_fail = AsyncMock()
            mock_response_fail.status = 500
            mock_response_success = AsyncMock()
            mock_response_success.status = 200
            
            mock_post.return_value.__aenter__.side_effect = [
                mock_response_fail,
                mock_response_success
            ]
            
            with patch('asyncio.sleep'):
                result = await notifier.send(drift_alert)
            
            assert result is True
            assert mock_post.call_count == 2

    @pytest.mark.asyncio
    async def test_send_max_retries_exceeded(self, webhook_config, drift_alert):
        notifier = WebhookNotifier(webhook_config)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_post.return_value.__aenter__.return_value = mock_response
            
            with patch('asyncio.sleep'):
                result = await notifier.send(drift_alert)
            
            assert result is False
            assert mock_post.call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_send_timeout_error(self, webhook_config, drift_alert):
        notifier = WebhookNotifier(webhook_config)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = asyncio.TimeoutError()
            
            result = await notifier.send(drift_alert)
            
            assert result is False

    @pytest.mark.asyncio
    async def test_payload_formatting(self, webhook_config, drift_alert):
        notifier = WebhookNotifier(webhook_config)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            await notifier.send(drift_alert)
            
            _, kwargs = mock_post.call_args
            payload = kwargs['json']
            
            assert 'alert_type' in payload
            assert 'timestamp' in payload
            assert payload['severity'] == 'HIGH'
            assert payload['current_value'] == 0.75
            assert payload['threshold'] == 0.85