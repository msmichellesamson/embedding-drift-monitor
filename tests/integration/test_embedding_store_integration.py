import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, patch
from src.core.embedding_store import EmbeddingStore
from src.core.redis_cache import RedisCache


@pytest.fixture
def mock_redis():
    return AsyncMock(spec=RedisCache)


@pytest.fixture
def embedding_store(mock_redis):
    with patch('asyncpg.connect') as mock_connect:
        mock_conn = AsyncMock()
        mock_connect.return_value = mock_conn
        store = EmbeddingStore(db_url="postgresql://test", redis_cache=mock_redis)
        store.conn = mock_conn
        return store, mock_conn


@pytest.mark.asyncio
async def test_store_and_retrieve_flow(embedding_store):
    """Test complete flow: store embedding -> retrieve -> verify cache hit"""
    store, mock_conn = embedding_store
    
    # Mock data
    model_id = "test-model"
    embedding = np.random.random(768).astype(np.float32)
    metadata = {"source": "test", "timestamp": "2024-01-01"}
    
    # Mock database responses
    mock_conn.execute.return_value = None
    mock_conn.fetch.return_value = [{
        "id": 1,
        "embedding": embedding.tobytes(),
        "metadata": metadata,
        "created_at": "2024-01-01T00:00:00"
    }]
    
    # Store embedding
    await store.store_embedding(model_id, embedding, metadata)
    
    # Verify database call
    mock_conn.execute.assert_called_once()
    call_args = mock_conn.execute.call_args[0]
    assert "INSERT INTO embeddings" in call_args[0]
    
    # Retrieve embeddings
    results = await store.get_embeddings(model_id, limit=1)
    
    assert len(results) == 1
    assert np.array_equal(results[0]["embedding"], embedding)
    assert results[0]["metadata"] == metadata


@pytest.mark.asyncio
async def test_cache_integration(embedding_store):
    """Test Redis cache integration during retrieval"""
    store, mock_conn = embedding_store
    model_id = "cached-model"
    
    # Mock cache hit
    cached_data = [{"embedding": np.random.random(768), "metadata": {}}]
    store.redis_cache.get.return_value = cached_data
    
    results = await store.get_embeddings(model_id)
    
    # Should return cached data without DB call
    assert results == cached_data
    mock_conn.fetch.assert_not_called()
    store.redis_cache.get.assert_called_once_with(f"embeddings:{model_id}")


@pytest.mark.asyncio
async def test_database_failure_handling(embedding_store):
    """Test handling of database connection failures"""
    store, mock_conn = embedding_store
    
    # Simulate DB failure
    mock_conn.execute.side_effect = Exception("Connection failed")
    
    with pytest.raises(Exception, match="Connection failed"):
        await store.store_embedding("test", np.random.random(768), {})


@pytest.mark.asyncio
async def test_concurrent_operations(embedding_store):
    """Test concurrent store operations"""
    store, mock_conn = embedding_store
    
    # Prepare multiple embeddings
    tasks = []
    for i in range(5):
        embedding = np.random.random(768).astype(np.float32)
        task = store.store_embedding(f"model-{i}", embedding, {"batch": i})
        tasks.append(task)
    
    # Execute concurrently
    await asyncio.gather(*tasks)
    
    # Verify all calls were made
    assert mock_conn.execute.call_count == 5