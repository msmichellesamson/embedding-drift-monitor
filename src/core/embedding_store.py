import logging
import time
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class EmbeddingStore:
    def __init__(self, connection_string: str, min_conn: int = 5, max_conn: int = 20):
        self.connection_string = connection_string
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                min_conn, max_conn, connection_string, cursor_factory=RealDictCursor
            )
            logger.info(f"Initialized connection pool with {min_conn}-{max_conn} connections")
        except psycopg2.Error as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = self.pool.getconn()
            if conn.closed:
                logger.warning("Got closed connection from pool, creating new one")
                conn = psycopg2.connect(self.connection_string, cursor_factory=RealDictCursor)
            yield conn
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn and not conn.closed:
                self.pool.putconn(conn)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def store_embedding(self, model_name: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> str:
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO embeddings (model_name, embedding_vector, metadata, created_at)
                    VALUES (%s, %s, %s, NOW())
                    RETURNING id
                    """,
                    (model_name, embedding.tolist(), metadata)
                )
                embedding_id = cursor.fetchone()['id']
                conn.commit()
                logger.debug(f"Stored embedding {embedding_id} for model {model_name}")
                return embedding_id
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_recent_embeddings(self, model_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, embedding_vector, metadata, created_at
                    FROM embeddings
                    WHERE model_name = %s AND created_at >= NOW() - INTERVAL '%s hours'
                    ORDER BY created_at DESC
                    """,
                    (model_name, hours)
                )
                embeddings = cursor.fetchall()
                logger.debug(f"Retrieved {len(embeddings)} embeddings for {model_name}")
                return embeddings
    
    def close(self):
        if self.pool:
            self.pool.closeall()
            logger.info("Closed all database connections")
